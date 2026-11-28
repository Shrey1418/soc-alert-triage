import argparse

import joblib
import pandas as pd

from soc_triage.data.schema import TARGET
from soc_triage.models.cost import cost_matrix, expected_cost, min_cost_predictions
from soc_triage.models.split import time_split
from soc_triage.monitoring.check import format_report, probabilities, run_drift_check
from soc_triage.monitoring.simulate import apply_shift
from soc_triage.utils.config import PROJECT_ROOT, load_config


def main() -> None:
    cfg = load_config("base", "drift")
    parser = argparse.ArgumentParser(description="Show the drift check on a simulated shift")
    parser.add_argument("--input", default="data/processed/incidents_full.parquet")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--unseen-share", type=float, default=0.3)
    parser.add_argument("--evidence-x", type=float, default=2.5)
    args = parser.parse_args()

    bundle = joblib.load(PROJECT_ROOT / args.model)
    classes = list(bundle["classes"])
    cost = cost_matrix(cfg["cost"], classes)
    table = pd.read_parquet(PROJECT_ROOT / args.input)
    _, valid, test = time_split(table, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    shifted_test = apply_shift(test, args.unseen_share, args.evidence_x, cfg["model"]["seed"])

    windows = {
        "test (as is)": test,
        "test (simulated shift)": shifted_test,
    }
    results = {
        name: run_drift_check(bundle, valid, valid, part, cfg) for name, part in windows.items()
    }

    rows = []
    for name, part in windows.items():
        result = results[name]
        counted = result["features"][result["features"]["counts_for_retrain"]]
        alerts = int((counted["status"] == "alert").sum())
        y = part[TARGET].reset_index(drop=True)
        decision = min_cost_predictions(probabilities(bundle, part), cost, classes)
        rows.append(
            {
                "window": name,
                "features_in_alert": alerts,
                "prediction_psi": result["prediction"]["psi"],
                "retrain": result["decision"]["retrain"],
                "cost_min_cost": expected_cost(y, decision, cost, classes),
                "cost_escalate_all": expected_cost(y, ["TruePositive"] * len(y), cost, classes),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
    print(
        f"Simulated shift: {args.unseen_share:.0%} of incidents get an alert title never seen "
        f"in training, and evidence rows are multiplied by {args.evidence_x}."
    )
    print(format_report(results["test (simulated shift)"]))


if __name__ == "__main__":
    main()
