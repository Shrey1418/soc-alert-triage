import argparse

import joblib
import pandas as pd

from soc_triage.data.schema import TARGET
from soc_triage.models.cost import cost_matrix, min_cost_predictions
from soc_triage.models.evaluate import evaluate_predictions
from soc_triage.models.report import calibration_table, sensitivity_table
from soc_triage.models.split import time_split
from soc_triage.models.train import transform_table
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate the saved model on one split")
    parser.add_argument("--input", default="data/processed/incidents_sample.parquet")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()

    bundle = joblib.load(PROJECT_ROOT / args.model)
    classes = bundle["classes"]
    table = pd.read_parquet(PROJECT_ROOT / args.input)
    _, valid, test = time_split(table, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    part = {"valid": valid, "test": test}[args.split]

    X = transform_table(part, bundle["encoder"], bundle["feature_names"])
    y = part[TARGET].reset_index(drop=True)
    proba = bundle["model"].predict_proba(X)

    cost = cost_matrix(cfg["cost"], classes)
    decision = min_cost_predictions(proba, cost, classes)
    result = evaluate_predictions(y, decision, classes)

    print(f"Split: {args.split} | incidents: {len(y):,}")
    print(f"Macro-F1 (min-cost rule): {result['macro_f1']:.4f}")
    print(f"Escalated as TruePositive: {(decision == 'TruePositive').mean():.1%}")
    print("\nPer class (min-cost rule):")
    print(pd.DataFrame(result["per_class"]).T.round(3).to_string())
    print("\nConfusion (rows = true grade, columns = decision):")
    print(pd.DataFrame(result["confusion"]).T.to_string())

    sensitivity = sensitivity_table(y, proba, cfg["cost"], classes)
    calibration = calibration_table(y, proba, classes)
    print("\nSensitivity to the cost of a missed threat:")
    print(sensitivity.to_string(index=False))
    print("\nCalibration of the TruePositive probability:")
    print(calibration.to_string())

    out_dir = PROJECT_ROOT / "artifacts"
    sensitivity.to_csv(out_dir / f"sensitivity_{args.split}.csv", index=False)
    calibration.to_csv(out_dir / f"calibration_{args.split}.csv")
    logger.info("Saved evaluation tables to %s", out_dir)


if __name__ == "__main__":
    main()
