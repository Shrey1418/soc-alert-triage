import argparse
import json

import joblib
import pandas as pd

from soc_triage.models.evaluate import summarize_results
from soc_triage.models.train import run_training
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train baselines and LightGBM")
    parser.add_argument("--input", default="data/processed/incidents_sample.parquet")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    table = pd.read_parquet(PROJECT_ROOT / args.input)
    run = run_training(table, cfg)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": run["models"]["lightgbm"],
        "encoder": run["encoder"],
        "classes": run["classes"],
        "feature_names": run["feature_names"],
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    metrics = {"results": run["results"], "escalate_everything_cost": run["escalate_all"]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    logger.info("Saved model and metrics to %s", out_dir)

    print(run["split_summary"].to_string(index=False))
    print(summarize_results(run["results"]).to_string(index=False))
    print("cost per incident if every incident is escalated:", run["escalate_all"])
    print(run["importance"].head(15).round(1).to_string())


if __name__ == "__main__":
    main()
