import argparse
import json

import joblib
import pandas as pd

from soc_triage.models.split import time_split
from soc_triage.monitoring.check import format_report, run_drift_check
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config("base", "drift")
    parser = argparse.ArgumentParser(description="Compare the newest window with a reference")
    parser.add_argument("--input", default="data/processed/incidents_full.parquet")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--reference", choices=["valid", "train"], default="valid")
    args = parser.parse_args()

    bundle = joblib.load(PROJECT_ROOT / args.model)
    table = pd.read_parquet(PROJECT_ROOT / args.input)
    train, valid, test = time_split(table, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    reference = {"valid": valid, "train": train}[args.reference]

    result = run_drift_check(bundle, reference, valid, test, cfg)
    print(f"Reference: {args.reference} ({len(reference):,} incidents)")
    print(f"Current: test ({len(test):,} incidents)")
    print(format_report(result))

    out_dir = PROJECT_ROOT / "artifacts"
    result["features"].to_csv(out_dir / "drift_features.csv", index=False)
    summary = {key: result[key] for key in ("unseen", "prediction", "decision")}
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Saved drift report to %s", out_dir)


if __name__ == "__main__":
    main()
