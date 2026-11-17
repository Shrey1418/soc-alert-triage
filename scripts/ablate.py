import argparse

import pandas as pd

from soc_triage.models.train import run_training
from soc_triage.utils.config import PROJECT_ROOT, load_config

VARIANTS = {
    "all features": {},
    "no time-of-day features": {"drop_cols": ("start_hour", "start_dow", "start_is_weekend")},
    "no alert title or detector": {"encode_cols": ()},
}


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Retrain with feature groups removed")
    parser.add_argument("--input", default="data/processed/incidents_sample.parquet")
    args = parser.parse_args()

    table = pd.read_parquet(PROJECT_ROOT / args.input)
    rows = []
    for name, options in VARIANTS.items():
        run = run_training(table, cfg, **options)
        scored = run["results"]["lightgbm"]
        for split in ("valid", "test"):
            rows.append(
                {
                    "variant": name,
                    "split": split,
                    "macro_f1": scored[split]["argmax"]["macro_f1"],
                    "cost_min_cost": scored[split]["min_cost"]["cost_per_incident"],
                }
            )
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("cost per incident if every incident is escalated:", run["escalate_all"])


if __name__ == "__main__":
    main()
