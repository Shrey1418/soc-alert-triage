import argparse

from soc_triage.data.sample import make_sample
from soc_triage.data.validate import validate_frame
from soc_triage.utils.config import PROJECT_ROOT, load_config


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Incident-level sample of the train Parquet")
    parser.add_argument("--fraction", type=float, default=cfg["sampling"]["fraction"])
    parser.add_argument("--out", default="data/processed/train_sample.parquet")
    args = parser.parse_args()

    src = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "train.parquet"
    sample = make_sample(src, PROJECT_ROOT / args.out, args.fraction, cfg["sampling"]["seed"])
    print(f"{len(sample):,} rows -> {args.out}")
    print(validate_frame(sample))


if __name__ == "__main__":
    main()
