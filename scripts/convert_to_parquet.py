import argparse

from soc_triage.data.download import find_raw_file
from soc_triage.data.to_parquet import csv_to_parquet
from soc_triage.utils.config import PROJECT_ROOT, load_config


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Convert a raw GUIDE CSV to Parquet")
    parser.add_argument("name", choices=["train", "test"])
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    csv_path = find_raw_file(raw_dir, cfg["data"][f"{args.name}_file"])
    out = PROJECT_ROOT / cfg["paths"]["processed_dir"] / f"{args.name}.parquet"
    rows = csv_to_parquet(csv_path, out)
    print(f"{rows:,} rows -> {out}")


if __name__ == "__main__":
    main()
