import argparse
from pathlib import Path

import pandas as pd

from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def profile(path: Path, nrows: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
            "example": df.iloc[0],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick profile of a raw CSV file")
    parser.add_argument("path", type=Path)
    parser.add_argument("--nrows", type=int, default=100_000)
    args = parser.parse_args()

    size_gb = args.path.stat().st_size / 1024**3
    logger.info("File: %s (%.2f GB)", args.path, size_gb)

    summary = profile(args.path, args.nrows)
    options = ("display.max_rows", None, "display.width", 200, "display.max_colwidth", 40)
    with pd.option_context(*options):
        print(summary)


if __name__ == "__main__":
    main()
