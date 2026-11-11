from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from soc_triage.data.schema import INCIDENT_KEY
from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)
_BUCKETS = 10_000


def sample_incidents(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    key = df[INCIDENT_KEY[0]].astype(str) + "_" + df[INCIDENT_KEY[1]].astype(str)
    hashed = pd.util.hash_pandas_object(key, index=False, hash_key=f"{seed:016d}")
    return df[(hashed % _BUCKETS) < round(fraction * _BUCKETS)]


def make_sample(
    parquet_path: Path, out_path: Path, fraction: float, seed: int, batch_size: int = 500_000
) -> pd.DataFrame:
    parts = []
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=batch_size):
        kept = sample_incidents(batch.to_pandas(), fraction, seed)
        if len(kept):
            parts.append(kept)
    if not parts:
        raise DataValidationError("Sample is empty. Increase the fraction (minimum 0.0001).")

    sample = pd.concat(parts, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)
    logger.info("Sample: %s rows -> %s", f"{len(sample):,}", out_path)
    return sample
