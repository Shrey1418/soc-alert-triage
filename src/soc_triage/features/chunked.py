import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from soc_triage.data.schema import INCIDENT_KEY
from soc_triage.features.aggregate import build_incident_table, read_columns
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def global_category_values(
    parquet_path, category_cols, top_n=12, max_categories=50, batch_size=1_000_000
):
    if not category_cols:
        return {}
    totals = {col: pd.Series(dtype="float64") for col in category_cols}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(category_cols)):
        frame = batch.to_pandas()
        for col in category_cols:
            totals[col] = totals[col].add(frame[col].value_counts(), fill_value=0)

    values = {}
    for col, counts in totals.items():
        if len(counts) > max_categories:
            logger.warning("Skipping %s: more than %d distinct values", col, max_categories)
            continue
        values[col] = list(counts.nlargest(top_n).index)
    return values


def partition_by_incident(parquet_path, work_dir, columns, n_buckets, batch_size=1_000_000):
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    writers = {}
    try:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
            table = pa.Table.from_batches([batch])
            keys = table.select(INCIDENT_KEY).to_pandas()
            bucket = (pd.util.hash_pandas_object(keys, index=False) % n_buckets).to_numpy()
            for k in range(n_buckets):
                mask = bucket == k
                if not mask.any():
                    continue
                if k not in writers:
                    path = work_dir / f"bucket_{k:03d}.parquet"
                    writers[k] = pq.ParquetWriter(path, table.schema, compression="zstd")
                writers[k].write_table(table.filter(pa.array(mask)))
    finally:
        for writer in writers.values():
            writer.close()
    return sorted(work_dir.glob("bucket_*.parquet"))


def build_incident_table_chunked(
    parquet_path,
    work_dir,
    base_columns,
    entity_cols,
    category_cols,
    n_buckets,
    top_n=12,
    max_categories=50,
):
    category_values = global_category_values(parquet_path, category_cols, top_n, max_categories)
    columns = list(dict.fromkeys([*base_columns, *entity_cols, *category_cols]))
    categorical = [*entity_cols, *category_cols, "MitreTechniques"]

    bucket_files = partition_by_incident(parquet_path, work_dir, columns, n_buckets)
    parts = []
    for i, path in enumerate(bucket_files, start=1):
        df = read_columns(path, columns, categorical)
        part = build_incident_table(df, entity_cols, category_cols, category_values=category_values)
        parts.append(part)
        logger.info("Bucket %d/%d done: %s incidents", i, len(bucket_files), f"{len(part):,}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return pd.concat(parts, ignore_index=True)
