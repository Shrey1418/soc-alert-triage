import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from soc_triage.data.schema import INT_COLUMNS, check_required_columns
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def csv_to_parquet(csv_path: Path, parquet_path: Path, block_mb: int = 64) -> int:
    csv_path, parquet_path = Path(csv_path), Path(parquet_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    check_required_columns(header)

    column_types = {c: pa.int64() if c in INT_COLUMNS else pa.string() for c in header}
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_mb * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=column_types, strings_can_be_null=True),
    )

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with pq.ParquetWriter(parquet_path, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    logger.info("Wrote %s rows to %s", f"{rows:,}", parquet_path)
    return rows
