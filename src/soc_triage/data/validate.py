import pandas as pd

from soc_triage.data.schema import (
    INCIDENT_KEY,
    TARGET,
    TARGET_CLASSES,
    check_required_columns,
)
from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def validate_frame(df: pd.DataFrame) -> dict[str, object]:
    check_required_columns(df.columns)

    labels = df[TARGET].dropna()
    unknown = sorted(set(labels.unique()) - set(TARGET_CLASSES))
    if unknown:
        raise DataValidationError(f"Unexpected {TARGET} values: {unknown}")

    mixed = df.groupby(INCIDENT_KEY)[TARGET].nunique().gt(1).sum()
    report = {
        "rows": len(df),
        "incidents": int(df[INCIDENT_KEY].drop_duplicates().shape[0]),
        "target_null_pct": round(float(df[TARGET].isna().mean() * 100), 3),
        "duplicate_ids": int(df["Id"].duplicated().sum()),
        "incidents_with_mixed_labels": int(mixed),
        "class_share": labels.value_counts(normalize=True).round(4).to_dict(),
    }
    logger.info("Validation report: %s", report)
    return report
