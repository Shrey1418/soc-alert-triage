from collections.abc import Iterable

from soc_triage.utils.exceptions import DataValidationError

TARGET = "IncidentGrade"
TARGET_CLASSES = ("TruePositive", "BenignPositive", "FalsePositive")
INCIDENT_KEY = ["OrgId", "IncidentId"]
TIME_COLUMN = "Timestamp"

INT_COLUMNS = ["Id", "OrgId", "IncidentId", "AlertId", "DetectorId", "AlertTitle"]

REQUIRED_COLUMNS = [
    *INT_COLUMNS,
    "Timestamp",
    "Category",
    "MitreTechniques",
    "IncidentGrade",
    "EntityType",
    "EvidenceRole",
]


def check_required_columns(columns: Iterable[str]) -> None:
    present = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. "
            "Check the dataset version or update REQUIRED_COLUMNS in schema.py."
        )
