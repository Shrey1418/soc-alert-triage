from pathlib import Path

from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def find_raw_file(raw_dir: Path, name: str) -> Path:
    matches = list(Path(raw_dir).rglob(name))
    if not matches:
        raise DataValidationError(
            f"{name} not found under {raw_dir}. "
            "Download the GUIDE dataset from Kaggle and extract it into that folder."
        )
    path = matches[0]
    logger.info("Found %s (%.2f GB)", path, path.stat().st_size / 1024**3)
    return path
