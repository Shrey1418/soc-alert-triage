from soc_triage.data.download import find_raw_file
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    find_raw_file(raw_dir, cfg["data"]["train_file"])
    try:
        find_raw_file(raw_dir, cfg["data"]["test_file"])
    except DataValidationError as exc:
        logger.warning("Optional test file missing: %s", exc)


if __name__ == "__main__":
    main()
