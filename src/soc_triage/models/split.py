import pandas as pd

from soc_triage.data.schema import TARGET
from soc_triage.utils.exceptions import ConfigError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def time_split(table, train_frac, valid_frac):
    if train_frac <= 0 or valid_frac <= 0 or train_frac + valid_frac >= 1:
        raise ConfigError("train_frac and valid_frac must be > 0 and sum to less than 1")

    usable = table[table[TARGET].notna() & table["start_time"].notna()]
    dropped = len(table) - len(usable)
    if dropped:
        logger.warning("Dropped %d incidents with no label or no start time", dropped)

    usable = usable.sort_values("start_time", kind="stable")
    n = len(usable)
    i = round(n * train_frac)
    j = round(n * (train_frac + valid_frac))
    return usable.iloc[:i], usable.iloc[i:j], usable.iloc[j:]


def summarize_splits(train, valid, test):
    rows = []
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        shares = part[TARGET].value_counts(normalize=True).round(4).to_dict()
        rows.append(
            {
                "split": name,
                "incidents": len(part),
                "start": part["start_time"].min(),
                "end": part["start_time"].max(),
                **shares,
            }
        )
    return pd.DataFrame(rows)
