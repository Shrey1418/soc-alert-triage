import re

import pandas as pd
import pyarrow.parquet as pq

from soc_triage.data.schema import INCIDENT_KEY, TARGET
from soc_triage.features.time_features import add_time_features
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


def usable_columns(parquet_path, wanted):
    names = set(pq.read_schema(parquet_path).names)
    found = [c for c in wanted if c in names]
    missing = [c for c in wanted if c not in names]
    return found, missing


def read_columns(parquet_path, columns, categorical=()):
    table = pq.read_table(parquet_path, columns=list(dict.fromkeys(columns)))
    return table.to_pandas(categories=list(categorical))


def _clean(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")


def _mitre_counts(df):
    pairs = df[[*INCIDENT_KEY, "MitreTechniques"]].dropna().drop_duplicates()
    pairs = pairs.assign(tech=pairs["MitreTechniques"].astype(str).str.split(";"))
    exploded = pairs.explode("tech")
    return exploded.groupby(INCIDENT_KEY)["tech"].nunique().rename("n_mitre_techniques")


def _category_shares(df, col, top_n, max_categories):
    if df[col].nunique() > max_categories:
        logger.warning("Skipping %s: more than %d distinct values", col, max_categories)
        return None
    counts = pd.crosstab([df[k] for k in INCIDENT_KEY], df[col])
    top = counts.sum().nlargest(top_n).index
    shares = counts[top].div(counts.sum(axis=1), axis=0)
    shares.columns = [f"{col}__{_clean(v)}" for v in top]
    return shares


def build_incident_table(df, entity_cols, category_cols, top_n=12, max_categories=50):
    start = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce", format="ISO8601")
    work = df.assign(_ts=start)

    aggs = {
        "n_evidence_rows": ("AlertId", "size"),
        "n_alerts": ("AlertId", "nunique"),
        "n_detectors": ("DetectorId", "nunique"),
        "n_alert_titles": ("AlertTitle", "nunique"),
        "start_time": ("_ts", "min"),
        "end_time": ("_ts", "max"),
        TARGET: (TARGET, "first"),
    }
    for col in entity_cols:
        aggs[f"n_distinct_{col}"] = (col, "nunique")

    table = work.groupby(INCIDENT_KEY, sort=False).agg(**aggs)
    table = table.join(_mitre_counts(df), how="left")
    for col in category_cols:
        shares = _category_shares(df, col, top_n, max_categories)
        if shares is not None:
            table = table.join(shares, how="left")

    table = table.reset_index()
    fill_cols = [c for c in table.columns if c == "n_mitre_techniques" or "__" in c]
    table[fill_cols] = table[fill_cols].fillna(0)
    table[TARGET] = table[TARGET].astype("object")
    return add_time_features(table)
