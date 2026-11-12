import argparse

from soc_triage.features.aggregate import build_incident_table, read_columns, usable_columns
from soc_triage.models.split import summarize_splits, time_split
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)

BASE_COLUMNS = [
    "OrgId",
    "IncidentId",
    "AlertId",
    "DetectorId",
    "AlertTitle",
    "Timestamp",
    "MitreTechniques",
    "IncidentGrade",
]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Build the incident-level feature table")
    parser.add_argument("--input", default="data/processed/train_sample.parquet")
    parser.add_argument("--out", default="data/processed/incidents_sample.parquet")
    args = parser.parse_args()

    src = PROJECT_ROOT / args.input
    entity_cols, missing_entity = usable_columns(src, cfg["features"]["entity_columns"])
    category_cols, missing_cat = usable_columns(src, cfg["features"]["category_columns"])
    if missing_entity or missing_cat:
        logger.warning("Configured columns not in data: %s", missing_entity + missing_cat)

    wanted = BASE_COLUMNS + entity_cols + category_cols
    categorical = [*entity_cols, *category_cols, "MitreTechniques"]
    df = read_columns(src, wanted, categorical)

    table = build_incident_table(df, entity_cols, category_cols)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"{len(table):,} incidents x {table.shape[1]} columns -> {out}")

    split_cfg = cfg["split"]
    train, valid, test = time_split(table, split_cfg["train_frac"], split_cfg["valid_frac"])
    print(summarize_splits(train, valid, test).to_string(index=False))


if __name__ == "__main__":
    main()
