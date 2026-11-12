import pandas as pd


def add_time_features(table: pd.DataFrame) -> pd.DataFrame:
    start = table["start_time"]
    table["duration_min"] = (table["end_time"] - start).dt.total_seconds() / 60
    table["start_hour"] = start.dt.hour
    table["start_dow"] = start.dt.dayofweek
    table["start_is_weekend"] = (table["start_dow"] >= 5).astype("int8")
    return table
