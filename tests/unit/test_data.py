import pandas as pd
import pytest

from soc_triage.data.sample import sample_incidents
from soc_triage.data.schema import INCIDENT_KEY
from soc_triage.data.to_parquet import csv_to_parquet
from soc_triage.data.validate import validate_frame
from soc_triage.utils.exceptions import DataValidationError

GRADES = ["TruePositive", "BenignPositive", "FalsePositive"]


def make_frame(n_incidents: int = 200, rows_per_incident: int = 5) -> pd.DataFrame:
    rows = []
    row_id = 0
    for inc in range(n_incidents):
        for _ in range(rows_per_incident):
            row_id += 1
            rows.append(
                {
                    "Id": row_id,
                    "OrgId": inc % 7,
                    "IncidentId": inc,
                    "AlertId": row_id * 10,
                    "Timestamp": "2024-06-04T22:56:25.000Z",
                    "DetectorId": 1,
                    "AlertTitle": 2,
                    "Category": "InitialAccess",
                    "MitreTechniques": "T1078;T1566",
                    "IncidentGrade": GRADES[inc % 3],
                    "EntityType": "Ip",
                    "EvidenceRole": "Related",
                    "CountryCode": "31",
                }
            )
    return pd.DataFrame(rows)


def test_csv_to_parquet_roundtrip(tmp_path):
    df = make_frame()
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)
    out_path = tmp_path / "out.parquet"

    rows = csv_to_parquet(csv_path, out_path)
    out = pd.read_parquet(out_path)

    assert rows == len(df)
    assert out["IncidentId"].dtype == "int64"
    assert pd.api.types.is_string_dtype(out["CountryCode"])


def test_csv_with_missing_required_column_fails(tmp_path):
    csv_path = tmp_path / "raw.csv"
    make_frame().drop(columns=["IncidentGrade"]).to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError):
        csv_to_parquet(csv_path, tmp_path / "out.parquet")


def test_sampling_keeps_whole_incidents_and_is_deterministic():
    df = make_frame()
    first = sample_incidents(df, 0.3, 42)
    second = sample_incidents(df, 0.3, 42)

    sizes = first.groupby(INCIDENT_KEY).size()
    assert 0 < len(first) < len(df)
    assert (sizes == 5).all()
    assert first["Id"].tolist() == second["Id"].tolist()


def test_validate_frame_report():
    report = validate_frame(make_frame())
    assert report["incidents"] == 200
    assert report["incidents_with_mixed_labels"] == 0
    assert report["duplicate_ids"] == 0


def test_validate_frame_flags_unknown_class_and_mixed_labels():
    bad = make_frame()
    bad.loc[0, "IncidentGrade"] = "Unknown"
    with pytest.raises(DataValidationError):
        validate_frame(bad)

    mixed = make_frame()
    mixed.loc[0, "IncidentGrade"] = "FalsePositive"
    assert validate_frame(mixed)["incidents_with_mixed_labels"] == 1
