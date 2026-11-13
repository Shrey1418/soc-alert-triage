import pandas as pd
import pytest

from soc_triage.features.aggregate import build_incident_table
from soc_triage.models.split import time_split
from soc_triage.utils.exceptions import ConfigError


def frame():
    return pd.DataFrame(
        {
            "OrgId": [1, 1, 1, 1, 2],
            "IncidentId": [10, 10, 10, 11, 10],
            "AlertId": [100, 100, 101, 102, 200],
            "DetectorId": [1, 1, 2, 1, 3],
            "AlertTitle": [5, 5, 6, 7, 8],
            "Timestamp": [
                "2024-06-04T10:00:00.000Z",
                "2024-06-04T10:30:00.000Z",
                "2024-06-04T12:00:00.000Z",
                "2024-06-05T09:00:00.000Z",
                "2024-06-06T08:00:00.000Z",
            ],
            "MitreTechniques": ["T1078;T1566", "T1078", None, "T1059", None],
            "IncidentGrade": [
                "TruePositive",
                "TruePositive",
                "TruePositive",
                "FalsePositive",
                None,
            ],
            "Category": ["A", "A", "B", "A", "B"],
            "IpAddress": ["1.1.1.1", "1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"],
        }
    )


def test_incident_table_counts():
    table = build_incident_table(frame(), ["IpAddress"], ["Category"])
    assert len(table) == 3

    row = table[(table["OrgId"] == 1) & (table["IncidentId"] == 10)].iloc[0]
    assert row["n_evidence_rows"] == 3
    assert row["n_alerts"] == 2
    assert row["n_detectors"] == 2
    assert row["n_mitre_techniques"] == 2
    assert row["n_distinct_IpAddress"] == 2
    assert row["duration_min"] == 120
    assert row["Category__A"] == pytest.approx(2 / 3)


def test_unlabelled_incident_keeps_null_label():
    table = build_incident_table(frame(), ["IpAddress"], ["Category"])
    other_org = table[(table["OrgId"] == 2) & (table["IncidentId"] == 10)].iloc[0]
    assert pd.isna(other_org["IncidentGrade"])


def test_time_split_is_ordered_and_disjoint():
    n = 100
    table = pd.DataFrame(
        {
            "OrgId": 1,
            "IncidentId": range(n),
            "start_time": pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")[::-1],
            "IncidentGrade": ["TruePositive"] * n,
        }
    )
    train, valid, test = time_split(table, 0.7, 0.15)

    assert (len(train), len(valid), len(test)) == (70, 15, 15)
    assert train["start_time"].max() <= valid["start_time"].min()
    assert valid["start_time"].max() <= test["start_time"].min()
    assert not set(train["IncidentId"]) & set(test["IncidentId"])


def test_time_split_drops_unlabelled_and_rejects_bad_fractions():
    table = pd.DataFrame(
        {
            "OrgId": [1, 1, 1, 1],
            "IncidentId": [1, 2, 3, 4],
            "start_time": pd.to_datetime(
                ["2024-06-01", "2024-06-02", None, "2024-06-04"], utc=True
            ),
            "IncidentGrade": ["TruePositive", None, "FalsePositive", "BenignPositive"],
        }
    )
    train, valid, test = time_split(table, 0.5, 0.25)
    assert len(train) + len(valid) + len(test) == 2

    with pytest.raises(ConfigError):
        time_split(table, 0.9, 0.2)
