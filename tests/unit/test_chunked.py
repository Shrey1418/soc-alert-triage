import numpy as np
import pandas as pd

from soc_triage.features.aggregate import build_incident_table
from soc_triage.features.chunked import build_incident_table_chunked

GRADES = ["TruePositive", "BenignPositive", "FalsePositive"]
BASE = [
    "OrgId",
    "IncidentId",
    "AlertId",
    "DetectorId",
    "AlertTitle",
    "Timestamp",
    "MitreTechniques",
    "IncidentGrade",
]


def synthetic_evidence(n_incidents=80, seed=0):
    rng = np.random.default_rng(seed)
    techniques = ["T1078;T1566", "T1059", None]
    rows = []
    for inc in range(n_incidents):
        org = int(rng.integers(0, 4))
        grade = GRADES[inc % 3] if inc % 10 else None
        for _ in range(int(rng.integers(2, 7))):
            day = int(rng.integers(1, 20))
            hour = int(rng.integers(0, 24))
            rows.append(
                {
                    "OrgId": org,
                    "IncidentId": inc,
                    "AlertId": inc * 10 + int(rng.integers(0, 6)),
                    "DetectorId": int(rng.integers(1, 4)),
                    "AlertTitle": int(rng.integers(1, 8)),
                    "Timestamp": f"2024-06-{day:02d}T{hour:02d}:00:00.000Z",
                    "MitreTechniques": techniques[int(rng.integers(0, 3))],
                    "IncidentGrade": grade,
                    "Category": ["A", "B", "C"][int(rng.integers(0, 3))],
                    "IpAddress": f"10.0.0.{int(rng.integers(1, 6))}",
                }
            )
    return pd.DataFrame(rows)


def test_chunked_build_matches_in_memory(tmp_path):
    df = synthetic_evidence()
    path = tmp_path / "evidence.parquet"
    df.to_parquet(path, index=False)

    whole = build_incident_table(df, ["IpAddress"], ["Category"])
    chunked = build_incident_table_chunked(
        path, tmp_path / "buckets", BASE, ["IpAddress"], ["Category"], 4
    )

    key = ["OrgId", "IncidentId"]
    a = whole.sort_values(key).reset_index(drop=True).sort_index(axis=1)
    b = chunked.sort_values(key).reset_index(drop=True).sort_index(axis=1)
    a["IncidentGrade"] = a["IncidentGrade"].fillna("none")
    b["IncidentGrade"] = b["IncidentGrade"].fillna("none")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
