import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from soc_triage.api.main import app
from soc_triage.api.predictor import ACTIONS
from soc_triage.features.aggregate import build_incident_table
from soc_triage.models.train import run_training, transform_table
from soc_triage.utils.config import load_config


def make_evidence(n_incidents=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for inc in range(n_incidents):
        n_alerts = int(rng.integers(1, 9))
        if n_alerts >= 6:
            grade = "TruePositive"
        elif n_alerts >= 3:
            grade = "BenignPositive"
        else:
            grade = "FalsePositive"
        day = 1 + inc * 25 // n_incidents
        for k in range(n_alerts):
            hour = int(rng.integers(0, 24))
            rows.append(
                {
                    "OrgId": inc % 4,
                    "IncidentId": inc,
                    "AlertId": inc * 100 + k,
                    "DetectorId": int(rng.integers(1, 4)),
                    "AlertTitle": int(rng.integers(1, 6)),
                    "Timestamp": f"2024-06-{day:02d}T{hour:02d}:00:00.000Z",
                    "MitreTechniques": ["T1078;T1566", "T1059", None][int(rng.integers(0, 3))],
                    "IncidentGrade": grade,
                    "Category": ["A", "B", "C"][int(rng.integers(0, 3))],
                    "IpAddress": f"10.0.0.{int(rng.integers(1, 6))}",
                }
            )
    return pd.DataFrame(rows)


def payload(evidence, incident_ids):
    rows = evidence[evidence["IncidentId"].isin(incident_ids)].drop(columns=["IncidentGrade"])
    return {"evidence": json.loads(rows.to_json(orient="records"))}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    evidence = make_evidence()
    table = build_incident_table(evidence, ["IpAddress"], ["Category"])
    cfg = load_config()
    cfg["model"]["lightgbm"]["n_estimators"] = 40
    run = run_training(table, cfg)

    bundle = {
        "model": run["models"]["lightgbm"],
        "encoder": run["encoder"],
        "classes": run["classes"],
        "feature_names": run["feature_names"],
    }
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(bundle, path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_PATH", str(path))
        with TestClient(app) as client:
            yield client, evidence, table, bundle


def test_health_and_root(served):
    client = served[0]
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert "message" in client.get("/").json()


def test_predict_returns_one_decision_per_incident(served):
    client, evidence, _, _ = served
    response = client.post("/predict", json=payload(evidence, [200, 201, 202]))

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["IncidentId"] for r in results] == [200, 201, 202]
    for r in results:
        assert r["action"] == ACTIONS[r["decision"]]
        assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_serving_matches_training_features(served):
    client, evidence, table, bundle = served
    incident = 57
    row = table[table["IncidentId"] == incident]
    X = transform_table(row, bundle["encoder"], bundle["feature_names"])
    expected = bundle["model"].predict_proba(X)[0]

    response = client.post("/predict", json=payload(evidence, [incident]))
    got = list(response.json()["results"][0]["probabilities"].values())
    assert np.allclose(got, expected, atol=1e-3)


def test_bad_requests_are_rejected(served):
    client, evidence, _, _ = served
    assert client.post("/predict", json={"evidence": []}).status_code == 422
    assert client.post("/predict", json={"evidence": [{"OrgId": 1}]}).status_code == 422

    bad = payload(evidence, [5])
    for row in bad["evidence"]:
        row["Timestamp"] = "not a time"
    assert client.post("/predict", json=bad).status_code == 422
