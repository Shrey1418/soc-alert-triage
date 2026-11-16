import numpy as np
import pandas as pd

from soc_triage.models.train import run_training
from soc_triage.utils.config import load_config

GRADES = ["TruePositive", "BenignPositive", "FalsePositive"]


def synthetic_table(n=900, seed=0):
    rng = np.random.default_rng(seed)
    n_alerts = rng.integers(1, 12, n)
    grade = np.where(n_alerts >= 4, "BenignPositive", "FalsePositive")
    grade = np.where(n_alerts >= 8, "TruePositive", grade)
    noisy = rng.random(n) < 0.05
    grade = np.where(noisy, rng.choice(GRADES, n), grade)

    start = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "OrgId": rng.integers(0, 5, n),
            "IncidentId": np.arange(n),
            "start_time": start,
            "end_time": start + pd.Timedelta(hours=1),
            "IncidentGrade": grade,
            "dominant_alert_title": rng.integers(0, 40, n),
            "dominant_detector": rng.integers(0, 10, n),
            "n_alerts": n_alerts,
            "n_evidence_rows": n_alerts * rng.integers(1, 4, n),
        }
    )


def test_run_training_end_to_end():
    cfg = load_config()
    cfg["model"]["lightgbm"]["n_estimators"] = 60
    run = run_training(synthetic_table(), cfg)

    results = run["results"]
    assert set(results) == {"majority", "logreg", "lightgbm"}
    lgbm = results["lightgbm"]["test"]["argmax"]["macro_f1"]
    majority = results["majority"]["test"]["argmax"]["macro_f1"]
    assert lgbm > majority + 0.2
    assert run["feature_names"][-1].startswith("dominant_detector__te_")
