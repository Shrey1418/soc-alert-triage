import numpy as np
import pandas as pd
import pytest

from soc_triage.models.split import time_split
from soc_triage.models.train import run_training
from soc_triage.monitoring.check import run_drift_check
from soc_triage.monitoring.drift import (
    feature_drift,
    psi_categorical,
    psi_numeric,
    status,
    unseen_rate,
)
from soc_triage.monitoring.retrain import retrain_decision
from soc_triage.monitoring.simulate import apply_shift
from soc_triage.utils.config import load_config

GRADES = ["TruePositive", "BenignPositive", "FalsePositive"]
CFG = {"psi": {"warn": 0.1, "alert": 0.25}, "ks": {"warn": 0.1, "alert": 0.2}}


def test_psi_numeric_zero_for_same_small_for_resample_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)

    assert psi_numeric(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_numeric(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi_numeric(ref, rng.normal(1.5, 1, 5000)) > 0.5


def test_psi_categorical_flags_new_categories():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 60)

    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(ref, cur) > 0.25


def test_binary_feature_uses_value_shares():
    ref = pd.Series([0] * 90 + [1] * 10)
    cur = pd.Series([0] * 50 + [1] * 50)
    assert psi_numeric(ref, cur) > 0.25


def test_unseen_rate_and_status_levels():
    assert unseen_rate([1, 2, 3], [1, 2, 9, 9]) == pytest.approx(0.5)
    assert status(0.05, 0.1, 0.25) == "ok"
    assert status(0.1, 0.1, 0.25) == "warn"
    assert status(0.25, 0.1, 0.25) == "alert"
    assert status(float("nan"), 0.1, 0.25) == "n/a"


def test_feature_drift_flags_only_the_shifted_column():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 4000),
            "moves": rng.normal(0, 1, 4000),
            "title": rng.integers(0, 20, 4000),
        }
    )
    cur = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 4000),
            "moves": rng.normal(2, 1, 4000),
            "title": rng.integers(0, 20, 4000),
        }
    )
    out = feature_drift(ref, cur, ["stable", "moves"], ["title"], CFG).set_index("feature")

    assert out.loc["stable", "status"] == "ok"
    assert out.loc["moves", "status"] == "alert"
    assert out.loc["title", "status"] == "ok"


def test_retrain_decision_rules():
    features = pd.DataFrame({"feature": list("abcde"), "status": ["alert"] * 5})
    quiet = {"psi": 0.01, "status": "ok"}
    unseen_ok = {"dominant_alert_title": {"rate": 0.05, "status": "ok"}}
    rules = {"min_current_incidents": 100, "retrain": {"min_alert_features": 5}}

    assert retrain_decision(features, quiet, unseen_ok, 1000, rules)["retrain"] is True
    assert retrain_decision(features.head(2), quiet, unseen_ok, 1000, rules)["retrain"] is False

    small = retrain_decision(features, quiet, unseen_ok, 50, rules)
    assert small["retrain"] is False
    assert small["enough_data"] is False

    unseen_bad = {"dominant_alert_title": {"rate": 0.4, "status": "alert"}}
    assert retrain_decision(features.head(1), quiet, unseen_bad, 1000, rules)["retrain"] is True


def test_apply_shift_changes_titles_and_counts_without_touching_the_input():
    table = pd.DataFrame(
        {"dominant_alert_title": np.arange(5000) % 10, "n_evidence_rows": np.ones(5000, dtype=int)}
    )
    shifted = apply_shift(table, 0.3, 3, seed=0)

    share = (shifted["dominant_alert_title"] >= 900_000_000).mean()
    assert 0.25 < share < 0.35
    assert (shifted["n_evidence_rows"] == 3).all()
    assert (table["n_evidence_rows"] == 1).all()
    assert (table["dominant_alert_title"] < 10).all()


def synthetic_table(n=3000, seed=0):
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


@pytest.fixture(scope="module")
def trained():
    cfg = load_config("base", "drift")
    cfg["model"]["lightgbm"]["n_estimators"] = 40
    cfg["drift"]["min_current_incidents"] = 100
    table = synthetic_table()
    run = run_training(table, cfg)
    bundle = {
        "model": run["models"]["lightgbm"],
        "encoder": run["encoder"],
        "classes": run["classes"],
        "feature_names": run["feature_names"],
    }
    _, valid, test = time_split(table, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    return cfg, bundle, valid, test


def test_drift_check_is_quiet_on_clean_data_and_loud_on_shifted_data(trained):
    cfg, bundle, valid, test = trained

    clean = run_drift_check(bundle, valid, valid, test, cfg)
    shifted = run_drift_check(bundle, valid, valid, apply_shift(test, 0.5, 4, seed=0), cfg)

    assert clean["decision"]["retrain"] is False
    assert shifted["unseen"]["dominant_alert_title"]["status"] == "alert"
    assert shifted["decision"]["retrain"] is True
