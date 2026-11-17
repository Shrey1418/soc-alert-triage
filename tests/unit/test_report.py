import numpy as np

from soc_triage.models.cost import cost_matrix
from soc_triage.models.report import calibration_table, scaled_cost, sensitivity_table

CLASSES = ["BenignPositive", "FalsePositive", "TruePositive"]
COST_CFG = {
    "TruePositive": {"TruePositive": 0, "BenignPositive": 20, "FalsePositive": 50},
    "BenignPositive": {"TruePositive": 2, "BenignPositive": 0, "FalsePositive": 1},
    "FalsePositive": {"TruePositive": 2, "BenignPositive": 1, "FalsePositive": 0},
}


def test_scaled_cost_only_changes_the_missed_threat_row():
    base = cost_matrix(COST_CFG, CLASSES)
    scaled = scaled_cost(COST_CFG, CLASSES, 2)

    assert scaled[2, 1] == 2 * base[2, 1]
    assert scaled[2, 0] == 2 * base[2, 0]
    assert scaled[2, 2] == 0
    assert (scaled[:2] == base[:2]).all()


def test_sensitivity_table_known_values():
    y = np.array(["TruePositive", "FalsePositive"])
    proba = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    out = sensitivity_table(y, proba, COST_CFG, CLASSES, factors=(1, 2))

    assert list(out["miss_penalty_x"]) == [1, 2]
    assert list(out["argmax"]) == [25.0, 50.0]
    assert list(out["escalate_all"]) == [1.0, 1.0]
    assert list(out["min_cost"]) == [1.0, 1.0]
    assert list(out["escalated_pct"]) == [100.0, 100.0]


def test_calibration_table_bins_and_rates():
    y = np.array(["TruePositive"] * 5 + ["FalsePositive"] * 5)
    proba = np.zeros((10, 3))
    proba[:, 2] = [0.9] * 5 + [0.1] * 5
    out = calibration_table(y, proba, CLASSES, n_bins=2)

    assert out["incidents"].tolist() == [5, 5]
    assert out["actual_rate"].tolist() == [0.0, 1.0]
    assert out["mean_predicted"].tolist() == [0.1, 0.9]
