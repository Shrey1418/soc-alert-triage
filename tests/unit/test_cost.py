import numpy as np
import pytest

from soc_triage.models.cost import cost_matrix, expected_cost, min_cost_predictions
from soc_triage.models.evaluate import evaluate_predictions

CLASSES = ["BenignPositive", "FalsePositive", "TruePositive"]
COST_CFG = {
    "TruePositive": {"TruePositive": 0, "BenignPositive": 20, "FalsePositive": 50},
    "BenignPositive": {"TruePositive": 2, "BenignPositive": 0, "FalsePositive": 1},
    "FalsePositive": {"TruePositive": 2, "BenignPositive": 1, "FalsePositive": 0},
}


def test_cost_matrix_order_and_expected_cost():
    cost = cost_matrix(COST_CFG, CLASSES)
    assert cost.shape == (3, 3)
    assert cost[2, 1] == COST_CFG["TruePositive"]["FalsePositive"]

    y = ["TruePositive", "FalsePositive"]
    assert expected_cost(y, y, cost, CLASSES) == 0
    wrong = ["FalsePositive", "FalsePositive"]
    assert expected_cost(y, wrong, cost, CLASSES) == pytest.approx(25.0)


def test_min_cost_escalates_when_missing_a_threat_is_expensive():
    cost = cost_matrix(COST_CFG, CLASSES)
    proba = np.array([[0.1, 0.8, 0.1]])

    assert CLASSES[proba.argmax()] == "FalsePositive"
    assert min_cost_predictions(proba, cost, CLASSES)[0] == "TruePositive"


def test_evaluate_perfect_predictions():
    y = ["TruePositive", "FalsePositive", "BenignPositive", "TruePositive"]
    result = evaluate_predictions(y, y, CLASSES)

    assert result["macro_f1"] == 1.0
    assert result["confusion"]["TruePositive"]["TruePositive"] == 2
