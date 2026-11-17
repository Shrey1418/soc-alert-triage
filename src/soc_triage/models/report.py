import numpy as np
import pandas as pd

from soc_triage.models.cost import cost_matrix, expected_cost, min_cost_predictions


def scaled_cost(cost_cfg, classes, factor):
    scaled = {true: dict(row) for true, row in cost_cfg.items()}
    for decision in ("BenignPositive", "FalsePositive"):
        scaled["TruePositive"][decision] = cost_cfg["TruePositive"][decision] * factor
    return cost_matrix(scaled, classes)


def sensitivity_table(y, proba, cost_cfg, classes, factors=(0.25, 0.5, 1, 2, 4)):
    y_arr = np.asarray(y)
    threats = y_arr == "TruePositive"
    argmax_pred = np.array(classes)[proba.argmax(axis=1)]
    escalate = ["TruePositive"] * len(y_arr)

    rows = []
    for factor in factors:
        cost = scaled_cost(cost_cfg, classes, factor)
        min_pred = min_cost_predictions(proba, cost, classes)
        base = expected_cost(y_arr, escalate, cost, classes)
        model = expected_cost(y_arr, min_pred, cost, classes)
        dismissed = (min_pred == "FalsePositive") & threats
        rows.append(
            {
                "miss_penalty_x": factor,
                "escalate_all": base,
                "argmax": expected_cost(y_arr, argmax_pred, cost, classes),
                "min_cost": model,
                "saving_vs_escalate_pct": (1 - model / base) * 100 if base else 0.0,
                "escalated_pct": float((min_pred == "TruePositive").mean() * 100),
                "threats_dismissed_pct": float(dismissed.sum() / max(threats.sum(), 1) * 100),
            }
        )
    return pd.DataFrame(rows).round(3)


def calibration_table(y, proba, classes, target="TruePositive", n_bins=10):
    predicted = proba[:, classes.index(target)]
    actual = (np.asarray(y) == target).astype(float)
    bins = pd.cut(predicted, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    frame = pd.DataFrame({"bin": bins, "predicted": predicted, "actual": actual})
    out = frame.groupby("bin", observed=True).agg(
        incidents=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    )
    return out.round(3)
