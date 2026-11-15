import numpy as np


def cost_matrix(cost_cfg, classes):
    return np.array([[cost_cfg[t][p] for p in classes] for t in classes], dtype=float)


def expected_cost(y_true, y_pred, cost, classes):
    index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([index[v] for v in y_true])
    pred_idx = np.array([index[v] for v in y_pred])
    return float(cost[true_idx, pred_idx].mean())


def min_cost_predictions(proba, cost, classes):
    expected = proba @ cost
    return np.array(classes)[expected.argmin(axis=1)]
