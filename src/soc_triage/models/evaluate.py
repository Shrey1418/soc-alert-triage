import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PER_CLASS_KEYS = ("precision", "recall", "f1-score", "support")


def evaluate_predictions(y_true, y_pred, classes):
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {c: {k: float(report[c][k]) for k in PER_CLASS_KEYS} for c in classes},
        "confusion": pd.DataFrame(matrix, index=classes, columns=classes).to_dict(orient="index"),
    }


def summarize_results(results):
    rows = []
    for model, splits in results.items():
        for split, scored in splits.items():
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "macro_f1": scored["argmax"]["macro_f1"],
                    "accuracy": scored["argmax"]["accuracy"],
                    "cost_argmax": scored["argmax"]["cost_per_incident"],
                    "macro_f1_min_cost": scored["min_cost"]["macro_f1"],
                    "cost_min_cost": scored["min_cost"]["cost_per_incident"],
                }
            )
    return pd.DataFrame(rows).round(4)
