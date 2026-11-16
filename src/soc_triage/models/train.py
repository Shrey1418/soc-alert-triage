import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from soc_triage.data.schema import TARGET, TARGET_CLASSES
from soc_triage.features.encoders import OutOfFoldTargetEncoder
from soc_triage.models.baselines import make_logreg, make_majority
from soc_triage.models.cost import cost_matrix, expected_cost, min_cost_predictions
from soc_triage.models.evaluate import evaluate_predictions
from soc_triage.models.split import summarize_splits, time_split
from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)

ENCODE_COLS = ["dominant_alert_title", "dominant_detector"]
DROP_COLS = ["OrgId", "IncidentId", "start_time", "end_time"]


def numeric_feature_columns(table):
    skip = {*DROP_COLS, *ENCODE_COLS, TARGET}
    return [c for c in table.columns if c not in skip]


def build_matrices(train, valid, test, classes, seed):
    train, valid, test = (part.reset_index(drop=True) for part in (train, valid, test))
    numeric = numeric_feature_columns(train)
    encoder = OutOfFoldTargetEncoder(ENCODE_COLS, classes, seed=seed)

    train_te = encoder.fit_transform(train[ENCODE_COLS], train[TARGET])
    X_train = pd.concat([train[numeric], train_te], axis=1)
    X_valid = pd.concat([valid[numeric], encoder.transform(valid[ENCODE_COLS])], axis=1)
    X_test = pd.concat([test[numeric], encoder.transform(test[ENCODE_COLS])], axis=1)
    return encoder, X_train, X_valid, X_test


def fit_lightgbm(X_train, y_train, X_valid, y_valid, params, seed):
    params = dict(params)
    stop_rounds = params.pop("early_stopping_rounds")
    model = LGBMClassifier(
        objective="multiclass", random_state=seed, n_jobs=-1, verbose=-1, **params
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(stop_rounds, verbose=False)],
    )
    logger.info("LightGBM best iteration: %s", model.best_iteration_)
    return model


def score_model(model, X, y, classes, cost):
    if list(model.classes_) != list(classes):
        raise ValueError(f"Class order mismatch: {list(model.classes_)} vs {list(classes)}")
    proba = model.predict_proba(X)
    decisions = {
        "argmax": np.array(classes)[proba.argmax(axis=1)],
        "min_cost": min_cost_predictions(proba, cost, classes),
    }
    scored = {}
    for name, pred in decisions.items():
        scored[name] = evaluate_predictions(y, pred, classes)
        scored[name]["cost_per_incident"] = expected_cost(y, pred, cost, classes)
    return scored


def run_training(table, cfg):
    missing = [c for c in ENCODE_COLS if c not in table.columns]
    if missing:
        raise DataValidationError(
            f"Columns {missing} missing. Rebuild the table: python scripts/build_features.py"
        )

    classes = sorted(TARGET_CLASSES)
    seed = cfg["model"]["seed"]
    cost = cost_matrix(cfg["cost"], classes)

    train, valid, test = time_split(table, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    split_summary = summarize_splits(train, valid, test)
    logger.info("Split sizes: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    encoder, X_train, X_valid, X_test = build_matrices(train, valid, test, classes, seed)
    y_train, y_valid, y_test = (
        part[TARGET].reset_index(drop=True) for part in (train, valid, test)
    )

    models = {
        "majority": make_majority().fit(X_train, y_train),
        "logreg": make_logreg(seed).fit(X_train, y_train),
        "lightgbm": fit_lightgbm(
            X_train, y_train, X_valid, y_valid, cfg["model"]["lightgbm"], seed
        ),
    }
    results = {
        name: {
            "valid": score_model(model, X_valid, y_valid, classes, cost),
            "test": score_model(model, X_test, y_test, classes, cost),
        }
        for name, model in models.items()
    }
    escalate_all = {
        split: expected_cost(y, ["TruePositive"] * len(y), cost, classes)
        for split, y in (("valid", y_valid), ("test", y_test))
    }

    gains = models["lightgbm"].booster_.feature_importance(importance_type="gain")
    importance = pd.Series(gains, index=X_train.columns).sort_values(ascending=False)

    return {
        "models": models,
        "encoder": encoder,
        "classes": classes,
        "feature_names": list(X_train.columns),
        "results": results,
        "escalate_all": escalate_all,
        "importance": importance,
        "split_summary": split_summary,
    }
