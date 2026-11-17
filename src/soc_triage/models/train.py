import inspect

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

ENCODE_COLS = ("dominant_alert_title", "dominant_detector")
DROP_COLS = ("OrgId", "IncidentId", "start_time", "end_time")


class CodedClassifier:
    def __init__(self, model, classes):
        self.model = model
        self.classes_ = np.array(classes)

    @property
    def booster_(self):
        return self.model.booster_

    @property
    def best_iteration_(self):
        return self.model.best_iteration_

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def numeric_feature_columns(table, drop_cols=()):
    skip = {*DROP_COLS, *ENCODE_COLS, *drop_cols, TARGET}
    return [c for c in table.columns if c not in skip]


def transform_table(table, encoder, feature_names):
    table = table.reset_index(drop=True)
    encoded = encoder.transform(table[encoder.columns])
    return pd.concat([table, encoded], axis=1).reindex(columns=feature_names, fill_value=0)


def build_matrices(train, valid, test, classes, seed, drop_cols=(), encode_cols=ENCODE_COLS):
    train, valid, test = (part.reset_index(drop=True) for part in (train, valid, test))
    numeric = numeric_feature_columns(train, drop_cols)
    encoder = OutOfFoldTargetEncoder(encode_cols, classes, seed=seed)

    train_te = encoder.fit_transform(train[encoder.columns], train[TARGET])
    X_train = pd.concat([train[numeric], train_te], axis=1)
    names = list(X_train.columns)
    X_valid = transform_table(valid, encoder, names)
    X_test = transform_table(test, encoder, names)
    return encoder, X_train, X_valid, X_test


def fit_lightgbm(X_train, y_train, X_valid, y_valid, params, seed, classes):
    params = dict(params)
    stop_rounds = params.pop("early_stopping_rounds")

    code = {c: i for i, c in enumerate(classes)}
    train_codes = y_train.map(code).to_numpy(dtype=int)
    valid_codes = y_valid.map(code).to_numpy(dtype=int)

    model = LGBMClassifier(
        objective="multiclass", random_state=seed, n_jobs=-1, verbose=-1, **params
    )
    if "eval_X" in inspect.signature(LGBMClassifier.fit).parameters:
        evaluation = {"eval_X": X_valid, "eval_y": valid_codes}
    else:
        evaluation = {"eval_set": [(X_valid, valid_codes)]}
    model.fit(
        X_train,
        train_codes,
        callbacks=[early_stopping(stop_rounds, verbose=False)],
        **evaluation,
    )
    logger.info("LightGBM best iteration: %s", model.best_iteration_)
    return CodedClassifier(model, classes)


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


def run_training(table, cfg, drop_cols=(), encode_cols=ENCODE_COLS):
    missing = [c for c in encode_cols if c not in table.columns]
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

    encoder, X_train, X_valid, X_test = build_matrices(
        train, valid, test, classes, seed, drop_cols, encode_cols
    )
    y_train, y_valid, y_test = (
        part[TARGET].reset_index(drop=True) for part in (train, valid, test)
    )

    models = {
        "majority": make_majority().fit(X_train, y_train),
        "logreg": make_logreg(seed).fit(X_train, y_train),
        "lightgbm": fit_lightgbm(
            X_train, y_train, X_valid, y_valid, cfg["model"]["lightgbm"], seed, classes
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
