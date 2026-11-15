import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def log1p_clip(values):
    return np.log1p(np.clip(values, 0, None))


def make_majority():
    return DummyClassifier(strategy="prior")


def make_logreg(seed=42):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log1p_clip)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=seed)),
        ]
    )
