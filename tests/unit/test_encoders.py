import numpy as np
import pandas as pd
import pytest

from soc_triage.features.encoders import OutOfFoldTargetEncoder

CLASSES = ["BenignPositive", "FalsePositive", "TruePositive"]


def make_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.integers(0, 5, n), "b": rng.integers(0, 3, n)})
    y = pd.Series(rng.choice(CLASSES, n))
    return X, y


def test_oof_encoding_has_expected_names_and_rows_sum_to_one():
    X, y = make_table()
    out = OutOfFoldTargetEncoder(["a", "b"], CLASSES, seed=1).fit_transform(X, y)

    assert list(out.columns) == [f"{c}__te_{k}" for c in ["a", "b"] for k in CLASSES]
    assert len(out) == len(X)
    for col in ["a", "b"]:
        block = out[[f"{col}__te_{k}" for k in CLASSES]]
        assert np.allclose(block.sum(axis=1), 1.0)


def test_singleton_category_gets_prior_not_its_own_label():
    X = pd.DataFrame({"a": [0] * 99 + [1]})
    y = pd.Series(
        ["TruePositive"] * 45 + ["BenignPositive"] * 45 + ["FalsePositive"] * 9 + ["FalsePositive"]
    )
    out = OutOfFoldTargetEncoder(["a"], CLASSES, n_splits=5, seed=0).fit_transform(X, y)

    assert out.iloc[99]["a__te_FalsePositive"] == pytest.approx(0.1, abs=1e-9)


def test_unseen_category_falls_back_to_prior():
    X, y = make_table()
    enc = OutOfFoldTargetEncoder(["a"], CLASSES, seed=1)
    enc.fit_transform(X, y)

    out = enc.transform(pd.DataFrame({"a": [999]}))
    prior = y.value_counts(normalize=True).reindex(CLASSES).to_numpy()
    assert np.allclose(out.iloc[0].to_numpy(), prior)


def test_encoder_with_no_columns_returns_empty_frame():
    X, y = make_table()
    enc = OutOfFoldTargetEncoder([], CLASSES, seed=1)

    assert enc.fit_transform(X, y).shape == (len(X), 0)
    assert enc.transform(X).shape == (len(X), 0)
