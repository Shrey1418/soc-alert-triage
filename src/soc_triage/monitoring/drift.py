import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from soc_triage.models.cost import min_cost_predictions

EPS = 1e-4
ORDER = {"ok": 0, "warn": 1, "alert": 2}


def status(value, warn, alert):
    if value is None or value != value:
        return "n/a"
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def worst(*levels):
    known = [level for level in levels if level in ORDER]
    return max(known, key=ORDER.get) if known else "n/a"


def _psi_from_shares(reference_share, current_share):
    ref = np.clip(np.asarray(reference_share, dtype=float), EPS, None)
    cur = np.clip(np.asarray(current_share, dtype=float), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference, current, top_n=30):
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if ref.empty or cur.empty:
        return float("nan")
    top = set(ref.value_counts().head(top_n).index)
    ref = ref.where(ref.isin(top), "__other__")
    cur = cur.where(cur.isin(top), "__other__")
    ref_share = ref.value_counts(normalize=True)
    cur_share = cur.value_counts(normalize=True)
    categories = ref_share.index.union(cur_share.index)
    return _psi_from_shares(
        ref_share.reindex(categories, fill_value=0),
        cur_share.reindex(categories, fill_value=0),
    )


def psi_numeric(reference, current, bins=10):
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if len(np.unique(ref)) <= bins:
        return psi_categorical(ref, cur)
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    size = len(inner) + 1
    ref_share = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=size) / len(ref)
    cur_share = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=size) / len(cur)
    return _psi_from_shares(ref_share, cur_share)


def ks_stat(reference, current):
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    return float(ks_2samp(ref, cur).statistic)


def unseen_rate(reference, current):
    cur = pd.Series(current).dropna()
    if cur.empty:
        return float("nan")
    seen = set(pd.Series(reference).dropna())
    return float((~cur.isin(seen)).mean())


def feature_drift(reference, current, numeric_cols, categorical_cols, cfg):
    rows = []
    for col in numeric_cols:
        psi = psi_numeric(reference[col], current[col])
        ks = ks_stat(reference[col], current[col])
        level = worst(
            status(psi, cfg["psi"]["warn"], cfg["psi"]["alert"]),
            status(ks, cfg["ks"]["warn"], cfg["ks"]["alert"]),
        )
        rows.append({"feature": col, "kind": "numeric", "psi": psi, "ks": ks, "status": level})
    for col in categorical_cols:
        psi = psi_categorical(reference[col], current[col])
        level = status(psi, cfg["psi"]["warn"], cfg["psi"]["alert"])
        rows.append(
            {"feature": col, "kind": "categorical", "psi": psi, "ks": np.nan, "status": level}
        )
    out = pd.DataFrame(rows)
    return out.sort_values("psi", ascending=False).reset_index(drop=True).round(4)


def unseen_report(reference, current, columns, thresholds):
    report = {}
    for col in columns:
        rate = unseen_rate(reference[col], current[col])
        limit = thresholds[col]
        report[col] = {"rate": rate, "status": status(rate, limit["warn"], limit["alert"])}
    return report


def prediction_drift(ref_proba, cur_proba, classes, cost, cfg):
    index = list(classes).index("TruePositive")
    psi = psi_numeric(ref_proba[:, index], cur_proba[:, index])
    ref_decision = min_cost_predictions(ref_proba, cost, classes)
    cur_decision = min_cost_predictions(cur_proba, cost, classes)
    return {
        "psi": psi,
        "status": status(psi, cfg["prediction_psi"]["warn"], cfg["prediction_psi"]["alert"]),
        "mean_true_positive_probability": {
            "reference": float(ref_proba[:, index].mean()),
            "current": float(cur_proba[:, index].mean()),
        },
        "escalated_share": {
            "reference": float((ref_decision == "TruePositive").mean()),
            "current": float((cur_decision == "TruePositive").mean()),
        },
    }
