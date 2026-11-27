import numpy as np


def apply_shift(table, unseen_title_share=0.3, evidence_multiplier=2.5, seed=42):
    rng = np.random.default_rng(seed)
    shifted = table.copy()
    mask = rng.random(len(shifted)) < unseen_title_share
    new_titles = 900_000_000 + rng.integers(0, 500, int(mask.sum()))
    shifted.loc[mask, "dominant_alert_title"] = new_titles
    shifted["n_evidence_rows"] = (shifted["n_evidence_rows"] * evidence_multiplier).round()
    return shifted
