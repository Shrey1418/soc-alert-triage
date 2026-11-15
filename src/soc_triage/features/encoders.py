import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class OutOfFoldTargetEncoder:
    def __init__(self, columns, classes, n_splits=5, smoothing=20.0, seed=42):
        self.columns = list(columns)
        self.classes = list(classes)
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.prior_ = None
        self.maps_ = None

    def feature_names(self):
        return [f"{col}__te_{cls}" for col in self.columns for cls in self.classes]

    def _onehot(self, y):
        return pd.DataFrame({c: (y == c).astype(float) for c in self.classes})

    def _fit_maps(self, X, onehot):
        prior = onehot.mean()
        maps = {}
        for col in self.columns:
            sums = onehot.groupby(X[col]).sum()
            counts = onehot.groupby(X[col]).size()
            maps[col] = sums.add(prior * self.smoothing, axis=1).div(
                counts + self.smoothing, axis=0
            )
        return maps, prior

    def _apply(self, X, maps, prior):
        parts = []
        for col in self.columns:
            encoded = maps[col].reindex(X[col].to_numpy()).fillna(prior)
            encoded.columns = [f"{col}__te_{c}" for c in self.classes]
            parts.append(encoded.reset_index(drop=True))
        return pd.concat(parts, axis=1)

    def fit_transform(self, X, y):
        Xr = X.reset_index(drop=True)
        yr = y.reset_index(drop=True)
        onehot = self._onehot(yr)
        out = np.zeros((len(Xr), len(self.columns) * len(self.classes)))

        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for fit_idx, enc_idx in folds.split(Xr, yr):
            maps, prior = self._fit_maps(Xr.iloc[fit_idx], onehot.iloc[fit_idx])
            out[enc_idx] = self._apply(Xr.iloc[enc_idx], maps, prior).to_numpy()

        self.maps_, self.prior_ = self._fit_maps(Xr, onehot)
        return pd.DataFrame(out, columns=self.feature_names(), index=X.index)

    def transform(self, X):
        out = self._apply(X, self.maps_, self.prior_)
        out.index = X.index
        return out
