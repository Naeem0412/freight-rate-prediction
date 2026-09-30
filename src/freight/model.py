from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import KFold

from . import config as C


def _hgb(loss: str, seed: int = 0, iters: int = 500, lr: float = 0.05) -> HGB:
    return HGB(loss=loss, max_iter=iters, learning_rate=lr, random_state=seed)


def flag_label_outliers(X: pd.DataFrame, y: pd.Series, cols: list[str]) -> np.ndarray:
    """Out-of-fold robust fit; rows whose log-residual is huge are corrupted labels
    (rate multiplied / divided by ~2-5x). Returns a boolean mask of CLEAN rows."""
    oof = np.zeros(len(X))
    for a, b in KFold(5, shuffle=True, random_state=C.RANDOM_STATE).split(X):
        m = _hgb("absolute_error", iters=300, lr=0.06).fit(X.iloc[a][cols], y.iloc[a])
        oof[b] = m.predict(X.iloc[b][cols])
    return np.abs(y.values - oof) < C.LABEL_OUTLIER_THRESHOLD


class RateModel:
    """Mean (in log space) of two gradient-boosting models trained with different losses."""

    def __init__(self, cols=None, losses=("squared_error", "absolute_error")):
        self.cols = cols or C.FEATURES
        self.losses = losses
        self.models = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RateModel":
        self.models = [_hgb(l).fit(X[self.cols], y) for l in self.losses]
        return self

    def predict_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X[self.cols]) for m in self.models], axis=0)
