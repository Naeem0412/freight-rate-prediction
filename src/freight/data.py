"""Loading and cleaning. Every rule here is applied identically to train and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_labeled() -> pd.DataFrame:
    return pd.read_csv(C.TRAIN_CSV, parse_dates=["date"])


def load_unlabeled() -> pd.DataFrame:
    return pd.read_csv(C.VALID_CSV, parse_dates=["date"])


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean_features(df: pd.DataFrame, weight_fill: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return the feature frame. `weight_fill` (median weight per equipment) is learned on
    training data only and re-used for every other frame."""
    d = df.copy()
    if weight_fill is None:
        weight_fill = d.assign(w=d["weight"].abs()).groupby("equipment")["w"].median().to_dict()

    # 1) weight: negative values are sign errors, NaN is missing, +/-47,500 is a cap
    d["weight_was_negative"] = (d["weight"] < 0).astype(int)
    d["weight_missing"] = d["weight"].isna().astype(int)
    d["weight"] = d["weight"].abs()
    d["weight"] = d["weight"].fillna(d["equipment"].map(weight_fill)).clip(upper=C.WEIGHT_CAP)

    # 2) geometry: distance is the reported road distance; haversine + ratio expose bad distances
    d["log_distance"] = np.log(d["distance"])
    d["haversine"] = haversine_miles(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["distance_ratio"] = d["distance"] / d["haversine"].clip(lower=1.0)

    d["equipment"] = d["equipment"].map(C.EQUIPMENT_CODES)
    d["dow"] = d["date"].dt.dayofweek
    return d, weight_fill


def target(df: pd.DataFrame) -> pd.Series:
    """Model log(rate per mile). Rate scales ~linearly with distance, so this removes the
    dominant effect and makes the error relative (percentage-like)."""
    return np.log(df["posted_rate"] / df["distance"])


def to_rate(log_rpm, distance) -> np.ndarray:
    return np.exp(log_rpm) * np.asarray(distance)
