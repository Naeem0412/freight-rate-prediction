"""Rolling-origin temporal validation, ablations, baseline and unseen-city stress test.

Why temporal: the real validation set is the future (Nov-Dec 2025) relative to the labeled
data (Jan-Oct 2025), so a random K-fold would leak market regime information and be
optimistic. Every fold here trains on the past and tests on the following months.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .data import clean_features, load_labeled, target, to_rate
from .model import RateModel, flag_label_outliers

# (last training month, test months)
FOLDS = [(6, [7, 8]), (7, [8, 9]), (8, [9, 10])]


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    ape = np.abs(err) / y
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE": float(ape.mean()),
        "MdAPE": float(np.median(ape)),
        "n": int(len(y)),
    }


def main() -> None:
    C.OUT.mkdir(exist_ok=True)
    raw = load_labeled()
    X, _ = clean_features(raw)
    y = target(raw)
    month = raw["date"].dt.month.values
    rate = raw["posted_rate"].values
    dist = raw["distance"].values

    # diagnostic mask (used only to *report* the error on non-corrupted labels)
    diag_clean = flag_label_outliers(X, y, C.FEATURES)
    print(f"Label outliers flagged: {(~diag_clean).sum()} ({(~diag_clean).mean():.2%})")

    configs = {
        "final (features as chosen)": C.FEATURES,
        "with quote_signal": C.FEATURES + ["quote_signal"],
        "without coordinates": [c for c in C.FEATURES if "lat" not in c and "lon" not in c],
        "without day-of-week": [c for c in C.FEATURES if c != "dow"],
        "without haversine/ratio": [c for c in C.FEATURES if c not in ("haversine", "distance_ratio")],
        "with market_index": C.FEATURES + ["market_index"],
        "with market_index (daily mean)": C.FEATURES + ["mi_day"],
    }
    day_mi = pd.concat([raw, ]).groupby("date")["market_index"].mean().rename("mi_day")
    X = X.join(day_mi, on="date")

    fold_rows, cache_masks = [], {}
    for last, test_months in FOLDS:
        tr = month <= last
        te = np.isin(month, test_months)
        # outlier filtering uses ONLY the training part of the fold (no test leakage)
        clean_tr = flag_label_outliers(X[tr].reset_index(drop=True), y[tr].reset_index(drop=True), C.FEATURES)
        idx_tr = np.where(tr)[0][clean_tr]
        cache_masks[last] = (idx_tr, np.where(te)[0])
        for name, cols in configs.items():
            m = RateModel(cols).fit(X.iloc[idx_tr], y.iloc[idx_tr])
            p = to_rate(m.predict_log_rpm(X.iloc[np.where(te)[0]]), dist[te])
            yt = rate[te]
            cm = diag_clean[te]
            for scope, mask in (("all_rows", np.ones(len(yt), bool)), ("clean_labels", cm)):
                r = metrics(p[mask], yt[mask]); r.update(fold=f"train<=M{last}->test M{test_months}", config=name, scope=scope)
                fold_rows.append(r)
        # baseline: median rate-per-mile per equipment x distance
        med = (raw.loc[idx_tr].assign(r=lambda d: d.posted_rate / d.distance).groupby("equipment")["r"].median())
        p = raw.loc[te, "equipment"].map(med).values * dist[te]
        for scope, mask in (("all_rows", np.ones(te.sum(), bool)), ("clean_labels", diag_clean[te])):
            r = metrics(p[mask], rate[te][mask]); r.update(fold=f"train<=M{last}->test M{test_months}", config="baseline: median rpm x distance", scope=scope)
            fold_rows.append(r)
        print(f"fold train<=M{last} done")

    df = pd.DataFrame(fold_rows)
    df.to_csv(C.OUT / "cv_folds.csv", index=False)
    summary = (df.groupby(["config", "scope"])[["MAE", "RMSE", "MAPE", "MdAPE"]].mean().round(4).reset_index())
    summary.to_csv(C.OUT / "cv_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ---- unseen-city stress test (validation has 8 cities never seen in training) ----
    last, test_months = FOLDS[-1]
    idx_tr, idx_te = cache_masks[last]
    cities = sorted(set(raw.pickup) | set(raw.delivery))
    rng = np.random.default_rng(C.RANDOM_STATE)
    held = set(rng.choice(cities, 8, replace=False))
    touches = raw[["pickup", "delivery"]].isin(held).any(axis=1).values
    idx_tr_unseen = idx_tr[~touches[idx_tr]]
    te_sel = idx_te[touches[idx_te]]
    res = {}
    for label, idx in (("cities seen in training", idx_tr), ("cities NOT seen in training", idx_tr_unseen)):
        m = RateModel().fit(X.iloc[idx], y.iloc[idx])
        p = to_rate(m.predict_log_rpm(X.iloc[te_sel]), dist[te_sel])
        cm = diag_clean[te_sel]
        res[label] = metrics(p[cm], rate[te_sel][cm])
    print("Unseen-city stress test (clean labels):", json.dumps(res, indent=1))

    out = {
        "held_out_cities": sorted(held),
        "unseen_city_test": res,
        "label_outliers_flagged": int((~diag_clean).sum()),
        "label_outlier_share": float((~diag_clean).mean()),
        "folds": [f"train<=M{a} -> test M{b}" for a, b in FOLDS],
    }
    (C.OUT / "validation_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
