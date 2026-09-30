"""Fit the final model on all cleaned labeled data, predict validation.csv and the December chart rows."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import clean_features, load_labeled, load_unlabeled, target, to_rate
from .model import RateModel, flag_label_outliers


def december_frame(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs for the fixed December lane. The chart file only has lane, equipment,
    weight, distance and date, so we add what the model also needs:
      * city coordinates -> looked up from the city's (constant) coordinates in the data
The final model uses no market_index / quote_signal, so nothing else has to be invented."""
    dec = pd.read_csv(C.DEC_CSV, parse_dates=["date"])
    coords = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].rename(columns=lambda c: c.replace("pickup", "city")),
        train[["delivery", "delivery_lat", "delivery_lon"]].rename(columns=lambda c: c.replace("delivery", "city")),
    ]).drop_duplicates("city").set_index("city")
    for side in ("pickup", "delivery"):
        dec[f"{side}_lat"] = dec[side].map(coords["city_lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["city_lon"])
    if dec[["pickup_lat", "delivery_lat"]].isna().any().any():
        raise SystemExit("Could not build December inputs (missing coordinates).")
    return dec


def main() -> None:
    C.OUT.mkdir(exist_ok=True)
    train, valid = load_labeled(), load_unlabeled()

    Xtr, weight_fill = clean_features(train)
    ytr = target(train)
    clean = flag_label_outliers(Xtr, ytr, C.FEATURES)
    print(f"Training rows: {len(train):,}; corrupted labels removed: {(~clean).sum()}")
    model = RateModel().fit(Xtr[clean], ytr[clean])
    joblib.dump({"model": model, "weight_fill": weight_fill}, C.OUT / "model.joblib")

    # ---- validation.csv ----
    Xv, _ = clean_features(valid, weight_fill)
    pred = to_rate(model.predict_log_rpm(Xv), valid["distance"].values)
    out = pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": np.round(pred, 2)})
    template = pd.read_csv(C.TEMPLATE_CSV)
    assert list(template.columns) == ["load_id", "predicted_rate"]
    assert set(template.load_id) == set(out.load_id) and len(out) == 12_000
    out = template[["load_id"]].merge(out, on="load_id", how="left")   # keep template order
    assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.to_csv(C.ROOT / "validation_predictions.csv", index=False)
    rpm = out.predicted_rate.values / valid.set_index("load_id").loc[out.load_id, "distance"].values
    print(f"validation_predictions.csv written. Predicted rate/mile: median {np.median(rpm):.3f} "
          f"(train median clean {np.exp(ytr[clean].median()):.3f})")

    # ---- December chart ----
    dec = december_frame(train, valid)
    Xd, _ = clean_features(dec.assign(weight=dec["weight"].astype(float)), weight_fill)
    dec["predicted_rate"] = np.round(to_rate(model.predict_log_rpm(Xd), dec["distance"].values), 2)
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    final = dec[cols].copy()
    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final.to_csv(C.OUT / "december_predictions.csv", index=False)
    print(final["predicted_rate"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
