from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

TRAIN_CSV = DATA / "train_test.csv"
VALID_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DEC_CSV = DATA / "december_chart_inputs.csv"

WEIGHT_CAP = 47_500.0           # weights pile up exactly at +/- this number (censored)
LABEL_OUTLIER_THRESHOLD = 0.30  # |log residual| above this => corrupted label
RANDOM_STATE = 0

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# Final feature set (chosen by rolling temporal CV, see src/freight/validate.py)
FEATURES = [
    "distance", "log_distance", "weight", "weight_missing", "weight_was_negative",
    "equipment", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "haversine", "distance_ratio", "dow",
]
# NOT used: market_index (hurts every temporal fold) and quote_signal (helps only when the
# training window already covers the test regime; hurts 2 of 3 rolling folds) - see report.
