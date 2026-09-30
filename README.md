# Freight Rate Prediction Challenge

Predicts `posted_rate` for 12,000 future loads (Nov-Dec 2025) from 48,000 labeled loads (Jan-Oct 2025).

## Run (VS Code terminal)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python run_validate.py             # optional (~5 min): rolling temporal CV -> outputs/cv_*.csv
python run_pipeline.py             # fits final model -> validation_predictions.csv + outputs/december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_predictions.csv
python -c "import sys; sys.path.insert(0,'src'); from freight.eda_plots import main; main()"
python reports/build_report.py     # rebuilds reports/Freight_Rate_Report.docx
```

Put the provided CSVs in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Layout

| Path | Purpose |
|---|---|
| `src/freight/data.py` | cleaning + feature engineering (same rules for train/validation) |
| `src/freight/model.py` | label-outlier filter + GBM ensemble |
| `src/freight/validate.py` | rolling-origin temporal CV, ablations, unseen-city test |
| `src/freight/train_predict.py` | final fit and predictions |
| `score.py` | provided scorer |

## Approach in short
* Temporal validation (train on past months, test on following months) because the target set is in the future.
* Target = log(rate per mile); final prediction = exp(pred) x distance.
* ~1.4% corrupted labels removed from training via an out-of-fold robust fit.
* weight fixes: abs(), median impute + flags, cap at 47,500.
* No city IDs, so the 8 cities unseen in training still work (coordinates + distance).
* `market_index` and `quote_signal` deliberately excluded (unstable across time; see report).
