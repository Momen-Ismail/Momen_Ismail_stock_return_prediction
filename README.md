# Stock Return Prediction Data Pipeline

This project builds a monthly stock-level panel for predicting next-month excess returns. Predictors dated at month `t` are used to predict the stock return in month `t+1` minus the month `t+1` risk-free rate.

## Pipeline

1. `01_build_clean_yahoo_daily.py` loads the locked stock universe, downloads daily Yahoo Finance stock data, and applies transparent price-quality filters.
2. `02_build_monthly_stock_features.py` creates monthly return, momentum, volatility, liquidity, market-risk, and trend variables and constructs the next-month target.
3. `03_add_fundamentals_and_macro.py` constructs annual Compustat characteristics and merges them only after a six-month reporting lag.
4. `04_build_raw_kelly_dataset.py` adds the eight Welch-Goyal macro states, characteristic-macro interactions, and SIC2 industry controls.
5. `05_clean_and_rank_normalize.py` drops observations without a target, documents target outliers, imputes predictors by monthly cross-sectional medians, and rank-normalizes predictors within each month. Train, validation, and test splits are created later by the model workflow.

Run the complete build from the project root:

```bash
pip install -r requirements.txt
./run_pipeline.sh
```

## Required inputs

- `input/raw/compustat_annual_1980_2025.csv`
- `input/stock_universe_locked.csv`
- `input/fama_french_rf_monthly.csv`
- `input/market_gspc_daily.csv`
- `input/market_vix_daily.csv`
- `input/external/welch_goyal_macro_1990_2025.csv`

Small external datasets are locked as permanent local inputs. One-time acquisition scripts live in `src/acquisition/`; normal pipeline scripts do not download Wikipedia, Fama-French, GSPC, VIX, or Welch-Goyal inputs.

## Documentation

- `docs/Data_Construction_Guide.pdf`: full data-construction methodology.
- `docs/Pipeline_Overview.pdf`: short visual pipeline overview.

The data-construction guide source is `docs/Data_Construction_Guide.tex`.

## Final outputs

The model-ready dataset is saved in `output/data/final/model_dataset_kelly_ranked_full_1990_2025.parquet`. Predictor names and preprocessing metadata are saved beside it as CSV files. Model scripts split the full panel chronologically into training, validation, and test samples.

The build scripts contain only data-construction and preprocessing logic. Exploratory inspection and manual outlier-check scripts are intentionally excluded from the production pipeline.
