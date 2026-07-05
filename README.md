# Stock Return Prediction Data Pipeline

This project builds a monthly stock-level panel for predicting next-month excess returns. Predictors dated at month `t` are used to predict the stock return in month `t+1` minus the month `t+1` risk-free rate.

## Pipeline

1. `01_build_clean_yahoo_daily.py` builds the historical S&P 500 universe, downloads daily Yahoo Finance data, and applies transparent price-quality filters.
2. `02_build_monthly_stock_features.py` creates monthly return, momentum, volatility, liquidity, market-risk, and trend variables and constructs the next-month target.
3. `03_add_fundamentals_and_macro.py` constructs annual Compustat characteristics and merges them only after a six-month reporting lag.
4. `04_build_raw_kelly_dataset.py` adds the eight Welch-Goyal macro states, characteristic-macro interactions, and SIC2 industry controls.
5. `05_clean_and_rank_normalize.py` applies chronological splitting, training-only target winsorization and missingness filtering, monthly median imputation, and monthly cross-sectional rank normalization.

Run the complete build from the project root:

```bash
pip install -r requirements.txt
./run_pipeline.sh
```

## Required inputs

- `input/raw/compustat_annual_1980_2025.csv`
- `input/external/welch_goyal_macro_1990_2025.csv`

Yahoo Finance and Fama-French data are downloaded during the build. The cleaned Welch-Goyal file is treated as a fixed external input; its one-off preparation script is intentionally excluded from the production pipeline.

## Final outputs

The modeling datasets are saved in `output/data/final/` as full, training (1990-2014), validation (2015-2019), and test (2020-2025) Parquet files. Predictor names and preprocessing metadata are saved beside them as CSV files.

The build scripts contain only data-construction and preprocessing logic. Exploratory inspection and manual outlier-check scripts are intentionally excluded from the production pipeline.
