# Permanent Inputs

The normal data pipeline reads these files directly and does not download them:

- `stock_universe_locked.csv`: locked current-plus-historical S&P 500 universe.
- `fama_french_rf_monthly.csv`: monthly Fama-French risk-free rate in decimal form.
- `market_gspc_daily.csv`: daily S&P 500 index prices.
- `market_vix_daily.csv`: daily VIX prices.
- `external/welch_goyal_macro_1990_2025.csv`: cleaned Welch-Goyal macro variables.
- `raw/compustat_annual_1980_2025.csv`: raw annual Compustat fundamentals.

One-time acquisition or validation scripts live in `src/acquisition/`.
