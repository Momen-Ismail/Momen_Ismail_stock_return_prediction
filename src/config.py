"""Central project configuration."""

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"
RAW_INPUT_DIR = INPUT_DIR / "raw"
EXTERNAL_INPUT_DIR = INPUT_DIR / "external"

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
INTERMEDIATE_DATA_DIR = DATA_OUTPUT_DIR / "intermediate"
FINAL_DATA_DIR = DATA_OUTPUT_DIR / "final"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"

for folder in [
    RAW_INPUT_DIR,
    EXTERNAL_INPUT_DIR,
    INTERMEDIATE_DATA_DIR,
    FINAL_DATA_DIR,
    MODEL_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)



SLEEP_SECONDS = 0.5
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

DROPPED_PREDICTORS_FILE = FINAL_DATA_DIR / "dropped_predictors_missing_train.csv"
TARGET_WINSOR_FILE = FINAL_DATA_DIR / "target_winsorization_limits_train.csv"
MONTHLY_MEDIAN_FILE = FINAL_DATA_DIR / "monthly_imputation_medians_summary.csv"

COMPUSTAT_RAW_FILE = RAW_INPUT_DIR / "compustat_annual_1980_2025.csv"
WELCH_GOYAL_CLEAN_FILE = EXTERNAL_INPUT_DIR / "welch_goyal_macro_1990_2025.csv"

TARGET_WINSOR_FILE = FINAL_DATA_DIR / "target_winsorization_cutoffs.csv"
EXTREME_TARGET_FILE = FINAL_DATA_DIR / "extreme_target_observations.csv"
EXTREME_TARGET_COUNT_FILE = FINAL_DATA_DIR / "extreme_target_counts.csv"
# ------------------------------------------------------------
# Sample period and target
# ------------------------------------------------------------
START_YEAR = 1990
END_YEAR = 2025

PRICE_START = "1987-01-01"
PRICE_END = "2026-02-01"  # Includes Jan 2026 to create Dec 2025 target

SAMPLE_START = pd.Timestamp("1990-01-31")

TARGET = "target_excess_return_next_1m"

TRAIN_END = pd.Timestamp("2014-12-31")
VALIDATION_END = pd.Timestamp("2019-12-31")


# ------------------------------------------------------------
# Cleaning and preprocessing
# ------------------------------------------------------------
MAX_MISSING_SHARE = 0.40
LOWER_Q = 0.01
UPPER_Q = 0.99


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------
COMPUSTAT_RAW_FILE = RAW_INPUT_DIR / "compustat_annual_1980_2025.csv"
WELCH_GOYAL_CLEAN_FILE = EXTERNAL_INPUT_DIR / "welch_goyal_macro_1990_2025.csv"


# ------------------------------------------------------------
# Intermediate and final data files
# ------------------------------------------------------------
TICKERS_RAW_FILE = INTERMEDIATE_DATA_DIR / "sp500_tickers_raw.csv"
DAILY_RAW_FILE = INTERMEDIATE_DATA_DIR / "daily_prices_raw_1987_2026.csv"
QUALITY_REPORT_FILE = INTERMEDIATE_DATA_DIR / "ticker_quality_report.csv"
MARKET_FILE = INTERMEDIATE_DATA_DIR / "gspc_vix_daily_1987_2026.csv"
FF_FILE = INTERMEDIATE_DATA_DIR / "fama_french_3_factors_monthly.csv"
COMPUSTAT_CLEAN_FILE = INTERMEDIATE_DATA_DIR / "compustat_annual_cleaned_1980_2025.csv"
MACRO_FILE = INTERMEDIATE_DATA_DIR / "fred_macro_monthly_1980_2026.csv"

TICKERS_CLEAN_FILE = FINAL_DATA_DIR / "sp500_tickers_clean.csv"
DAILY_CLEAN_FILE = FINAL_DATA_DIR / "daily_prices_clean_1987_2026.csv"
REMOVED_TICKERS_FILE = FINAL_DATA_DIR / "removed_tickers.csv"

MONTHLY_STOCK_FILE = FINAL_DATA_DIR / "monthly_stock_panel_with_targets_1990_2025.csv"
PANEL_WITH_FUNDAMENTALS_FILE = INTERMEDIATE_DATA_DIR / "monthly_panel_with_compustat_macro_1990_2025.csv"

RAW_KELLY_FILE = FINAL_DATA_DIR / "model_dataset_kelly_raw_full_1990_2025.csv"
RAW_PREDICTOR_FILE = FINAL_DATA_DIR / "predictor_columns_kelly_raw.csv"
RAW_KELLY_SUMMARY_FILE = FINAL_DATA_DIR / "kelly_raw_dataset_summary.csv"

CLEAN_FULL_FILE = FINAL_DATA_DIR / "model_dataset_kelly_ranked_full_1990_2025.parquet"
CLEAN_TRAIN_FILE = FINAL_DATA_DIR / "model_train_kelly_ranked_1990_2014.parquet"
CLEAN_VALIDATION_FILE = FINAL_DATA_DIR / "model_validation_kelly_ranked_2015_2019.parquet"
CLEAN_TEST_FILE = FINAL_DATA_DIR / "model_test_kelly_ranked_2020_2025.parquet"
CLEAN_PREDICTOR_FILE = FINAL_DATA_DIR / "predictor_columns_kelly_ranked.csv"
CLEANING_SUMMARY_FILE = FINAL_DATA_DIR / "cleaning_summary.csv"
