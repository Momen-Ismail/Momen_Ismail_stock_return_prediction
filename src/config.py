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
STAGING_DATA_DIR = DATA_OUTPUT_DIR / "staging"
FINAL_DATA_DIR = DATA_OUTPUT_DIR / "final"
QUALITY_OUTPUT_DIR = OUTPUT_DIR / "quality"

MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
TUNING_OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"
SELECTION_OUTPUT_DIR = MODEL_OUTPUT_DIR / "selection"
TEST_OUTPUT_DIR = MODEL_OUTPUT_DIR / "test"
INTERPRETATION_OUTPUT_DIR = (
    MODEL_OUTPUT_DIR
    / "interpretation"
)
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"

for folder in [
    INPUT_DIR,
    RAW_INPUT_DIR,
    EXTERNAL_INPUT_DIR,
    STAGING_DATA_DIR,
    FINAL_DATA_DIR,
    QUALITY_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
    TUNING_OUTPUT_DIR,
    SELECTION_OUTPUT_DIR,
    TEST_OUTPUT_DIR,
    INTERPRETATION_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# External sources
# ------------------------------------------------------------
WIKI_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/"
    "faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)

SLEEP_SECONDS = 0.5


# ------------------------------------------------------------
# Sample period and target
# ------------------------------------------------------------
START_YEAR = 1990
END_YEAR = 2025

PRICE_START = "1987-01-01"
PRICE_END = "2026-02-01"

SAMPLE_START = pd.Timestamp(
    "1990-01-31"
)

SAMPLE_END = pd.Timestamp(
    "2025-12-31"
)

TARGET = (
    "target_excess_return_next_1m"
)

TRAIN_END = pd.Timestamp(
    "2014-12-31"
)

VALIDATION_END = pd.Timestamp(
    "2019-12-31"
)


# ------------------------------------------------------------
# Cleaning parameters
# ------------------------------------------------------------
DAILY_BAD_ROW_SHARE_THRESHOLD = 0.10
MAX_COMPUSTAT_AGE_MONTHS = 12
# ------------------------------------------------------------
# Cleaning parameters
# ------------------------------------------------------------
DAILY_BAD_ROW_SHARE_THRESHOLD = 0.10
MAX_COMPUSTAT_AGE_MONTHS = 12

PREDICTOR_WINSOR_LOWER = 0.01
PREDICTOR_WINSOR_UPPER = 0.99

# ------------------------------------------------------------
# File 05: Final winsorized model dataset
# ------------------------------------------------------------
CLEAN_FULL_FILE = (
    FINAL_DATA_DIR
    / "006_model_dataset_kelly_winsorized_1990_2025.parquet"
)

CLEAN_PREDICTOR_FILE = (
    FINAL_DATA_DIR
    / "006_predictor_columns_kelly_winsorized.csv"
)

CLEANING_SUMMARY_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_cleaning_summary.csv"
)




# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------
INPUT_MANIFEST_FILE = (
    INPUT_DIR
    / "input_manifest.csv"
)

STOCK_UNIVERSE_FILE = (
    INPUT_DIR
    / "stock_universe_locked.csv"
)

FAMA_FRENCH_RF_FILE = (
    INPUT_DIR
    / "fama_french_rf_monthly.csv"
)

GSPC_DAILY_FILE = (
    INPUT_DIR
    / "market_gspc_daily.csv"
)

VIX_DAILY_FILE = (
    INPUT_DIR
    / "market_vix_daily.csv"
)

COMPUSTAT_RAW_FILE = (
    RAW_INPUT_DIR
    / "compustat_annual_1980_2025.csv"
)

WELCH_GOYAL_INPUT_FILE = (
    EXTERNAL_INPUT_DIR
    / "welch_goyal_macro_1990_2025.csv"
)


# ------------------------------------------------------------
# File 01: Daily Yahoo data
# ------------------------------------------------------------
DAILY_CLEAN_FILE = (
    STAGING_DATA_DIR
    / "001_daily_prices_clean.parquet"
)

YAHOO_DOWNLOAD_REPORT_FILE = (
    QUALITY_OUTPUT_DIR
    / "001_yahoo_download_report.csv"
)

DAILY_QUALITY_REPORT_FILE = (
    QUALITY_OUTPUT_DIR
    / "001_daily_ticker_quality.csv"
)


# ------------------------------------------------------------
# File 02: Monthly stock panel and target
# ------------------------------------------------------------
MONTHLY_STOCK_FILE = (
    STAGING_DATA_DIR
    / "002_monthly_stock_panel.parquet"
)


# ------------------------------------------------------------
# File 03: Compustat fundamentals
# ------------------------------------------------------------
COMPUSTAT_CLEAN_FILE = (
    STAGING_DATA_DIR
    / "003_compustat_annual_clean.parquet"
)

PANEL_WITH_FUNDAMENTALS_FILE = (
    STAGING_DATA_DIR
    / "004_monthly_panel_with_fundamentals.parquet"
)


# ------------------------------------------------------------
# File 04: Raw Kelly-style base dataset
# ------------------------------------------------------------
RAW_KELLY_FILE = (
    STAGING_DATA_DIR
    / "005_raw_kelly_base.parquet"
)

RAW_PREDICTOR_FILE = (
    STAGING_DATA_DIR
    / "005_raw_kelly_predictors.csv"
)

RAW_KELLY_SUMMARY_FILE = (
    QUALITY_OUTPUT_DIR
    / "005_raw_kelly_summary.csv"
)


# ------------------------------------------------------------
# File 05: Final cleaned Kelly-style dataset
# ------------------------------------------------------------
CLEAN_FULL_FILE = (
    FINAL_DATA_DIR
    / "006_model_dataset_kelly_winsorized_1990_2025.parquet"
)

CLEAN_PREDICTOR_FILE = (
    FINAL_DATA_DIR
    / "006_predictor_columns_kelly_winsorized.csv"
)

CLEANING_SUMMARY_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_cleaning_summary.csv"
)

DROPPED_PREDICTORS_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_dropped_predictors.csv"
)

DROPPED_STATE_ROWS_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_dropped_state_rows.csv"
)

MONTHLY_MEDIAN_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_monthly_imputation_medians.csv"
)

BINARY_IMPUTATION_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_binary_imputation_summary.csv"
)

EXTREME_TARGET_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_extreme_target_observations.csv"
)

EXTREME_TARGET_COUNT_FILE = (
    QUALITY_OUTPUT_DIR
    / "006_extreme_target_counts.csv"
)


# ------------------------------------------------------------
# Model and report outputs
# ------------------------------------------------------------
FINAL_SELECTION_FILE = (
    MODEL_OUTPUT_DIR
    / "final_selection.json"
)
