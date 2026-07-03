"""Clean and normalize the raw Kelly-style model dataset.

This file is the documented cleaning/preprocessing step.

Steps:
1. Read the raw Kelly-style dataset from file 04
2. Drop rows with missing target
3. Split chronologically into train, validation, and test
4. Drop predictors with more than 40% missing values in training data
5. Winsorize the target using training quantiles
6. Impute missing predictors using monthly cross-sectional medians
7. Cross-sectionally rank-normalize predictors by month to [-1, 1]
8. Save clean train, validation, and test files as Parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# 1) Settings
DATA_DIR = Path("output/data")
RAW_DIR = DATA_DIR / "final/kelly_style"
CLEAN_DIR = DATA_DIR / "final/kelly_style_clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "model_dataset_kelly_raw_full_1990_2025.csv"
RAW_PREDICTOR_FILE = RAW_DIR / "predictor_columns_kelly_raw.csv"

FULL_FILE = CLEAN_DIR / "model_dataset_kelly_ranked_full_1990_2025.parquet"
TRAIN_FILE = CLEAN_DIR / "model_train_kelly_ranked_1990_2014.parquet"
VALIDATION_FILE = CLEAN_DIR / "model_validation_kelly_ranked_2015_2019.parquet"
TEST_FILE = CLEAN_DIR / "model_test_kelly_ranked_2020_2025.parquet"

# Optional CSV outputs only for quick checking, not for modeling.
SUMMARY_FILE = CLEAN_DIR / "cleaning_summary.csv"
PREDICTOR_FILE = CLEAN_DIR / "predictor_columns_kelly_ranked.csv"
DROPPED_FILE = CLEAN_DIR / "dropped_predictors_missing_train.csv"
TARGET_WINSOR_FILE = CLEAN_DIR / "target_winsorization_limits_train.csv"
MONTHLY_MEDIAN_FILE = CLEAN_DIR / "monthly_imputation_medians_summary.csv"

TARGET = "target_excess_return_next_1m"

TRAIN_END = pd.Timestamp("2014-12-31")
VALIDATION_END = pd.Timestamp("2019-12-31")

MAX_MISSING_SHARE = 0.40
LOWER_Q = 0.01
UPPER_Q = 0.99


# 2) Helper functions
def split_data(data):
    train = data[data.month <= TRAIN_END].copy()
    validation = data[(data.month > TRAIN_END) & (data.month <= VALIDATION_END)].copy()
    test = data[data.month > VALIDATION_END].copy()

    return train, validation, test


def winsorize_target_using_train(train, validation, test):
    lower, upper = train[TARGET].quantile([LOWER_Q, UPPER_Q])

    for sample in (train, validation, test):
        sample[TARGET] = sample[TARGET].clip(lower, upper)

    return {
        "column": TARGET,
        "lower_1pct_train": lower,
        "upper_99pct_train": upper,
    }


def rank_to_minus_one_plus_one(values):
    """Transform cross-sectional ranks in one month to [-1, 1]."""
    ranks = values.rank(method="average", na_option="keep")
    count = ranks.notna().sum()

    if count <= 1:
        return pd.Series(0.0, index=values.index)

    return 2.0 * (ranks - 1.0) / (count - 1.0) - 1.0


def impute_and_rank_by_month(data, predictors):
    """Use monthly median imputation, then monthly rank normalization.

    This follows the Gu-Kelly-Xiu style logic used in the reference repo:
    within each month, missing characteristics are replaced by that month's
    cross-sectional median, and then characteristics are ranked to [-1, 1].
    """
    data = data.sort_values(["month", "ticker"]).copy()

    median_rows = []

    for month, index in data.groupby("month").groups.items():
        monthly = data.loc[index, predictors].copy()

        medians = monthly.median()
        monthly = monthly.fillna(medians)

        # If a variable is missing for the whole month, use 0 before ranking.
        monthly = monthly.fillna(0.0)

        for name in predictors:
            data.loc[index, name] = rank_to_minus_one_plus_one(monthly[name]).astype("float32")

        median_rows.append({
            "month": month,
            "predictors": len(predictors),
            "all_missing_predictor_months": int(medians.isna().sum()),
        })

    return data.sort_values(["ticker", "month"]).reset_index(drop=True), median_rows


def check_final_data(data, name):
    missing_values = int(data.isna().sum().sum())
    duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if missing_values or duplicates:
        raise ValueError(
            f"{name} checks failed: missing_values={missing_values}, "
            f"duplicate_ticker_months={duplicates}"
        )


def save_parquet(data, path):
    try:
        data.to_parquet(path, index=False)
    except ImportError as error:
        raise ImportError(
            "Saving Parquet requires pyarrow or fastparquet. "
            "Install pyarrow with: pip install pyarrow"
        ) from error


# 3) Run cleaning pipeline
def main():
    print("=" * 80)
    print("05_clean_and_normalize_model_data.py")
    print("=" * 80)

    data = pd.read_csv(RAW_FILE, low_memory=False)
    data["month"] = pd.to_datetime(data.month)
    data["ticker"] = data.ticker.astype(str).str.upper().str.strip()

    predictors = pd.read_csv(RAW_PREDICTOR_FILE)["predictor"].astype(str).tolist()
    predictors = [name for name in predictors if name in data.columns]

    raw_rows = len(data)
    raw_predictors = len(predictors)
    raw_missing_targets = int(data[TARGET].isna().sum())
    raw_duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if raw_duplicates:
        raise ValueError(f"Raw data has duplicate ticker-month rows: {raw_duplicates}")

    # Drop missing target rows.
    data = data[data[TARGET].notna()].copy()

    # Keep numeric predictors only.
    predictors = [
        name for name in predictors
        if pd.api.types.is_numeric_dtype(data[name])
    ]

    # Split before learning any rule based on training sample.
    train, validation, test = split_data(data)

    # Drop high-missing predictors based on training data only.
    missing_share = train[predictors].isna().mean()
    dropped = missing_share[missing_share > MAX_MISSING_SHARE].index.tolist()
    predictors = [name for name in predictors if name not in dropped]

    # Keep final columns.
    columns = ["ticker", "month", TARGET] + predictors
    train = train[columns].copy()
    validation = validation[columns].copy()
    test = test[columns].copy()

    # Winsorize target using training target distribution only.
    target_winsor_row = winsorize_target_using_train(train, validation, test)

    # Monthly median imputation + monthly rank normalization for predictors.
    train, train_medians = impute_and_rank_by_month(train, predictors)
    validation, validation_medians = impute_and_rank_by_month(validation, predictors)
    test, test_medians = impute_and_rank_by_month(test, predictors)

    full = (
        pd.concat([train, validation, test], ignore_index=True)
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    check_final_data(train, "Train")
    check_final_data(validation, "Validation")
    check_final_data(test, "Test")
    check_final_data(full, "Full")

    # Save light model files.
    save_parquet(full, FULL_FILE)
    save_parquet(train, TRAIN_FILE)
    save_parquet(validation, VALIDATION_FILE)
    save_parquet(test, TEST_FILE)

    # Save metadata as CSV.
    pd.DataFrame({"predictor": predictors}).to_csv(PREDICTOR_FILE, index=False)

    pd.DataFrame({
        "predictor": dropped,
        "missing_share_train": missing_share[dropped].values,
    }).to_csv(DROPPED_FILE, index=False)

    pd.DataFrame([target_winsor_row]).to_csv(TARGET_WINSOR_FILE, index=False)

    monthly_medians = pd.DataFrame(
        train_medians + validation_medians + test_medians
    )
    monthly_medians.to_csv(MONTHLY_MEDIAN_FILE, index=False)

    summary = {
        "raw_rows": raw_rows,
        "raw_predictors": raw_predictors,
        "raw_missing_targets": raw_missing_targets,
        "rows_after_dropping_missing_target": len(data),
        "final_rows": len(full),
        "final_columns": full.shape[1],
        "final_predictors": len(predictors),
        "dropped_high_missing_predictors": len(dropped),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_tickers": train.ticker.nunique(),
        "validation_tickers": validation.ticker.nunique(),
        "test_tickers": test.ticker.nunique(),
        "first_month": full.month.min(),
        "last_month": full.month.max(),
        "missing_values_final": int(full.isna().sum().sum()),
        "duplicate_ticker_months_final": int(full.duplicated(["ticker", "month"]).sum()),
        "normalization": "monthly_cross_sectional_median_imputation_then_rank_to_minus_one_plus_one",
        "file_format": "parquet",
    }

    pd.DataFrame(summary.items(), columns=["item", "value"]).to_csv(SUMMARY_FILE, index=False)

    print(f"Saved cleaned ranked full dataset: {full.shape}")
    print(f"Final predictors: {len(predictors)}")
    print(f"Dropped high-missing predictors: {len(dropped)}")
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Date range: {full.month.min()} to {full.month.max()}")
    print(f"Missing values final: {summary['missing_values_final']}")
    print(f"Duplicate ticker-months final: {summary['duplicate_ticker_months_final']}")
    print(f"Saved Parquet files in: {CLEAN_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()