"""Clean and rank-normalize the raw Kelly-style model dataset.

This file is the documented preprocessing step.

Steps:
1. Read the raw Kelly-style dataset from file 04
2. Drop rows with missing target
3. Automatically flag extreme target observations for documentation
4. Keep numeric predictors only
5. Split chronologically into train, validation, and test
6. Winsorize the target using training-sample cutoffs only
7. Drop predictors with more than 40% missing values in training data
8. Impute missing predictors using monthly cross-sectional medians
9. Cross-sectionally rank-normalize predictors by month to [-1, 1]
10. Save clean full, train, validation, and test files as Parquet

Important:
- Predictors are not winsorized because rank-normalization already reduces
  the influence of extreme predictor values.
- The target is winsorized at the 1st and 99th percentiles using training-sample
  cutoffs only. This avoids look-ahead bias.
- Extreme target observations are flagged and saved for documentation, but they
  are not mechanically deleted.
- This follows the reference repo style for predictors: monthly median imputation
  followed by monthly cross-sectional rank-normalization.
"""

from pathlib import Path
import sys

import pandas as pd


# Allow direct execution from the project root:
# python src/data/05_clean_and_rank_normalize.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    TRAIN_END,
    VALIDATION_END,
    MAX_MISSING_SHARE,
    LOWER_Q,
    UPPER_Q,
    RAW_KELLY_FILE,
    RAW_PREDICTOR_FILE,
    CLEAN_FULL_FILE,
    CLEAN_TRAIN_FILE,
    CLEAN_VALIDATION_FILE,
    CLEAN_TEST_FILE,
    CLEAN_PREDICTOR_FILE,
    CLEANING_SUMMARY_FILE,
    DROPPED_PREDICTORS_FILE,
    MONTHLY_MEDIAN_FILE,
    TARGET_WINSOR_FILE,
    EXTREME_TARGET_FILE,
    EXTREME_TARGET_COUNT_FILE,
)


# ---------------------------------------------------------------------
# 1) Split, target, and rank-normalization helpers
# ---------------------------------------------------------------------
def split_data(data):
    """Split the panel chronologically."""
    train = data[data["month"] <= TRAIN_END].copy()

    validation = data[
        (data["month"] > TRAIN_END)
        & (data["month"] <= VALIDATION_END)
    ].copy()

    test = data[data["month"] > VALIDATION_END].copy()

    return train, validation, test


def flag_extreme_targets(data):
    """Flag extreme target observations for documentation.

    This function does not delete observations.

    Some extreme target returns may be real market events, while others may be
    caused by adjustment, split, ticker-history, or data-quality problems.
    Because we cannot automatically know which are real and which are errors,
    we save an objective report and then use winsorization to limit their
    influence on squared-error models.
    """
    thresholds = [0.25, 0.50, 1.00, 2.00, 5.00, 10.00]

    data = data.copy()

    extreme_rows = data[data[TARGET].abs() > 1.00][
        ["ticker", "month", TARGET]
    ].copy()

    extreme_rows = extreme_rows.sort_values(
        TARGET,
        key=lambda values: values.abs(),
        ascending=False,
    )

    count_rows = []

    for threshold in thresholds:
        count_rows.append({
            "threshold_abs_return": threshold,
            "count_above_positive_threshold": int((data[TARGET] > threshold).sum()),
            "count_below_negative_threshold": int((data[TARGET] < -threshold).sum()),
            "share_above_positive_threshold": (data[TARGET] > threshold).mean(),
            "share_below_negative_threshold": (data[TARGET] < -threshold).mean(),
        })

    extreme_counts = pd.DataFrame(count_rows)

    return extreme_rows, extreme_counts


def winsorize_target_using_train(train, validation, test):
    """Winsorize the target using training-sample cutoffs only.

    The lower and upper cutoffs are computed only from the training sample.
    The same cutoffs are then applied to train, validation, and test.

    This avoids look-ahead bias because validation and test information is not
    used to choose the winsorization thresholds.
    """
    lower_cutoff = train[TARGET].quantile(LOWER_Q)
    upper_cutoff = train[TARGET].quantile(UPPER_Q)

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train[TARGET] = train[TARGET].clip(lower_cutoff, upper_cutoff)
    validation[TARGET] = validation[TARGET].clip(lower_cutoff, upper_cutoff)
    test[TARGET] = test[TARGET].clip(lower_cutoff, upper_cutoff)

    cutoffs = {
        "lower_quantile": LOWER_Q,
        "upper_quantile": UPPER_Q,
        "lower_cutoff_train": lower_cutoff,
        "upper_cutoff_train": upper_cutoff,
        "target_treatment": (
            "target_winsorized_at_training_sample_"
            f"{LOWER_Q:.2f}_{UPPER_Q:.2f}_quantiles"
        ),
    }

    return train, validation, test, cutoffs


def rank_to_minus_one_plus_one(values):
    """Transform cross-sectional ranks in one month to [-1, 1].

    Formula:
        rank_scaled = 2 * (rank - 1) / (N - 1) - 1

    The smallest value becomes -1, the largest value becomes +1,
    and the middle observations are placed between them.
    """
    ranks = values.rank(method="average", na_option="keep")
    count = ranks.notna().sum()

    if count <= 1:
        return pd.Series(0.0, index=values.index)

    return 2.0 * (ranks - 1.0) / (count - 1.0) - 1.0


def impute_and_rank_by_month(data, predictors):
    """Impute predictors by monthly median and rank-normalize by month.

    For every month separately:
    1. Take all firms observed in that month.
    2. Replace missing predictor values with that month's cross-sectional median.
    3. If a predictor is missing for the whole month, replace remaining missing
       values by 0.
    4. Rank-normalize the predictor cross-section to [-1, 1].

    This is the main preprocessing step learned from the reference repo.
    """
    data = data.sort_values(["month", "ticker"]).copy()
    data[predictors] = data[predictors].astype("float32")

    median_rows = []

    for month, index in data.groupby("month").groups.items():
        monthly_values = data.loc[index, predictors].copy()

        medians = monthly_values.median()
        monthly_values = monthly_values.fillna(medians)

        # If a predictor is missing for the full month, use 0 before ranking.
        monthly_values = monthly_values.fillna(0.0)

        for predictor in predictors:
            data.loc[index, predictor] = (
                rank_to_minus_one_plus_one(monthly_values[predictor])
                .astype("float32")
            )

        median_rows.append({
            "month": month,
            "predictors": len(predictors),
            "all_missing_predictor_months": int(medians.isna().sum()),
        })

    data = data.sort_values(["ticker", "month"]).reset_index(drop=True)
    data[predictors] = data[predictors].astype("float32")

    return data, median_rows


# ---------------------------------------------------------------------
# 2) Final checks and saving
# ---------------------------------------------------------------------
def check_final_data(data, name):
    """Check that the final model data has no missing values or duplicates."""
    missing_values = int(data.isna().sum().sum())
    duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if missing_values or duplicates:
        raise ValueError(
            f"{name} checks failed: "
            f"missing_values={missing_values}, "
            f"duplicate_ticker_months={duplicates}"
        )


def save_parquet(data, path):
    """Save a DataFrame as Parquet."""
    try:
        data.to_parquet(path, index=False)

    except ImportError as error:
        raise ImportError(
            "Saving Parquet requires pyarrow or fastparquet. "
            "Install pyarrow with: pip install pyarrow"
        ) from error


# ---------------------------------------------------------------------
# 3) Run cleaning pipeline
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("05_clean_and_rank_normalize.py")
    print("=" * 80)

    data = pd.read_csv(RAW_KELLY_FILE, low_memory=False)

    data["month"] = pd.to_datetime(data["month"])
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()

    predictors = (
        pd.read_csv(RAW_PREDICTOR_FILE)["predictor"]
        .astype(str)
        .tolist()
    )

    predictors = [name for name in predictors if name in data.columns]

    raw_rows = len(data)
    raw_predictors = len(predictors)
    raw_missing_targets = int(data[TARGET].isna().sum())
    raw_duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if raw_duplicates:
        raise ValueError(
            f"Raw data has duplicate ticker-month rows: {raw_duplicates}"
        )

    # Drop rows where the next-month target is unavailable.
    data = data[data[TARGET].notna()].copy()

    rows_after_dropping_missing_target = len(data)

    # Automatically flag extreme target observations for documentation.
    extreme_targets, extreme_target_counts = flag_extreme_targets(data)

    # Keep numeric predictors only.
    predictors = [
        name
        for name in predictors
        if pd.api.types.is_numeric_dtype(data[name])
    ]

    # Split chronologically before learning any preprocessing rule.
    train, validation, test = split_data(data)

    # Winsorize the target using training-sample cutoffs only.
    train, validation, test, target_cutoffs = winsorize_target_using_train(
        train,
        validation,
        test,
    )

    # Drop high-missing predictors based only on training data.
    missing_share = train[predictors].isna().mean()

    dropped = (
        missing_share[missing_share > MAX_MISSING_SHARE]
        .index
        .tolist()
    )

    predictors = [name for name in predictors if name not in dropped]

    # Keep only final modeling columns.
    columns = ["ticker", "month", TARGET] + predictors

    train = train[columns].copy()
    validation = validation[columns].copy()
    test = test[columns].copy()

    # Rank-normalization creates float values, so cast predictors first.
    train[predictors] = train[predictors].astype("float32")
    validation[predictors] = validation[predictors].astype("float32")
    test[predictors] = test[predictors].astype("float32")

    # Repo-style preprocessing:
    # monthly cross-sectional median imputation + monthly rank-normalization.
    train, train_medians = impute_and_rank_by_month(train, predictors)

    validation, validation_medians = impute_and_rank_by_month(
        validation,
        predictors,
    )

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

    # Save clean model files.
    save_parquet(full, CLEAN_FULL_FILE)
    save_parquet(train, CLEAN_TRAIN_FILE)
    save_parquet(validation, CLEAN_VALIDATION_FILE)
    save_parquet(test, CLEAN_TEST_FILE)

    # Save metadata files.
    pd.DataFrame({"predictor": predictors}).to_csv(
        CLEAN_PREDICTOR_FILE,
        index=False,
    )

    pd.DataFrame({
        "predictor": dropped,
        "missing_share_train": missing_share[dropped].values,
    }).to_csv(DROPPED_PREDICTORS_FILE, index=False)

    monthly_medians = pd.DataFrame(
        train_medians + validation_medians + test_medians
    )

    monthly_medians.to_csv(MONTHLY_MEDIAN_FILE, index=False)

    pd.DataFrame(
        target_cutoffs.items(),
        columns=["item", "value"],
    ).to_csv(TARGET_WINSOR_FILE, index=False)

    extreme_targets.to_csv(EXTREME_TARGET_FILE, index=False)
    extreme_target_counts.to_csv(EXTREME_TARGET_COUNT_FILE, index=False)

    summary = {
        "raw_rows": raw_rows,
        "raw_predictors": raw_predictors,
        "raw_missing_targets": raw_missing_targets,
        "rows_after_dropping_missing_target": rows_after_dropping_missing_target,
        "extreme_target_abs_above_100pct": len(extreme_targets),
        "target_lower_quantile": target_cutoffs["lower_quantile"],
        "target_upper_quantile": target_cutoffs["upper_quantile"],
        "target_lower_cutoff_train": target_cutoffs["lower_cutoff_train"],
        "target_upper_cutoff_train": target_cutoffs["upper_cutoff_train"],
        "final_rows": len(full),
        "final_columns": full.shape[1],
        "final_predictors": len(predictors),
        "dropped_high_missing_predictors": len(dropped),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_tickers": train["ticker"].nunique(),
        "validation_tickers": validation["ticker"].nunique(),
        "test_tickers": test["ticker"].nunique(),
        "first_month": full["month"].min(),
        "last_month": full["month"].max(),
        "missing_values_final": int(full.isna().sum().sum()),
        "missing_targets_final": int(full[TARGET].isna().sum()),
        "duplicate_ticker_months_final": int(
            full.duplicated(["ticker", "month"]).sum()
        ),
        "target_treatment": target_cutoffs["target_treatment"],
        "target_outlier_handling": (
            "extreme_targets_flagged_for_documentation_not_deleted"
        ),
        "predictor_imputation": "monthly_cross_sectional_median",
        "predictor_normalization": (
            "monthly_cross_sectional_rank_to_minus_one_plus_one"
        ),
        "file_format": "parquet",
    }

    pd.DataFrame(
        summary.items(),
        columns=["item", "value"],
    ).to_csv(CLEANING_SUMMARY_FILE, index=False)

    print(f"Saved cleaned ranked full dataset: {full.shape}")
    print(f"Final predictors: {len(predictors)}")
    print(f"Dropped high-missing predictors: {len(dropped)}")
    print(f"Extreme targets flagged abs(return) > 100%: {len(extreme_targets)}")
    print(
        "Target winsorization cutoffs from train: "
        f"{target_cutoffs['lower_cutoff_train']:.6f}, "
        f"{target_cutoffs['upper_cutoff_train']:.6f}"
    )
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Date range: {full['month'].min()} to {full['month'].max()}")
    print(f"Missing values final: {summary['missing_values_final']}")
    print(f"Missing targets final: {summary['missing_targets_final']}")
    print(
        "Duplicate ticker-months final: "
        f"{summary['duplicate_ticker_months_final']}"
    )
    print(f"Saved Parquet files in: {CLEAN_FULL_FILE.parent}")
    print("=" * 80)


if __name__ == "__main__":
    main()