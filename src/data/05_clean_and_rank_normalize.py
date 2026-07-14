"""Clean and rank-normalize the raw Kelly-style predictor dataset.

This file does not split the data. Train, validation, and test samples
are defined later inside the model workflow.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    RAW_KELLY_FILE,
    RAW_PREDICTOR_FILE,
    CLEAN_FULL_FILE,
    CLEAN_PREDICTOR_FILE,
    CLEANING_SUMMARY_FILE,
    DROPPED_PREDICTORS_FILE,
    MONTHLY_MEDIAN_FILE,
    EXTREME_TARGET_FILE,
    EXTREME_TARGET_COUNT_FILE,
)
from src.feature_definitions import (  # noqa: E402
    LOCKED_CHARACTERISTIC_COLUMNS,
    LOCKED_MARKET_COLUMNS,
    LOCKED_MACRO_COLUMNS,
)


# ---------------------------------------------------------------------
# 1) Document extreme raw targets
# ---------------------------------------------------------------------
def target_diagnostics(data):
    """Create reports for unusually large target returns."""
    thresholds = [0.25, 0.50, 1.00, 2.00]

    extreme = (
        data.loc[
            data[TARGET].abs().gt(1.0),
            ["ticker", "month", TARGET],
        ]
        .assign(abs_target=lambda frame: frame[TARGET].abs())
        .sort_values("abs_target", ascending=False)
    )

    counts = pd.DataFrame([
        {
            "abs_target_threshold": threshold,
            "count": int(data[TARGET].abs().gt(threshold).sum()),
            "share": float(data[TARGET].abs().gt(threshold).mean()),
        }
        for threshold in thresholds
    ])

    return extreme, counts


# ---------------------------------------------------------------------
# 2) Impute and rank predictors within each month
# ---------------------------------------------------------------------
def impute_and_rank(data, predictors):
    """Impute monthly medians and map cross-sectional ranks to [-1, 1]."""
    data = data.sort_values(["month", "ticker"]).copy()
    data[predictors] = data[predictors].astype("float32")

    diagnostics = []

    for month, index in data.groupby("month", sort=False).groups.items():
        values = data.loc[index, predictors]
        medians = values.median()

        values = values.fillna(medians).fillna(0.0)
        ranks = values.rank(method="average")

        if len(values) > 1:
            values = 2 * (ranks - 1) / (len(values) - 1) - 1
        else:
            values.loc[:, :] = 0.0

        data.loc[index, predictors] = values.astype("float32")

        diagnostics.append({
            "month": month,
            "firms": len(index),
            "all_missing_predictors": int(medians.isna().sum()),
        })

    return (
        data.sort_values(["ticker", "month"]).reset_index(drop=True),
        diagnostics,
    )


# ---------------------------------------------------------------------
# 3) Validate the final dataset
# ---------------------------------------------------------------------
def check_final_data(data):
    """Check missing values, duplicates, and predictor ranges."""
    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if missing or duplicates:
        raise ValueError(
            f"Final checks failed: missing={missing}, "
            f"duplicate_ticker_months={duplicates}"
        )

    predictors = data.columns.drop(["ticker", "month", TARGET])
    outside_range = (
        data[predictors].lt(-1).any().any()
        or data[predictors].gt(1).any().any()
    )

    if outside_range:
        raise ValueError("Rank-normalized predictors are outside [-1, 1].")


# ---------------------------------------------------------------------
# 4) Run cleaning pipeline
# ---------------------------------------------------------------------
def main():
    """Create and save the full model-ready dataset."""
    data = pd.read_csv(RAW_KELLY_FILE, low_memory=False)

    data["month"] = pd.to_datetime(data["month"])
    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    requested = (
        pd.read_csv(RAW_PREDICTOR_FILE)["predictor"]
        .astype(str)
        .tolist()
    )

    locked_interactions = [
        f"{characteristic}_x_{macro}"
        for macro in LOCKED_MACRO_COLUMNS
        for characteristic in LOCKED_CHARACTERISTIC_COLUMNS
    ]

    locked_predictors = (
        LOCKED_CHARACTERISTIC_COLUMNS
        + LOCKED_MARKET_COLUMNS
        + locked_interactions
    )

    missing_locked = [
        name for name in locked_predictors
        if name not in requested or name not in data.columns
    ]

    if missing_locked:
        print(
            "Warning: missing locked predictors before cleaning: "
            f"{missing_locked}"
        )

    raw_rows = len(data)
    raw_missing_targets = int(data[TARGET].isna().sum())
    duplicates = int(data.duplicated(["ticker", "month"]).sum())

    if duplicates:
        raise ValueError(
            f"Raw data contains {duplicates} duplicate ticker-month rows."
        )

    # A supervised model cannot use observations without an outcome.
    data = data[data[TARGET].notna()].copy()

    # Keep usable numeric predictors.
    predictors = [
        name
        for name in requested
        if name in data.columns
        and pd.api.types.is_numeric_dtype(data[name])
    ]

    # Drop only completely empty predictors.
    dropped = [
        name
        for name in predictors
        if data[name].isna().all()
    ]

    dropped_locked = [
        name for name in dropped
        if name in locked_predictors
    ]

    if dropped_locked:
        print(
            "Warning: locked predictors are completely empty and will be "
            f"reported as dropped: {dropped_locked}"
        )

    predictors = [
        name
        for name in predictors
        if name not in dropped
    ]

    extreme_targets, extreme_counts = target_diagnostics(data)

    columns = ["ticker", "month", TARGET] + predictors
    clean = data[columns].copy()

    clean, monthly_diagnostics = impute_and_rank(
        clean,
        predictors,
    )

    check_final_data(clean)

    clean.to_parquet(CLEAN_FULL_FILE, index=False)

    pd.DataFrame({
        "predictor": predictors
    }).to_csv(
        CLEAN_PREDICTOR_FILE,
        index=False,
    )

    pd.DataFrame({
        "predictor": dropped,
        "reason": "completely_missing",
    }).to_csv(
        DROPPED_PREDICTORS_FILE,
        index=False,
    )

    pd.DataFrame(monthly_diagnostics).to_csv(
        MONTHLY_MEDIAN_FILE,
        index=False,
    )

    extreme_targets.to_csv(
        EXTREME_TARGET_FILE,
        index=False,
    )

    extreme_counts.to_csv(
        EXTREME_TARGET_COUNT_FILE,
        index=False,
    )

    summary = {
        "raw_rows": raw_rows,
        "raw_missing_targets": raw_missing_targets,
        "rows_after_target_drop": len(clean),
        "final_columns": clean.shape[1],
        "final_predictors": len(predictors),
        "locked_stock_characteristics": len(LOCKED_CHARACTERISTIC_COLUMNS),
        "locked_market_variables": len(LOCKED_MARKET_COLUMNS),
        "locked_macro_variables": len(LOCKED_MACRO_COLUMNS),
        "locked_interactions": len(locked_interactions),
        "missing_locked_predictors": len(missing_locked),
        "dropped_empty_locked_predictors": len(dropped_locked),
        "dropped_empty_predictors": len(dropped),
        "unique_tickers": clean["ticker"].nunique(),
        "first_month": clean["month"].min(),
        "last_month": clean["month"].max(),
        "missing_values_final": int(clean.isna().sum().sum()),
        "missing_targets_final": int(clean[TARGET].isna().sum()),
        "duplicate_ticker_months_final": int(
            clean.duplicated(["ticker", "month"]).sum()
        ),
        "predictor_imputation": "monthly_cross_sectional_median",
        "predictor_normalization": "monthly_rank_to_minus_one_plus_one",
        "target_treatment": "raw_target_not_winsorized",
        "sample_splitting": "deferred_to_model_workflow",
    }

    pd.DataFrame(
        summary.items(),
        columns=["item", "value"],
    ).to_csv(
        CLEANING_SUMMARY_FILE,
        index=False,
    )

    print(f"Saved: {CLEAN_FULL_FILE}")
    print(f"Shape: {clean.shape}")
    print(f"Predictors: {len(predictors)}")
    print(f"Tickers: {clean['ticker'].nunique()}")
    print(f"Dropped empty predictors: {len(dropped)}")
    print(
        f"Date range: {clean['month'].min()} "
        f"to {clean['month'].max()}"
    )
    print(f"Missing values: {summary['missing_values_final']}")
    print(
        f"Duplicate ticker-months: "
        f"{summary['duplicate_ticker_months_final']}"
    )


if __name__ == "__main__":
    main()
