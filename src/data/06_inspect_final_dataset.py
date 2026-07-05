"""Inspect the final cleaned/ranked stock-return ML dataset.

This script checks the final modeling data created by:
    src/data/05_clean_and_rank_normalize.py

Main purpose:
1. Confirm that the final dataset has no missing values.
2. Confirm that the target has no missing values.
3. Confirm that there are no duplicate ticker-month rows.
4. Check train/validation/test sample sizes.
5. Check target distribution after winsorization.
6. Check that ranked predictors are inside [-1, 1].
7. Save summary tables and diagnostic plots.

Important:
- This script is diagnostic only.
- It does not change the data.
- It should be rerun after the final cleaning file is rerun.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Allow direct execution from project root:
# python src/data/06_inspect_final_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    TRAIN_END,
    VALIDATION_END,
    CLEAN_FULL_FILE,
    CLEAN_PREDICTOR_FILE,
    REPORT_OUTPUT_DIR,
)


# ---------------------------------------------------------------------
# 1) Helpers
# ---------------------------------------------------------------------
def ensure_report_dir():
    """Create report output directory if it does not exist."""
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table(dataframe, filename):
    """Save a DataFrame to the report folder."""
    path = REPORT_OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_plot(filename):
    """Save the current matplotlib figure to the report folder."""
    path = REPORT_OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {path}")


def load_data():
    """Load final full dataset and recreate chronological splits."""
    full = pd.read_parquet(CLEAN_FULL_FILE)
    full["month"] = pd.to_datetime(full["month"])

    train = full[full["month"] <= TRAIN_END].copy()
    validation = full[
        (full["month"] > TRAIN_END)
        & (full["month"] <= VALIDATION_END)
    ].copy()
    test = full[full["month"] > VALIDATION_END].copy()

    predictors = (
        pd.read_csv(CLEAN_PREDICTOR_FILE)["predictor"]
        .astype(str)
        .tolist()
    )

    predictors = [name for name in predictors if name in full.columns]

    return full, train, validation, test, predictors


def target_summary(dataframe, name):
    """Create target summary for one sample."""
    y = dataframe[TARGET]

    return {
        "sample": name,
        "rows": len(dataframe),
        "tickers": dataframe["ticker"].nunique(),
        "first_month": dataframe["month"].min(),
        "last_month": dataframe["month"].max(),
        "target_mean": y.mean(),
        "target_std": y.std(),
        "target_min": y.min(),
        "target_p01": y.quantile(0.01),
        "target_p05": y.quantile(0.05),
        "target_p50": y.quantile(0.50),
        "target_p95": y.quantile(0.95),
        "target_p99": y.quantile(0.99),
        "target_max": y.max(),
    }


# ---------------------------------------------------------------------
# 2) Dataset checks
# ---------------------------------------------------------------------
def create_dataset_summary(full, train, validation, test, predictors):
    """Create one summary table for the final dataset."""
    duplicated_rows = full.duplicated(["ticker", "month"]).sum()

    summary = pd.DataFrame(
        [
            {
                "item": "full_rows",
                "value": len(full),
            },
            {
                "item": "full_columns",
                "value": full.shape[1],
            },
            {
                "item": "predictors",
                "value": len(predictors),
            },
            {
                "item": "unique_tickers",
                "value": full["ticker"].nunique(),
            },
            {
                "item": "first_month",
                "value": full["month"].min(),
            },
            {
                "item": "last_month",
                "value": full["month"].max(),
            },
            {
                "item": "train_rows",
                "value": len(train),
            },
            {
                "item": "validation_rows",
                "value": len(validation),
            },
            {
                "item": "test_rows",
                "value": len(test),
            },
            {
                "item": "missing_values_final",
                "value": int(full.isna().sum().sum()),
            },
            {
                "item": "missing_targets_final",
                "value": int(full[TARGET].isna().sum()),
            },
            {
                "item": "duplicate_ticker_months",
                "value": int(duplicated_rows),
            },
        ]
    )

    save_table(summary, "final_dataset_summary.csv")

    print("\nSTEP 1: FINAL DATASET SUMMARY")
    print(summary.to_string(index=False))


def create_split_target_summary(train, validation, test, full):
    """Create target distribution table by sample."""
    summary = pd.DataFrame(
        [
            target_summary(train, "train"),
            target_summary(validation, "validation"),
            target_summary(test, "test"),
            target_summary(full, "full"),
        ]
    )

    save_table(summary, "target_summary_by_sample.csv")

    print("\nSTEP 2: TARGET SUMMARY BY SAMPLE")
    print(summary.to_string(index=False))


def create_monthly_panel_summary(full):
    """Create monthly stock-count summary."""
    monthly = (
        full.groupby("month")
        .agg(
            rows=("ticker", "size"),
            tickers=("ticker", "nunique"),
            target_mean=(TARGET, "mean"),
            target_std=(TARGET, "std"),
        )
        .reset_index()
    )

    save_table(monthly, "monthly_panel_summary.csv")

    print("\nSTEP 3: MONTHLY PANEL SUMMARY")
    print(
        monthly[["rows", "tickers", "target_mean", "target_std"]]
        .describe()
        .to_string()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["tickers"])
    plt.title("Number of Stocks by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Stocks")
    save_plot("stock_count_by_month.png")

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["target_mean"])
    plt.title("Mean Target Return by Month")
    plt.xlabel("Month")
    plt.ylabel("Mean next-month excess return")
    save_plot("target_mean_by_month.png")

    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["target_std"])
    plt.title("Target Volatility by Month")
    plt.xlabel("Month")
    plt.ylabel("Cross-sectional target standard deviation")
    save_plot("target_volatility_by_month.png")


def check_predictor_ranges(full, predictors):
    """Check that ranked predictors are inside [-1, 1]."""
    mins = full[predictors].min()
    maxs = full[predictors].max()

    range_check = pd.DataFrame(
        {
            "predictor": predictors,
            "min": mins.values,
            "max": maxs.values,
            "below_minus_one": (mins.values < -1).astype(int),
            "above_plus_one": (maxs.values > 1).astype(int),
        }
    )

    range_check["outside_rank_range"] = (
        range_check["below_minus_one"].astype(bool)
        | range_check["above_plus_one"].astype(bool)
    )

    save_table(range_check, "predictor_rank_range_check.csv")

    outside = int(range_check["outside_rank_range"].sum())

    print("\nSTEP 4: PREDICTOR RANK RANGE CHECK")
    print(f"Predictors checked: {len(predictors)}")
    print(f"Predictors outside [-1, 1]: {outside}")

    if outside > 0:
        print(
            range_check[range_check["outside_rank_range"]]
            .head(30)
            .to_string(index=False)
        )


def create_predictor_distribution_summary(full, predictors):
    """Summarize final ranked predictor distributions."""
    selected = predictors[: min(100, len(predictors))]

    summary = full[selected].describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    ).T.reset_index()

    summary = summary.rename(columns={"index": "predictor"})

    save_table(summary, "predictor_distribution_summary_first_100.csv")

    negative_summary = pd.DataFrame(
        {
            "predictor": predictors,
            "share_negative": [(full[col] < 0).mean() for col in predictors],
            "share_zero": [(full[col] == 0).mean() for col in predictors],
            "share_positive": [(full[col] > 0).mean() for col in predictors],
        }
    )

    save_table(negative_summary, "predictor_negative_value_summary.csv")

    print("\nSTEP 5: PREDICTOR DISTRIBUTION SUMMARY")
    print("Saved distribution summaries for ranked predictors.")


def create_target_diagnostics(full):
    """Create target distribution diagnostics."""
    y = full[TARGET]

    largest = (
        full[["ticker", "month", TARGET]]
        .sort_values(TARGET, ascending=False)
        .head(30)
    )

    smallest = (
        full[["ticker", "month", TARGET]]
        .sort_values(TARGET, ascending=True)
        .head(30)
    )

    save_table(largest, "largest_target_returns.csv")
    save_table(smallest, "smallest_target_returns.csv")

    thresholds = [0.25, 0.50, 1.00, 2.00, 5.00, 10.00]

    counts = pd.DataFrame(
        {
            "abs_target_threshold": thresholds,
            "count": [(y.abs() > threshold).sum() for threshold in thresholds],
            "share": [(y.abs() > threshold).mean() for threshold in thresholds],
        }
    )

    save_table(counts, "target_extreme_return_counts.csv")

    print("\nSTEP 6: TARGET DIAGNOSTICS")
    print("Largest target values:")
    print(largest.to_string(index=False))
    print("\nSmallest target values:")
    print(smallest.to_string(index=False))
    print("\nExtreme target counts:")
    print(counts.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=100)
    plt.title("Target Distribution")
    plt.xlabel("Next-month excess return")
    plt.ylabel("Frequency")
    save_plot("target_distribution.png")

    trimmed = y[(y >= y.quantile(0.01)) & (y <= y.quantile(0.99))]

    plt.figure(figsize=(8, 5))
    plt.hist(trimmed, bins=100)
    plt.title("Target Distribution Trimmed to 1st-99th Percentiles")
    plt.xlabel("Next-month excess return")
    plt.ylabel("Frequency")
    save_plot("target_distribution_trimmed.png")


def create_selected_variable_plots(full, predictors):
    """Plot a few common variables if they exist."""
    selected_candidates = [
        "mom12m",
        "mom6m",
        "retvol_12m",
        "beta_12m",
        "idiovol_12m",
        "amihud_1m",
        "bm_comp",
        "op_at",
        "debt_at",
    ]

    existing = [col for col in selected_candidates if col in predictors]

    examples = []

    for col in existing:
        examples.append(
            {
                "predictor": col,
                "min": full[col].min(),
                "p01": full[col].quantile(0.01),
                "p50": full[col].quantile(0.50),
                "p99": full[col].quantile(0.99),
                "max": full[col].max(),
                "share_negative": (full[col] < 0).mean(),
                "share_positive": (full[col] > 0).mean(),
            }
        )

        plt.figure(figsize=(8, 5))
        plt.hist(full[col], bins=100)
        plt.title(f"Ranked Distribution: {col}")
        plt.xlabel(f"{col} ranked value")
        plt.ylabel("Frequency")
        save_plot(f"{col}_rank_distribution.png")

    if examples:
        save_table(pd.DataFrame(examples), "selected_variable_examples.csv")


# ---------------------------------------------------------------------
# 3) Main
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("06_inspect_final_dataset.py")
    print("=" * 80)

    ensure_report_dir()

    full, train, validation, test, predictors = load_data()

    create_dataset_summary(full, train, validation, test, predictors)
    create_split_target_summary(train, validation, test, full)
    create_monthly_panel_summary(full)
    check_predictor_ranges(full, predictors)
    create_predictor_distribution_summary(full, predictors)
    create_target_diagnostics(full)
    create_selected_variable_plots(full, predictors)

    print("\n" + "=" * 80)
    print("Inspection complete.")
    print(f"Reports saved in: {REPORT_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
