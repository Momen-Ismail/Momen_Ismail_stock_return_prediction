"""Inspect the final cleaned modeling dataset.

This file does not change or clean the data.

It checks whether the final model dataset is reasonable and saves diagnostic
tables and graphs that can be used in the project report.

Main checks:
1. Basic shape and sample periods
2. Missing values and duplicate ticker-month rows
3. Train / validation / test split sizes
4. Target distribution and target outliers
5. Predictor rank-normalization range [-1, 1]
6. Explanation of negative predictor values
7. Number of stocks per month
8. Examples of selected variables
9. Output tables and plots for documentation

Important interpretation:
- The target variable is raw next-month excess return.
- The predictors are rank-normalized by month to [-1, 1].
- Therefore, negative predictor values are expected and are not data errors.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Allow direct execution from the project root:
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
# 1) Load final data
# ---------------------------------------------------------------------
def load_data():
    """Load the cleaned full dataset and recreate the chronological splits.

    Reading only the full Parquet file is enough for inspection and avoids
    unnecessary repeated Parquet reads.
    """
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


# ---------------------------------------------------------------------
# 2) Basic dataset checks
# ---------------------------------------------------------------------
def create_sample_summary(full, train, validation, test, predictors):
    """Create summary for full, train, validation, and test samples."""
    rows = []

    samples = {
        "full": full,
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, data in samples.items():
        rows.append({
            "sample": name,
            "rows": len(data),
            "columns": data.shape[1],
            "predictors": len(predictors),
            "tickers": data["ticker"].nunique(),
            "first_month": data["month"].min(),
            "last_month": data["month"].max(),
            "missing_values": int(data.isna().sum().sum()),
            "missing_targets": int(data[TARGET].isna().sum()),
            "duplicate_ticker_months": int(
                data.duplicated(["ticker", "month"]).sum()
            ),
            "target_mean": data[TARGET].mean(),
            "target_std": data[TARGET].std(),
            "target_min": data[TARGET].min(),
            "target_p01": data[TARGET].quantile(0.01),
            "target_p05": data[TARGET].quantile(0.05),
            "target_p50": data[TARGET].quantile(0.50),
            "target_p95": data[TARGET].quantile(0.95),
            "target_p99": data[TARGET].quantile(0.99),
            "target_max": data[TARGET].max(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_OUTPUT_DIR / "final_dataset_summary.csv", index=False)

    return summary


def create_monthly_panel_summary(full):
    """Check number of observations and stocks per month."""
    monthly = (
        full.groupby("month")
        .agg(
            rows=("ticker", "size"),
            tickers=("ticker", "nunique"),
            target_mean=(TARGET, "mean"),
            target_std=(TARGET, "std"),
            target_min=(TARGET, "min"),
            target_max=(TARGET, "max"),
        )
        .reset_index()
    )

    monthly.to_csv(REPORT_OUTPUT_DIR / "monthly_panel_summary.csv", index=False)

    return monthly


# ---------------------------------------------------------------------
# 3) Target checks
# ---------------------------------------------------------------------
def create_target_outlier_report(full):
    """Save the largest and smallest target observations.

    The target is not rank-normalized or winsorized. Therefore, very large
    positive or negative returns should be inspected manually.
    """
    columns = ["ticker", "month", TARGET]

    largest = (
        full[columns]
        .sort_values(TARGET, ascending=False)
        .head(50)
        .copy()
    )

    smallest = (
        full[columns]
        .sort_values(TARGET, ascending=True)
        .head(50)
        .copy()
    )

    largest.to_csv(REPORT_OUTPUT_DIR / "largest_target_returns.csv", index=False)
    smallest.to_csv(REPORT_OUTPUT_DIR / "smallest_target_returns.csv", index=False)

    return largest, smallest


def create_target_extreme_count(full):
    """Count how many target returns are economically extreme."""
    thresholds = [0.25, 0.50, 1.00, 2.00, 5.00, 10.00]

    rows = []

    for threshold in thresholds:
        rows.append({
            "threshold_abs_return": threshold,
            "count_above_positive_threshold": int((full[TARGET] > threshold).sum()),
            "count_below_negative_threshold": int((full[TARGET] < -threshold).sum()),
            "share_above_positive_threshold": (full[TARGET] > threshold).mean(),
            "share_below_negative_threshold": (full[TARGET] < -threshold).mean(),
        })

    extreme_counts = pd.DataFrame(rows)
    extreme_counts.to_csv(
        REPORT_OUTPUT_DIR / "target_extreme_return_counts.csv",
        index=False,
    )

    return extreme_counts


# ---------------------------------------------------------------------
# 4) Predictor checks
# ---------------------------------------------------------------------
def create_predictor_range_check(full, predictors):
    """Check that rank-normalized predictors lie inside [-1, 1]."""
    ranges = full[predictors].agg(["min", "max", "mean", "std"]).T.reset_index()
    ranges = ranges.rename(columns={"index": "predictor"})

    ranges["below_minus_one"] = ranges["min"] < -1.0001
    ranges["above_plus_one"] = ranges["max"] > 1.0001
    ranges["inside_rank_range"] = (
        ~ranges["below_minus_one"] & ~ranges["above_plus_one"]
    )

    ranges.to_csv(REPORT_OUTPUT_DIR / "predictor_rank_range_check.csv", index=False)

    return ranges


def create_predictor_distribution_summary(full, predictors):
    """Save percentile summary of all predictors."""
    percentiles = [0.00, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]

    summary = (
        full[predictors]
        .quantile(percentiles)
        .T
        .reset_index()
        .rename(columns={"index": "predictor"})
    )

    summary.columns = [
        "predictor",
        "p00_min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "p100_max",
    ]

    summary["mean"] = full[predictors].mean().values
    summary["std"] = full[predictors].std().values

    summary.to_csv(
        REPORT_OUTPUT_DIR / "predictor_distribution_summary.csv",
        index=False,
    )

    return summary


def create_negative_value_explanation(full, predictors):
    """Count negative, zero, and positive values for each predictor.

    Negative predictor values are expected because predictors are rank-normalized
    to [-1, 1]. A negative value means the firm is below the cross-sectional
    middle for that predictor in that month.
    """
    rows = []

    for predictor in predictors:
        values = full[predictor]

        rows.append({
            "predictor": predictor,
            "share_negative": (values < 0).mean(),
            "share_zero": (values == 0).mean(),
            "share_positive": (values > 0).mean(),
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
        })

    negative_summary = pd.DataFrame(rows)

    negative_summary.to_csv(
        REPORT_OUTPUT_DIR / "predictor_negative_value_summary.csv",
        index=False,
    )

    return negative_summary


def create_selected_variable_examples(full, predictors):
    """Save detailed examples for important variables."""
    selected = [
        TARGET,
        "mom1m",
        "mom12m",
        "beta_12m",
        "idiovol_12m",
        "amihud_1m",
        "bm_comp",
        "gp_at",
        "debt_at",
        "asset_growth",
        "wg_dp",
        "wg_ep",
        "wg_bm",
        "wg_tbl",
        "wg_tms",
        "wg_dfy",
        "wg_svar",
    ]

    selected = [name for name in selected if name in full.columns]

    rows = []

    for name in selected:
        values = full[name]

        rows.append({
            "variable": name,
            "is_target": name == TARGET,
            "is_predictor": name in predictors,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p01": values.quantile(0.01),
            "p05": values.quantile(0.05),
            "p50": values.quantile(0.50),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "max": values.max(),
            "interpretation": variable_interpretation(name),
        })

    examples = pd.DataFrame(rows)
    examples.to_csv(REPORT_OUTPUT_DIR / "selected_variable_examples.csv", index=False)

    return examples


def variable_interpretation(name):
    """Short explanation of selected variables."""
    explanations = {
        TARGET: (
            "Raw next-month stock excess return. Negative values mean the stock "
            "underperformed the risk-free rate next month."
        ),
        "mom1m": (
            "Rank-normalized current-month return. Negative means relatively low "
            "return compared with other stocks in the same month."
        ),
        "mom12m": (
            "Rank-normalized past 12-month momentum excluding the current month."
        ),
        "beta_12m": "Rank-normalized rolling market beta.",
        "idiovol_12m": (
            "Rank-normalized residual volatility from a rolling market regression."
        ),
        "amihud_1m": "Rank-normalized Amihud illiquidity measure.",
        "bm_comp": "Rank-normalized book-to-market ratio.",
        "gp_at": "Rank-normalized gross profitability scaled by assets.",
        "debt_at": "Rank-normalized total debt scaled by assets.",
        "asset_growth": "Rank-normalized asset growth.",
        "wg_dp": "Lagged Welch-Goyal dividend-price ratio macro variable.",
        "wg_ep": "Lagged Welch-Goyal earnings-price ratio macro variable.",
        "wg_bm": "Lagged Welch-Goyal aggregate book-to-market macro variable.",
        "wg_tbl": "Lagged Treasury bill rate macro variable.",
        "wg_tms": "Lagged term spread macro variable.",
        "wg_dfy": "Lagged default spread macro variable.",
        "wg_svar": "Lagged stock variance macro variable.",
    }

    return explanations.get(name, "Rank-normalized predictor or macro variable.")


# ---------------------------------------------------------------------
# 5) Plots
# ---------------------------------------------------------------------
def plot_stock_count_by_month(monthly):
    """Plot number of stocks per month."""
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["tickers"])
    plt.title("Number of stocks per month")
    plt.xlabel("Month")
    plt.ylabel("Number of tickers")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / "stock_count_by_month.png", dpi=300)
    plt.close()


def plot_target_distribution(full):
    """Plot distribution of the target."""
    plt.figure(figsize=(8, 5))
    plt.hist(full[TARGET], bins=100)
    plt.title("Distribution of next-month excess returns")
    plt.xlabel("Next-month excess return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / "target_distribution.png", dpi=300)
    plt.close()


def plot_target_distribution_trimmed(full):
    """Plot target distribution after trimming only the graph range.

    This graph does not change the data. It only zooms in on the central part
    of the distribution to make the histogram readable.
    """
    lower = full[TARGET].quantile(0.01)
    upper = full[TARGET].quantile(0.99)

    trimmed = full[(full[TARGET] >= lower) & (full[TARGET] <= upper)]

    plt.figure(figsize=(8, 5))
    plt.hist(trimmed[TARGET], bins=100)
    plt.title("Distribution of next-month excess returns, 1st-99th percentile")
    plt.xlabel("Next-month excess return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / "target_distribution_trimmed.png", dpi=300)
    plt.close()


def plot_target_mean_by_month(monthly):
    """Plot average target return by month."""
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["target_mean"])
    plt.title("Average next-month excess return by month")
    plt.xlabel("Month")
    plt.ylabel("Average next-month excess return")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / "target_mean_by_month.png", dpi=300)
    plt.close()


def plot_target_volatility_by_month(monthly):
    """Plot cross-sectional target volatility by month."""
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["target_std"])
    plt.title("Cross-sectional volatility of next-month excess returns")
    plt.xlabel("Month")
    plt.ylabel("Cross-sectional standard deviation")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / "target_volatility_by_month.png", dpi=300)
    plt.close()


def plot_example_predictor_distribution(full, variable):
    """Plot one example predictor distribution."""
    if variable not in full.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(full[variable], bins=50)
    plt.title(f"Distribution of rank-normalized {variable}")
    plt.xlabel(f"{variable} after rank-normalization")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT_DIR / f"{variable}_rank_distribution.png", dpi=300)
    plt.close()


def plot_macro_variables(full):
    """Plot the clean Welch-Goyal macro variables over time."""
    macro_variables = [
        "wg_dp",
        "wg_ep",
        "wg_bm",
        "wg_ntis",
        "wg_tbl",
        "wg_tms",
        "wg_dfy",
        "wg_svar",
    ]

    macro_variables = [name for name in macro_variables if name in full.columns]

    macro = (
        full[["month"] + macro_variables]
        .drop_duplicates("month")
        .sort_values("month")
    )

    for variable in macro_variables:
        plt.figure(figsize=(10, 5))
        plt.plot(macro["month"], macro[variable])
        plt.title(f"Welch-Goyal macro variable: {variable}")
        plt.xlabel("Month")
        plt.ylabel(variable)
        plt.tight_layout()
        plt.savefig(REPORT_OUTPUT_DIR / f"{variable}_time_series.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# 6) Main inspection pipeline
# ---------------------------------------------------------------------
def main():
    full, train, validation, test, predictors = load_data()

    summary = create_sample_summary(full, train, validation, test, predictors)
    monthly = create_monthly_panel_summary(full)

    ranges = create_predictor_range_check(full, predictors)
    predictor_summary = create_predictor_distribution_summary(full, predictors)
    negative_summary = create_negative_value_explanation(full, predictors)
    examples = create_selected_variable_examples(full, predictors)
    largest_targets, smallest_targets = create_target_outlier_report(full)
    extreme_counts = create_target_extreme_count(full)

    plot_stock_count_by_month(monthly)
    plot_target_distribution(full)
    plot_target_distribution_trimmed(full)
    plot_target_mean_by_month(monthly)
    plot_target_volatility_by_month(monthly)

    plot_example_predictor_distribution(full, "mom12m")
    plot_example_predictor_distribution(full, "bm_comp")
    plot_example_predictor_distribution(full, "beta_12m")

    plot_macro_variables(full)

    outside_range = int(
        ranges["below_minus_one"].sum() + ranges["above_plus_one"].sum()
    )

    print("=" * 80)
    print("06_inspect_final_dataset.py")
    print("=" * 80)

    print("\nSTEP 1: Basic final dataset summary")
    print(summary.to_string(index=False))

    print("\nSTEP 2: Panel coverage")
    print(f"First month: {full['month'].min()}")
    print(f"Last month:  {full['month'].max()}")
    print(f"Average tickers per month: {monthly['tickers'].mean():.1f}")
    print(f"Minimum tickers in a month: {monthly['tickers'].min()}")
    print(f"Maximum tickers in a month: {monthly['tickers'].max()}")

    print("\nSTEP 3: Missing values and duplicates")
    print(f"Missing values in full dataset: {int(full.isna().sum().sum())}")
    print(f"Missing targets in full dataset: {int(full[TARGET].isna().sum())}")
    print(
        "Duplicate ticker-month rows: "
        f"{int(full.duplicated(['ticker', 'month']).sum())}"
    )

    print("\nSTEP 4: Target behavior")
    print(f"Target mean: {full[TARGET].mean():.6f}")
    print(f"Target standard deviation: {full[TARGET].std():.6f}")
    print(f"Target 1st percentile: {full[TARGET].quantile(0.01):.6f}")
    print(f"Target median: {full[TARGET].median():.6f}")
    print(f"Target 99th percentile: {full[TARGET].quantile(0.99):.6f}")
    print(f"Target maximum: {full[TARGET].max():.6f}")
    print(f"Target minimum: {full[TARGET].min():.6f}")

    print("\nSTEP 4B: Target outlier inspection")
    print("Largest next-month excess returns:")
    print(largest_targets.head(10).to_string(index=False))
    print()
    print("Smallest next-month excess returns:")
    print(smallest_targets.head(10).to_string(index=False))

    print("\nSTEP 4C: Count of extreme target returns")
    print(extreme_counts.to_string(index=False))

    print("\nSTEP 5: Predictor rank-normalization check")
    print(f"Number of predictors: {len(predictors)}")
    print(f"Predictors outside [-1, 1]: {outside_range}")
    print(
        "Negative predictor values are expected. They mean the stock is below "
        "the cross-sectional middle for that predictor in that month."
    )

    print("\nSTEP 6: Example variables")
    display_columns = [
        "variable",
        "is_target",
        "is_predictor",
        "mean",
        "std",
        "min",
        "p50",
        "max",
    ]
    print(examples[display_columns].to_string(index=False))

    print("\nSTEP 7: Saved diagnostic files")
    print(f"Reports saved in: {REPORT_OUTPUT_DIR}")

    print("\nCreated main tables:")
    print("- final_dataset_summary.csv")
    print("- monthly_panel_summary.csv")
    print("- predictor_rank_range_check.csv")
    print("- predictor_distribution_summary.csv")
    print("- predictor_negative_value_summary.csv")
    print("- selected_variable_examples.csv")
    print("- largest_target_returns.csv")
    print("- smallest_target_returns.csv")
    print("- target_extreme_return_counts.csv")

    print("\nCreated main graphs:")
    print("- stock_count_by_month.png")
    print("- target_distribution.png")
    print("- target_distribution_trimmed.png")
    print("- target_mean_by_month.png")
    print("- target_volatility_by_month.png")
    print("- mom12m_rank_distribution.png")
    print("- bm_comp_rank_distribution.png")
    print("- beta_12m_rank_distribution.png")
    print("- wg_*_time_series.png")
    print("=" * 80)


if __name__ == "__main__":
    main()