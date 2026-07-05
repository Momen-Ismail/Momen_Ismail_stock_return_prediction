"""Inspect raw price windows around extreme target-return observations.

Purpose:
This script is diagnostic only. It helps explain whether very large raw target
returns come from the original monthly stock return data.

Important:
- This file does not clean the data.
- It does not delete observations.
- It checks raw monthly returns before final target winsorization.
- Final outlier handling is done in 05_clean_and_rank_normalize.py.

The main idea:
If target at month t is next-month return, then an extreme target at t should
correspond to an extreme ret_1m at t+1 for the same ticker.
"""

from pathlib import Path
import sys

import pandas as pd


# Allow direct execution from project root:
# python src/data/07_check_target_outlier_prices.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    MONTHLY_STOCK_FILE,
    REPORT_OUTPUT_DIR,
)


OUTPUT_FILE = REPORT_OUTPUT_DIR / "target_outlier_price_windows.csv"


def load_monthly_stock_panel():
    """Load raw monthly stock panel created by file 02."""
    data = pd.read_csv(MONTHLY_STOCK_FILE)
    data["month"] = pd.to_datetime(data["month"])

    data = data.sort_values(["ticker", "month"]).reset_index(drop=True)

    return data


def find_extreme_targets(data, threshold=1.0, top_n=20):
    """Find observations with very large absolute raw target returns."""
    if TARGET not in data.columns:
        raise ValueError(f"Target column not found in monthly stock file: {TARGET}")

    extreme = data[data[TARGET].abs() > threshold].copy()

    if extreme.empty:
        return extreme

    extreme["abs_target"] = extreme[TARGET].abs()

    extreme = extreme.sort_values("abs_target", ascending=False)

    return extreme.head(top_n)


def collect_price_window(data, ticker, event_month, window=3):
    """Collect months around one extreme target observation."""
    stock = data[data["ticker"] == ticker].copy()
    stock = stock.sort_values("month").reset_index(drop=True)

    event_month = pd.to_datetime(event_month)

    event_positions = stock.index[stock["month"] == event_month].tolist()

    if not event_positions:
        return pd.DataFrame()

    event_position = event_positions[0]

    start = max(0, event_position - window)
    end = min(len(stock), event_position + window + 2)

    keep_columns = [
        "ticker",
        "month",
        "last_adj_close",
        "ret_1m",
        TARGET,
    ]

    existing_columns = [col for col in keep_columns if col in stock.columns]

    window_data = stock.loc[start:end, existing_columns].copy()

    window_data["event_ticker"] = ticker
    window_data["event_month"] = event_month
    window_data["is_event_month"] = window_data["month"].eq(event_month)

    return window_data


def create_outlier_price_windows(data, threshold=1.0, top_n=20, window=3):
    """Create a combined price-window table for the largest outliers."""
    extreme = find_extreme_targets(data, threshold=threshold, top_n=top_n)

    if extreme.empty:
        return pd.DataFrame(), extreme

    windows = []

    for _, row in extreme.iterrows():
        ticker = row["ticker"]
        event_month = row["month"]

        window_data = collect_price_window(
            data=data,
            ticker=ticker,
            event_month=event_month,
            window=window,
        )

        if not window_data.empty:
            window_data["event_target"] = row[TARGET]
            window_data["event_abs_target"] = abs(row[TARGET])
            windows.append(window_data)

    if not windows:
        return pd.DataFrame(), extreme

    combined = pd.concat(windows, ignore_index=True)

    return combined, extreme


def main():
    print("=" * 80)
    print("07_check_target_outlier_prices.py")
    print("=" * 80)

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_monthly_stock_panel()

    print(f"Loaded monthly stock panel: {data.shape}")
    print(f"Date range: {data['month'].min()} to {data['month'].max()}")
    print(f"Tickers: {data['ticker'].nunique()}")

    windows, extreme = create_outlier_price_windows(
        data=data,
        threshold=1.0,
        top_n=30,
        window=4,
    )

    extreme_file = REPORT_OUTPUT_DIR / "raw_extreme_target_observations_top30.csv"
    extreme.to_csv(extreme_file, index=False)

    print(f"Saved extreme target list: {extreme_file}")

    if windows.empty:
        print("No extreme target price windows created.")
    else:
        windows.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved price windows: {OUTPUT_FILE}")

        print("\nLargest raw target observations:")
        show_columns = ["ticker", "month", TARGET, "abs_target"]

        existing_show_columns = [
            col for col in show_columns if col in extreme.columns
        ]

        print(extreme[existing_show_columns].head(30).to_string(index=False))

        print("\nExample price windows:")
        print(windows.head(80).to_string(index=False))

    print("\nImportant interpretation:")
    print(
        "This script is diagnostic only. It checks the raw monthly panel before "
        "final target winsorization. It should not be used to delete observations."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
