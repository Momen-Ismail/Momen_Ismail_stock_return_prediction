"""Inspect raw monthly prices around extreme target returns.

This file checks whether very large target returns are real or caused by
possible price-adjustment, split, ticker, or data-quality problems.

It does not modify the model dataset.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    MONTHLY_STOCK_FILE,
    REPORT_OUTPUT_DIR,
)


OUTLIERS_TO_CHECK = [
    ("NVR", "1993-09-30"),
    ("GME", "2020-12-31"),
    ("MI", "2021-02-28"),
    ("STI", "2024-02-29"),
    ("EP", "2003-02-28"),
    ("REGN", "2000-01-31"),
    ("POM", "2025-11-30"),
    ("STI", "2024-01-31"),
    ("AIG", "2008-08-31"),
]


def inspect_ticker_window(monthly, ticker, month, window=3):
    """Return observations around the outlier month for one ticker."""
    month = pd.Timestamp(month)

    start = month - pd.DateOffset(months=window)
    end = month + pd.DateOffset(months=window + 1)

    ticker_data = monthly[
        (monthly["ticker"] == ticker)
        & (monthly["month"] >= start)
        & (monthly["month"] <= end)
    ].copy()

    useful_columns = [
        "ticker",
        "month",
        "last_adj_close",
        "ret_1m",
        TARGET,
    ]

    useful_columns = [col for col in useful_columns if col in ticker_data.columns]

    return ticker_data[useful_columns].sort_values("month")


def main():
    monthly = pd.read_csv(MONTHLY_STOCK_FILE, low_memory=False)

    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["ticker"] = monthly["ticker"].astype(str).str.upper().str.strip()

    all_rows = []

    for ticker, month in OUTLIERS_TO_CHECK:
        window_data = inspect_ticker_window(monthly, ticker, month)

        print("=" * 80)
        print(f"{ticker}, outlier month t = {month}")
        print("Remember: target at month t is the return in month t+1.")
        print("=" * 80)
        print(window_data.to_string(index=False))
        print()

        window_data["outlier_month_t"] = month
        all_rows.append(window_data)

    report = pd.concat(all_rows, ignore_index=True)

    report.to_csv(
        REPORT_OUTPUT_DIR / "target_outlier_price_windows.csv",
        index=False,
    )

    print("=" * 80)
    print("Saved:")
    print(REPORT_OUTPUT_DIR / "target_outlier_price_windows.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
