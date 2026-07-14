"""Download permanent Yahoo market inputs for GSPC and VIX."""

from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.acquisition.manifest import update_input_manifest  # noqa: E402
from src.config import (  # noqa: E402
    PRICE_START,
    PRICE_END,
    GSPC_DAILY_FILE,
    VIX_DAILY_FILE,
    INPUT_MANIFEST_FILE,
)


STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def download_yahoo_daily(ticker):
    """Download one Yahoo daily price series with standard columns."""
    data = yf.download(
        ticker,
        start=PRICE_START,
        end=PRICE_END,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename_axis("date").reset_index()
    data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]

    missing = [column for column in STANDARD_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Yahoo data for {ticker} is missing columns: {missing}")

    data = data[STANDARD_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"])

    return data.sort_values("date").reset_index(drop=True)


def save_market_input(ticker, output_file, source_name):
    """Download, save, and register one market input."""
    data = download_yahoo_daily(ticker)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)

    update_input_manifest(
        manifest_file=INPUT_MANIFEST_FILE,
        input_file=output_file,
        source=source_name,
        coverage_start=data["date"].min().date(),
        coverage_end=data["date"].max().date(),
        notes=f"Yahoo daily market input for {ticker}; auto_adjust=False.",
    )

    print(f"Saved {output_file}: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")


def main():
    save_market_input("^GSPC", GSPC_DAILY_FILE, "Yahoo Finance ^GSPC")
    save_market_input("^VIX", VIX_DAILY_FILE, "Yahoo Finance ^VIX")


if __name__ == "__main__":
    main()
