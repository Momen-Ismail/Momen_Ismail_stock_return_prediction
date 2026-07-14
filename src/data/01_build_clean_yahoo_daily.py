"""Build and clean the Yahoo Finance daily price dataset."""

from pathlib import Path
import sys
import time

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    PRICE_START,
    PRICE_END,
    SLEEP_SECONDS,
    DAILY_BAD_ROW_SHARE_THRESHOLD,
    STOCK_UNIVERSE_FILE,
    TICKERS_RAW_FILE,
    DAILY_RAW_FILE,
    QUALITY_REPORT_FILE,
    TICKERS_CLEAN_FILE,
    DAILY_CLEAN_FILE,
    REMOVED_TICKERS_FILE,
)


# ---------------------------------------------------------------------
# 1) Load the permanent ticker universe
# ---------------------------------------------------------------------
def load_locked_universe():
    """Load unique Yahoo-format tickers."""
    universe = pd.read_csv(STOCK_UNIVERSE_FILE)

    if "ticker" not in universe.columns:
        raise ValueError("The stock-universe file must contain a 'ticker' column.")

    return sorted(
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )


# ---------------------------------------------------------------------
# 2) Download daily Yahoo prices
# ---------------------------------------------------------------------
def download_daily_prices(tickers):
    """Download prices and return the data and failed ticker list."""
    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    frames = []
    failed = []

    for number, ticker in enumerate(tickers, start=1):
        print(f"Downloading {number}/{len(tickers)}: {ticker}")

        try:
            data = yf.download(
                ticker,
                start=PRICE_START,
                end=PRICE_END,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if data.empty:
                failed.append(ticker)
                print("  Failed: no data")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.rename_axis("date").reset_index()
            data.columns = [
                str(column).lower().replace(" ", "_").replace("-", "_")
                for column in data.columns
            ]

            missing = [column for column in required if column not in data.columns]

            if missing:
                failed.append(ticker)
                print(f"  Failed: missing columns {missing}")
                continue

            data["ticker"] = ticker
            frames.append(data[["ticker"] + required])

        except Exception as error:
            failed.append(ticker)
            print(f"  Failed: {error}")

        finally:
            time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("Yahoo returned no usable price data.")

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    return daily, failed


# ---------------------------------------------------------------------
# 3) Clean daily data and create a ticker-quality report
# ---------------------------------------------------------------------
def clean_daily_prices(daily):
    """Remove bad rows and ticker histories with excessive bad data."""
    data = daily.sort_values(["ticker", "date"]).copy()

    data["daily_ret"] = data.groupby("ticker")["adj_close"].pct_change()

    data["bad_ohlc"] = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    data["missing_core"] = data[
        ["open", "high", "low", "close", "adj_close", "volume"]
    ].isna().any(axis=1)

    data["nonpositive_price"] = data["adj_close"].le(0)

    data["extreme_return"] = (
        data["daily_ret"].gt(3.0)
        | data["daily_ret"].lt(-0.95)
    )

    data["duplicate_date"] = data.duplicated(
        ["ticker", "date"],
        keep=False,
    )

    quality_columns = [
        "bad_ohlc",
        "missing_core",
        "nonpositive_price",
        "extreme_return",
        "duplicate_date",
    ]

    data["bad_row"] = data[quality_columns].any(axis=1)

    report = (
        data.groupby("ticker", as_index=False)
        .agg(
            rows=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            bad_ohlc_rows=("bad_ohlc", "sum"),
            missing_core_rows=("missing_core", "sum"),
            nonpositive_price_rows=("nonpositive_price", "sum"),
            extreme_return_rows=("extreme_return", "sum"),
            duplicate_date_rows=("duplicate_date", "sum"),
            bad_rows=("bad_row", "sum"),
            max_daily_return=("daily_ret", "max"),
            min_daily_return=("daily_ret", "min"),
        )
    )

    report["bad_row_share"] = report["bad_rows"] / report["rows"]
    report["remove_ticker"] = (
        report["bad_row_share"] > DAILY_BAD_ROW_SHARE_THRESHOLD
    )

    removed = report[report["remove_ticker"]].copy()
    removed["removal_reason"] = "too_many_bad_rows"

    clean = data[
        ~data["bad_row"]
        & ~data["ticker"].isin(removed["ticker"])
    ].copy()

    clean = clean[
        [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ].sort_values(["ticker", "date"]).reset_index(drop=True)

    clean_tickers = sorted(clean["ticker"].unique())

    return clean, clean_tickers, report, removed


# ---------------------------------------------------------------------
# 4) Run the complete daily-data pipeline
# ---------------------------------------------------------------------
def main():
    """Download, clean, and save the daily stock dataset."""
    universe = load_locked_universe()

    pd.DataFrame({"ticker": universe}).to_csv(
        TICKERS_RAW_FILE,
        index=False,
    )

    daily, failed = download_daily_prices(universe)
    daily.to_csv(DAILY_RAW_FILE, index=False)

    clean, clean_tickers, report, removed = clean_daily_prices(daily)

    report.to_csv(QUALITY_REPORT_FILE, index=False)
    pd.DataFrame({"ticker": clean_tickers}).to_csv(
        TICKERS_CLEAN_FILE,
        index=False,
    )
    clean.to_csv(DAILY_CLEAN_FILE, index=False)
    removed.to_csv(REMOVED_TICKERS_FILE, index=False)

    print("\nFinal summary")
    print(f"Requested tickers: {len(universe)}")
    print(f"Successfully downloaded: {daily['ticker'].nunique()}")
    print(f"Failed downloads: {len(failed)}")
    print(f"Removed by quality filter: {len(removed)}")
    print(f"Final clean tickers: {len(clean_tickers)}")
    print(f"Final clean rows: {len(clean):,}")

    if failed:
        print(f"Failed tickers: {', '.join(failed)}")

    if not removed.empty:
        print("\nRemoved tickers:")
        print(
            removed[["ticker", "removal_reason"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()