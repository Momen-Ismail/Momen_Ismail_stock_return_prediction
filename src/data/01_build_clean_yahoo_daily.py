"""Download and clean daily Yahoo Finance stock data."""

from pathlib import Path
import sys
import time

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    PRICE_START,
    PRICE_END,
    SLEEP_SECONDS,
    DAILY_BAD_ROW_SHARE_THRESHOLD,
    STOCK_UNIVERSE_FILE,
    DAILY_CLEAN_FILE,
    YAHOO_DOWNLOAD_REPORT_FILE,
    DAILY_QUALITY_REPORT_FILE,
)


DAILY_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


# ---------------------------------------------------------------------
# 1) Load the permanent ticker universe
# ---------------------------------------------------------------------
def load_locked_universe():
    universe = pd.read_csv(STOCK_UNIVERSE_FILE)

    return (
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


# ---------------------------------------------------------------------
# 2) Download daily Yahoo prices
# ---------------------------------------------------------------------
def download_daily_prices(tickers):
    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }

    frames = []
    records = []

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
                records.append({
                    "ticker": ticker,
                    "status": "failed",
                    "reason": "no_data",
                    "rows": 0,
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                })
                time.sleep(SLEEP_SECONDS)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.rename_axis("date").reset_index()
            data.columns = [
                str(column).lower().strip().replace(" ", "_").replace("-", "_")
                for column in data.columns
            ]

            missing_columns = sorted(required_columns.difference(data.columns))

            if missing_columns:
                records.append({
                    "ticker": ticker,
                    "status": "failed",
                    "reason": f"missing_columns:{','.join(missing_columns)}",
                    "rows": len(data),
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                })
                time.sleep(SLEEP_SECONDS)
                continue

            data["ticker"] = ticker
            data["date"] = pd.to_datetime(data["date"])

            frames.append(data[DAILY_COLUMNS])

            records.append({
                "ticker": ticker,
                "status": "downloaded",
                "reason": "",
                "rows": len(data),
                "first_date": data["date"].min(),
                "last_date": data["date"].max(),
            })

        except Exception as error:
            records.append({
                "ticker": ticker,
                "status": "failed",
                "reason": str(error),
                "rows": 0,
                "first_date": pd.NaT,
                "last_date": pd.NaT,
            })

        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("Yahoo Finance returned no usable stock data.")

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    return daily, pd.DataFrame(records)

# ---------------------------------------------------------------------
# 3) Clean daily data and create a ticker-quality report
# ---------------------------------------------------------------------
def clean_daily_prices(daily):
    data = daily.sort_values(["ticker", "date"]).copy()

    duplicate_rows = data.duplicated(["ticker", "date"], keep="last")

    duplicate_report = (
        data.loc[duplicate_rows]
        .groupby("ticker")
        .size()
        .rename("duplicate_rows")
    )

    data = (
        data.drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )

    price_columns = ["open", "high", "low", "close", "adj_close"]
    core_columns = price_columns + ["volume"]

    data["daily_ret"] = (
        data.groupby("ticker")["adj_close"]
        .pct_change(fill_method=None)
    )

    data["bad_ohlc"] = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    data["missing_core"] = data[core_columns].isna().any(axis=1)
    data["nonpositive_price"] = data[price_columns].le(0).any(axis=1)

    data["extreme_return"] = (
        data["daily_ret"].gt(3.0)
        | data["daily_ret"].lt(-0.95)
    )

    data["bad_row"] = (
        data["bad_ohlc"]
        | data["missing_core"]
        | data["nonpositive_price"]
    )

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
            bad_rows=("bad_row", "sum"),
            minimum_daily_return=("daily_ret", "min"),
            maximum_daily_return=("daily_ret", "max"),
        )
    )

    report["duplicate_rows"] = (
        report["ticker"]
        .map(duplicate_report)
        .fillna(0)
        .astype(int)
    )

    report["bad_row_share"] = report["bad_rows"] / report["rows"]

    bad_share = (
        report["bad_row_share"] > DAILY_BAD_ROW_SHARE_THRESHOLD
    )

    extreme_history = report["extreme_return_rows"] > 0

    report["remove_ticker"] = bad_share | extreme_history
    report["status"] = "retained"
    report.loc[report["remove_ticker"], "status"] = "removed_quality"

    report["removal_reason"] = ""
    report.loc[
        bad_share & ~extreme_history,
        "removal_reason",
    ] = "bad_row_share_above_threshold"

    report.loc[
        extreme_history & ~bad_share,
        "removal_reason",
    ] = "extreme_adjusted_return"

    report.loc[
        bad_share & extreme_history,
        "removal_reason",
    ] = "bad_row_share_and_extreme_return"

    removed_tickers = report.loc[
        report["remove_ticker"],
        "ticker",
    ]

    clean = data.loc[
        ~data["bad_row"]
        & ~data["ticker"].isin(removed_tickers),
        DAILY_COLUMNS,
    ]

    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)

    return clean, report
# ---------------------------------------------------------------------
# 4) Run the complete daily-data pipeline
# ---------------------------------------------------------------------
def main():
    universe = load_locked_universe()

    daily, download_report = download_daily_prices(universe)
    clean, quality_report = clean_daily_prices(daily)

    DAILY_CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    YAHOO_DOWNLOAD_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAILY_QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean.to_parquet(DAILY_CLEAN_FILE, index=False)
    download_report.to_csv(YAHOO_DOWNLOAD_REPORT_FILE, index=False)
    quality_report.to_csv(DAILY_QUALITY_REPORT_FILE, index=False)

    downloaded = download_report["status"].eq("downloaded").sum()
    failed = download_report["status"].eq("failed").sum()
    removed = quality_report["remove_ticker"].sum()

    print("\nFinal summary")
    print(f"Requested tickers: {len(universe)}")
    print(f"Downloaded tickers: {downloaded}")
    print(f"Failed downloads: {failed}")
    print(f"Removed by quality filter: {removed}")
    print(f"Final clean tickers: {clean['ticker'].nunique()}")
    print(f"Final clean rows: {len(clean):,}")
    print(f"Saved: {DAILY_CLEAN_FILE}")
    print(f"Saved: {YAHOO_DOWNLOAD_REPORT_FILE}")
    print(f"Saved: {DAILY_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    main()