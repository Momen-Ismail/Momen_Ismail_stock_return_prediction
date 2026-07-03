"""Build and clean the Yahoo Finance daily price dataset.

Output:
- raw S&P 500 ticker list
- raw Yahoo daily prices
- ticker quality report
- cleaned daily prices
- cleaned ticker list
- removed ticker report
"""

from io import StringIO
from pathlib import Path
import sys
import time

import pandas as pd
import requests
import urllib3
import yfinance as yf


# Allow direct execution from the project root:
# python src/data/01_build_clean_yahoo_daily.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    START_YEAR,
    END_YEAR,
    PRICE_START,
    PRICE_END,
    SLEEP_SECONDS,
    WIKI_URL,
    TICKERS_RAW_FILE,
    DAILY_RAW_FILE,
    QUALITY_REPORT_FILE,
    TICKERS_CLEAN_FILE,
    DAILY_CLEAN_FILE,
    REMOVED_TICKERS_FILE,
)


# ---------------------------------------------------------------------
# 1) Build the stock universe
# ---------------------------------------------------------------------
def clean_ticker(value):
    """Convert ticker to Yahoo format, for example BRK.B -> BRK-B."""
    if pd.isna(value):
        return None

    ticker = str(value).upper().strip().replace(".", "-").replace(" ", "")

    if ticker in {"", "NAN", "NONE"}:
        return None

    return ticker


def flatten_columns(table):
    """Flatten Wikipedia tables with multi-level column names."""
    table.columns = [
        " ".join(str(item) for item in column if str(item) != "nan").strip()
        if isinstance(column, tuple)
        else str(column).strip()
        for column in table.columns
    ]

    return table


def find_column(table, words):
    """Find the first column containing all requested words."""
    for column in table.columns:
        if all(word in column.lower() for word in words):
            return column

    return None


def build_sp500_universe():
    """Combine current and historical Wikipedia S&P 500 tickers."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    response = requests.get(
        WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        verify=False,
        timeout=30,
    )
    response.raise_for_status()

    current, changes = map(
        flatten_columns,
        pd.read_html(StringIO(response.text))[:2],
    )

    if "Symbol" not in current.columns:
        raise ValueError("Wikipedia current-members table has no Symbol column.")

    date_column = find_column(changes, ["effective", "date"])

    if date_column is None:
        raise ValueError("Wikipedia changes table has no effective-date column.")

    changes[date_column] = pd.to_datetime(changes[date_column], errors="coerce")
    changes = changes[changes[date_column].dt.year.between(START_YEAR, END_YEAR)]

    added_column = find_column(changes, ["added", "ticker"])
    removed_column = find_column(changes, ["removed", "ticker"])

    ticker_series = [current["Symbol"]]

    if added_column:
        ticker_series.append(changes[added_column])

    if removed_column:
        ticker_series.append(changes[removed_column])

    tickers = {
        clean_ticker(value)
        for series in ticker_series
        for value in series
    }

    return sorted(tickers - {None})


# ---------------------------------------------------------------------
# 2) Download Yahoo prices
# ---------------------------------------------------------------------
def yahoo_download(ticker, start, end):
    """Download one ticker from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def yahoo_has_data(ticker):
    """Check whether Yahoo has usable data for a ticker."""
    try:
        data = yahoo_download(
            ticker,
            start=f"{START_YEAR}-01-01",
            end=f"{END_YEAR}-12-31",
        )

        return not data.empty

    except Exception:
        return False


def download_daily_prices(tickers):
    """Download daily OHLCV prices for all tickers."""
    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    frames = []

    for number, ticker in enumerate(tickers, start=1):
        print(f"Downloading {number}/{len(tickers)}: {ticker}")

        try:
            data = yahoo_download(ticker, PRICE_START, PRICE_END)

            if data.empty:
                continue

            data = data.rename_axis("date").reset_index()
            data.columns = [
                str(column).lower().replace(" ", "_").replace("-", "_")
                for column in data.columns
            ]

            missing = [column for column in required_columns if column not in data.columns]

            if missing:
                print(f"  Skipped: missing columns {missing}")
                continue

            data["ticker"] = ticker
            frames.append(data[["ticker"] + required_columns])

        except Exception as error:
            print(f"  Skipped: {error}")

        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("Yahoo returned no price data.")

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])

    return daily.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 3) Apply transparent data-quality rules
# ---------------------------------------------------------------------
def create_quality_report(daily):
    """Create ticker-level quality report and removal flags."""
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

    data["nonpositive_adj_close"] = data["adj_close"] <= 0
    data["extreme_positive_return"] = data["daily_ret"] > 3.0
    data["extreme_negative_return"] = data["daily_ret"] < -0.95

    report = data.groupby("ticker").agg(
        rows=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        bad_ohlc_rows=("bad_ohlc", "sum"),
        missing_core_rows=("missing_core", "sum"),
        nonpositive_adj_close_rows=("nonpositive_adj_close", "sum"),
        extreme_positive_return_rows=("extreme_positive_return", "sum"),
        extreme_negative_return_rows=("extreme_negative_return", "sum"),
        max_daily_return=("daily_ret", "max"),
        min_daily_return=("daily_ret", "min"),
    ).reset_index()

    duplicates = data.duplicated(["ticker", "date"]).groupby(data["ticker"]).sum()

    report["duplicate_ticker_date_rows"] = (
        report["ticker"].map(duplicates).fillna(0).astype(int)
    )

    report["remove_bad_ohlc"] = report["bad_ohlc_rows"] > 0
    report["remove_duplicate_dates"] = report["duplicate_ticker_date_rows"] > 0
    report["remove_missing_core"] = report["missing_core_rows"] > 0
    report["remove_nonpositive_price"] = report["nonpositive_adj_close_rows"] > 0
    report["remove_repeated_extreme_returns"] = (
        (report["extreme_positive_return_rows"] >= 2)
        | (report["extreme_negative_return_rows"] >= 2)
    )

    removal_rules = {
        "remove_bad_ohlc": "bad_ohlc",
        "remove_duplicate_dates": "duplicate_ticker_date",
        "remove_missing_core": "missing_core_price_or_volume",
        "remove_nonpositive_price": "nonpositive_adjusted_close",
        "remove_repeated_extreme_returns": "repeated_extreme_daily_returns",
    }

    rule_columns = list(removal_rules.keys())
    report["remove_ticker"] = report[rule_columns].any(axis=1)

    report["removal_reason"] = report.apply(
        lambda row: "; ".join(
            reason
            for rule, reason in removal_rules.items()
            if row[rule]
        ),
        axis=1,
    )

    return report


def apply_quality_filter(daily, report):
    """Remove tickers failing the quality rules."""
    removed = report[report["remove_ticker"]].copy()

    clean = daily[~daily["ticker"].isin(removed["ticker"])].copy()
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)

    clean_tickers = sorted(clean["ticker"].unique())

    return clean, clean_tickers, removed


# ---------------------------------------------------------------------
# 4) Run pipeline
# ---------------------------------------------------------------------
def main():
    universe = build_sp500_universe()
    print(f"Wikipedia universe: {len(universe)} tickers")

    available = []

    for number, ticker in enumerate(universe, start=1):
        print(f"Checking {number}/{len(universe)}: {ticker}")

        if yahoo_has_data(ticker):
            available.append(ticker)

        time.sleep(SLEEP_SECONDS)

    available = sorted(available)
    pd.DataFrame({"ticker": available}).to_csv(TICKERS_RAW_FILE, index=False)

    daily = download_daily_prices(available)
    daily.to_csv(DAILY_RAW_FILE, index=False)

    report = create_quality_report(daily)
    report.to_csv(QUALITY_REPORT_FILE, index=False)

    clean, clean_tickers, removed = apply_quality_filter(daily, report)

    pd.DataFrame({"ticker": clean_tickers}).to_csv(TICKERS_CLEAN_FILE, index=False)
    clean.to_csv(DAILY_CLEAN_FILE, index=False)
    removed.to_csv(REMOVED_TICKERS_FILE, index=False)

    print(f"Clean data: {len(clean):,} rows, {len(clean_tickers)} tickers")

    if removed.empty:
        print("No tickers removed.")
    else:
        print(removed[["ticker", "removal_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()