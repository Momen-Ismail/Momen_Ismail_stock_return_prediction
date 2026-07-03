"""Build and clean the Yahoo Finance daily-price dataset."""

from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests
import urllib3
import yfinance as yf


# 1) Settings
START_YEAR, END_YEAR = 1990, 2025
PRICE_START, PRICE_END = "1987-01-01", "2026-02-01"
SLEEP_SECONDS = 0.5
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DATA_DIR = Path("output/data")
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
for folder in (INTERMEDIATE_DIR, FINAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TICKERS_RAW_FILE = INTERMEDIATE_DIR / "sp500_tickers_raw.csv"
DAILY_RAW_FILE = INTERMEDIATE_DIR / "daily_prices_raw_1987_2026.csv"
QUALITY_REPORT_FILE = INTERMEDIATE_DIR / "ticker_quality_report.csv"
TICKERS_CLEAN_FILE = FINAL_DIR / "sp500_tickers_clean.csv"
DAILY_CLEAN_FILE = FINAL_DIR / "daily_prices_clean_1987_2026.csv"
REMOVED_TICKERS_FILE = FINAL_DIR / "removed_tickers.csv"


# 2) Build the stock universe
def clean_ticker(value):
    """Convert a ticker to Yahoo format, for example BRK.B -> BRK-B."""
    if pd.isna(value):
        return None
    ticker = str(value).upper().strip().replace(".", "-").replace(" ", "")
    return None if ticker in {"", "NAN", "NONE"} else ticker


def flatten_columns(table):
    table.columns = [
        " ".join(str(item) for item in column if str(item) != "nan").strip()
        if isinstance(column, tuple) else str(column).strip()
        for column in table.columns
    ]
    return table


def find_column(table, words):
    return next(
        (column for column in table if all(word in column.lower() for word in words)),
        None,
    )


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
    current, changes = map(flatten_columns, pd.read_html(StringIO(response.text))[:2])

    if "Symbol" not in current:
        raise ValueError("Wikipedia current-members table has no Symbol column.")

    date_column = find_column(changes, ["effective", "date"])
    if date_column is None:
        raise ValueError("Wikipedia changes table has no effective-date column.")

    changes[date_column] = pd.to_datetime(changes[date_column], errors="coerce")
    changes = changes[changes[date_column].dt.year.between(START_YEAR, END_YEAR)]

    ticker_columns = [
        "Symbol",
        find_column(changes, ["added", "ticker"]),
        find_column(changes, ["removed", "ticker"]),
    ]
    series = [current["Symbol"]] + [changes[col] for col in ticker_columns[1:] if col]
    return sorted({clean_ticker(value) for values in series for value in values} - {None})


# 3) Download Yahoo prices
def yahoo_download(ticker, start, end):
    data = yf.download(
        ticker, start=start, end=end, progress=False,
        auto_adjust=False, threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def yahoo_has_data(ticker):
    try:
        return not yahoo_download(
            ticker, f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"
        ).empty
    except Exception:
        return False


def download_daily_prices(tickers):
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    frames = []

    for number, ticker in enumerate(tickers, 1):
        print(f"Downloading {number}/{len(tickers)}: {ticker}")
        try:
            data = yahoo_download(ticker, PRICE_START, PRICE_END)
            if data.empty:
                continue
            data = data.rename_axis("date").reset_index()
            data.columns = [str(col).lower().replace(" ", "_").replace("-", "_") for col in data]
            if missing := [col for col in required if col not in data]:
                print(f"  Skipped: missing {missing}")
                continue
            data["ticker"] = ticker
            frames.append(data[["ticker"] + required])
        except Exception as error:
            print(f"  Skipped: {error}")
        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("Yahoo returned no price data.")
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


# 4) Apply transparent data-quality rules
def create_quality_report(daily):
    data = daily.sort_values(["ticker", "date"]).copy()
    data["daily_ret"] = data.groupby("ticker")["adj_close"].pct_change()
    data["bad_ohlc"] = (
        (data.high < data.low) | (data.high < data.open) | (data.high < data.close)
        | (data.low > data.open) | (data.low > data.close)
    )
    data["missing_core"] = data[["open", "high", "low", "close", "adj_close", "volume"]].isna().any(axis=1)
    data["nonpositive_adj_close"] = data.adj_close <= 0
    data["extreme_positive_return"] = data.daily_ret > 3.0
    data["extreme_negative_return"] = data.daily_ret < -0.95

    report = data.groupby("ticker").agg(
        rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
        bad_ohlc_rows=("bad_ohlc", "sum"), missing_core_rows=("missing_core", "sum"),
        nonpositive_adj_close_rows=("nonpositive_adj_close", "sum"),
        extreme_positive_return_rows=("extreme_positive_return", "sum"),
        extreme_negative_return_rows=("extreme_negative_return", "sum"),
        max_daily_return=("daily_ret", "max"), min_daily_return=("daily_ret", "min"),
    ).reset_index()
    duplicates = data.duplicated(["ticker", "date"]).groupby(data.ticker).sum()
    report["duplicate_ticker_date_rows"] = report.ticker.map(duplicates).fillna(0).astype(int)

    rules = {
        "remove_bad_ohlc": report.bad_ohlc_rows > 0,
        "remove_duplicate_dates": report.duplicate_ticker_date_rows > 0,
        "remove_missing_core": report.missing_core_rows > 0,
        "remove_nonpositive_price": report.nonpositive_adj_close_rows > 0,
        "remove_repeated_extreme_returns": (
            (report.extreme_positive_return_rows >= 2)
            | (report.extreme_negative_return_rows >= 2)
        ),
    }
    for name, condition in rules.items():
        report[name] = condition
    report["remove_ticker"] = pd.DataFrame(rules).any(axis=1)

    reason_names = {
        "remove_bad_ohlc": "bad_ohlc",
        "remove_duplicate_dates": "duplicate_ticker_date",
        "remove_missing_core": "missing_core_price_or_volume",
        "remove_nonpositive_price": "nonpositive_adjusted_close",
        "remove_repeated_extreme_returns": "repeated_extreme_daily_returns",
    }
    report["removal_reason"] = report.apply(
        lambda row: "; ".join(reason for rule, reason in reason_names.items() if row[rule]),
        axis=1,
    )
    return report


# 5) Run the pipeline
def main():
    universe = build_sp500_universe()
    print(f"Wikipedia universe: {len(universe)} tickers")

    available = []
    for number, ticker in enumerate(universe, 1):
        print(f"Checking {number}/{len(universe)}: {ticker}")
        if yahoo_has_data(ticker):
            available.append(ticker)
        time.sleep(SLEEP_SECONDS)
    available = sorted(available)
    pd.DataFrame({"ticker": available}).to_csv(TICKERS_RAW_FILE, index=False)

    daily = download_daily_prices(available)
    daily["date"] = pd.to_datetime(daily.date)
    daily.to_csv(DAILY_RAW_FILE, index=False)

    report = create_quality_report(daily)
    report.to_csv(QUALITY_REPORT_FILE, index=False)
    removed = report[report.remove_ticker].copy()
    clean = daily[~daily.ticker.isin(removed.ticker)].copy()
    clean_tickers = sorted(clean.ticker.unique())

    pd.DataFrame({"ticker": clean_tickers}).to_csv(TICKERS_CLEAN_FILE, index=False)
    clean.to_csv(DAILY_CLEAN_FILE, index=False)
    removed.to_csv(REMOVED_TICKERS_FILE, index=False)

    print(f"Clean data: {len(clean):,} rows, {len(clean_tickers)} tickers")
    print(removed[["ticker", "removal_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
