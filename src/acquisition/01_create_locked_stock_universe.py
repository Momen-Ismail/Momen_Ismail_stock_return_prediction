"""Create the permanent locked stock universe input.

This is a one-time acquisition script. The normal data pipeline reads the
saved CSV and does not access Wikipedia.
"""

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.acquisition.manifest import update_input_manifest  # noqa: E402
from src.config import (  # noqa: E402
    START_YEAR,
    END_YEAR,
    WIKI_URL,
    INPUT_MANIFEST_FILE,
    STOCK_UNIVERSE_FILE,
)


def clean_ticker(value):
    """Convert ticker to Yahoo format, for example BRK.B -> BRK-B."""
    if pd.isna(value):
        return None

    ticker = str(value).upper().strip().replace(".", "-").replace(" ", "")

    if ticker in {"", "NAN", "NONE"}:
        return None

    return ticker


def normalize_columns(table):
    """Return a copy with simple string column names."""
    table = table.copy()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            " ".join(str(item) for item in column if str(item) != "nan").strip()
            for column in table.columns
        ]
    else:
        table.columns = [str(column).strip() for column in table.columns]

    return table


def find_column(table, words):
    """Find the first column containing all requested words."""
    for column in table.columns:
        if all(word in column.lower() for word in words):
            return column

    return None


def build_sp500_universe():
    """Combine current and historical Wikipedia S&P 500 tickers."""
    try:
        current, changes = map(normalize_columns, pd.read_html(WIKI_URL)[:2])
    except Exception:
        response = requests.get(
            WIKI_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        current, changes = map(normalize_columns, pd.read_html(response.text)[:2])

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


def main():
    tickers = build_sp500_universe()
    STOCK_UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(STOCK_UNIVERSE_FILE, index=False)

    update_input_manifest(
        manifest_file=INPUT_MANIFEST_FILE,
        input_file=STOCK_UNIVERSE_FILE,
        source=WIKI_URL,
        coverage_start=START_YEAR,
        coverage_end=END_YEAR,
        notes="Locked current-plus-historical S&P 500 universe in Yahoo ticker format.",
    )

    print(f"Saved {STOCK_UNIVERSE_FILE}")
    print(f"Tickers: {len(tickers):,}")
    print(f"Created on: {date.today().isoformat()}")


if __name__ == "__main__":
    main()
