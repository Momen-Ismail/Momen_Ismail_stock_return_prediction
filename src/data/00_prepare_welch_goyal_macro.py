"""Prepare clean Welch-Goyal macro predictors.

Input:
- original Welch-Goyal Excel file

Output:
- clean monthly macro CSV with variables used for interactions

This script is run once. After creating the clean CSV, the original Excel
file can be deleted. The main pipeline uses only the clean CSV.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow direct execution from the project root:
# python src/data/00_prepare_welch_goyal_macro.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    WELCH_GOYAL_RAW_FILE,
    WELCH_GOYAL_CLEAN_FILE,
)


def prepare_welch_goyal_macro():
    """Create clean Welch-Goyal macro variables."""
    if WELCH_GOYAL_RAW_FILE.suffix.lower() == ".csv":
        macro = pd.read_csv(WELCH_GOYAL_RAW_FILE)
    else:
        macro = pd.read_excel(WELCH_GOYAL_RAW_FILE)
    macro.columns = macro.columns.astype(str).str.strip()

    required_columns = [
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "infl",
        "svar",
    ]

    missing = [name for name in required_columns if name not in macro.columns]

    if missing:
        raise ValueError(f"Missing Welch-Goyal columns: {missing}")

    macro["yyyymm"] = macro["yyyymm"].astype(str).str.extract(r"(\d{6})")[0]
    macro = macro[macro["yyyymm"].notna()].copy()

    macro["month"] = (
        pd.to_datetime(macro["yyyymm"], format="%Y%m")
        + pd.offsets.MonthEnd(0)
    )

    for column in required_columns:
        if column != "yyyymm":
            macro[column] = pd.to_numeric(macro[column], errors="coerce")

    clean = pd.DataFrame()
    clean["month"] = macro["month"]

    clean["wg_dp"] = np.log(macro["D12"] / macro["Index"])
    clean["wg_ep"] = np.log(macro["E12"] / macro["Index"])
    clean["wg_bm"] = macro["b/m"]
    clean["wg_ntis"] = macro["ntis"]
    clean["wg_tbl"] = macro["tbl"]
    clean["wg_tms"] = macro["lty"] - macro["tbl"]
    clean["wg_dfy"] = macro["BAA"] - macro["AAA"]
    clean["wg_svar"] = macro["svar"]
    clean["wg_infl"] = macro["infl"]

    macro_names = [
        "wg_dp",
        "wg_ep",
        "wg_bm",
        "wg_ntis",
        "wg_tbl",
        "wg_tms",
        "wg_dfy",
        "wg_svar",
        "wg_infl",
    ]

    clean = clean.sort_values("month").copy()

    # Lag all macro variables by one month to avoid look-ahead bias.
    for name in macro_names:
        clean[name] = clean[name].shift(1)

    clean[macro_names] = (
        clean[macro_names]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    clean = clean[
        (clean["month"] >= pd.Timestamp("1990-01-31"))
        & (clean["month"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    clean.to_csv(WELCH_GOYAL_CLEAN_FILE, index=False)

    print(f"Saved {WELCH_GOYAL_CLEAN_FILE}: {clean.shape}")
    print(f"Date range: {clean['month'].min()} to {clean['month'].max()}")
    print(f"Missing values: {int(clean.isna().sum().sum())}")
    print("Variables:")
    print(macro_names)


def main():
    prepare_welch_goyal_macro()


if __name__ == "__main__":
    main()
