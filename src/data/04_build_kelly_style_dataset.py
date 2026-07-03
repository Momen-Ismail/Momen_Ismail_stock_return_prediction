"""Create the raw Kelly-style predictor dataset.

This file constructs the raw model panel:
1. Add SIC2 industry dummies
2. Create macro-state variables
3. Select stock characteristics
4. Create characteristic x macro interactions
5. Save the raw full dataset

Final cleaning, splitting, winsorization, imputation, and rank-normalization
are done in file 05.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# 1) Settings
DATA_DIR = Path("output/data")
INPUT_FILE = DATA_DIR / "intermediate/monthly_panel_with_compustat_macro_1990_2025.csv"
OUTPUT_DIR = DATA_DIR / "final/kelly_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_FILE = OUTPUT_DIR / "model_dataset_kelly_raw_full_1990_2025.csv"
PREDICTOR_FILE = OUTPUT_DIR / "predictor_columns_kelly_raw.csv"
SUMMARY_FILE = OUTPUT_DIR / "kelly_raw_dataset_summary.csv"

TARGET = "target_excess_return_next_1m"


# 2) Create industry and macro-state variables
def aggregate_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce").sum(min_count=1)
    denominator = pd.to_numeric(denominator, errors="coerce").sum(min_count=1)

    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan

    return numerator / denominator


def first_available(group, column):
    if column not in group:
        return np.nan

    values = pd.to_numeric(group[column], errors="coerce").dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


def add_industry_dummies(data):
    """Kelly-style industry controls using SIC2 dummies."""
    sic2 = pd.get_dummies(
        pd.to_numeric(data.sic2, errors="coerce").fillna(-1).astype(int),
        prefix="sic2",
        dtype=int,
    ).rename(columns={"sic2_-1": "sic2_missing"})

    data = pd.concat([data, sic2], axis=1)

    return data, list(sic2.columns)


def add_macro_proxies(data):
    """Create the 8 Welch-Goyal-style macro variables used for interactions."""
    rows = []

    for month, group in data.groupby("month"):
        equity_issuance = pd.to_numeric(group.equity_issuance_at, errors="coerce")
        repurchases = pd.to_numeric(group.repurchase_at, errors="coerce")

        rows.append({
            "month": month,
            "wg_bm_proxy": aggregate_ratio(group.book_equity, group.comp_market_equity),
            "wg_ep_proxy": aggregate_ratio(group.comp_ni, group.comp_market_equity),
            "wg_dp_proxy": pd.to_numeric(group.dividend_yield_comp, errors="coerce").median(),
            "wg_ntis_proxy": (equity_issuance - repurchases).median(),
            "wg_tbl": first_available(group, "macro_tbill_3m_rate_dec_lag1"),
            "wg_tms": first_available(group, "macro_term_spread_lag1"),
            "wg_dfy": first_available(group, "macro_default_spread_lag1"),
            "wg_svar_proxy": first_available(group, "market_vol_1m") ** 2,
        })

    names = [
        "wg_bm_proxy",
        "wg_ep_proxy",
        "wg_dp_proxy",
        "wg_ntis_proxy",
        "wg_tbl",
        "wg_tms",
        "wg_dfy",
        "wg_svar_proxy",
    ]

    macro = pd.DataFrame(rows).sort_values("month")
    macro[names] = macro[names].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    data = data.merge(macro, on="month", how="left")

    return data, names


# 3) Select stock characteristics and create interactions
def select_characteristics(data, macro_names):
    """
    Select stock-level characteristics c_i,t.

    We exclude identifiers, targets, macro variables, market-wide variables,
    industry codes, and construction flags.
    """
    exclude = {
        "ticker",
        "month",
        "first_date",
        "last_date",

        TARGET,
        "target_return_next_1m",
        "RF_next_1m",

        "gvkey",
        "comp_conm",
        "comp_cusip",
        "comp_cik",
        "comp_datadate",
        "comp_available_month",
        "comp_fyear",
        "comp_fyr",
        "comp_sic",
        "comp_naics",
        "comp_gsector",
        "comp_ggroup",
        "comp_gind",
        "comp_gsubind",
        "sic2",

        "Mkt_RF",
        "SMB",
        "HML",
        "RF",

        "gspc_last_adj_close",
        "market_ret_1m",
        "market_vol_1m",
        "market_maxret_1m",
        "market_minret_1m",

        "vix_last_close",
        "vix_avg_1m",
        "vix_max_1m",
        "vix_min_1m",
        "vix_change_1m",

        "has_compustat_annual",
        "xrd_missing",
        "xsga_missing",
        "beta_obs_daily",
    } | set(macro_names)

    bad_prefixes = (
        "target_",
        "macro_",
        "wg_",
        "sic2_",
        "sector_",
    )

    characteristics = [
        name
        for name in data.select_dtypes(include=[np.number, "bool"]).columns
        if name not in exclude
        and not name.startswith(bad_prefixes)
        and not data[name].isna().all()
    ]

    return characteristics


def add_interactions(data, characteristics, macro_names):
    """Create characteristic x macro-state interactions."""
    blocks = []
    interaction_names = []

    for macro in macro_names:
        macro_values = pd.to_numeric(data[macro], errors="coerce").astype("float32")

        block = pd.DataFrame({
            f"{char}_x_{macro}":
            pd.to_numeric(data[char], errors="coerce").astype("float32") * macro_values
            for char in characteristics
        }, index=data.index)

        blocks.append(block)
        interaction_names.extend(block.columns.tolist())

    interactions = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan)
    data = pd.concat([data, interactions], axis=1)

    return data, interaction_names


# 4) Save raw model data
def prepare_model_data(data, characteristics, interactions, sic2_dummies):
    """Save the raw Kelly-style model dataset.

    Cleaning and normalization are deliberately not done here.
    They are moved to file 05 so the cleaning stage is visible.
    """
    predictors = characteristics + interactions + sic2_dummies

    predictors = [
        name
        for name in predictors
        if name in data and not data[name].isna().all()
    ]

    columns = ["ticker", "month", TARGET] + predictors

    full = (
        data[columns]
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    duplicate_rows = int(full.duplicated(["ticker", "month"]).sum())
    missing_targets = int(full[TARGET].isna().sum())

    if duplicate_rows:
        raise ValueError(f"Duplicate ticker-month rows found: {duplicate_rows}")

    full.to_csv(FULL_FILE, index=False)
    pd.DataFrame({"predictor": predictors}).to_csv(PREDICTOR_FILE, index=False)

    summary = {
        "raw_rows": len(full),
        "raw_columns": full.shape[1],
        "raw_predictors": len(predictors),
        "stock_characteristics_raw": len(characteristics),
        "interactions_raw": len(interactions),
        "sic2_dummies_raw": len(sic2_dummies),
        "missing_targets_raw": missing_targets,
        "duplicate_ticker_months_raw": duplicate_rows,
        "first_month": full.month.min(),
        "last_month": full.month.max(),
    }

    pd.DataFrame(summary.items(), columns=["item", "value"]).to_csv(SUMMARY_FILE, index=False)

    return full, predictors, summary


# 5) Run the pipeline
def main():
    data = pd.read_csv(INPUT_FILE, low_memory=False)

    data["month"] = pd.to_datetime(data.month)
    data["ticker"] = data.ticker.astype(str).str.upper().str.strip()

    data, sic2_dummies = add_industry_dummies(data)
    data, macro_names = add_macro_proxies(data)

    characteristics = select_characteristics(data, macro_names)
    data, interactions = add_interactions(data, characteristics, macro_names)

    full, predictors, summary = prepare_model_data(
        data,
        characteristics,
        interactions,
        sic2_dummies,
    )

    print(f"Saved raw Kelly dataset: {full.shape}")
    print(f"Raw predictors: {len(predictors)}")
    print(f"Stock characteristics raw: {summary['stock_characteristics_raw']}")
    print(f"Interactions raw: {summary['interactions_raw']}")
    print(f"SIC2 dummies raw: {summary['sic2_dummies_raw']}")
    print(f"Missing targets raw: {summary['missing_targets_raw']}")
    print(f"Duplicate ticker-months raw: {summary['duplicate_ticker_months_raw']}")
    print(f"Date range: {full.month.min()} to {full.month.max()}")


if __name__ == "__main__":
    main()