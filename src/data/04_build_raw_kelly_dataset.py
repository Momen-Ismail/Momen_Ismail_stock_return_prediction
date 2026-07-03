"""Create the raw Kelly-style predictor dataset.

This file constructs the raw model panel:

1. Add SIC2 industry dummies
2. Create Welch-Goyal-style macro-state proxies
3. Select stock characteristics
4. Create stock characteristic x macro-state interactions
5. Save the raw full dataset

Final cleaning, splitting, winsorization, imputation, and rank-normalization
are done in file 05.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow direct execution from the project root:
# python src/data/04_build_raw_kelly_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    PANEL_WITH_FUNDAMENTALS_FILE,
    WELCH_GOYAL_CLEAN_FILE,
    RAW_KELLY_FILE,
    RAW_PREDICTOR_FILE,
    RAW_KELLY_SUMMARY_FILE,
)

CHARACTERISTIC_COLUMNS = [
    # ------------------------------------------------------------
    # Return and momentum characteristics
    # ------------------------------------------------------------
    "ret_1m",
    "mom1m",
    "mom3m",
    "mom6m",
    "mom12m",
    "mom36m",
    "chmom",

    # ------------------------------------------------------------
    # Volatility, downside/upside return, and market risk
    # ------------------------------------------------------------
    "retvol_1m",
    "retvol3m",
    "retvol6m",
    "retvol12m",
    "rvol_21d",
    "maxret_1m",
    "minret_1m",
    "rmax1_21d",
    "rmax5_21d",
    "beta_12m",
    "betasq_12m",
    "idiovol_12m",

    # ------------------------------------------------------------
    # Liquidity and trading activity
    # ------------------------------------------------------------
    "avg_volume_1m",
    "std_volume_1m",
    "avg_dolvol_1m",
    "std_dolvol_1m",
    "avg_log_dolvol_1m",
    "amihud_1m",
    "ami_126d",
    "zerotrade_1m",
    "zero_trades_21d",
    "zero_trades_126d",
    "dolvol_126d",
    "dolvol_var_126d",
    "volume_growth_1m",
    "dolvol_growth_1m",

    # ------------------------------------------------------------
    # Price level and trend variables
    # ------------------------------------------------------------
    "last_adj_close",
    "last_close",
    "price_to_ma3",
    "price_to_ma12",
    "ma3_to_ma12",
    "dist_from_high_12m",
    "avg_range_1m",
    "max_range_1m",

    # ------------------------------------------------------------
    # Valuation characteristics
    # ------------------------------------------------------------
    "bm_comp",
    "be_me",
    "at_me",
    "sale_me",
    "ni_me",
    "ocf_me",
    "debt_me",
    "log_comp_market_equity",

    # ------------------------------------------------------------
    # Profitability characteristics
    # ------------------------------------------------------------
    "profitability_oiadp_at",
    "profitability_oibdp_at",
    "gp_at",
    "op_at",
    "roa_ni_at",
    "roa_ib_at",
    "roe_ni_be",
    "ni_be",
    "ocf_at",

    # ------------------------------------------------------------
    # Leverage, balance-sheet strength, and tangibility
    # ------------------------------------------------------------
    "leverage_debt_at",
    "leverage_lt_at",
    "debt_at",
    "cash_at",
    "working_capital_at",
    "current_ratio",
    "tangibility",

    # ------------------------------------------------------------
    # Investment and asset composition
    # ------------------------------------------------------------
    "capx_at",
    "rd_at",
    "sga_at",
    "ppe_at",
    "gross_ppe_at",
    "intangibles_at",
    "goodwill_at",
    "inventory_at",
    "receivables_at",
    "payables_at",

    # ------------------------------------------------------------
    # Margins and operating efficiency
    # ------------------------------------------------------------
    "gross_margin",
    "operating_margin",
    "net_margin",
    "asset_turnover",

    # ------------------------------------------------------------
    # Accruals, payout, and financing
    # ------------------------------------------------------------
    "accruals_at",
    "oaccruals_at",
    "dividends_at",
    "dividend_yield_comp",
    "dividend_dummy",
    "debt_issuance_at",
    "debt_reduction_at",
    "equity_issuance_at",
    "repurchase_at",
    "share_turnover_comp",

    # ------------------------------------------------------------
    # Growth characteristics
    # ------------------------------------------------------------
    "asset_growth",
    "sales_growth",
    "revenue_growth",
    "book_equity_growth",
    "market_equity_growth_comp",
    "be_gr1a",
    "inv_gr1a",
    "ppeinv_gr1a",
    "investment_asset_growth",
]

# ---------------------------------------------------------------------
# 1) Helper functions
# ---------------------------------------------------------------------
def aggregate_ratio(numerator, denominator):
    """Compute a cross-sectional aggregate ratio."""
    numerator = pd.to_numeric(numerator, errors="coerce").sum(min_count=1)
    denominator = pd.to_numeric(denominator, errors="coerce").sum(min_count=1)

    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan

    return numerator / denominator


def first_available(group, column):
    """Take the first non-missing value of a monthly macro or market variable."""
    if column not in group.columns:
        return np.nan

    values = pd.to_numeric(group[column], errors="coerce").dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


# ---------------------------------------------------------------------
# 2) Industry dummies and macro-state variables
# ---------------------------------------------------------------------
def add_industry_dummies(data):
    """Add Kelly-style industry controls using SIC2 dummies."""
    sic2 = pd.get_dummies(
        pd.to_numeric(data["sic2"], errors="coerce").fillna(-1).astype(int),
        prefix="sic2",
        dtype=int,
    )

    sic2 = sic2.rename(columns={"sic2_-1": "sic2_missing"})

    data = pd.concat([data, sic2], axis=1)

    return data, list(sic2.columns)


def add_welch_goyal_macro(data):
    """Add clean Welch-Goyal macro variables used for interactions."""
    macro = pd.read_csv(WELCH_GOYAL_CLEAN_FILE)
    macro["month"] = pd.to_datetime(macro["month"])

    macro_names = [
        "wg_dp",
        "wg_ep",
        "wg_bm",
        "wg_ntis",
        "wg_tbl",
        "wg_tms",
        "wg_dfy",
        "wg_svar",
    ]

    missing = [name for name in macro_names if name not in macro.columns]

    if missing:
        raise ValueError(f"Missing clean Welch-Goyal variables: {missing}")

    data = data.merge(
        macro[["month"] + macro_names],
        on="month",
        how="left",
    )

    return data, macro_names


# ---------------------------------------------------------------------
# 3) Select stock characteristics and create interactions
# ---------------------------------------------------------------------
def select_characteristics(data, macro_names):
    """Select manually chosen stock characteristics.

    The goal is not to exactly reproduce all Kelly/Gu-Xiu characteristics.
    Some original characteristics require CRSP bid-ask spreads, quarterly
    earnings-announcement data, analyst-type data, or other unavailable fields.

    Instead, we keep only characteristics that we constructed directly from
    Yahoo daily prices and Compustat annual fundamentals and that we can defend.
    """
    characteristics = [
        name
        for name in CHARACTERISTIC_COLUMNS
        if name in data.columns
        and pd.api.types.is_numeric_dtype(data[name])
        and not data[name].isna().all()
    ]

    missing_from_data = [
        name
        for name in CHARACTERISTIC_COLUMNS
        if name not in data.columns
    ]

    if missing_from_data:
        print("Requested characteristics not found and skipped:")
        print(missing_from_data)

    print(f"Selected defensible stock characteristics: {len(characteristics)}")

    return characteristics

def add_interactions(data, characteristics, macro_names):
    """Create stock characteristic x macro-state interactions."""
    blocks = []
    interaction_names = []

    for macro in macro_names:
        macro_values = pd.to_numeric(
            data[macro],
            errors="coerce",
        ).astype("float32")

        block = pd.DataFrame(
            {
                f"{characteristic}_x_{macro}": (
                    pd.to_numeric(
                        data[characteristic],
                        errors="coerce",
                    ).astype("float32")
                    * macro_values
                )
                for characteristic in characteristics
            },
            index=data.index,
        )

        blocks.append(block)
        interaction_names.extend(block.columns.tolist())

    interactions = pd.concat(blocks, axis=1)
    interactions = interactions.replace([np.inf, -np.inf], np.nan)

    data = pd.concat([data, interactions], axis=1)

    return data, interaction_names


# ---------------------------------------------------------------------
# 4) Save raw model data
# ---------------------------------------------------------------------
def prepare_model_data(data, characteristics, interactions, sic2_dummies):
    """Save the raw Kelly-style model dataset.

    Cleaning and normalization are deliberately not done here.
    They are done in file 05.
    """
    predictors = characteristics + interactions + sic2_dummies

    predictors = [
        name
        for name in predictors
        if name in data.columns and not data[name].isna().all()
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

    full.to_csv(RAW_KELLY_FILE, index=False)
    pd.DataFrame({"predictor": predictors}).to_csv(RAW_PREDICTOR_FILE, index=False)

    summary = {
        "raw_rows": len(full),
        "raw_columns": full.shape[1],
        "raw_predictors": len(predictors),
        "stock_characteristics_raw": len(characteristics),
        "interactions_raw": len(interactions),
        "sic2_dummies_raw": len(sic2_dummies),
        "missing_targets_raw": missing_targets,
        "duplicate_ticker_months_raw": duplicate_rows,
        "first_month": full["month"].min(),
        "last_month": full["month"].max(),
    }

    pd.DataFrame(
        summary.items(),
        columns=["item", "value"],
    ).to_csv(RAW_KELLY_SUMMARY_FILE, index=False)

    return full, predictors, summary


# ---------------------------------------------------------------------
# 5) Run pipeline
# ---------------------------------------------------------------------
def main():
    data = pd.read_csv(PANEL_WITH_FUNDAMENTALS_FILE, low_memory=False)

    data["month"] = pd.to_datetime(data["month"])
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()

    data, sic2_dummies = add_industry_dummies(data)
    data, macro_names = add_welch_goyal_macro(data)
    characteristics = select_characteristics(data, macro_names)
    data, interactions = add_interactions(data, characteristics, macro_names)

    full, predictors, summary = prepare_model_data(
        data=data,
        characteristics=characteristics,
        interactions=interactions,
        sic2_dummies=sic2_dummies,
    )

    print(f"Saved raw Kelly dataset: {full.shape}")
    print(f"Raw predictors: {len(predictors)}")
    print(f"Stock characteristics raw: {summary['stock_characteristics_raw']}")
    print(f"Interactions raw: {summary['interactions_raw']}")
    print(f"SIC2 dummies raw: {summary['sic2_dummies_raw']}")
    print(f"Missing targets raw: {summary['missing_targets_raw']}")
    print(f"Duplicate ticker-months raw: {summary['duplicate_ticker_months_raw']}")
    print(f"Date range: {full['month'].min()} to {full['month'].max()}")


if __name__ == "__main__":
    main()