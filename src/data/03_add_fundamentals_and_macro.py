"""Add lagged Compustat fundamentals to the monthly stock panel.

Input:
- monthly stock panel from file 02
- raw Compustat annual fundamentals

Output:
- monthly panel with lagged accounting variables

Important timing rule:
Compustat variables are merged only after a six-month reporting lag.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow direct execution from the project root:
# python src/data/03_add_fundamentals_and_macro.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    MONTHLY_STOCK_FILE,
    COMPUSTAT_RAW_FILE,
    COMPUSTAT_CLEAN_FILE,
    PANEL_WITH_FUNDAMENTALS_FILE,
)


# ---------------------------------------------------------------------
# 1) Small cleaning helpers
# ---------------------------------------------------------------------
def column(data, name):
    """Return a column if it exists, otherwise return a missing-value series."""
    if name in data.columns:
        return data[name]

    return pd.Series(np.nan, index=data.index)


def ratio(numerator, denominator):
    """Safely compute a ratio and remove infinite values."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)

    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------
# 2) Clean Compustat and create accounting predictors
# ---------------------------------------------------------------------
def clean_compustat():
    """Clean Compustat annual data and create accounting predictors."""
    comp = pd.read_csv(COMPUSTAT_RAW_FILE, low_memory=False)
    comp.columns = comp.columns.str.lower().str.strip()

    required_columns = ["tic", "gvkey", "datadate"]
    missing = [name for name in required_columns if name not in comp.columns]

    if missing:
        raise ValueError(f"Missing Compustat columns: {missing}")

    comp["ticker"] = comp["tic"].astype(str).str.upper().str.strip()
    comp["gvkey"] = comp["gvkey"].astype(str).str.strip()
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

    comp = comp[
        comp["datadate"].notna()
        & ~comp["ticker"].isin(["", "NAN", "NONE"])
    ].copy()

    # Keep standard industrial, consolidated, USD observations when available.
    standard_filters = {
        "consol": "C",
        "indfmt": "INDL",
        "datafmt": "STD",
        "curcd": "USD",
    }

    for name, value in standard_filters.items():
        if name in comp.columns:
            comp = comp[comp[name].eq(value)].copy()

    numeric_columns = [
        "at", "act", "lct", "lt", "ceq", "seq", "teq",
        "txditc", "txdb", "itcb",
        "che", "dlc", "dltt", "dt",
        "pstk", "pstkrv", "pstkl",
        "invt", "rect", "ap", "ppegt", "ppent", "intan", "gdwl",
        "csho", "sale", "revt", "gp", "cogs", "xsga", "xrd",
        "oiadp", "oibdp", "ib", "ni", "oancf",
        "capx", "dvt", "dltis", "dltr", "sstk", "prstkc",
        "mkvalt", "prcc_c", "prcc_f", "cshtr_f",
        "fyear", "fyr", "sic", "naics",
        "gsector", "ggroup", "gind", "gsubind",
    ]

    for name in set(numeric_columns) & set(comp.columns):
        comp[name] = pd.to_numeric(comp[name], errors="coerce")

    comp = (
        comp.sort_values(["ticker", "datadate", "gvkey"])
        .drop_duplicates(["ticker", "datadate"], keep="last")
        .copy()
    )

    # ------------------------------------------------------------
    # Book equity
    # book equity = shareholders' equity + deferred taxes - preferred stock
    # ------------------------------------------------------------
    preferred_stock = (
        column(comp, "pstkrv")
        .fillna(column(comp, "pstkl"))
        .fillna(column(comp, "pstk"))
        .fillna(0)
    )

    deferred_taxes = (
        column(comp, "txditc")
        .fillna(column(comp, "txdb").fillna(0) + column(comp, "itcb").fillna(0))
        .fillna(0)
    )

    shareholders_equity = (
        column(comp, "seq")
        .fillna(column(comp, "ceq") + preferred_stock)
        .fillna(column(comp, "at") - column(comp, "lt"))
        .fillna(column(comp, "teq"))
    )

    comp["book_equity"] = shareholders_equity + deferred_taxes - preferred_stock
    comp.loc[comp["book_equity"] <= 0, "book_equity"] = np.nan

    comp["comp_market_equity"] = (
        column(comp, "mkvalt")
        .fillna(column(comp, "prcc_f").abs() * column(comp, "csho"))
        .fillna(column(comp, "prcc_c").abs() * column(comp, "csho"))
    )

    comp["debt_total"] = (
        column(comp, "dt")
        .fillna(column(comp, "dlc").fillna(0) + column(comp, "dltt").fillna(0))
    )

    comp["xrd_missing"] = column(comp, "xrd").isna().astype(int)
    comp["xsga_missing"] = column(comp, "xsga").isna().astype(int)
    comp["xrd_filled"] = column(comp, "xrd").fillna(0)

    comp["gross_profit_clean"] = column(comp, "gp").fillna(
        column(comp, "sale") - column(comp, "cogs")
    )

    comp["operating_profit_clean"] = column(comp, "oiadp").fillna(
        column(comp, "sale")
        - column(comp, "cogs")
        - column(comp, "xsga").fillna(0)
        + comp["xrd_filled"]
    )

    comp["tangibility"] = ratio(
        column(comp, "che")
        + 0.715 * column(comp, "rect")
        + 0.547 * column(comp, "invt")
        + 0.535 * column(comp, "ppegt"),
        column(comp, "at"),
    )

    # ------------------------------------------------------------
    # Accounting and valuation ratios
    # ------------------------------------------------------------
    ratios = {
        # Valuation
        "bm_comp": (comp["book_equity"], comp["comp_market_equity"]),
        "be_me": (comp["book_equity"], comp["comp_market_equity"]),
        "at_me": (column(comp, "at"), comp["comp_market_equity"]),
        "sale_me": (column(comp, "sale"), comp["comp_market_equity"]),
        "ni_me": (column(comp, "ni"), comp["comp_market_equity"]),
        "ocf_me": (column(comp, "oancf"), comp["comp_market_equity"]),
        "debt_me": (comp["debt_total"], comp["comp_market_equity"]),

        # Profitability
        "profitability_oiadp_at": (column(comp, "oiadp"), column(comp, "at")),
        "profitability_oibdp_at": (column(comp, "oibdp"), column(comp, "at")),
        "gp_at": (comp["gross_profit_clean"], column(comp, "at")),
        "op_at": (comp["operating_profit_clean"], column(comp, "at")),
        "roa_ni_at": (column(comp, "ni"), column(comp, "at")),
        "roa_ib_at": (column(comp, "ib"), column(comp, "at")),
        "roe_ni_be": (column(comp, "ni"), comp["book_equity"]),
        "ni_be": (column(comp, "ni"), comp["book_equity"]),
        "ocf_at": (column(comp, "oancf"), column(comp, "at")),

        # Leverage and balance sheet
        "leverage_debt_at": (comp["debt_total"], column(comp, "at")),
        "leverage_lt_at": (column(comp, "lt"), column(comp, "at")),
        "debt_at": (comp["debt_total"], column(comp, "at")),
        "cash_at": (column(comp, "che"), column(comp, "at")),
        "working_capital_at": (
            column(comp, "act") - column(comp, "lct"),
            column(comp, "at"),
        ),
        "current_ratio": (column(comp, "act"), column(comp, "lct")),
        "capx_at": (column(comp, "capx"), column(comp, "at")),
        "rd_at": (comp["xrd_filled"], column(comp, "at")),
        "sga_at": (column(comp, "xsga"), column(comp, "at")),
        "ppe_at": (column(comp, "ppent"), column(comp, "at")),
        "gross_ppe_at": (column(comp, "ppegt"), column(comp, "at")),
        "intangibles_at": (column(comp, "intan"), column(comp, "at")),
        "goodwill_at": (column(comp, "gdwl"), column(comp, "at")),
        "inventory_at": (column(comp, "invt"), column(comp, "at")),
        "receivables_at": (column(comp, "rect"), column(comp, "at")),
        "payables_at": (column(comp, "ap"), column(comp, "at")),

        # Margins and efficiency
        "gross_margin": (comp["gross_profit_clean"], column(comp, "sale")),
        "operating_margin": (column(comp, "oiadp"), column(comp, "sale")),
        "net_margin": (column(comp, "ni"), column(comp, "sale")),
        "asset_turnover": (column(comp, "sale"), column(comp, "at")),

        # Accruals and payout
        "accruals_at": (column(comp, "ib") - column(comp, "oancf"), column(comp, "at")),
        "oaccruals_at": (column(comp, "ib") - column(comp, "oancf"), column(comp, "at")),
        "dividends_at": (column(comp, "dvt"), column(comp, "at")),
        "dividend_yield_comp": (column(comp, "dvt"), comp["comp_market_equity"]),

        # Financing
        "debt_issuance_at": (column(comp, "dltis"), column(comp, "at")),
        "debt_reduction_at": (column(comp, "dltr"), column(comp, "at")),
        "equity_issuance_at": (column(comp, "sstk"), column(comp, "at")),
        "repurchase_at": (column(comp, "prstkc"), column(comp, "at")),
        "share_turnover_comp": (column(comp, "cshtr_f"), column(comp, "csho")),
    }

    for name, (numerator, denominator) in ratios.items():
        comp[name] = ratio(numerator, denominator)

    comp["log_comp_market_equity"] = np.log(
        comp["comp_market_equity"].where(comp["comp_market_equity"] > 0)
    )

    comp["dividend_dummy"] = column(comp, "dvt").fillna(0).gt(0).astype(int)
    comp["sic2"] = np.floor(column(comp, "sic") / 100)

    # ------------------------------------------------------------
    # Growth variables: sorted by gvkey and report date
    # ------------------------------------------------------------
    comp = comp.sort_values(["gvkey", "datadate"]).copy()

    lag_variables = [
        "at",
        "sale",
        "revt",
        "book_equity",
        "comp_market_equity",
        "invt",
        "ppegt",
    ]

    for name in lag_variables:
        comp[f"{name}_lag1"] = comp.groupby("gvkey")[name].shift(1)

    comp["asset_growth"] = ratio(
        column(comp, "at") - comp["at_lag1"],
        comp["at_lag1"],
    )

    comp["sales_growth"] = ratio(
        column(comp, "sale") - comp["sale_lag1"],
        comp["sale_lag1"],
    )

    comp["revenue_growth"] = ratio(
        column(comp, "revt") - comp["revt_lag1"],
        comp["revt_lag1"],
    )

    comp["book_equity_growth"] = ratio(
        comp["book_equity"] - comp["book_equity_lag1"],
        comp["book_equity_lag1"],
    )

    comp["market_equity_growth_comp"] = ratio(
        comp["comp_market_equity"] - comp["comp_market_equity_lag1"],
        comp["comp_market_equity_lag1"],
    )

    # JKP-style growth scaled by lagged assets.
    comp["be_gr1a"] = ratio(
        comp["book_equity"] - comp["book_equity_lag1"],
        comp["at_lag1"],
    )

    comp["inv_gr1a"] = ratio(
        column(comp, "invt") - comp["invt_lag1"],
        comp["at_lag1"],
    )

    comp["ppeinv_gr1a"] = ratio(
        (column(comp, "ppegt") + column(comp, "invt"))
        - (comp["ppegt_lag1"] + comp["invt_lag1"]),
        comp["at_lag1"],
    )

    comp["investment_asset_growth"] = comp["asset_growth"]

    # Six-month lag to avoid using accounting data before it is public.
    comp["comp_available_month"] = (
        comp["datadate"]
        + pd.DateOffset(months=6)
        + pd.offsets.MonthEnd(0)
    )

    keep_columns = [
        "ticker", "gvkey", "datadate", "fyear", "fyr", "conm", "cusip", "cik",
        "sic", "naics", "gsector", "ggroup", "gind", "gsubind",
        "comp_available_month",

        "book_equity", "comp_market_equity", "debt_total",
        "at", "sale", "revt", "ni", "ib", "oiadp", "oibdp", "capx", "oancf",
        "xrd", "xsga", "csho", "prcc_f", "mkvalt",

        "bm_comp", "be_me", "at_me", "sale_me", "ni_me", "ocf_me", "debt_me",
        "log_comp_market_equity",

        "profitability_oiadp_at", "profitability_oibdp_at",
        "gp_at", "op_at", "roa_ni_at", "roa_ib_at", "roe_ni_be", "ni_be",
        "ocf_at",

        "leverage_debt_at", "leverage_lt_at", "debt_at",
        "cash_at", "working_capital_at", "current_ratio",
        "capx_at", "rd_at", "sga_at", "ppe_at", "gross_ppe_at",
        "intangibles_at", "goodwill_at", "inventory_at", "receivables_at",
        "payables_at", "tangibility",

        "gross_margin", "operating_margin", "net_margin", "asset_turnover",
        "accruals_at", "oaccruals_at",

        "dividends_at", "dividend_yield_comp", "dividend_dummy",
        "debt_issuance_at", "debt_reduction_at", "equity_issuance_at",
        "repurchase_at", "share_turnover_comp",

        "asset_growth", "sales_growth", "revenue_growth",
        "book_equity_growth", "market_equity_growth_comp",
        "be_gr1a", "inv_gr1a", "ppeinv_gr1a",
        "investment_asset_growth",

        "xrd_missing", "xsga_missing", "sic2",
    ]

    rename_columns = {
        "datadate": "comp_datadate",
        "fyear": "comp_fyear",
        "fyr": "comp_fyr",
        "conm": "comp_conm",
        "cusip": "comp_cusip",
        "cik": "comp_cik",
        "sic": "comp_sic",
        "naics": "comp_naics",
        "gsector": "comp_gsector",
        "ggroup": "comp_ggroup",
        "gind": "comp_gind",
        "gsubind": "comp_gsubind",
        "at": "comp_at",
        "sale": "comp_sale",
        "revt": "comp_revt",
        "ni": "comp_ni",
        "ib": "comp_ib",
        "oiadp": "comp_oiadp",
        "oibdp": "comp_oibdp",
        "capx": "comp_capx",
        "oancf": "comp_oancf",
        "xrd": "comp_xrd",
        "xsga": "comp_xsga",
        "csho": "comp_csho",
        "prcc_f": "comp_prcc_f",
        "mkvalt": "comp_mkvalt",
    }

    comp = comp[[name for name in keep_columns if name in comp.columns]].rename(
        columns=rename_columns
    )

    comp.to_csv(COMPUSTAT_CLEAN_FILE, index=False)

    return comp


# ---------------------------------------------------------------------
# 3) Merge Compustat with the monthly panel
# ---------------------------------------------------------------------
def merge_compustat(monthly, comp):
    """Merge lagged Compustat data to monthly stock observations."""
    monthly = monthly.sort_values(["ticker", "month"]).copy()
    monthly["ticker"] = monthly["ticker"].astype(str).str.upper().str.strip()

    comp = comp.sort_values(["ticker", "comp_available_month"]).copy()

    parts = []

    for ticker, stock in monthly.groupby("ticker"):
        accounting = comp[comp["ticker"].eq(ticker)].copy()
        stock = stock.sort_values("month").copy()

        if accounting.empty:
            for name in comp.columns.drop("ticker"):
                stock[name] = np.nan

            parts.append(stock)
            continue

        merged = pd.merge_asof(
            stock,
            accounting.sort_values("comp_available_month"),
            by="ticker",
            left_on="month",
            right_on="comp_available_month",
            direction="backward",
        )

        parts.append(merged)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["ticker", "month"]).reset_index(drop=True)

    panel["has_compustat_annual"] = panel["comp_datadate"].notna().astype(int)

    return panel


# ---------------------------------------------------------------------
# 4) Run pipeline
# ---------------------------------------------------------------------
def main():
    monthly = pd.read_csv(MONTHLY_STOCK_FILE, low_memory=False)
    monthly["month"] = pd.to_datetime(monthly["month"])

    comp = clean_compustat()
    panel = merge_compustat(monthly, comp)

    panel = panel.sort_values(["ticker", "month"]).reset_index(drop=True)
    panel.to_csv(PANEL_WITH_FUNDAMENTALS_FILE, index=False)

    print(f"Saved {PANEL_WITH_FUNDAMENTALS_FILE}: {panel.shape}")
    print(f"Compustat matches: {panel['has_compustat_annual'].sum():,}")
    print(f"Missing targets: {panel[TARGET].isna().sum():,}")


if __name__ == "__main__":
    main()
