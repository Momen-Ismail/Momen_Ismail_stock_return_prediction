"""Add lagged annual Compustat fundamentals to the monthly stock panel."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
# 1) Helpers
# ---------------------------------------------------------------------
def column(data, name):
    """Return a column or a missing-value series."""
    if name in data.columns:
        return data[name]

    return pd.Series(np.nan, index=data.index)


def ratio(numerator, denominator):
    """Calculate a ratio safely."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = (
        pd.to_numeric(denominator, errors="coerce")
        .replace(0, np.nan)
    )

    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], np.nan)
    )


# ---------------------------------------------------------------------
# 2) Clean Compustat and construct accounting characteristics
# ---------------------------------------------------------------------
def clean_compustat():
    """Create compact annual accounting predictors with a six-month lag."""
    comp = pd.read_csv(COMPUSTAT_RAW_FILE, low_memory=False)
    comp.columns = comp.columns.str.lower().str.strip()

    required = [
        "tic",
        "gvkey",
        "datadate",
        "act",
        "lct",
        "invt",
        "rect",
        "che",
        "sale",
        "capx",
        "oancf",
    ]
    missing = [name for name in required if name not in comp.columns]

    if missing:
        raise ValueError(f"Missing Compustat columns: {missing}")

    comp["ticker"] = comp["tic"].astype(str).str.upper().str.strip()
    comp["gvkey"] = comp["gvkey"].astype(str).str.strip()
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

    comp = comp[
        comp["datadate"].notna()
        & ~comp["ticker"].isin(["", "NAN", "NONE"])
    ].copy()

    # Keep standard consolidated industrial USD statements
    for name, value in {
        "consol": "C",
        "indfmt": "INDL",
        "datafmt": "STD",
        "curcd": "USD",
    }.items():
        if name in comp.columns:
            comp = comp[comp[name].eq(value)].copy()

    numeric_columns = [
        "act", "lct",
        "at", "lt", "ceq", "seq", "teq",
        "txditc", "txdb", "itcb",
        "che", "dlc", "dltt", "dt",
        "pstk", "pstkrv", "pstkl",
        "invt", "rect", "ppegt", "ppent",
        "csho", "sale", "cogs", "xsga", "xrd",
        "oiadp", "ib", "oancf", "capx", "dvt",
        "sstk", "prstkc", "mkvalt", "prcc_c", "prcc_f",
        "sic",
    ]

    for name in numeric_columns:
        if name in comp.columns:
            comp[name] = pd.to_numeric(comp[name], errors="coerce")

    comp = (
        comp.sort_values(["ticker", "datadate", "gvkey"])
        .drop_duplicates(["ticker", "datadate"], keep="last")
        .copy()
    )

    # Book equity
    preferred_stock = (
        column(comp, "pstkrv")
        .fillna(column(comp, "pstkl"))
        .fillna(column(comp, "pstk"))
        .fillna(0)
    )

    deferred_taxes = (
        column(comp, "txditc")
        .fillna(
            column(comp, "txdb").fillna(0)
            + column(comp, "itcb").fillna(0)
        )
        .fillna(0)
    )

    shareholders_equity = (
        column(comp, "seq")
        .fillna(column(comp, "ceq") + preferred_stock)
        .fillna(column(comp, "at") - column(comp, "lt"))
        .fillna(column(comp, "teq"))
    )

    comp["book_equity"] = (
        shareholders_equity
        + deferred_taxes
        - preferred_stock
    )

    comp.loc[comp["book_equity"] <= 0, "book_equity"] = np.nan

    # Market equity and debt
    comp["comp_market_equity"] = (
        column(comp, "mkvalt")
        .fillna(column(comp, "prcc_f").abs() * column(comp, "csho"))
        .fillna(column(comp, "prcc_c").abs() * column(comp, "csho"))
    )

    comp["debt_total"] = (
        column(comp, "dt")
        .fillna(
            column(comp, "dlc").fillna(0)
            + column(comp, "dltt").fillna(0)
        )
    )

    # Operating profitability
    comp["xrd_missing"] = column(comp, "xrd").isna().astype(int)
    comp["xrd_filled"] = column(comp, "xrd").fillna(0)

    comp["operating_profit"] = (
        column(comp, "oiadp")
        .fillna(
            column(comp, "sale")
            - column(comp, "cogs")
            - column(comp, "xsga").fillna(0)
            + comp["xrd_filled"]
        )
    )

    # Final accounting characteristics
    ratios = {
        "be_me": (
            comp["book_equity"],
            comp["comp_market_equity"],
        ),
        "ocf_me": (
            column(comp, "oancf"),
            comp["comp_market_equity"],
        ),
        "op_at": (
            comp["operating_profit"],
            column(comp, "at"),
        ),
        "ocf_at": (
            column(comp, "oancf"),
            column(comp, "at"),
        ),
        "debt_at": (
            comp["debt_total"],
            column(comp, "at"),
        ),
        "cash_at": (
            column(comp, "che"),
            column(comp, "at"),
        ),
        "cashflow_to_debt": (
            column(comp, "oancf"),
            comp["debt_total"],
        ),
        "current_ratio": (
            column(comp, "act"),
            column(comp, "lct"),
        ),
        "quick_ratio": (
            column(comp, "act") - column(comp, "invt"),
            column(comp, "lct"),
        ),
        "capx_at": (
            column(comp, "capx"),
            column(comp, "at"),
        ),
        "rd_at": (
            comp["xrd_filled"],
            column(comp, "at"),
        ),
        "ppe_at": (
            column(comp, "ppent"),
            column(comp, "at"),
        ),
        "tangibility": (
            column(comp, "che")
            + 0.715 * column(comp, "rect")
            + 0.547 * column(comp, "invt")
            + 0.535 * column(comp, "ppegt"),
            column(comp, "at"),
        ),
        "asset_turnover": (
            column(comp, "sale"),
            column(comp, "at"),
        ),
        "sales_to_inventory": (
            column(comp, "sale"),
            column(comp, "invt"),
        ),
        "sales_to_cash": (
            column(comp, "sale"),
            column(comp, "che"),
        ),
        "sales_to_receivables": (
            column(comp, "sale"),
            column(comp, "rect"),
        ),
        "accruals_at": (
            column(comp, "ib") - column(comp, "oancf"),
            column(comp, "at"),
        ),
        "equity_issuance_at": (
            column(comp, "sstk"),
            column(comp, "at"),
        ),
        "repurchase_at": (
            column(comp, "prstkc"),
            column(comp, "at"),
        ),
    }

    for name, (numerator, denominator) in ratios.items():
        comp[name] = ratio(numerator, denominator)

    comp["log_comp_market_equity"] = np.log(
        comp["comp_market_equity"].where(
            comp["comp_market_equity"] > 0
        )
    )

    comp["dividend_dummy"] = (
        column(comp, "dvt")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    comp["sic2"] = np.floor(column(comp, "sic") / 100)

    # Annual growth
    comp = comp.sort_values(["gvkey", "datadate"]).copy()
    grouped = comp.groupby("gvkey")

    comp["at_lag1"] = grouped["at"].shift(1)
    comp["sale_lag1"] = grouped["sale"].shift(1)
    comp["invt_lag1"] = grouped["invt"].shift(1)
    comp["ppegt_lag1"] = grouped["ppegt"].shift(1)
    comp["capx_lag1"] = grouped["capx"].shift(1)

    comp["asset_growth"] = ratio(
        column(comp, "at") - comp["at_lag1"],
        comp["at_lag1"],
    )

    comp["sales_growth"] = ratio(
        column(comp, "sale") - comp["sale_lag1"],
        comp["sale_lag1"],
    )

    comp["capx_growth"] = ratio(
        column(comp, "capx") - comp["capx_lag1"],
        comp["capx_lag1"],
    )

    comp["ppeinv_gr1a"] = ratio(
        (
            column(comp, "ppegt")
            + column(comp, "invt")
        )
        - (
            comp["ppegt_lag1"]
            + comp["invt_lag1"]
        ),
        comp["at_lag1"],
    )

    # Accounting data becomes usable six months after fiscal year-end
    comp["comp_available_month"] = (
        comp["datadate"]
        + pd.DateOffset(months=6)
        + pd.offsets.MonthEnd(0)
    )

    keep_columns = [
        "ticker",
        "datadate",
        "comp_available_month",
        "sic2",
        "be_me",
        "ocf_me",
        "log_comp_market_equity",
        "op_at",
        "ocf_at",
        "debt_at",
        "cash_at",
        "cashflow_to_debt",
        "current_ratio",
        "quick_ratio",
        "capx_at",
        "capx_growth",
        "rd_at",
        "ppe_at",
        "tangibility",
        "asset_turnover",
        "sales_to_inventory",
        "sales_to_cash",
        "sales_to_receivables",
        "accruals_at",
        "dividend_dummy",
        "equity_issuance_at",
        "repurchase_at",
        "asset_growth",
        "sales_growth",
        "ppeinv_gr1a",
        "xrd_missing",
    ]

    comp = (
        comp[keep_columns]
        .rename(columns={"datadate": "comp_datadate"})
        .sort_values(["ticker", "comp_available_month"])
        .reset_index(drop=True)
    )

    comp.to_csv(COMPUSTAT_CLEAN_FILE, index=False)

    return comp


# ---------------------------------------------------------------------
# 3) Merge the latest available fundamentals into each stock-month
# ---------------------------------------------------------------------
def merge_compustat(monthly, comp):
    """Attach the latest available annual Compustat report."""
    monthly = monthly.copy()
    monthly["ticker"] = (
        monthly["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    monthly = monthly.sort_values(["month", "ticker"])
    comp = comp.sort_values(["comp_available_month", "ticker"])

    panel = pd.merge_asof(
        monthly,
        comp,
        by="ticker",
        left_on="month",
        right_on="comp_available_month",
        direction="backward",
    )

    panel["has_compustat_annual"] = (
        panel["comp_datadate"].notna().astype(int)
    )

    return panel.sort_values(
        ["ticker", "month"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# 4) Run pipeline
# ---------------------------------------------------------------------
def main():
    """Merge annual Compustat fundamentals into the monthly panel."""
    monthly = pd.read_csv(MONTHLY_STOCK_FILE, low_memory=False)
    monthly["month"] = pd.to_datetime(monthly["month"])

    panel = merge_compustat(
        monthly,
        clean_compustat(),
    )

    panel.to_csv(PANEL_WITH_FUNDAMENTALS_FILE, index=False)

    print(f"Saved: {PANEL_WITH_FUNDAMENTALS_FILE}")
    print(f"Shape: {panel.shape}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Compustat matches: {panel['has_compustat_annual'].sum():,}")
    print(f"Missing targets: {panel[TARGET].isna().sum():,}")


if __name__ == "__main__":
    main()
