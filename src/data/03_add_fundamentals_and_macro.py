"""Add lagged annual Compustat fundamentals to the monthly stock panel."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    MAX_COMPUSTAT_AGE_MONTHS,
    MONTHLY_STOCK_FILE,
    COMPUSTAT_RAW_FILE,
    COMPUSTAT_CLEAN_FILE,
    PANEL_WITH_FUNDAMENTALS_FILE,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def column(data, name):
    return (
        data[name]
        if name in data.columns
        else pd.Series(np.nan, index=data.index)
    )


def ratio(numerator, denominator):
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = (
        pd.to_numeric(
            denominator,
            errors="coerce",
        )
        .replace(0, np.nan)
    )

    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], np.nan)
    )


# ---------------------------------------------------------------------
# Clean annual Compustat data
# ---------------------------------------------------------------------
def clean_compustat():
    comp = pd.read_csv(
        COMPUSTAT_RAW_FILE,
        low_memory=False,
    )

    comp.columns = (
        comp.columns
        .str.lower()
        .str.strip()
    )

    comp["ticker"] = (
        comp["tic"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    comp["gvkey"] = (
        comp["gvkey"]
        .astype(str)
        .str.strip()
    )

    comp["datadate"] = pd.to_datetime(
        comp["datadate"],
        errors="coerce",
    )

    comp = comp.loc[
        comp["datadate"].notna()
        & ~comp["ticker"].isin(
            ["", "NAN", "NONE"]
        )
    ].copy()

    filters = {
        "consol": "C",
        "indfmt": "INDL",
        "datafmt": "STD",
        "popsrc": "D",
        "curcd": "USD",
    }

    for name, value in filters.items():
        if name in comp.columns:
            comp = comp.loc[
                comp[name].eq(value)
            ].copy()

    numeric_columns = [
        "act",
        "lct",
        "at",
        "lt",
        "ceq",
        "seq",
        "teq",
        "txditc",
        "txdb",
        "itcb",
        "che",
        "dlc",
        "dltt",
        "dt",
        "pstk",
        "pstkrv",
        "pstkl",
        "invt",
        "rect",
        "ppegt",
        "ppent",
        "csho",
        "sale",
        "cogs",
        "xsga",
        "xrd",
        "oiadp",
        "ib",
        "oancf",
        "capx",
        "dvt",
        "sstk",
        "prstkc",
        "mkvalt",
        "prcc_c",
        "prcc_f",
        "sic",
    ]

    for name in numeric_columns:
        if name in comp.columns:
            comp[name] = pd.to_numeric(
                comp[name],
                errors="coerce",
            )

    comp = (
        comp.sort_values(
            ["gvkey", "datadate", "ticker"]
        )
        .drop_duplicates(
            ["gvkey", "datadate"],
            keep="last",
        )
        .reset_index(drop=True)
    )

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
        .fillna(
            column(comp, "ceq")
            + preferred_stock
        )
        .fillna(
            column(comp, "at")
            - column(comp, "lt")
        )
        .fillna(column(comp, "teq"))
    )

    comp["book_equity"] = (
        shareholders_equity
        + deferred_taxes
        - preferred_stock
    )

    comp.loc[
        comp["book_equity"].le(0),
        "book_equity",
    ] = np.nan

    comp["comp_market_equity"] = (
        column(comp, "mkvalt")
        .fillna(
            column(comp, "prcc_f").abs()
            * column(comp, "csho")
        )
        .fillna(
            column(comp, "prcc_c").abs()
            * column(comp, "csho")
        )
    )

    comp["debt_total"] = (
        column(comp, "dt")
        .fillna(
            column(comp, "dlc").fillna(0)
            + column(comp, "dltt").fillna(0)
        )
    )

    comp["xrd_missing"] = (
        column(comp, "xrd")
        .isna()
        .astype(int)
    )

    xrd_filled = (
        column(comp, "xrd")
        .fillna(0)
    )

    comp["operating_profit"] = (
        column(comp, "oiadp")
        .fillna(
            column(comp, "sale")
            - column(comp, "cogs")
            - column(comp, "xsga").fillna(0)
            + xrd_filled
        )
    )

    ratio_definitions = {
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
            column(comp, "act")
            - column(comp, "invt"),
            column(comp, "lct"),
        ),
        "capx_at": (
            column(comp, "capx"),
            column(comp, "at"),
        ),
        "rd_at": (
            xrd_filled,
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
            column(comp, "ib")
            - column(comp, "oancf"),
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

    for name, values in ratio_definitions.items():
        numerator, denominator = values

        comp[name] = ratio(
            numerator,
            denominator,
        )

    comp["log_comp_market_equity"] = np.log(
        comp["comp_market_equity"].where(
            comp["comp_market_equity"].gt(0)
        )
    )

    comp["dividend_dummy"] = (
        column(comp, "dvt")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    comp["sic2"] = np.floor(
        column(comp, "sic") / 100
    )

    comp = (
        comp.sort_values(
            ["gvkey", "datadate"]
        )
        .reset_index(drop=True)
    )

    grouped = comp.groupby("gvkey")

    previous_date = (
        grouped["datadate"]
        .shift(1)
    )

    previous_assets = (
        grouped["at"]
        .shift(1)
    )

    previous_sales = (
        grouped["sale"]
        .shift(1)
    )

    previous_inventory = (
        grouped["invt"]
        .shift(1)
    )

    previous_ppe = (
        grouped["ppegt"]
        .shift(1)
    )

    previous_capx = (
        grouped["capx"]
        .shift(1)
    )

    annual_gap = (
        comp["datadate"] - previous_date
    ).dt.days.between(300, 430)

    comp["asset_growth"] = ratio(
        column(comp, "at")
        - previous_assets,
        previous_assets,
    )

    comp["sales_growth"] = ratio(
        column(comp, "sale")
        - previous_sales,
        previous_sales,
    )

    comp["capx_growth"] = ratio(
        column(comp, "capx")
        - previous_capx,
        previous_capx,
    )

    comp["ppeinv_gr1a"] = ratio(
        (
            column(comp, "ppegt")
            + column(comp, "invt")
        )
        - (
            previous_ppe
            + previous_inventory
        ),
        previous_assets,
    )

    growth_columns = [
        "asset_growth",
        "sales_growth",
        "capx_growth",
        "ppeinv_gr1a",
    ]

    comp.loc[
        ~annual_gap,
        growth_columns,
    ] = np.nan

    comp["comp_available_month"] = (
        comp["datadate"]
        + pd.DateOffset(months=6)
        + pd.offsets.MonthEnd(0)
    )

    keep_columns = [
        "ticker",
        "gvkey",
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
        .rename(
            columns={
                "gvkey": "comp_gvkey",
                "datadate": "comp_datadate",
            }
        )
        .sort_values(
            [
                "ticker",
                "comp_available_month",
                "comp_datadate",
                "comp_gvkey",
            ]
        )
        .drop_duplicates(
            ["ticker", "comp_available_month"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return comp


# ---------------------------------------------------------------------
# Merge most recent available Compustat report
# ---------------------------------------------------------------------
def merge_compustat(monthly, comp):
    monthly = monthly.copy()

    monthly["ticker"] = (
        monthly["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    monthly = (
        monthly.sort_values(
            ["month", "ticker"]
        )
        .reset_index(drop=True)
    )

    comp = (
        comp.sort_values(
            ["comp_available_month", "ticker"]
        )
        .reset_index(drop=True)
    )

    panel = pd.merge_asof(
        monthly,
        comp,
        by="ticker",
        left_on="month",
        right_on="comp_available_month",
        direction="backward",
    )

    month_number = (
        panel["month"].dt.year * 12
        + panel["month"].dt.month
    )

    available_month_number = (
        panel["comp_available_month"].dt.year * 12
        + panel["comp_available_month"].dt.month
    )

    accounting_age = (
        month_number
        - available_month_number
    )

    stale_report = (
        panel["comp_datadate"].notna()
        & accounting_age.gt(
            MAX_COMPUSTAT_AGE_MONTHS
        )
    )

    compustat_columns = [
        name
        for name in comp.columns
        if name != "ticker"
    ]

    panel.loc[
        stale_report,
        compustat_columns,
    ] = np.nan

    panel["has_compustat_annual"] = (
        panel["comp_datadate"]
        .notna()
        .astype(int)
    )

    panel = (
        panel.sort_values(
            ["ticker", "month"]
        )
        .reset_index(drop=True)
    )

    return panel, int(stale_report.sum())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    monthly = pd.read_parquet(
        MONTHLY_STOCK_FILE
    )

    monthly["month"] = pd.to_datetime(
        monthly["month"]
    )

    comp = clean_compustat()

    panel, stale_reports = merge_compustat(
        monthly,
        comp,
    )

    for output_file in [
        COMPUSTAT_CLEAN_FILE,
        PANEL_WITH_FUNDAMENTALS_FILE,
    ]:
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    comp.to_parquet(
        COMPUSTAT_CLEAN_FILE,
        index=False,
    )

    panel.to_parquet(
        PANEL_WITH_FUNDAMENTALS_FILE,
        index=False,
    )

    timing_violations = (
        panel["comp_available_month"]
        .gt(panel["month"])
        .sum()
    )

    print("\nFinal summary")
    print(f"Clean Compustat rows: {len(comp):,}")
    print(
        "Clean Compustat tickers: "
        f"{comp['ticker'].nunique():,}"
    )
    print(f"Panel rows: {len(panel):,}")
    print(f"Panel columns: {panel.shape[1]:,}")
    print(
        "Panel tickers: "
        f"{panel['ticker'].nunique():,}"
    )
    print(
        "Compustat matches: "
        f"{panel['has_compustat_annual'].sum():,}"
    )
    print(
        "Unmatched observations: "
        f"{panel['has_compustat_annual'].eq(0).sum():,}"
    )
    print(
        "Stale reports invalidated: "
        f"{stale_reports:,}"
    )
    print(
        "Timing violations: "
        f"{timing_violations:,}"
    )
    print(
        "Missing targets: "
        f"{panel[TARGET].isna().sum():,}"
    )
    print(f"Saved: {COMPUSTAT_CLEAN_FILE}")
    print(
        f"Saved: "
        f"{PANEL_WITH_FUNDAMENTALS_FILE}"
    )


if __name__ == "__main__":
    main()