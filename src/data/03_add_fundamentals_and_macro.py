"""Merge lagged Compustat fundamentals and FRED macro data."""

from io import StringIO
from pathlib import Path
import ssl
import urllib.request

import certifi
import numpy as np
import pandas as pd
import requests


# 1) Settings
DATA_DIR = Path("output/data")
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
for folder in (INTERMEDIATE_DIR, FINAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MONTHLY_FILE = FINAL_DIR / "monthly_stock_panel_with_targets_1990_2025.csv"
COMPUSTAT_RAW_FILE = Path("data_inputs/raw/compustat_annual_1980_2025.csv")
COMPUSTAT_CLEAN_FILE = INTERMEDIATE_DIR / "compustat_annual_cleaned_1980_2025.csv"
MACRO_FILE = INTERMEDIATE_DIR / "fred_macro_monthly_1980_2026.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "monthly_panel_with_compustat_macro_1990_2025.csv"
TARGET = "target_excess_return_next_1m"


# 2) Small cleaning helpers
def column(data, name):
    return data[name] if name in data else pd.Series(np.nan, index=data.index)


def ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


# 3) Clean Compustat and create accounting predictors
def clean_compustat():
    comp = pd.read_csv(COMPUSTAT_RAW_FILE, low_memory=False)
    comp.columns = comp.columns.str.lower().str.strip()

    missing = [name for name in ["tic", "gvkey", "datadate"] if name not in comp]
    if missing:
        raise ValueError(f"Missing Compustat columns: {missing}")

    comp["ticker"] = comp.tic.astype(str).str.upper().str.strip()
    comp["gvkey"] = comp.gvkey.astype(str).str.strip()
    comp["datadate"] = pd.to_datetime(comp.datadate, errors="coerce")
    comp = comp[comp.datadate.notna() & ~comp.ticker.isin(["", "NAN"])].copy()

    for name, value in {
        "consol": "C",
        "indfmt": "INDL",
        "datafmt": "STD",
        "curcd": "USD",
    }.items():
        if name in comp:
            comp = comp[comp[name].eq(value)].copy()

    numeric = [
        "at", "act", "lct", "lt", "ceq", "seq", "teq",
        "txditc", "txdb", "itcb",
        "che", "dlc", "dltt", "dt",
        "pstk", "pstkrv", "pstkl",
        "invt", "rect", "ap", "ppegt", "ppent", "intan", "gdwl",
        "csho", "sale", "revt", "gp", "cogs", "xsga", "xrd",
        "oiadp", "oibdp", "ib", "ni", "oancf",
        "capx", "dvt", "dltis", "dltr", "sstk", "prstkc",
        "mkvalt", "prcc_c", "prcc_f", "cshtr_f",
        "fyear", "fyr", "sic", "naics", "gsector", "ggroup", "gind", "gsubind",
    ]

    for name in set(numeric) & set(comp.columns):
        comp[name] = pd.to_numeric(comp[name], errors="coerce")

    comp = (
        comp.sort_values(["ticker", "datadate", "gvkey"])
        .drop_duplicates(["ticker", "datadate"], keep="last")
    )

    # ------------------------------------------------------------
    # Book equity, following the usual Compustat logic:
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
    comp.loc[comp.book_equity <= 0, "book_equity"] = np.nan

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

    # Gross profit fallback.
    comp["gross_profit_clean"] = column(comp, "gp").fillna(
        column(comp, "sale") - column(comp, "cogs")
    )

    # Operating profitability proxy. We keep it simple and transparent.
    comp["operating_profit_clean"] = column(comp, "oiadp").fillna(
        column(comp, "sale")
        - column(comp, "cogs")
        - column(comp, "xsga").fillna(0)
        + comp["xrd_filled"]
    )

    # Tangibility proxy from the JKP-style formula.
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
        # Old names kept
        "bm_comp": (comp.book_equity, comp.comp_market_equity),
        "profitability_oiadp_at": (column(comp, "oiadp"), column(comp, "at")),
        "profitability_oibdp_at": (column(comp, "oibdp"), column(comp, "at")),
        "roa_ni_at": (column(comp, "ni"), column(comp, "at")),
        "roa_ib_at": (column(comp, "ib"), column(comp, "at")),
        "roe_ni_be": (column(comp, "ni"), comp.book_equity),
        "leverage_debt_at": (comp.debt_total, column(comp, "at")),
        "leverage_lt_at": (column(comp, "lt"), column(comp, "at")),

        # Clearer JKP-style names
        "be_me": (comp.book_equity, comp.comp_market_equity),
        "at_me": (column(comp, "at"), comp.comp_market_equity),
        "sale_me": (column(comp, "sale"), comp.comp_market_equity),
        "ni_me": (column(comp, "ni"), comp.comp_market_equity),
        "ocf_me": (column(comp, "oancf"), comp.comp_market_equity),
        "debt_me": (comp.debt_total, comp.comp_market_equity),

        "gp_at": (comp.gross_profit_clean, column(comp, "at")),
        "op_at": (comp.operating_profit_clean, column(comp, "at")),
        "ni_be": (column(comp, "ni"), comp.book_equity),
        "ocf_at": (column(comp, "oancf"), column(comp, "at")),
        "debt_at": (comp.debt_total, column(comp, "at")),

        # Balance sheet and investment
        "cash_at": (column(comp, "che"), column(comp, "at")),
        "working_capital_at": (column(comp, "act") - column(comp, "lct"), column(comp, "at")),
        "current_ratio": (column(comp, "act"), column(comp, "lct")),
        "capx_at": (column(comp, "capx"), column(comp, "at")),
        "rd_at": (comp.xrd_filled, column(comp, "at")),
        "sga_at": (column(comp, "xsga"), column(comp, "at")),
        "ppe_at": (column(comp, "ppent"), column(comp, "at")),
        "gross_ppe_at": (column(comp, "ppegt"), column(comp, "at")),
        "intangibles_at": (column(comp, "intan"), column(comp, "at")),
        "goodwill_at": (column(comp, "gdwl"), column(comp, "at")),
        "inventory_at": (column(comp, "invt"), column(comp, "at")),
        "receivables_at": (column(comp, "rect"), column(comp, "at")),
        "payables_at": (column(comp, "ap"), column(comp, "at")),

        # Margins and efficiency
        "gross_margin": (comp.gross_profit_clean, column(comp, "sale")),
        "operating_margin": (column(comp, "oiadp"), column(comp, "sale")),
        "net_margin": (column(comp, "ni"), column(comp, "sale")),
        "asset_turnover": (column(comp, "sale"), column(comp, "at")),

        # Accruals and payout
        "accruals_at": (column(comp, "ib") - column(comp, "oancf"), column(comp, "at")),
        "oaccruals_at": (column(comp, "ib") - column(comp, "oancf"), column(comp, "at")),
        "dividends_at": (column(comp, "dvt"), column(comp, "at")),
        "dividend_yield_comp": (column(comp, "dvt"), comp.comp_market_equity),

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
        comp.comp_market_equity.where(comp.comp_market_equity > 0)
    )
    comp["dividend_dummy"] = column(comp, "dvt").fillna(0).gt(0).astype(int)
    comp["sic2"] = np.floor(column(comp, "sic") / 100)

    # ------------------------------------------------------------
    # Growth variables
    # ------------------------------------------------------------
    comp = comp.sort_values(["gvkey", "datadate"]).copy()

    for name in [
        "at", "sale", "revt", "book_equity", "comp_market_equity",
        "invt", "ppegt",
    ]:
        comp[f"{name}_lag1"] = comp.groupby("gvkey")[name].shift(1)

    comp["asset_growth"] = ratio(column(comp, "at") - comp["at_lag1"], comp["at_lag1"])
    comp["sales_growth"] = ratio(column(comp, "sale") - comp["sale_lag1"], comp["sale_lag1"])
    comp["revenue_growth"] = ratio(column(comp, "revt") - comp["revt_lag1"], comp["revt_lag1"])
    comp["book_equity_growth"] = ratio(
        comp.book_equity - comp["book_equity_lag1"],
        comp["book_equity_lag1"],
    )
    comp["market_equity_growth_comp"] = ratio(
        comp.comp_market_equity - comp["comp_market_equity_lag1"],
        comp["comp_market_equity_lag1"],
    )

    # JKP-style growth scaled by lagged assets.
    comp["be_gr1a"] = ratio(comp.book_equity - comp["book_equity_lag1"], comp["at_lag1"])
    comp["inv_gr1a"] = ratio(column(comp, "invt") - comp["invt_lag1"], comp["at_lag1"])
    comp["ppeinv_gr1a"] = ratio(
        (column(comp, "ppegt") + column(comp, "invt"))
        - (comp["ppegt_lag1"] + comp["invt_lag1"]),
        comp["at_lag1"],
    )

    comp["investment_asset_growth"] = comp.asset_growth
    comp["comp_available_month"] = comp.datadate + pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)

    keep = [
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

    comp = comp[[name for name in keep if name in comp]].rename(columns={
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
    })

    comp.to_csv(COMPUSTAT_CLEAN_FILE, index=False)
    return comp

def merge_compustat(monthly, comp):
    monthly = monthly.sort_values(["ticker", "month"]).copy()
    monthly["ticker"] = monthly.ticker.astype(str).str.upper().str.strip()
    comp = comp.sort_values(["ticker", "comp_available_month"])
    parts = []
    for ticker, stock in monthly.groupby("ticker"):
        accounting = comp[comp.ticker.eq(ticker)]
        if accounting.empty:
            stock = stock.copy()
            for name in comp.columns.drop("ticker"):
                stock[name] = np.nan
            parts.append(stock)
        else:
            parts.append(pd.merge_asof(
                stock.sort_values("month"), accounting.sort_values("comp_available_month"),
                by="ticker", left_on="month", right_on="comp_available_month", direction="backward",
            ))
    panel = pd.concat(parts, ignore_index=True).sort_values(["ticker", "month"])
    panel["has_compustat_annual"] = panel.comp_datadate.notna().astype(int)
    return panel


# 4) Download and lag FRED variables
def download_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        text = requests.get(url, timeout=30, verify=certifi.where()).text
    except Exception:
        with urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=30) as response:
            text = response.read().decode()
    data = pd.read_csv(StringIO(text)).rename(columns={"observation_date": "date", series_id: "value"})
    data["date"] = pd.to_datetime(data.date)
    data["value"] = pd.to_numeric(data.value, errors="coerce")
    data["month"] = data.date + pd.offsets.MonthEnd(0)
    return data[["month", "value"]].drop_duplicates("month", keep="last")


def add_macro(panel):
    series = {
        "UNRATE": "unemployment_rate", "CPIAUCSL": "cpi", "INDPRO": "industrial_production",
        "FEDFUNDS": "fed_funds_rate", "TB3MS": "tbill_3m_rate", "GS10": "treasury_10y_rate",
        "AAA": "aaa_corporate_yield", "BAA": "baa_corporate_yield",
    }
    macro = None
    for series_id, name in series.items():
        data = download_fred(series_id).rename(columns={"value": name})
        macro = data if macro is None else macro.merge(data, on="month", how="outer")
    macro = macro.sort_values("month").ffill()

    for name in ["fed_funds_rate", "tbill_3m_rate", "treasury_10y_rate", "aaa_corporate_yield", "baa_corporate_yield"]:
        macro[f"{name}_dec"] = macro[name] / 100
    macro["inflation_12m"] = macro.cpi.pct_change(12)
    macro["inflation_1m"] = macro.cpi.pct_change()
    macro["industrial_production_growth_12m"] = macro.industrial_production.pct_change(12)
    macro["industrial_production_growth_1m"] = macro.industrial_production.pct_change()
    macro["unemployment_rate_dec"] = macro.unemployment_rate / 100
    macro["unemployment_change_12m"] = macro.unemployment_rate_dec.diff(12)
    macro["unemployment_change_1m"] = macro.unemployment_rate_dec.diff()
    macro["term_spread"] = macro.treasury_10y_rate_dec - macro.tbill_3m_rate_dec
    macro["default_spread"] = macro.baa_corporate_yield_dec - macro.aaa_corporate_yield_dec
    macro["fedfunds_change_1m"] = macro.fed_funds_rate_dec.diff()
    macro["fedfunds_change_12m"] = macro.fed_funds_rate_dec.diff(12)

    variables = [
        "unemployment_rate_dec", "inflation_12m", "inflation_1m",
        "industrial_production_growth_12m", "industrial_production_growth_1m", "fed_funds_rate_dec",
        "tbill_3m_rate_dec", "treasury_10y_rate_dec", "aaa_corporate_yield_dec",
        "baa_corporate_yield_dec", "term_spread", "default_spread", "unemployment_change_12m",
        "unemployment_change_1m", "fedfunds_change_1m", "fedfunds_change_12m",
    ]
    lagged = []
    for name in variables:
        lagged_name = f"macro_{name}_lag1"
        macro[lagged_name] = macro[name].shift(1)
        lagged.append(lagged_name)
    macro.to_csv(MACRO_FILE, index=False)
    return panel.merge(macro[["month"] + lagged], on="month", how="left")


# 5) Run the pipeline (no split or model preprocessing here)
def main():
    monthly = pd.read_csv(MONTHLY_FILE, low_memory=False)
    monthly["month"] = pd.to_datetime(monthly.month)
    comp = clean_compustat()
    panel = add_macro(merge_compustat(monthly, comp))
    panel = panel.sort_values(["ticker", "month"]).reset_index(drop=True)
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}: {panel.shape}")
    print(f"Compustat matches: {panel.has_compustat_annual.sum():,}")
    print(f"Missing targets: {panel[TARGET].isna().sum():,}")


if __name__ == "__main__":
    main()
