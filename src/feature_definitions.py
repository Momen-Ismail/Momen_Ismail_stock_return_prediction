"""Locked predictor definitions for the stock-return ML project."""

LOCKED_CHARACTERISTIC_COLUMNS = [
    # Returns and momentum
    "ret_1m",
    "mom6m",
    "mom12m",
    "long_term_return_36m",

    # Risk and extreme returns
    "retvol_1m",
    "retvol12m",
    "minret_1m",
    "rmax5_21d",
    "beta_12m",
    "betasq_12m",
    "idiovol_12m",

    # Liquidity and trading activity
    "avg_dolvol_1m",
    "std_dolvol_1m",
    "avg_log_dolvol_1m",
    "amihud_1m",
    "zerotrade_1m",
    "dolvol_growth_1m",

    # Price trend and trading range
    "price_to_ma12",
    "dist_from_high_12m",
    "avg_range_1m",

    # Valuation and firm size
    "be_me",
    "ocf_me",
    "log_comp_market_equity",

    # Profitability
    "op_at",
    "ocf_at",

    # Leverage and liquidity
    "debt_at",
    "cash_at",
    "cashflow_to_debt",
    "current_ratio",
    "quick_ratio",

    # Investment, efficiency, and asset structure
    "capx_at",
    "capx_growth",
    "rd_at",
    "ppe_at",
    "tangibility",
    "asset_turnover",
    "sales_to_inventory",
    "sales_to_cash",
    "sales_to_receivables",

    # Accruals, payout, and financing
    "accruals_at",
    "dividend_dummy",
    "equity_issuance_at",
    "repurchase_at",

    # Growth
    "asset_growth",
    "sales_growth",
    "ppeinv_gr1a",

    # Missing-R&D indicator
    "xrd_missing",
]


LOCKED_MARKET_COLUMNS = [
    "market_ret_1m",
    "market_vol_1m",
    "vix_avg_1m",
    "vix_change_1m",
]


LOCKED_MACRO_COLUMNS = [
    "wg_dp",
    "wg_ep",
    "wg_bm",
    "wg_ntis",
    "wg_tbl",
    "wg_tms",
    "wg_dfy",
    "wg_svar",
]
