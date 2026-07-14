"""Build monthly stock predictors and the next-month excess-return target."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    SAMPLE_START,
    SAMPLE_END,
    TARGET,
    DAILY_CLEAN_FILE,
    GSPC_DAILY_FILE,
    VIX_DAILY_FILE,
    FAMA_FRENCH_RF_FILE,
    MONTHLY_QUALITY_REPORT_FILE,
    MONTHLY_REMOVED_TICKERS_FILE,
    MONTHLY_STOCK_FILE,
)


# ---------------------------------------------------------------------
# 1) Build monthly stock variables
# ---------------------------------------------------------------------
def build_monthly_panel(daily):
    """Aggregate daily prices into monthly stock characteristics."""
    daily = daily.sort_values(["ticker", "date"]).copy()

    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")
    daily["stock_daily_ret"] = (
        daily.groupby("ticker")["adj_close"].pct_change()
    )

    daily["dolvol"] = daily["adj_close"] * daily["volume"]
    daily["log_dolvol"] = np.log1p(daily["dolvol"])

    daily["amihud_daily"] = (
        daily["stock_daily_ret"].abs()
        / daily["dolvol"].where(daily["dolvol"] > 0)
    )

    daily["range_daily"] = (
        (daily["high"] - daily["low"])
        / daily["close"].where(daily["close"] > 0)
    )

    daily["zero_volume"] = daily["volume"].eq(0)

    monthly = (
        daily.groupby(["ticker", "month"], as_index=False)
        .agg(
            last_adj_close=("adj_close", "last"),
            avg_dolvol_1m=("dolvol", "mean"),
            std_dolvol_1m=("dolvol", "std"),
            avg_log_dolvol_1m=("log_dolvol", "mean"),
            retvol_1m=("stock_daily_ret", "std"),
            minret_1m=("stock_daily_ret", "min"),
            rmax5_21d=(
                "stock_daily_ret",
                lambda values: values.nlargest(5).mean(),
            ),
            amihud_1m=("amihud_daily", "mean"),
            zerotrade_1m=("zero_volume", "mean"),
            avg_range_1m=("range_daily", "mean"),
        )
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    monthly["ret_1m"] = (
        monthly.groupby("ticker")["last_adj_close"].pct_change()
    )

    return monthly, daily


# ---------------------------------------------------------------------
# 2) Remove unreliable ticker histories
# ---------------------------------------------------------------------
def remove_bad_monthly_tickers(monthly):
    """Remove tickers with repeated implausible monthly returns."""
    monthly = monthly.copy()
    monthly["implausible_monthly_return"] = (
        monthly["ret_1m"].gt(3.0)
        | monthly["ret_1m"].lt(-0.95)
    )

    report = (
        monthly.groupby("ticker", as_index=False)
        .agg(
            rows=("month", "size"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            implausible_months=("implausible_monthly_return", "sum"),
            max_monthly_return=("ret_1m", "max"),
            min_monthly_return=("ret_1m", "min"),
        )
    )

    report["remove_ticker"] = report["implausible_months"].ge(2)
    report.to_csv(MONTHLY_QUALITY_REPORT_FILE, index=False)

    removed = report[report["remove_ticker"]].copy()
    removed["removal_reason"] = "repeated_implausible_monthly_returns"
    removed.to_csv(MONTHLY_REMOVED_TICKERS_FILE, index=False)

    bad_tickers = removed["ticker"].tolist()

    monthly = monthly[
        ~monthly["ticker"].isin(bad_tickers)
    ].drop(columns="implausible_monthly_return").copy()

    print(f"Removed monthly-quality tickers: {len(bad_tickers)}")

    if bad_tickers:
        print(f"Removed tickers: {', '.join(bad_tickers)}")

    return monthly, bad_tickers


# ---------------------------------------------------------------------
# 3) Add rolling stock predictors
# ---------------------------------------------------------------------
def compound(returns):
    """Compound simple returns."""
    return np.prod(1.0 + returns) - 1.0


def add_stock_features(monthly):
    """Create momentum, volatility, liquidity, and trend predictors."""
    monthly = monthly.sort_values(["ticker", "month"]).copy()
    grouped = monthly.groupby("ticker")

    lagged_return = grouped["ret_1m"].shift(1)

    for name, window in {
        "mom6m": 6,
        "mom12m": 12,
        "long_term_return_36m": 36,
    }.items():
        monthly[name] = (
            lagged_return.groupby(monthly["ticker"])
            .rolling(window, min_periods=window)
            .apply(compound, raw=True)
            .reset_index(level=0, drop=True)
        )

    monthly["retvol12m"] = (
        grouped["retvol_1m"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    monthly["dolvol_growth_1m"] = (
        grouped["avg_dolvol_1m"].pct_change()
    )

    ma12 = (
        grouped["last_adj_close"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    high12 = (
        grouped["last_adj_close"]
        .rolling(12, min_periods=12)
        .max()
        .reset_index(level=0, drop=True)
    )

    monthly["price_to_ma12"] = monthly["last_adj_close"] / ma12
    monthly["dist_from_high_12m"] = (
        monthly["last_adj_close"] / high12 - 1.0
    )

    return monthly[
        monthly["month"] >= SAMPLE_START
    ].copy()


# ---------------------------------------------------------------------
# 4) Add market and VIX variables
# ---------------------------------------------------------------------
def load_market_data():
    """Load permanent daily S&P 500 and VIX files."""
    gspc = pd.read_csv(GSPC_DAILY_FILE)
    gspc["series"] = "GSPC"

    vix = pd.read_csv(VIX_DAILY_FILE)
    vix["series"] = "VIX"

    market = pd.concat([gspc, vix], ignore_index=True)
    market["date"] = pd.to_datetime(market["date"])

    market = market.sort_values(
        ["series", "date"]
    ).reset_index(drop=True)

    gspc_rows = market["series"].eq("GSPC")

    market.loc[gspc_rows, "market_daily_ret"] = (
        market.loc[gspc_rows, "adj_close"]
        .pct_change()
        .to_numpy()
    )

    return market


def add_market_features(monthly, market):
    """Add monthly market return, market volatility, and VIX variables."""
    market = market.copy()
    market["month"] = market["date"].dt.to_period("M").dt.to_timestamp("M")

    gspc = market[market["series"].eq("GSPC")]
    vix = market[market["series"].eq("VIX")]

    market_monthly = (
        gspc.groupby("month", as_index=False)
        .agg(
            gspc_last_adj_close=("adj_close", "last"),
            market_vol_1m=("market_daily_ret", "std"),
        )
        .sort_values("month")
    )

    market_monthly["market_ret_1m"] = (
        market_monthly["gspc_last_adj_close"].pct_change()
    )

    vix_monthly = (
        vix.groupby("month", as_index=False)
        .agg(
            vix_last_close=("close", "last"),
            vix_avg_1m=("close", "mean"),
        )
        .sort_values("month")
    )

    vix_monthly["vix_change_1m"] = (
        vix_monthly["vix_last_close"].pct_change()
    )

    market_monthly = market_monthly[
        ["month", "market_ret_1m", "market_vol_1m"]
    ]

    vix_monthly = vix_monthly[
        ["month", "vix_avg_1m", "vix_change_1m"]
    ]

    return (
        monthly
        .merge(market_monthly, on="month", how="left")
        .merge(vix_monthly, on="month", how="left")
    )

# ---------------------------------------------------------------------
# 5) Add beta and idiosyncratic volatility
# ---------------------------------------------------------------------
def add_beta_idiovol(monthly, daily, market):
    """Create rolling market beta and idiosyncratic volatility."""
    market_returns = market.loc[
        market["series"].eq("GSPC"),
        ["date", "market_daily_ret"],
    ]

    returns = (
        daily[["ticker", "date", "stock_daily_ret"]]
        .merge(market_returns, on="date", how="inner")
        .dropna()
        .sort_values(["ticker", "date"])
    )

    parts = []

    for _, group in returns.groupby("ticker", sort=False):
        stock = group["stock_daily_ret"]
        market_ret = group["market_daily_ret"]

        market_var = market_ret.rolling(252, min_periods=126).var()
        stock_var = stock.rolling(252, min_periods=126).var()
        covariance = stock.rolling(252, min_periods=126).cov(market_ret)

        beta = covariance / market_var
        residual_var = stock_var - beta.pow(2) * market_var

        group = group.copy()
        group["beta_12m"] = beta.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        group["betasq_12m"] = group["beta_12m"].pow(2)
        group["idiovol_12m"] = np.sqrt(
            residual_var.clip(lower=0)
        )

        parts.append(group)

    risk = pd.concat(parts, ignore_index=True)
    risk["month"] = risk["date"].dt.to_period("M").dt.to_timestamp("M")

    risk = (
        risk.groupby(["ticker", "month"], as_index=False)
        .agg(
            beta_12m=("beta_12m", "last"),
            betasq_12m=("betasq_12m", "last"),
            idiovol_12m=("idiovol_12m", "last"),
        )
    )

    return monthly.merge(
        risk,
        on=["ticker", "month"],
        how="left",
    )


# ---------------------------------------------------------------------
# 6) Create the next-month excess-return target
# ---------------------------------------------------------------------
def load_fama_french_rf():
    """Load the permanent monthly risk-free rate."""
    rf = pd.read_csv(
        FAMA_FRENCH_RF_FILE,
        usecols=["month", "RF"],
    )

    rf["month"] = pd.to_datetime(rf["month"])
    rf["RF"] = pd.to_numeric(rf["RF"], errors="coerce")

    return (
        rf.dropna()
        .drop_duplicates("month")
        .sort_values("month")
        .reset_index(drop=True)
    )


def add_target(monthly, rf):
    """Create next-month stock and excess returns."""
    data = monthly.sort_values(["ticker", "month"]).copy()

    data["target_return_next_1m"] = (
        data.groupby("ticker")["ret_1m"].shift(-1)
    )

    future_rf = rf.copy()
    future_rf["RF_next_1m"] = future_rf["RF"].shift(-1)

    data = data.merge(
        future_rf[["month", "RF_next_1m"]],
        on="month",
        how="left",
    )

    data[TARGET] = (
        data["target_return_next_1m"]
        - data["RF_next_1m"]
    )

    return data


# ---------------------------------------------------------------------
# 7) Run pipeline
# ---------------------------------------------------------------------
def main():
    """Build and save the monthly stock panel."""
    daily = pd.read_csv(DAILY_CLEAN_FILE)
    daily["date"] = pd.to_datetime(daily["date"])

    monthly, daily_features = build_monthly_panel(daily)

    monthly, bad_tickers = remove_bad_monthly_tickers(monthly)

    daily_features = daily_features[
        ~daily_features["ticker"].isin(bad_tickers)
    ].copy()

    monthly = add_stock_features(monthly)

    market = load_market_data()
    monthly = add_market_features(monthly, market)
    monthly = add_beta_idiovol(
        monthly,
        daily_features,
        market,
    )

    rf = load_fama_french_rf()
    monthly = add_target(monthly, rf)
    monthly = monthly[
        (monthly["month"] <= SAMPLE_END)
        & monthly[TARGET].notna()
    ].copy()

    monthly = (
        monthly
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    monthly.to_csv(MONTHLY_STOCK_FILE, index=False)

    print(f"\nSaved: {MONTHLY_STOCK_FILE}")
    print(f"Shape: {monthly.shape}")
    print(f"Tickers: {monthly['ticker'].nunique()}")
    print(
        f"Date range: {monthly['month'].min()} "
        f"to {monthly['month'].max()}"
    )
    print(f"Missing targets: {monthly[TARGET].isna().sum():,}")


if __name__ == "__main__":
    main()
