"""Build monthly stock predictors and the next-month excess-return target."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    SAMPLE_START,
    SAMPLE_END,
    TARGET,
    DAILY_CLEAN_FILE,
    GSPC_DAILY_FILE,
    VIX_DAILY_FILE,
    FAMA_FRENCH_RF_FILE,
    MONTHLY_STOCK_FILE,
)


# ---------------------------------------------------------------------
# 1) Build monthly stock variables
# ---------------------------------------------------------------------
def build_monthly_panel(daily):
    daily = daily.sort_values(["ticker", "date"]).copy()

    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")

    daily["stock_daily_ret"] = (
        daily.groupby("ticker")["adj_close"]
        .pct_change(fill_method=None)
    )

    daily["dolvol"] = daily["adj_close"] * daily["volume"]
    daily["log_dolvol"] = np.log1p(daily["dolvol"])

    daily["amihud_daily"] = (
        daily["stock_daily_ret"].abs()
        / daily["dolvol"].where(daily["dolvol"] > 0)
    )

    daily["range_daily"] = (
        (daily["high"] - daily["low"])
        / daily["close"]
    )

    daily["zero_volume"] = daily["volume"].eq(0)

    monthly = (
        daily.groupby(["ticker", "month"], as_index=False)
        .agg(
            n_trading_days=("date", "size"),
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

    invalid_volume = monthly["avg_dolvol_1m"].eq(0)

    monthly.loc[
        invalid_volume,
        [
            "avg_dolvol_1m",
            "std_dolvol_1m",
            "avg_log_dolvol_1m",
            "amihud_1m",
            "zerotrade_1m",
        ],
    ] = np.nan

    grouped = monthly.groupby("ticker")

    previous_month = grouped["month"].shift(1)
    previous_price = grouped["last_adj_close"].shift(1)
    expected_month = previous_month + pd.offsets.MonthEnd(1)

    monthly["ret_1m"] = (
        monthly["last_adj_close"] / previous_price - 1.0
    )

    gap_returns = (
        previous_month.notna()
        & monthly["month"].ne(expected_month)
    )

    monthly.loc[gap_returns, "ret_1m"] = np.nan

    return monthly, daily, int(gap_returns.sum())

# ---------------------------------------------------------------------
# 2) Remove fragmented ticker histories
# ---------------------------------------------------------------------
def remove_fragmented_tickers(monthly, daily):
    data = monthly.sort_values(["ticker", "month"]).copy()

    data["month_number"] = (
        data["month"].dt.year * 12
        + data["month"].dt.month
    )

    data["previous_month_number"] = (
        data.groupby("ticker")["month_number"].shift(1)
    )

    data["missing_months"] = (
        data["month_number"]
        - data["previous_month_number"]
        - 1
    )

    maximum_gap = (
        data.groupby("ticker")["missing_months"]
        .max()
    )

    bad_tickers = maximum_gap[
        maximum_gap > 2
    ].index.tolist()

    monthly = monthly[
        ~monthly["ticker"].isin(bad_tickers)
    ].copy()

    daily = daily[
        ~daily["ticker"].isin(bad_tickers)
    ].copy()

    return monthly, daily, bad_tickers

# ---------------------------------------------------------------------
# 3) Compound simple returns
# ---------------------------------------------------------------------
def compound(returns):
    return np.prod(1.0 + returns) - 1.0


# ---------------------------------------------------------------------
# 4) Add rolling stock predictors
# ---------------------------------------------------------------------
def add_stock_features(monthly):
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

    previous_dolvol = grouped["avg_dolvol_1m"].shift(1)

    monthly["dolvol_growth_1m"] = (
        monthly["avg_dolvol_1m"]
        / previous_dolvol.where(previous_dolvol > 0)
        - 1.0
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

    return monthly


# ---------------------------------------------------------------------
# 5) Load daily market and VIX data
# ---------------------------------------------------------------------
def load_market_data():
    gspc = pd.read_csv(GSPC_DAILY_FILE)
    vix = pd.read_csv(VIX_DAILY_FILE)

    gspc["date"] = pd.to_datetime(gspc["date"])
    vix["date"] = pd.to_datetime(vix["date"])

    gspc = gspc.sort_values("date").reset_index(drop=True)
    vix = vix.sort_values("date").reset_index(drop=True)

    gspc["market_daily_ret"] = (
        gspc["adj_close"]
        .pct_change(fill_method=None)
    )

    return gspc, vix


# ---------------------------------------------------------------------
# 6) Add monthly market and VIX variables
# ---------------------------------------------------------------------
def add_market_features(monthly, gspc, vix):
    gspc = gspc.copy()
    vix = vix.copy()

    gspc["month"] = gspc["date"].dt.to_period("M").dt.to_timestamp("M")
    vix["month"] = vix["date"].dt.to_period("M").dt.to_timestamp("M")

    market_monthly = (
        gspc.groupby("month", as_index=False)
        .agg(
            gspc_last_adj_close=("adj_close", "last"),
            market_vol_1m=("market_daily_ret", "std"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    previous_market_month = market_monthly["month"].shift(1)
    previous_market_price = market_monthly["gspc_last_adj_close"].shift(1)

    market_monthly["market_ret_1m"] = (
        market_monthly["gspc_last_adj_close"]
        / previous_market_price
        - 1.0
    )

    market_monthly.loc[
        market_monthly["month"].ne(
            previous_market_month + pd.offsets.MonthEnd(1)
        ),
        "market_ret_1m",
    ] = np.nan

    vix_monthly = (
        vix.groupby("month", as_index=False)
        .agg(
            vix_last_close=("close", "last"),
            vix_avg_1m=("close", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    previous_vix_month = vix_monthly["month"].shift(1)
    previous_vix_close = vix_monthly["vix_last_close"].shift(1)

    vix_monthly["vix_change_1m"] = (
        vix_monthly["vix_last_close"]
        / previous_vix_close
        - 1.0
    )

    vix_monthly.loc[
        vix_monthly["month"].ne(
            previous_vix_month + pd.offsets.MonthEnd(1)
        ),
        "vix_change_1m",
    ] = np.nan

    market_monthly = market_monthly[
        ["month", "market_ret_1m", "market_vol_1m"]
    ]

    vix_monthly = vix_monthly[
        ["month", "vix_avg_1m", "vix_change_1m"]
    ]

    monthly = monthly.merge(
        market_monthly,
        on="month",
        how="left",
    )

    monthly = monthly.merge(
        vix_monthly,
        on="month",
        how="left",
    )

    return monthly


# ---------------------------------------------------------------------
# 7) Add beta and idiosyncratic volatility
# ---------------------------------------------------------------------
def add_beta_idiovol(monthly, daily, gspc):
    market_returns = gspc[
        ["date", "market_daily_ret"]
    ]

    returns = (
        daily[["ticker", "date", "stock_daily_ret"]]
        .merge(market_returns, on="date", how="inner")
        .dropna()
        .sort_values(["ticker", "date"])
    )

    parts = []

    for _, group in returns.groupby("ticker", sort=False):
        stock_return = group["stock_daily_ret"]
        market_return = group["market_daily_ret"]

        market_variance = (
            market_return
            .rolling(252, min_periods=126)
            .var()
        )

        stock_variance = (
            stock_return
            .rolling(252, min_periods=126)
            .var()
        )

        covariance = (
            stock_return
            .rolling(252, min_periods=126)
            .cov(market_return)
        )

        beta = covariance / market_variance

        residual_variance = (
            stock_variance
            - beta.pow(2) * market_variance
        )

        group = group.copy()

        group["beta_12m"] = beta.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        group["betasq_12m"] = group["beta_12m"].pow(2)

        group["idiovol_12m"] = np.sqrt(
            residual_variance.clip(lower=0)
        )

        parts.append(group)

    risk = pd.concat(parts, ignore_index=True)

    risk["month"] = (
        risk["date"]
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )

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
# 8) Load the monthly Fama-French risk-free rate
# ---------------------------------------------------------------------
def load_fama_french_rf():
    rf = pd.read_csv(
        FAMA_FRENCH_RF_FILE,
        usecols=["month", "RF"],
    )

    rf["month"] = (
        pd.to_datetime(rf["month"])
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )

    rf["RF"] = pd.to_numeric(
        rf["RF"],
        errors="coerce",
    )

    return (
        rf.dropna()
        .drop_duplicates("month")
        .sort_values("month")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# 9) Create the next-month excess-return target
# ---------------------------------------------------------------------
def add_target(monthly, rf):
    data = monthly.sort_values(["ticker", "month"]).copy()
    grouped = data.groupby("ticker")

    next_month = grouped["month"].shift(-1)
    next_return = grouped["ret_1m"].shift(-1)
    expected_next_month = data["month"] + pd.offsets.MonthEnd(1)

    valid_target = next_month.eq(expected_next_month)

    data["target_return_next_1m"] = (
        next_return.where(valid_target)
    )

    data["target_month"] = expected_next_month

    future_rf = rf.rename(
        columns={
            "month": "target_month",
            "RF": "RF_next_1m",
        }
    )

    data = data.merge(
        future_rf[["target_month", "RF_next_1m"]],
        on="target_month",
        how="left",
    )

    data[TARGET] = (
        data["target_return_next_1m"]
        - data["RF_next_1m"]
    )

    gap_targets = int(
        (next_month.notna() & ~valid_target).sum()
    )

    data = data.drop(columns="target_month")

    return data, gap_targets


# ---------------------------------------------------------------------
# 10) Run the monthly-data pipeline
# ---------------------------------------------------------------------
def main():
    daily = pd.read_parquet(DAILY_CLEAN_FILE)
    daily["date"] = pd.to_datetime(daily["date"])

    monthly, daily_features, gap_returns = build_monthly_panel(daily)

    monthly, daily_features, fragmented_tickers = (
        remove_fragmented_tickers(
            monthly,
            daily_features,
        )
    )

    monthly = add_stock_features(monthly)

    gspc, vix = load_market_data()

    monthly = add_market_features(
        monthly,
        gspc,
        vix,
    )

    monthly = add_beta_idiovol(
        monthly,
        daily_features,
        gspc,
    )

    rf = load_fama_french_rf()
    monthly, gap_targets = add_target(monthly, rf)

    sample_start = pd.Timestamp(SAMPLE_START)
    sample_end = pd.Timestamp(SAMPLE_END)

    monthly = monthly.loc[
        monthly["month"].between(sample_start, sample_end)
        & monthly[TARGET].notna()
    ].copy()

    monthly = (
        monthly.sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    MONTHLY_STOCK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_parquet(
        MONTHLY_STOCK_FILE,
        index=False,
    )

    print("\nFinal summary")
    print(f"Calendar gaps detected before ticker removal: {gap_returns}")
    print(f"Fragmented ticker histories removed: {len(fragmented_tickers)}")

    if fragmented_tickers:
        print(f"Removed fragmented tickers: {', '.join(fragmented_tickers)}")

    print(f"Calendar-gap targets blocked: {gap_targets}")
    print(f"Final rows: {len(monthly):,}")
    print(f"Final columns: {monthly.shape[1]}")
    print(f"Final tickers: {monthly['ticker'].nunique()}")
    print(
        f"Date range: {monthly['month'].min()} "
        f"to {monthly['month'].max()}"
    )
    print(f"Missing targets: {monthly[TARGET].isna().sum():,}")
    print(f"Saved: {MONTHLY_STOCK_FILE}")


if __name__ == "__main__":
    main()