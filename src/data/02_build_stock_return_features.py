"""Create monthly stock predictors and the next-month excess-return target."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# 1) Settings
DATA_DIR = Path("output/data")
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
for folder in (INTERMEDIATE_DIR, FINAL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DAILY_FILE = FINAL_DIR / "daily_prices_clean_1987_2026.csv"
MARKET_FILE = INTERMEDIATE_DIR / "gspc_vix_daily_1987_2025.csv"
FF_FILE = INTERMEDIATE_DIR / "fama_french_3_factors_monthly.csv"
OUTPUT_FILE = FINAL_DIR / "monthly_stock_panel_with_targets_1990_2025.csv"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
SAMPLE_START = pd.Timestamp("1990-01-31")


# 2) Convert daily prices to monthly observations
def compound(returns):
    return np.prod(1 + returns) - 1


def build_monthly_panel(daily):
    daily = daily.sort_values(["ticker", "date"]).copy()
    daily["month"] = daily.date.dt.to_period("M").dt.to_timestamp("M")

    daily["daily_ret"] = daily.groupby("ticker").adj_close.pct_change()
    daily["dolvol"] = daily.adj_close * daily.volume
    daily["log_dolvol"] = np.log1p(daily.dolvol)

    daily["amihud_daily"] = daily.daily_ret.abs().div(
        daily.dolvol.where(daily.dolvol > 0)
    )
    daily["range_daily"] = (daily.high - daily.low).div(
        daily.close.where(daily.close > 0)
    )
    daily["zero_volume"] = daily.volume.eq(0).astype(int)

    monthly = daily.groupby(["ticker", "month"]).agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_trading_days=("date", "size"),

        last_adj_close=("adj_close", "last"),
        last_close=("close", "last"),

        avg_volume_1m=("volume", "mean"),
        std_volume_1m=("volume", "std"),
        avg_dolvol_1m=("dolvol", "mean"),
        std_dolvol_1m=("dolvol", "std"),
        avg_log_dolvol_1m=("log_dolvol", "mean"),

        retvol_1m=("daily_ret", "std"),
        maxret_1m=("daily_ret", "max"),
        minret_1m=("daily_ret", "min"),

        # JKP-style one-month daily variables
        rvol_21d=("daily_ret", "std"),
        rmax1_21d=("daily_ret", "max"),
        rmax5_21d=("daily_ret", lambda x: x.nlargest(5).mean()),

        amihud_1m=("amihud_daily", "mean"),
        zerotrade_1m=("zero_volume", "mean"),
        zero_trades_21d=("zero_volume", "mean"),

        avg_range_1m=("range_daily", "mean"),
        max_range_1m=("range_daily", "max"),
    ).reset_index().sort_values(["ticker", "month"])

    monthly["ret_1m"] = monthly.groupby("ticker").last_adj_close.pct_change()

    return monthly


# 3) Add stock-level rolling predictors
def add_stock_features(monthly):
    monthly = monthly.sort_values(["ticker", "month"]).copy()
    grouped = monthly.groupby("ticker")

    # Current-month return. This is useful as short-term reversal.
    monthly["mom1m"] = monthly.ret_1m

    # Momentum variables: use lagged returns, so mom3m/mom6m/mom12m skip the current month.
    # Example: mom12m uses returns from t-12 to t-1, not the current month t.
    monthly["ret_lag1"] = grouped.ret_1m.shift(1)

    for window in (3, 6, 12, 36):
        monthly[f"mom{window}m"] = (
            monthly.groupby("ticker")["ret_lag1"]
            .rolling(window, min_periods=window)
            .apply(compound, raw=True)
            .reset_index(level=0, drop=True)
        )

    monthly["chmom"] = monthly.mom6m - monthly.mom12m

    # Rolling volatility from monthly daily-volatility measures.
    for window in (3, 6, 12):
        monthly[f"retvol{window}m"] = (
            grouped.retvol_1m
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # JKP-style 126 trading-day proxies using roughly 6 monthly observations.
    # We keep them simple because the Yahoo file has no daily shares outstanding.
    monthly["ami_126d"] = (
        grouped.amihud_1m
        .rolling(6, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    monthly["dolvol_126d"] = (
        grouped.avg_dolvol_1m
        .rolling(6, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    monthly["dolvol_var_126d"] = (
        grouped.avg_dolvol_1m
        .rolling(6, min_periods=4)
        .std()
        .reset_index(level=0, drop=True)
        / monthly["dolvol_126d"]
    )
    monthly["zero_trades_126d"] = (
        grouped.zerotrade_1m
        .rolling(6, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    monthly["volume_growth_1m"] = grouped.avg_volume_1m.pct_change()
    monthly["dolvol_growth_1m"] = grouped.avg_dolvol_1m.pct_change()

    for window in (3, 12):
        monthly[f"ma{window}_adj_close"] = (
            grouped.last_adj_close
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    monthly["price_to_ma3"] = monthly.last_adj_close / monthly.ma3_adj_close
    monthly["price_to_ma12"] = monthly.last_adj_close / monthly.ma12_adj_close
    monthly["ma3_to_ma12"] = monthly.ma3_adj_close / monthly.ma12_adj_close

    monthly["high_12m_adj_close"] = (
        grouped.last_adj_close
        .rolling(12, min_periods=12)
        .max()
        .reset_index(level=0, drop=True)
    )
    monthly["dist_from_high_12m"] = monthly.last_adj_close / monthly.high_12m_adj_close - 1

    monthly = monthly.drop(columns=["ret_lag1"])

    return monthly[monthly.month >= SAMPLE_START].copy()


# 4) Add market, VIX, beta, and idiosyncratic volatility
def download_market_data():
    frames = []
    for name, ticker in {"GSPC": "^GSPC", "VIX": "^VIX"}.items():
        data = yf.download(
            ticker, start="1987-01-01", end="2026-02-01",
            progress=False, auto_adjust=False, threads=False,
        )
        if data.empty:
            raise ValueError(f"Yahoo returned no data for {ticker}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename_axis("date").reset_index()
        data.columns = [str(col).lower().replace(" ", "_") for col in data]
        data["series"] = name
        frames.append(data[["series", "date", "open", "high", "low", "close", "adj_close", "volume"]])
    market = pd.concat(frames, ignore_index=True).sort_values(["series", "date"])
    market["date"] = pd.to_datetime(market.date)
    market.to_csv(MARKET_FILE, index=False)
    return market


def add_market_features(monthly, market):
    market = market.copy()
    market["month"] = market.date.dt.to_period("M").dt.to_timestamp("M")
    gspc = market[market.series.eq("GSPC")].sort_values("date").copy()
    vix = market[market.series.eq("VIX")].sort_values("date").copy()
    gspc["market_daily_ret"] = gspc.adj_close.pct_change()

    market_monthly = gspc.groupby("month").agg(
        gspc_last_adj_close=("adj_close", "last"), market_vol_1m=("market_daily_ret", "std"),
        market_maxret_1m=("market_daily_ret", "max"), market_minret_1m=("market_daily_ret", "min"),
    ).reset_index().sort_values("month")
    market_monthly["market_ret_1m"] = market_monthly.gspc_last_adj_close.pct_change()

    vix_monthly = vix.groupby("month").agg(
        vix_last_close=("close", "last"), vix_avg_1m=("close", "mean"),
        vix_max_1m=("close", "max"), vix_min_1m=("close", "min"),
    ).reset_index().sort_values("month")
    vix_monthly["vix_change_1m"] = vix_monthly.vix_last_close.pct_change()
    return monthly.merge(market_monthly, on="month", how="left").merge(vix_monthly, on="month", how="left")


def add_beta_idiovol(monthly, daily, market):
    gspc = market[market.series.eq("GSPC")].sort_values("date").copy()
    gspc["market_daily_ret"] = gspc.adj_close.pct_change()
    stocks = daily[["ticker", "date", "adj_close"]].sort_values(["ticker", "date"]).copy()
    stocks["stock_daily_ret"] = stocks.groupby("ticker").adj_close.pct_change()
    returns = stocks.merge(gspc[["date", "market_daily_ret"]], on="date", how="left").dropna()
    returns["month"] = returns.date.dt.to_period("M").dt.to_timestamp("M")

    def rolling_regression(group):
        group = group.sort_values("date").copy()
        x, y = group.market_daily_ret, group.stock_daily_ret
        stats = pd.DataFrame({"x": x, "y": y, "x2": x*x, "y2": y*y, "xy": x*y}).rolling(252, min_periods=126)
        count = stats.x.count()
        sum_x, sum_y = stats.x.sum(), stats.y.sum()
        sum_x2, sum_y2, sum_xy = stats.x2.sum(), stats.y2.sum(), stats.xy.sum()
        beta = (sum_xy - sum_x*sum_y/count) / (sum_x2 - sum_x**2/count)
        alpha = sum_y/count - beta*sum_x/count
        sse = sum_y2 - 2*alpha*sum_y - 2*beta*sum_xy + count*alpha**2 + 2*alpha*beta*sum_x + beta**2*sum_x2
        group["beta_12m"] = beta.replace([np.inf, -np.inf], np.nan)
        group["idiovol_12m"] = np.sqrt(sse / (count - 2)).replace([np.inf, -np.inf], np.nan)
        group["betasq_12m"] = group.beta_12m**2
        return group

    beta = returns.groupby("ticker", group_keys=False).apply(rolling_regression)
    beta = beta.groupby(["ticker", "month"]).agg(
        beta_obs_daily=("date", "size"), beta_12m=("beta_12m", "last"),
        betasq_12m=("betasq_12m", "last"), idiovol_12m=("idiovol_12m", "last"),
    ).reset_index()
    return monthly.merge(beta, on=["ticker", "month"], how="left")


# 5) Add Fama-French factors and the future target
def download_fama_french():
    response = requests.get(FF_URL, timeout=30)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        lines = archive.read(archive.namelist()[0]).decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.split(",")[0].strip().isdigit())
    end = next((i for i in range(start, len(lines)) if lines[i].split(",")[0].strip() in {"", "Annual Factors: January-December"}), len(lines))
    text = lines[start - 1] + "\n" + "\n".join(lines[start:end])
    factors = pd.read_csv(BytesIO(text.encode()))
    factors = factors.rename(columns={factors.columns[0]: "yyyymm", "Mkt-RF": "Mkt_RF"})
    factors = factors[factors.yyyymm.astype(str).str.fullmatch(r"\d{6}")].copy()
    factors["month"] = pd.to_datetime(factors.yyyymm.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    for column in ["Mkt_RF", "SMB", "HML", "RF"]:
        factors[column] = pd.to_numeric(factors[column], errors="coerce") / 100
    factors = factors[["month", "Mkt_RF", "SMB", "HML", "RF"]].sort_values("month")
    factors.to_csv(FF_FILE, index=False)
    return factors


def add_target(monthly, factors):
    data = monthly.merge(factors, on="month", how="left").sort_values(["ticker", "month"])
    data["target_return_next_1m"] = data.groupby("ticker").ret_1m.shift(-1)
    future_rf = factors[["month", "RF"]].copy()
    future_rf["RF_next_1m"] = future_rf.RF.shift(-1)
    data = data.merge(future_rf[["month", "RF_next_1m"]], on="month", how="left")
    data["target_excess_return_next_1m"] = data.target_return_next_1m - data.RF_next_1m
    return data


# 6) Run the pipeline
def main():
    daily = pd.read_csv(DAILY_FILE)
    daily["date"] = pd.to_datetime(daily.date)
    monthly = add_stock_features(build_monthly_panel(daily))
    market = download_market_data()
    monthly = add_market_features(monthly, market)
    monthly = add_beta_idiovol(monthly, daily, market)
    monthly = add_target(monthly, download_fama_french())
    monthly = monthly.sort_values(["ticker", "month"]).reset_index(drop=True)
    monthly.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}: {monthly.shape}")
    print(f"Date range: {monthly.month.min()} to {monthly.month.max()}")
    print(f"Missing targets: {monthly.target_excess_return_next_1m.isna().sum():,}")


if __name__ == "__main__":
    main()
