"""Cross-sectional portfolio-sort evaluation helpers."""

import numpy as np
import pandas as pd


def assign_portfolios(data, groups=10):
    """Assign stocks to monthly portfolios using predicted-return ranks."""
    ranked = data.copy()
    percentile = ranked.groupby("month")["prediction"].rank(
        method="first", pct=True
    )
    ranked["portfolio"] = np.ceil(percentile * groups).clip(1, groups).astype(int)
    return ranked


def portfolio_statistics(monthly):
    """Annualize monthly moments and compute conventional t-statistics."""
    statistics = {}
    for name in ("long", "short", "long_short"):
        returns = monthly[name].dropna()
        mean, volatility, count = returns.mean(), returns.std(), returns.count()
        annual_return = 12.0 * mean
        annual_volatility = np.sqrt(12.0) * volatility
        statistics.update({
            f"{name}_mean_monthly": mean,
            f"{name}_mean_annual": annual_return,
            f"{name}_vol_annual": annual_volatility,
            f"{name}_sharpe": (
                annual_return / annual_volatility if annual_volatility > 0 else np.nan
            ),
            f"{name}_t_stat": (
                mean / (volatility / np.sqrt(count))
                if volatility > 0 and count > 1 else np.nan
            ),
            f"{name}_positive_month_share": (returns > 0).mean(),
        })
    return statistics


def evaluate_portfolio(data, target, groups=10):
    """Build equal-weighted extreme portfolios for one model and sample."""
    data = data.dropna(subset=[target, "prediction"]).copy()
    metadata = {
        name: data[name].iloc[0]
        for name in ("stage", "model_group", "model", "sample")
    }
    base = {
        **metadata,
        "months": data["month"].nunique(),
        "observations": len(data),
        "prediction_std": data["prediction"].std(),
    }
    if base["prediction_std"] < 1e-12:
        return None, {**base, "valid_portfolio": False, "reason": "constant_predictions"}

    ranked = assign_portfolios(data, groups)
    returns = ranked.pivot_table(
        index="month", columns="portfolio", values=target, aggfunc="mean"
    ).sort_index()
    if 1 not in returns or groups not in returns:
        return None, {
            **base,
            "valid_portfolio": False,
            "reason": "missing_extreme_portfolio",
        }

    monthly = pd.DataFrame({
        "long": returns[groups],
        "short": -returns[1],
        "long_short": returns[groups] - returns[1],
        "equal_weight": returns.mean(axis=1),
    })
    for name, value in metadata.items():
        monthly[name] = value
    return monthly.reset_index(), {
        **base,
        "valid_portfolio": True,
        "reason": "",
        **portfolio_statistics(monthly),
    }


def rank_portfolios(summary, sample):
    """Rank valid portfolios by long-short Sharpe and annual return."""
    ranking = (
        summary[summary["sample"].eq(sample) & summary["valid_portfolio"]]
        .sort_values(
            ["long_short_sharpe", "long_short_mean_annual"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank_by_long_short_sharpe", ranking.index + 1)
    return ranking
