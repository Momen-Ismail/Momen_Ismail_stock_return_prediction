"""Form monthly portfolios from final-test predictions only."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.portfolio import (  # noqa: E402
    block_bootstrap_mean, evaluate_portfolio, rank_portfolios,
)

FINAL_TEST_DIR = MODEL_OUTPUT_DIR / "test"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "portfolio"
PREDICTION_FILE = FINAL_TEST_DIR / "final_test_predictions.parquet"


def prediction_groups():
    """Yield final-test model predictions."""
    required = [
        "ticker", "month", "sample", "model", "realized_target", "prediction"
    ]
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Run final test evaluation before portfolio analysis: {PREDICTION_FILE}"
        )
    data = pd.read_parquet(PREDICTION_FILE)
    data = data[data["sample"].eq("test")]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {PREDICTION_FILE}: {sorted(missing)}")
    if data.duplicated(["ticker", "month", "model"]).any():
        raise ValueError("Final-test predictions contain duplicate ticker-month rows.")
    data = data[required].rename(columns={"realized_target": TARGET})
    data["month"] = pd.to_datetime(data["month"])
    data["stage"], data["model_group"] = "test", "prespecified_models"
    yield from (
        model_sample
        for _, model_sample in data.groupby(
            ["stage", "model_group", "model", "sample"], sort=False
        )
    )


def main():
    monthly_frames, summaries = [], []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group in prediction_groups():
        monthly, summary = evaluate_portfolio(group, TARGET)
        summaries.append(summary)
        if monthly is not None:
            monthly_frames.append(monthly)

    summary = pd.DataFrame(summaries)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    ranking = rank_portfolios(summary, "test")

    bootstrap_rows = []
    if not ranking.empty:
        selected = ranking.iloc[0]
        selected_returns = monthly[
            monthly["stage"].eq(selected["stage"])
            & monthly["model"].eq(selected["model"])
        ]
        for sample, returns in selected_returns.groupby("sample"):
            bootstrap_rows.append({
                "stage": selected["stage"],
                "model": selected["model"],
                "sample": sample,
                **block_bootstrap_mean(returns["long_short"]),
            })

    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "portfolio_monthly_returns.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "portfolio_test_ranking.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT_DIR / "selected_portfolio_block_bootstrap.csv", index=False
    )
    print(ranking[[
        "rank_by_long_short_sharpe", "stage", "model",
        "long_short_mean_annual", "long_short_sharpe", "long_short_t_stat",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
