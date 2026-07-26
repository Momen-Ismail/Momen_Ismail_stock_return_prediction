"""Combine final-test prediction and portfolio results."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.portfolio import rank_portfolios  # noqa: E402

FINAL_TEST_DIR = MODEL_OUTPUT_DIR / "test"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "portfolio"


def main():
    prediction_file = FINAL_TEST_DIR / "final_test_metrics.csv"
    portfolio_file = OUTPUT_DIR / "final_portfolio_summary.csv"
    missing = [str(path) for path in (prediction_file, portfolio_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run final test and portfolio scripts first: {missing}")

    prediction = pd.read_csv(prediction_file)
    prediction = prediction[prediction["sample"].eq("test")]
    portfolio = pd.read_csv(portfolio_file).drop(
        columns="prediction_std", errors="ignore"
    )
    combined = prediction.merge(portfolio, on=["model", "sample"], how="left")
    ranking = rank_portfolios(combined, "test")

    combined.to_csv(
        OUTPUT_DIR / "final_prediction_portfolio_summary.csv", index=False
    )
    ranking.to_csv(
        OUTPUT_DIR / "final_prediction_portfolio_test_ranking.csv", index=False
    )
    print("Final-test prediction and portfolio summary:\n", ranking[[
        "rank_by_long_short_sharpe", "model", "monthly_mse",
        "oos_r2", "long_short_sharpe", "long_short_hac_t_stat",
    ]])


if __name__ == "__main__":
    main()
