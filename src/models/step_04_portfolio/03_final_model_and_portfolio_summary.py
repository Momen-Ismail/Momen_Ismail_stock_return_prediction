"""Combine statistical prediction and economic portfolio results."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402
from src.models.utils.portfolio import rank_portfolios  # noqa: E402

OPTIMIZATION_DIR = MODEL_OUTPUT_DIR / "optimization"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "portfolio"


def combined_ranking(combined, sample):
    """Add separate predictive and portfolio ranks for one sample."""
    ranking = rank_models(combined, sample).rename(
        columns={"rank_by_oos_r2": "rank_by_prediction_oos_r2"}
    )
    portfolio_ranks = rank_portfolios(combined, sample)[
        ["stage", "model", "rank_by_long_short_sharpe"]
    ]
    return ranking.merge(portfolio_ranks, on=["stage", "model"], how="left")


def main():
    prediction_file = OPTIMIZATION_DIR / "fixed_vs_optimized_all_metrics.csv"
    portfolio_file = OUTPUT_DIR / "final_portfolio_summary.csv"
    missing = [str(path) for path in (prediction_file, portfolio_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run the preceding comparison scripts: {missing}")

    prediction = pd.read_csv(prediction_file)
    prediction = prediction[prediction["sample"].isin(["validation", "test"])]
    prediction = prediction.rename(
        columns={"model_group": "prediction_model_group"}
    )
    portfolio = pd.read_csv(portfolio_file).drop(
        columns="prediction_std", errors="ignore"
    )
    portfolio = portfolio.rename(
        columns={"model_group": "portfolio_model_group"}
    )
    keys = ["stage", "model", "sample"]
    combined = prediction.merge(portfolio, on=keys, how="left")
    combined["valid_portfolio"] = combined["valid_portfolio"].eq(True)

    validation = combined_ranking(combined, "validation")
    test = combined_ranking(combined, "test")
    combined.to_csv(
        OUTPUT_DIR / "final_prediction_portfolio_summary.csv", index=False
    )
    validation.to_csv(
        OUTPUT_DIR / "final_prediction_portfolio_validation_ranking.csv", index=False
    )
    test.to_csv(
        OUTPUT_DIR / "final_prediction_portfolio_test_ranking.csv", index=False
    )

    columns = [
        "rank_by_prediction_oos_r2", "rank_by_long_short_sharpe",
        "stage", "model", "oos_r2_vs_train_mean", "long_short_sharpe",
    ]
    print("Validation results (use for selection):\n", validation[columns])
    print("Test results (final evaluation only):\n", test[columns])


if __name__ == "__main__":
    main()
