"""Create final-test portfolio ranking."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.portfolio import rank_portfolios  # noqa: E402

OUTPUT_DIR = MODEL_OUTPUT_DIR / "portfolio"
SUMMARY_FILE = OUTPUT_DIR / "portfolio_summary.csv"
FINAL_COLUMNS = [
    "rank_by_long_short_sharpe", "stage", "model_group", "model", "sample",
    "valid_portfolio", "months", "observations", "prediction_std",
    "long_mean_annual", "short_mean_annual", "long_short_mean_annual",
    "long_short_vol_annual", "long_short_sharpe", "long_short_t_stat",
    "long_short_hac_standard_error", "long_short_hac_t_stat",
    "long_short_hac_p_value", "long_short_positive_month_share",
]


def select_columns(data):
    return data[[column for column in FINAL_COLUMNS if column in data]].copy()


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Run 01_portfolio_sorts.py first: {SUMMARY_FILE}")

    summary = pd.read_csv(SUMMARY_FILE)
    if set(summary["sample"]) != {"test"}:
        raise ValueError("Portfolio summary must contain final-test results only.")

    ranking = rank_portfolios(summary, "test")
    clean = summary.sort_values(
        ["valid_portfolio", "long_short_sharpe"],
        ascending=[False, False],
    )

    select_columns(clean).to_csv(
        OUTPUT_DIR / "final_portfolio_summary.csv", index=False
    )
    select_columns(ranking).to_csv(
        OUTPUT_DIR / "final_portfolio_test_ranking.csv", index=False
    )
    print("Final-test portfolio ranking:\n", ranking[[
        "rank_by_long_short_sharpe", "stage", "model",
        "long_short_mean_annual", "long_short_sharpe", "long_short_t_stat",
    ]])


if __name__ == "__main__":
    main()
