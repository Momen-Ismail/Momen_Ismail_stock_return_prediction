"""Form monthly prediction-sorted decile portfolios."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.portfolio import (  # noqa: E402
    evaluate_portfolio, rank_portfolios,
)

FIXED_DIR = MODEL_OUTPUT_DIR / "fixed"
OPTIMIZATION_DIR = MODEL_OUTPUT_DIR / "optimization"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "portfolio"
SOURCES = [
    ("fixed", "linear", FIXED_DIR / "fixed_linear_model_predictions.parquet"),
    ("fixed", "tree", FIXED_DIR / "fixed_tree_model_predictions.parquet"),
    (
        "optimized", "linear",
        OPTIMIZATION_DIR / "optimized_linear_model_predictions.parquet",
    ),
    (
        "optimized", "tree",
        OPTIMIZATION_DIR / "optimized_tree_model_predictions.parquet",
    ),
]


def prediction_groups():
    """Yield model samples one prediction file at a time to limit memory use."""
    required = ["ticker", "month", "sample", "model", TARGET, "prediction"]
    for stage, group, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"Run the corresponding model script: {path}")
        data = pd.read_parquet(path)
        data = data[data["sample"].isin(["validation", "test"])]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        data = data[required].copy()
        data["month"] = pd.to_datetime(data["month"])
        data["stage"], data["model_group"] = stage, group
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

    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "portfolio_monthly_returns.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "portfolio_test_ranking.csv", index=False)
    print(ranking[[
        "rank_by_long_short_sharpe", "stage", "model",
        "long_short_mean_annual", "long_short_sharpe", "long_short_t_stat",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
