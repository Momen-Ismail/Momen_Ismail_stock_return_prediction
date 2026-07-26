"""Rank optimized models using validation data only."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402


OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"
ACTIVE_MODELS = {
    "historical_mean",
    "ols_3",
    "pls_optimized",
    "elastic_net_optimized",
    "decision_tree_optimized",
    "random_forest_optimized",
}


def main():
    linear_metrics = pd.read_csv(
        OUTPUT_DIR / "optimized_linear_model_metrics.csv"
    )

    tree_metrics = pd.read_csv(
        OUTPUT_DIR / "optimized_tree_model_metrics.csv"
    )

    if linear_metrics["sample"].eq("test").any() or tree_metrics["sample"].eq("test").any():
        raise ValueError("Optimized-model comparison must not contain test rows.")

    linear_metrics["model_group"] = "linear"
    tree_metrics["model_group"] = "tree"

    metrics = pd.concat(
        [
            linear_metrics,
            tree_metrics,
        ],
        ignore_index=True,
    )
    metrics = metrics[metrics["model"].isin(ACTIVE_MODELS)].copy()

    validation_ranking = rank_models(
        metrics,
        "validation",
    )

    metrics.to_csv(
        OUTPUT_DIR / "optimized_all_model_metrics.csv",
        index=False,
    )

    validation_ranking.to_csv(
        OUTPUT_DIR / "optimized_model_validation_ranking.csv",
        index=False,
    )

    print("\nOptimized-model validation ranking:")
    print(
        validation_ranking[
            [
                "rank",
                "model",
                "model_group",
                "monthly_mse",
                "monthly_rmse",
                "oos_r2",
                "prediction_target_correlation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
