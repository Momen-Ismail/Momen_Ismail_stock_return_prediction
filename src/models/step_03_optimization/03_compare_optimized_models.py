"""Rank optimized models using validation data only."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402


OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"

ACTIVE_MODELS = {
    "historical_mean",
    "ols_3",
    "pls_optimized",
    "elastic_net_optimized",
    "random_forest_optimized",
    "gradient_boosting_optimized",
}


def main():
    linear_metrics = pd.read_csv(
        OUTPUT_DIR / "optimized_linear_model_metrics.csv"
    )

    tree_metrics = pd.read_csv(
        OUTPUT_DIR / "optimized_tree_model_metrics.csv"
    )

    if (
        linear_metrics["sample"].eq("test").any()
        or tree_metrics["sample"].eq("test").any()
    ):
        raise ValueError(
            "Optimized-model comparison must not contain test rows."
        )

    linear_metrics["model_group"] = "linear"
    tree_metrics["model_group"] = "tree_ensemble"

    metrics = pd.concat(
        [
            linear_metrics,
            tree_metrics,
        ],
        ignore_index=True,
    )

    metrics = metrics[
        metrics["model"].isin(ACTIVE_MODELS)
    ].copy()

    missing_models = (
        ACTIVE_MODELS
        - set(metrics["model"].unique())
    )

    if missing_models:
        raise ValueError(
            "Missing optimized-model results for: "
            f"{sorted(missing_models)}"
        )

    duplicate_rows = metrics.duplicated(
        ["model", "sample"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            "Duplicate optimized metric rows: "
            f"{duplicate_rows}"
        )

    validation_ranking = rank_models(
        metrics,
        sample="validation",
    )

    metrics.to_csv(
        OUTPUT_DIR / "optimized_all_model_metrics.csv",
        index=False,
    )

    validation_ranking.to_csv(
        OUTPUT_DIR
        / "optimized_model_validation_ranking.csv",
        index=False,
    )

    display_columns = [
        "rank",
        "model",
        "model_group",
        "monthly_mse",
        "monthly_rmse",
        "monthly_oos_r2",
        "pooled_rmse",
        "oos_r2",
        "prediction_target_correlation",
    ]

    print("\nOptimized-model validation ranking:")

    print(
        validation_ranking[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved optimized-model comparison:")
    print(
        OUTPUT_DIR / "optimized_all_model_metrics.csv"
    )
    print(
        OUTPUT_DIR
        / "optimized_model_validation_ranking.csv"
    )


if __name__ == "__main__":
    main()