"""Compare fixed and optimized models using validation data only."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402


FIXED_DIR = MODEL_OUTPUT_DIR / "fixed"
OPTIMIZATION_DIR = MODEL_OUTPUT_DIR / "optimization"

FIXED_MODELS = {
    "historical_mean",
    "ols_3",
    "pls_fixed",
    "elastic_net_fixed",
    "random_forest_fixed",
    "gradient_boosting_fixed",
}

OPTIMIZED_MODELS = {
    "historical_mean",
    "ols_3",
    "pls_optimized",
    "elastic_net_optimized",
    "random_forest_optimized",
    "gradient_boosting_optimized",
}


def main():
    fixed_metrics = pd.read_csv(
        FIXED_DIR / "fixed_all_model_metrics.csv"
    )

    optimized_metrics = pd.read_csv(
        OPTIMIZATION_DIR / "optimized_all_model_metrics.csv"
    )

    if (
        fixed_metrics["sample"].eq("test").any()
        or optimized_metrics["sample"].eq("test").any()
    ):
        raise ValueError(
            "Validation comparison must not contain test rows."
        )

    fixed_metrics = fixed_metrics[
        fixed_metrics["model"].isin(FIXED_MODELS)
    ].copy()

    optimized_metrics = optimized_metrics[
        optimized_metrics["model"].isin(OPTIMIZED_MODELS)
    ].copy()

    missing_fixed = (
        FIXED_MODELS
        - set(fixed_metrics["model"].unique())
    )

    missing_optimized = (
        OPTIMIZED_MODELS
        - set(optimized_metrics["model"].unique())
    )

    if missing_fixed:
        raise ValueError(
            "Missing fixed-model results for: "
            f"{sorted(missing_fixed)}"
        )

    if missing_optimized:
        raise ValueError(
            "Missing optimized-model results for: "
            f"{sorted(missing_optimized)}"
        )

    fixed_validation = fixed_metrics[
        fixed_metrics["sample"].eq("validation")
    ].copy()

    optimized_validation = optimized_metrics[
        optimized_metrics["sample"].eq("validation")
    ].copy()

    fixed_validation["stage"] = "fixed"
    optimized_validation["stage"] = "optimized"

    fixed_validation["model_group"] = fixed_validation[
        "model"
    ].map({
        "historical_mean": "benchmark",
        "ols_3": "linear",
        "pls_fixed": "linear",
        "elastic_net_fixed": "linear",
        "random_forest_fixed": "tree_ensemble",
        "gradient_boosting_fixed": "tree_ensemble",
    })

    optimized_validation["model_group"] = optimized_validation[
        "model"
    ].map({
        "historical_mean": "benchmark",
        "ols_3": "linear",
        "pls_optimized": "linear",
        "elastic_net_optimized": "linear",
        "random_forest_optimized": "tree_ensemble",
        "gradient_boosting_optimized": "tree_ensemble",
    })

    # Historical mean and OLS-3 appear in both files.
    # Keep only the optimized-stage copies to avoid duplicates.
    fixed_models_only = fixed_validation[
        fixed_validation["model"].str.endswith("_fixed")
    ].copy()

    combined_validation = pd.concat(
        [
            optimized_validation,
            fixed_models_only,
        ],
        ignore_index=True,
    )

    duplicate_models = combined_validation[
        "model"
    ].duplicated().sum()

    if duplicate_models:
        raise ValueError(
            "Duplicate models in fixed-versus-optimized comparison: "
            f"{duplicate_models}"
        )

    validation_ranking = rank_models(
        combined_validation,
        sample="validation",
    )

    validation_ranking.to_csv(
        OPTIMIZATION_DIR
        / "fixed_vs_optimized_validation_ranking.csv",
        index=False,
    )

    combined_validation.to_csv(
        OPTIMIZATION_DIR
        / "fixed_vs_optimized_all_metrics.csv",
        index=False,
    )

    # Direct family-by-family comparison.
    fixed_family = fixed_validation[
        fixed_validation["model"].str.endswith("_fixed")
    ][
        [
            "model",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "oos_r2",
            "prediction_target_correlation",
        ]
    ].copy()

    fixed_family["model_family"] = (
        fixed_family["model"]
        .str.replace(
            "_fixed",
            "",
            regex=False,
        )
    )

    fixed_family = fixed_family.rename(
        columns={
            "model": "fixed_model",
            "monthly_mse": "fixed_monthly_mse",
            "monthly_rmse": "fixed_monthly_rmse",
            "monthly_oos_r2": "fixed_monthly_oos_r2",
            "oos_r2": "fixed_oos_r2",
            "prediction_target_correlation": (
                "fixed_prediction_target_correlation"
            ),
        }
    )

    optimized_family = optimized_validation[
        optimized_validation["model"].str.endswith(
            "_optimized"
        )
    ][
        [
            "model",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "oos_r2",
            "prediction_target_correlation",
        ]
    ].copy()

    optimized_family["model_family"] = (
        optimized_family["model"]
        .str.replace(
            "_optimized",
            "",
            regex=False,
        )
    )

    optimized_family = optimized_family.rename(
        columns={
            "model": "optimized_model",
            "monthly_mse": "optimized_monthly_mse",
            "monthly_rmse": "optimized_monthly_rmse",
            "monthly_oos_r2": "optimized_monthly_oos_r2",
            "oos_r2": "optimized_oos_r2",
            "prediction_target_correlation": (
                "optimized_prediction_target_correlation"
            ),
        }
    )

    family_comparison = fixed_family.merge(
        optimized_family,
        on="model_family",
        how="inner",
        validate="one_to_one",
    )

    expected_families = {
        "pls",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
    }

    missing_families = (
        expected_families
        - set(family_comparison["model_family"])
    )

    if missing_families:
        raise ValueError(
            "Missing fixed-versus-optimized families: "
            f"{sorted(missing_families)}"
        )

    family_comparison["monthly_mse_improvement"] = (
        family_comparison["fixed_monthly_mse"]
        - family_comparison["optimized_monthly_mse"]
    )

    family_comparison[
        "monthly_mse_improvement_percent"
    ] = (
        family_comparison["monthly_mse_improvement"]
        / family_comparison["fixed_monthly_mse"]
        * 100
    )

    family_comparison["monthly_oos_r2_change"] = (
        family_comparison["optimized_monthly_oos_r2"]
        - family_comparison["fixed_monthly_oos_r2"]
    )

    family_comparison["oos_r2_change"] = (
        family_comparison["optimized_oos_r2"]
        - family_comparison["fixed_oos_r2"]
    )

    family_comparison["optimization_improved"] = (
        family_comparison["monthly_mse_improvement"] > 0
    )

    family_comparison = family_comparison.sort_values(
        "optimized_monthly_mse"
    )[
        [
            "model_family",
            "fixed_model",
            "optimized_model",
            "fixed_monthly_mse",
            "optimized_monthly_mse",
            "monthly_mse_improvement",
            "monthly_mse_improvement_percent",
            "fixed_monthly_oos_r2",
            "optimized_monthly_oos_r2",
            "monthly_oos_r2_change",
            "fixed_oos_r2",
            "optimized_oos_r2",
            "oos_r2_change",
            "optimization_improved",
        ]
    ]

    family_comparison.to_csv(
        OPTIMIZATION_DIR
        / "fixed_vs_optimized_family_summary.csv",
        index=False,
    )

    print("\nFixed-versus-optimized validation ranking:")

    print(
        validation_ranking[
            [
                "rank",
                "stage",
                "model",
                "model_group",
                "monthly_mse",
                "monthly_rmse",
                "monthly_oos_r2",
                "oos_r2",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nFamily-by-family comparison:")

    print(
        family_comparison[
            [
                "model_family",
                "fixed_monthly_mse",
                "optimized_monthly_mse",
                "monthly_mse_improvement",
                "monthly_mse_improvement_percent",
                "fixed_monthly_oos_r2",
                "optimized_monthly_oos_r2",
                "optimization_improved",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved fixed-versus-optimized comparison:")
    print(
        OPTIMIZATION_DIR
        / "fixed_vs_optimized_validation_ranking.csv"
    )
    print(
        OPTIMIZATION_DIR
        / "fixed_vs_optimized_family_summary.csv"
    )


if __name__ == "__main__":
    main()