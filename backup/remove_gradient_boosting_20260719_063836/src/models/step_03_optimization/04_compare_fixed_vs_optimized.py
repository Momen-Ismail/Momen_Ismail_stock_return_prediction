"""Compare fixed and optimized models using validation data only."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402


FIXED_DIR = MODEL_OUTPUT_DIR / "fixed"
OPTIMIZATION_DIR = MODEL_OUTPUT_DIR / "optimization"


def main():
    fixed_metrics = pd.read_csv(
        FIXED_DIR / "fixed_all_model_metrics.csv"
    )

    optimized_metrics = pd.read_csv(
        OPTIMIZATION_DIR / "optimized_all_model_metrics.csv"
    )

    if fixed_metrics["sample"].eq("test").any() or optimized_metrics["sample"].eq("test").any():
        raise ValueError("Validation comparison must not contain test rows.")

    fixed_validation = fixed_metrics[
        fixed_metrics["sample"] == "validation"
    ].copy()

    optimized_validation = optimized_metrics[
        optimized_metrics["sample"] == "validation"
    ].copy()

    fixed_validation["stage"] = "fixed"
    optimized_validation["stage"] = "optimized"

    # Historical mean and OLS-3 appear in both files.
    # Keep only the optimized-file copies to avoid duplicate benchmarks.
    fixed_models_only = fixed_validation[
        fixed_validation["model"].str.endswith("_fixed")
    ].copy()

    combined_validation = pd.concat(
        [
            optimized_validation,
            fixed_models_only,
        ],
        ignore_index=True,
    ).drop_duplicates("model", keep="first")

    validation_ranking = rank_models(
        combined_validation,
        "validation",
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

    # Direct family-by-family comparison
    fixed_family = fixed_validation[
        fixed_validation["model"].str.endswith("_fixed")
    ][
        [
            "model",
            "monthly_mse",
            "monthly_rmse",
            "oos_r2",
            "prediction_target_correlation",
        ]
    ].copy()

    fixed_family["model_family"] = (
        fixed_family["model"]
        .str.replace("_fixed", "", regex=False)
    )

    fixed_family = fixed_family.rename(
        columns={
            "model": "fixed_model",
            "monthly_mse": "fixed_monthly_mse",
            "monthly_rmse": "fixed_monthly_rmse",
            "oos_r2": "fixed_oos_r2",
            "prediction_target_correlation": (
                "fixed_prediction_target_correlation"
            ),
        }
    )

    optimized_family = optimized_validation[
        optimized_validation["model"].str.endswith("_optimized")
    ][
        [
            "model",
            "monthly_mse",
            "monthly_rmse",
            "oos_r2",
            "prediction_target_correlation",
        ]
    ].copy()

    optimized_family["model_family"] = (
        optimized_family["model"]
        .str.replace("_optimized", "", regex=False)
    )

    optimized_family = optimized_family.rename(
        columns={
            "model": "optimized_model",
            "monthly_mse": "optimized_monthly_mse",
            "monthly_rmse": "optimized_monthly_rmse",
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
    )

    family_comparison["monthly_mse_improvement"] = (
        family_comparison["fixed_monthly_mse"]
        - family_comparison["optimized_monthly_mse"]
    )

    family_comparison["monthly_mse_improvement_percent"] = (
        family_comparison["monthly_mse_improvement"]
        / family_comparison["fixed_monthly_mse"]
        * 100
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
                "monthly_mse",
                "monthly_rmse",
                "oos_r2",
            ]
        ].to_string(index=False)
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
                "fixed_oos_r2",
                "optimized_oos_r2",
                "optimization_improved",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
