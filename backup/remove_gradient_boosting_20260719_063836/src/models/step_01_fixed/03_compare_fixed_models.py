"""Combine and rank fixed-model validation results."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402


OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"


def main():
    linear_metrics = pd.read_csv(
        OUTPUT_DIR / "fixed_linear_model_metrics.csv"
    )

    tree_metrics = pd.read_csv(
        OUTPUT_DIR / "fixed_tree_model_metrics.csv"
    )

    if linear_metrics["sample"].eq("test").any() or tree_metrics["sample"].eq("test").any():
        raise ValueError("Fixed-model comparison must not contain test rows.")

    metrics = pd.concat(
        [linear_metrics, tree_metrics],
        ignore_index=True,
    )

    ranking = rank_models(
        metrics,
        sample="validation",
    )

    metrics.to_csv(
        OUTPUT_DIR / "fixed_all_model_metrics.csv",
        index=False,
    )

    ranking.to_csv(
        OUTPUT_DIR / "fixed_model_validation_ranking.csv",
        index=False,
    )

    print(
        ranking[
            [
                "rank",
                "model",
                "monthly_mse",
                "pooled_rmse",
                "oos_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
