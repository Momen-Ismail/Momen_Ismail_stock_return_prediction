"""Combine and rank all optimized-model results."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import (  # noqa: E402
    rank_models, wide_summary,
)

OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"


def main():
    inputs = {
        "linear": OUTPUT_DIR / "optimized_linear_model_metrics.csv",
        "tree": OUTPUT_DIR / "optimized_tree_model_metrics.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run both optimized-model scripts first: {missing}")

    frames = []
    for group, path in inputs.items():
        frame = pd.read_csv(path)
        frame["model_group"] = group
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    validation = rank_models(metrics, "validation")
    test = rank_models(metrics, "test")

    metrics.to_csv(OUTPUT_DIR / "optimized_all_model_metrics.csv", index=False)
    validation.to_csv(
        OUTPUT_DIR / "optimized_model_validation_ranking.csv", index=False
    )
    test.to_csv(OUTPUT_DIR / "optimized_model_test_ranking.csv", index=False)
    wide_summary(metrics).to_csv(
        OUTPUT_DIR / "optimized_model_summary_wide.csv", index=False
    )

    columns = ["rank_by_oos_r2", "model", "rmse", "oos_r2_vs_train_mean"]
    print("Validation ranking:\n", validation[columns])
    print("Test ranking:\n", test[columns])


if __name__ == "__main__":
    main()
