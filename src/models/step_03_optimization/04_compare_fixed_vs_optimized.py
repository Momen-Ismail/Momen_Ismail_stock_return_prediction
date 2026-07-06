"""Compare fixed and optimized models without using test results for selection."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402
from src.models.utils.evaluation import rank_models  # noqa: E402

FIXED_DIR = MODEL_OUTPUT_DIR / "fixed"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"


def best_by_stage(metrics):
    """Summarize each stage's best validation and test model."""
    rows = []
    selected = metrics[metrics["sample"].isin(["validation", "test"])]
    for (stage, sample), group in selected.groupby(["stage", "sample"]):
        best = rank_models(group, sample).iloc[0]
        rows.append({
            "stage": stage,
            "sample": sample,
            "best_model": best["model"],
            "model_group": best["model_group"],
            "oos_r2_vs_train_mean": best["oos_r2_vs_train_mean"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "prediction_target_correlation": best["prediction_target_correlation"],
            "prediction_std": best["prediction_std"],
        })
    return pd.DataFrame(rows)


def main():
    inputs = {
        "fixed": FIXED_DIR / "fixed_all_model_metrics.csv",
        "optimized": OUTPUT_DIR / "optimized_all_model_metrics.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run the fixed and optimized comparisons: {missing}")

    frames = []
    for stage, path in inputs.items():
        frame = pd.read_csv(path)
        frame["stage"] = stage
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    validation = rank_models(metrics, "validation")
    test = rank_models(metrics, "test")
    summary = best_by_stage(metrics)

    metrics.to_csv(OUTPUT_DIR / "fixed_vs_optimized_all_metrics.csv", index=False)
    validation.to_csv(
        OUTPUT_DIR / "fixed_vs_optimized_validation_ranking.csv", index=False
    )
    test.to_csv(OUTPUT_DIR / "fixed_vs_optimized_test_ranking.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_modeling_summary.csv", index=False)

    columns = ["rank_by_oos_r2", "stage", "model", "oos_r2_vs_train_mean"]
    print("Validation ranking (use for model selection):\n", validation[columns])
    print("Test ranking (final evaluation only):\n", test[columns])
    print("Best by stage:\n", summary)


if __name__ == "__main__":
    main()
