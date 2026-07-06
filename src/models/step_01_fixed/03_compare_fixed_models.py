"""Combine and rank the fixed-model results."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402

OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"


def rank_models(metrics, sample):
    """Rank one sample by out-of-sample R-squared, then RMSE and MAE."""
    ranking = (
        metrics.loc[metrics["sample"].eq(sample)]
        .sort_values(
            ["oos_r2_vs_train_mean", "rmse", "mae"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking


def wide_summary(metrics):
    """Place validation and test metrics side by side."""
    columns = [
        "rmse",
        "mae",
        "oos_r2_vs_train_mean",
        "prediction_target_correlation",
        "prediction_std",
    ]
    wide = metrics[metrics["sample"].isin(["validation", "test"])].pivot(
        index=["model", "model_group"], columns="sample", values=columns
    )
    wide.columns = [f"{metric}_{sample}" for metric, sample in wide.columns]
    return wide.reset_index().sort_values(
        "oos_r2_vs_train_mean_test", ascending=False
    )


def main():
    inputs = {
        "linear": OUTPUT_DIR / "fixed_linear_model_metrics.csv",
        "tree": OUTPUT_DIR / "fixed_tree_model_metrics.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run both fixed-model scripts first: {missing}")

    frames = []
    for group, path in inputs.items():
        frame = pd.read_csv(path)
        frame["model_group"] = group
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    validation = rank_models(metrics, "validation")
    test = rank_models(metrics, "test")

    metrics.to_csv(OUTPUT_DIR / "fixed_all_model_metrics.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "fixed_model_validation_ranking.csv", index=False)
    test.to_csv(OUTPUT_DIR / "fixed_model_test_ranking.csv", index=False)
    wide_summary(metrics).to_csv(OUTPUT_DIR / "fixed_model_summary_wide.csv", index=False)

    print("Validation ranking:\n", validation[["rank", "model", "rmse", "oos_r2_vs_train_mean"]])
    print("Test ranking:\n", test[["rank", "model", "rmse", "oos_r2_vs_train_mean"]])


if __name__ == "__main__":
    main()
