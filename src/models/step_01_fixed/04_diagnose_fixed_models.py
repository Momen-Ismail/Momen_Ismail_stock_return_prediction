"""Diagnose fixed-model predictions on train and validation data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import load_model_data  # noqa: E402
from src.models.utils.evaluation import evaluate_predictions  # noqa: E402


FIXED_DIR = MODEL_OUTPUT_DIR / "fixed"
OUTPUT_DIR = FIXED_DIR / "diagnostics"


def load_predictions():
    """Load and combine fixed linear and tree predictions."""
    linear = pd.read_parquet(
        FIXED_DIR / "fixed_linear_model_predictions.parquet"
    )

    tree = pd.read_parquet(
        FIXED_DIR / "fixed_tree_model_predictions.parquet"
    )

    return pd.concat(
        [linear, tree],
        ignore_index=True,
    )


def target_summary(samples):
    """Summarize the target in train and validation samples."""
    rows = []

    for sample, data in samples.items():
        target = data[TARGET]

        rows.append({
            "sample": sample,
            "observations": len(data),
            "months": data["month"].nunique(),
            "tickers": data["ticker"].nunique(),
            "mean": target.mean(),
            "std": target.std(),
            "minimum": target.min(),
            "p01": target.quantile(0.01),
            "median": target.median(),
            "p99": target.quantile(0.99),
            "maximum": target.max(),
        })

    return pd.DataFrame(rows)


def prediction_summary(predictions):
    """Summarize prediction distributions for every model and sample."""
    rows = []

    for (model, sample), data in predictions.groupby(
        ["model", "sample"]
    ):
        prediction = data["prediction"]

        rows.append({
            "model": model,
            "sample": sample,
            "observations": len(data),
            "mean": prediction.mean(),
            "std": prediction.std(),
            "minimum": prediction.min(),
            "p01": prediction.quantile(0.01),
            "median": prediction.median(),
            "p99": prediction.quantile(0.99),
            "maximum": prediction.max(),
            "unique_predictions": prediction.nunique(),
        })

    return pd.DataFrame(rows)


def validation_by_year(predictions, train_mean):
    """Evaluate each fixed model separately for every validation year."""
    validation = predictions[
        predictions["sample"] == "validation"
    ].copy()

    validation["year"] = validation["month"].dt.year

    rows = []

    for (model, year), data in validation.groupby(
        ["model", "year"]
    ):
        benchmark = np.full(
            len(data),
            train_mean,
        )

        metrics = evaluate_predictions(
            data["realized_target"],
            data["prediction"],
            benchmark,
            data["month"],
        )

        rows.append({
            "model": model,
            "year": year,
            "observations": len(data),
            **metrics,
        })

    return pd.DataFrame(rows)


def extreme_predictions(predictions):
    """Return the largest absolute fixed-model predictions."""
    data = predictions.copy()
    data["absolute_prediction"] = data["prediction"].abs()

    return (
        data.sort_values(
            "absolute_prediction",
            ascending=False,
        )
        .groupby(
            ["model", "sample"],
            group_keys=False,
        )
        .head(25)
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples, _ = load_model_data(
        ("train", "validation")
    )

    predictions = load_predictions()
    predictions["month"] = pd.to_datetime(
        predictions["month"]
    )

    train_mean = samples["train"][TARGET].mean()

    targets = target_summary(samples)
    prediction_distribution = prediction_summary(predictions)
    yearly_results = validation_by_year(
        predictions,
        train_mean,
    )
    extremes = extreme_predictions(predictions)

    targets.to_csv(
        OUTPUT_DIR / "target_summary.csv",
        index=False,
    )

    prediction_distribution.to_csv(
        OUTPUT_DIR / "prediction_summary.csv",
        index=False,
    )

    yearly_results.to_csv(
        OUTPUT_DIR / "validation_performance_by_year.csv",
        index=False,
    )

    extremes.to_csv(
        OUTPUT_DIR / "extreme_predictions.csv",
        index=False,
    )

    decision_tree = prediction_distribution[
        (
            prediction_distribution["model"]
            == "decision_tree_fixed"
        )
        & (
            prediction_distribution["sample"]
            == "validation"
        )
    ]

    print("\nTarget summary:")
    print(targets.to_string(index=False))

    print("\nPrediction summary:")
    print(prediction_distribution.to_string(index=False))

    print("\nDecision Tree validation prediction behavior:")
    print(decision_tree.to_string(index=False))

    print(f"\nDiagnostic files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()