"""Estimate fixed linear train/validation benchmarks."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import (  # noqa: E402
    arrays,
    load_model_data,
    ols3_predictors,
)
from src.models.utils.evaluation import evaluate_model  # noqa: E402


OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"


def fixed_models(all_predictors):
    """Define the fixed linear model specifications."""
    return {
        "ols_3": (
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
            ols3_predictors(),
        ),
        "pls_fixed": (
            make_pipeline(
                StandardScaler(),
                PLSRegression(
                    n_components=20,
                    scale=False,
                ),
            ),
            all_predictors,
        ),
        "elastic_net_fixed": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=1e-4,
                    l1_ratio=0.5,
                    max_iter=20_000,
                    tol=1e-4,
                ),
            ),
            all_predictors,
        ),
    }


def main():
    samples, all_predictors = load_model_data(
        ("train", "validation")
    )

    models = fixed_models(all_predictors)

    all_metrics = []
    all_predictions = []
    all_coefficients = []

    train_mean = samples["train"][TARGET].mean()

    mean_predictions = {
        sample: np.full(len(data), train_mean)
        for sample, data in samples.items()
    }

    metrics, predictions = evaluate_model(
        "historical_mean",
        samples,
        mean_predictions,
        TARGET,
        train_mean,
    )

    all_metrics.append(metrics)
    all_predictions.append(predictions)

    print(metrics.to_string(index=False))

    for name, (model, predictors) in models.items():
        print(f"Estimating {name} ({len(predictors)} predictors)")

        model_arrays = arrays(samples, predictors)

        X_train, y_train = model_arrays["train"]
        model.fit(X_train, y_train)

        model_predictions = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }

        metrics, predictions = evaluate_model(
            name,
            samples,
            model_predictions,
            TARGET,
            train_mean,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)

        estimator = model[-1]
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        if len(coefficients) == len(predictors):
            all_coefficients.append(
                pd.DataFrame({
                    "model": name,
                    "predictor": predictors,
                    "coefficient": coefficients,
                }).sort_values("coefficient", key=abs, ascending=False)
            )

        print(metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat(
        all_metrics,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "fixed_linear_model_metrics.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_parquet(
        OUTPUT_DIR / "fixed_linear_model_predictions.parquet",
        index=False,
    )

    pd.concat(
        all_coefficients,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "fixed_linear_model_coefficients.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
