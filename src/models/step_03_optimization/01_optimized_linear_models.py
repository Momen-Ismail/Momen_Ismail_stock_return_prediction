"""Estimate optimized linear models on train and validation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import (  # noqa: E402
    arrays,
    load_model_data,
    ols3_predictors,
)
from src.models.utils.estimation import load_best_parameters  # noqa: E402
from src.models.utils.evaluation import evaluate_model  # noqa: E402


TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"


def optimized_models(all_predictors):
    """Define the optimized linear-model specifications."""
    pls_parameters = load_best_parameters(
        TUNING_DIR / "pls_best_parameters.csv"
    )

    elastic_net_parameters = load_best_parameters(
        TUNING_DIR / "elastic_net_best_parameters.csv"
    )

    models = {
        "ols_3": (
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
            ols3_predictors(),
        ),
        "pls_optimized": (
            make_pipeline(
                StandardScaler(),
                PLSRegression(
                    n_components=int(
                        pls_parameters["n_components"]
                    ),
                    scale=False,
                ),
            ),
            all_predictors,
        ),
        "elastic_net_optimized": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=float(
                        elastic_net_parameters["alpha"]
                    ),
                    l1_ratio=float(
                        elastic_net_parameters["l1_ratio"]
                    ),
                    max_iter=20_000,
                    tol=1e-4,
                ),
            ),
            all_predictors,
        ),
    }

    specifications = pd.DataFrame([
        {
            "model": "ols_3",
            "parameters": str({
                "predictors": ols3_predictors(),
            }),
        },
        {
            "model": "pls_optimized",
            "parameters": str(pls_parameters),
        },
        {
            "model": "elastic_net_optimized",
            "parameters": str(elastic_net_parameters),
        },
    ])

    return models, specifications


def main():
    samples, all_predictors = load_model_data(
        ("train", "validation")
    )

    models, specifications = optimized_models(
        all_predictors
    )

    all_metrics = []
    all_predictions = []
    all_coefficients = []
    coefficient_summary = []

    train_mean = samples["train"][TARGET].mean()

    mean_predictions = {
        sample: np.full(
            len(data),
            train_mean,
        )
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
        print(
            f"Estimating {name} "
            f"({len(predictors)} predictors)"
        )

        model_arrays = arrays(
            samples,
            predictors,
        )

        X_train, y_train = model_arrays["train"]

        model.fit(
            X_train,
            y_train,
        )

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

        coefficients = np.asarray(
            estimator.coef_
        ).reshape(-1)

        if len(coefficients) == len(predictors):
            coefficient_frame = pd.DataFrame({
                "model": name,
                "predictor": predictors,
                "standardized_coefficient": coefficients,
            }).sort_values(
                "standardized_coefficient",
                key=abs,
                ascending=False,
            )

            all_coefficients.append(
                coefficient_frame
            )

            coefficient_summary.append({
                "model": name,
                "total_coefficients": len(coefficients),
                "nonzero_coefficients": int(
                    np.count_nonzero(coefficients)
                ),
                "zero_coefficients": int(
                    np.sum(coefficients == 0)
                ),
                "maximum_absolute_coefficient": float(
                    np.max(np.abs(coefficients))
                ),
            })

        print(metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.concat(
        all_metrics,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "optimized_linear_model_metrics.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_parquet(
        OUTPUT_DIR
        / "optimized_linear_model_predictions.parquet",
        index=False,
    )

    pd.concat(
        all_coefficients,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "optimized_linear_model_coefficients.csv",
        index=False,
    )

    pd.DataFrame(
        coefficient_summary
    ).to_csv(
        OUTPUT_DIR
        / "optimized_linear_model_coefficient_summary.csv",
        index=False,
    )

    specifications.to_csv(
        OUTPUT_DIR
        / "optimized_linear_model_specifications.csv",
        index=False,
    )

    print("\nSaved optimized linear-model outputs:")
    print(
        OUTPUT_DIR
        / "optimized_linear_model_metrics.csv"
    )
    print(
        OUTPUT_DIR
        / "optimized_linear_model_predictions.parquet"
    )
    print(
        OUTPUT_DIR
        / "optimized_linear_model_coefficients.csv"
    )
    print(
        OUTPUT_DIR
        / "optimized_linear_model_coefficient_summary.csv"
    )
    print(
        OUTPUT_DIR
        / "optimized_linear_model_specifications.csv"
    )


if __name__ == "__main__":
    main()