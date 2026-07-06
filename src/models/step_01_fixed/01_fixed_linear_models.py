"""Estimate fixed linear benchmarks before hyperparameter tuning."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import (  # noqa: E402
    arrays,
    load_model_data,
    ols3_predictors,
)
from src.models.utils.evaluation import (  # noqa: E402
    evaluate_model,
    ranked_effects,
)

OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"
RANDOM_STATE = 42


def fixed_models(all_predictors, ols3):
    """Define the untuned models and the predictors used by each."""
    return {
        "historical_mean": (None, []),
        "ols_plus_h": (LinearRegression(), all_predictors),
        "ols_3_plus_h": (LinearRegression(), ols3),
        "pcr_20": (
            make_pipeline(PCA(20, random_state=RANDOM_STATE), LinearRegression()),
            all_predictors,
        ),
        "pls_20": (PLSRegression(20), all_predictors),
        "lasso_plus_h": (
            Lasso(alpha=1e-5, max_iter=20_000, random_state=RANDOM_STATE),
            all_predictors,
        ),
        "enet_plus_h": (
            ElasticNet(
                alpha=1e-5,
                l1_ratio=0.5,
                max_iter=20_000,
                random_state=RANDOM_STATE,
            ),
            all_predictors,
        ),
    }


def main():
    samples, all_predictors = load_model_data()
    models = fixed_models(all_predictors, ols3_predictors(all_predictors))
    train_mean = samples["train"][TARGET].mean()
    metrics, prediction_frames, coefficients = [], [], []

    for name, (model, predictors) in models.items():
        print(f"Estimating {name} ({len(predictors)} predictors)")

        if model is None:
            predictions = {
                sample: np.full(len(data), train_mean, dtype=np.float32)
                for sample, data in samples.items()
            }
        else:
            model_arrays = arrays(samples, predictors)
            model.fit(*model_arrays["train"])
            predictions = {
                sample: model.predict(X).reshape(-1)
                for sample, (X, _) in model_arrays.items()
            }

            if hasattr(model, "coef_"):
                values = np.asarray(model.coef_).reshape(-1)
                if len(values) == len(predictors):
                    coefficients.append(
                        ranked_effects(name, predictors, values, "coefficient")
                    )

        model_metrics, model_predictions = evaluate_model(
            name, samples, predictions, TARGET
        )
        metrics.append(model_metrics)
        prediction_frames.append(model_predictions)
        print(model_metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(metrics).to_csv(OUTPUT_DIR / "fixed_linear_model_metrics.csv", index=False)
    pd.concat(prediction_frames).to_parquet(
        OUTPUT_DIR / "fixed_linear_model_predictions.parquet", index=False
    )
    if coefficients:
        pd.concat(coefficients).to_csv(
            OUTPUT_DIR / "fixed_linear_model_coefficients.csv", index=False
        )


if __name__ == "__main__":
    main()
