"""Shared model fitting and annual walk-forward tuning helpers."""

from ast import literal_eval
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.config import TARGET
from src.models.utils.data import arrays
from src.models.utils.evaluation import evaluate_model, monthly_mse


def fit_models(models, samples, predictors, target=TARGET, effect=None):
    """Fit models on train and evaluate train and validation."""
    model_arrays = arrays(samples, predictors, target)
    X_train, y_train = model_arrays["train"]
    train_mean = samples["train"][target].mean()
    metrics = []
    predictions = []
    effects = []

    for name, model in models.items():
        print(f"Estimating {name}")
        model.fit(X_train, y_train)
        model_predictions = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }
        model_metrics, prediction_frame = evaluate_model(
            name,
            samples,
            model_predictions,
            target,
            train_mean,
        )
        metrics.append(model_metrics)
        predictions.append(prediction_frame)

        estimator = model[-1] if hasattr(model, "steps") else model
        if effect and hasattr(estimator, effect[0]):
            values = np.asarray(getattr(estimator, effect[0])).reshape(-1)
            if len(values) == len(predictors):
                effects.append(
                    pd.DataFrame({
                        "model": name,
                        "predictor": predictors,
                        effect[1]: values,
                    }).sort_values(effect[1], key=abs, ascending=False)
                )

    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(),
    )


def tune_grid(family, grid, make_model, data, predictors, folds, target=TARGET):
    """Test all hyperparameter combinations and select the lowest CV MSE."""
    results = []
    best_score = np.inf
    best_params = None

    for params in ParameterGrid(grid):
        params = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in params.items()
        }
        print(f"Testing {family}: {params}")
        fold_scores = []

        for fold in folds:
            train = data.iloc[fold["train_index"]]
            validation = data.iloc[fold["validation_index"]]
            X_train = train[predictors].to_numpy(dtype=np.float32)
            y_train = train[target].to_numpy(dtype=np.float32)
            X_validation = validation[predictors].to_numpy(dtype=np.float32)
            y_validation = validation[target].to_numpy(dtype=np.float32)

            model = make_model(family, params)
            model.fit(X_train, y_train)
            prediction = model.predict(X_validation).reshape(-1)
            fold_scores.append(
                monthly_mse(y_validation, prediction, validation["month"])
            )

        fold_scores = np.asarray(
            fold_scores,
            dtype=float,
        )

        average_mse = fold_scores.mean()

        mse_std = (
            fold_scores.std(ddof=1)
            if len(fold_scores) > 1
            else 0.0
        )

        mse_se = (
            mse_std / np.sqrt(len(fold_scores))
            if len(fold_scores) > 0
            else np.nan
        )

        results.append({
            "model_family": family,
            "parameters": str(params),
            **params,
            "cv_monthly_mse": average_mse,
            "cv_monthly_mse_std": mse_std,
            "cv_monthly_mse_se": mse_se,
            "cv_folds": len(fold_scores),
        })

        if average_mse < best_score:
            best_score = average_mse
            best_params = params

    return pd.DataFrame(results), best_params


def save_results(
    family,
    results,
    best_params,
    results_file,
    parameters_file,
):
    """Save tuning results and the best parameters."""
    results.to_csv(results_file, index=False)
    pd.DataFrame([{
        "model_family": family,
        "parameters": str(best_params),
    }]).to_csv(parameters_file, index=False)


def load_best_parameters(path, required_families=None):
    """Load one parameter dictionary or a dictionary by family."""
    rows = pd.read_csv(path)

    def parse(value):
        value = re.sub(r"np\.(?:float|int)\d+\(([^()]*)\)", r"\1", value)
        return literal_eval(value)

    if required_families is None:
        return parse(rows.iloc[0]["parameters"])

    parameters = {
        row.model_family: parse(row.parameters)
        for row in rows.itertuples()
    }
    missing = set(required_families) - parameters.keys()
    if missing:
        raise ValueError(f"Missing tuned parameters for: {sorted(missing)}")
    return parameters
