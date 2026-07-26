"""Shared fitting and annual walk-forward tuning helpers."""

from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.config import TARGET
from src.models.utils.data import arrays
from src.models.utils.evaluation import (
    evaluate_model,
    monthly_mse,
)

def fit_models(models, samples, predictors, target=TARGET):
    """Fit models on development data and evaluate their predictions."""
    model_arrays = arrays(samples, predictors, target)

    metrics = []
    predictions = []

    X_development, y_development = model_arrays["development"]

    for name, model in models.items():
        print(f"Estimating {name}")

        model.fit(X_development, y_development)

        model_predictions = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }

        model_metrics, model_prediction_frame = evaluate_model(
            name,
            samples,
            model_predictions,
            target,
        )

        metrics.append(model_metrics)
        predictions.append(model_prediction_frame)

    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
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

            score = monthly_mse(
                y_validation,
                prediction,
                validation["month"],
            )

            fold_scores.append(score)

        average_mse = np.mean(fold_scores)

        results.append({
            "model_family": family,
            "parameters": str(params),
            **params,
            "cv_monthly_mse": average_mse,
        })

        if average_mse < best_score:
            best_score = average_mse
            best_params = params
            
        for fold_number, fold in enumerate(folds, start=1):
            print(
                f"  Fold {fold_number}/{len(folds)}",
                flush=True,
            )

            train = data.iloc[fold["train_index"]]
            validation = data.iloc[fold["validation_index"]]

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
    
    
    

def load_best_parameters(path):
    """Load the saved best parameter dictionary."""
    row = pd.read_csv(path).iloc[0]
    return literal_eval(row["parameters"])
