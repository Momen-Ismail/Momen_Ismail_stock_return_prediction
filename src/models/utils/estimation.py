"""Shared model fitting and validation-grid tuning helpers."""

from ast import literal_eval
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.models.utils.data import arrays
from src.models.utils.evaluation import (
    evaluate_model,
    evaluate_predictions,
    ranked_effects,
)


def fit_models(models, samples, predictors, target, effect=None):
    """Fit named models on training data and evaluate every sample."""
    model_arrays = arrays(samples, predictors, target)
    metrics, predictions, effects = [], [], []

    for name, model in models.items():
        print(f"Estimating {name} ({len(predictors)} predictors)")
        start = perf_counter()
        model.fit(*model_arrays["train"])
        model_predictions = {
            sample: np.asarray(model.predict(X)).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }
        model_metrics, prediction_frame = evaluate_model(
            name, samples, model_predictions, target
        )
        model_metrics["elapsed_seconds"] = perf_counter() - start
        metrics.append(model_metrics)
        predictions.append(prediction_frame)

        if effect and hasattr(model, effect[0]):
            values = np.asarray(getattr(model, effect[0])).reshape(-1)
            if len(values) == len(predictors):
                effects.append(ranked_effects(name, predictors, values, effect[1]))
        print(model_metrics.to_string(index=False))

    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(),
    )


def tune_grid(family, grid, make_model, train, validation, benchmark):
    """Fit one family and evaluate every candidate on validation data."""
    X_train, y_train = train
    X_validation, y_validation = validation
    rows = []
    for params in ParameterGrid(grid):
        print(f"Testing {family}: {params}", flush=True)
        start = perf_counter()
        model = make_model(family, params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_validation)).reshape(-1)
        rows.append({
            "model_family": family,
            "parameters": str(params),
            **params,
            **evaluate_predictions(y_validation, predictions, benchmark),
            "elapsed_seconds": perf_counter() - start,
        })
    return rows


def best_by_family(results):
    """Select each family's highest validation out-of-sample R-squared."""
    ranked = results.sort_values(
        ["oos_r2_vs_train_mean", "rmse", "mae"],
        ascending=[False, True, True],
    )
    best = ranked.groupby("model_family", group_keys=False).head(1).copy()
    best["selection_rule"] = "highest_validation_oos_r2"
    return best.reset_index(drop=True)


def save_results(rows, results_file, parameters_file):
    """Save candidate results and current family winners."""
    results = pd.DataFrame(rows)
    results.to_csv(results_file, index=False)
    best_by_family(results).to_csv(parameters_file, index=False)


def load_best_parameters(path, required_families):
    """Load validation-selected parameter dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Run the corresponding tuning script first: {path}")
    parameters = {
        row.model_family: literal_eval(row.parameters)
        for row in pd.read_csv(path).itertuples()
    }
    missing = set(required_families) - parameters.keys()
    if missing:
        raise ValueError(f"Missing tuned parameters for: {sorted(missing)}")
    return parameters
