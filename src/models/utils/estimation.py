"""Shared model fitting and validation-grid tuning helpers."""

from ast import literal_eval
import re
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

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
        if hasattr(model, "oob_score_"):
            model_metrics["oob_r2_train"] = model.oob_score_
        metrics.append(model_metrics)
        predictions.append(prediction_frame)

        estimator = model[-1] if hasattr(model, "steps") else model
        if effect and hasattr(estimator, effect[0]):
            values = np.asarray(getattr(estimator, effect[0])).reshape(-1)
            if len(values) == len(predictors):
                effects.append(ranked_effects(name, predictors, values, effect[1]))
        print(model_metrics.to_string(index=False))

    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(),
    )


def tune_grid(
    family, grid, make_model, data, predictors, target, folds,
    train_sample_fraction=1.0, random_state=42,
):
    """Tune one family using expanding month-blocked cross-validation."""
    rows = []
    for params in ParameterGrid(grid):
        params = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in params.items()
        }
        print(f"Testing {family}: {params}", flush=True)
        start = perf_counter()
        realized, predictions, benchmarks, fold_mse, fold_oob = [], [], [], [], []

        for fold_number, fold in enumerate(folds, start=1):
            train = data.loc[fold["train_index"]]
            validation = data.loc[fold["validation_index"]]
            if train_sample_fraction < 1.0:
                train = train.sample(
                    frac=train_sample_fraction,
                    random_state=random_state + fold_number,
                )

            X_train = train[predictors].to_numpy(dtype=np.float32)
            y_train = train[target].to_numpy(dtype=np.float32)
            X_validation = validation[predictors].to_numpy(dtype=np.float32)
            y_validation = validation[target].to_numpy(dtype=np.float32)

            model = make_model(family, params)
            model.fit(X_train, y_train)
            if hasattr(model, "oob_score_"):
                fold_oob.append(model.oob_score_)
            realized.append(y_validation)
            fold_prediction = np.asarray(model.predict(X_validation)).reshape(-1)
            predictions.append(fold_prediction)
            fold_mse.append(mean_squared_error(y_validation, fold_prediction))
            benchmarks.append(
                np.full(len(validation), y_train.mean(), dtype=np.float32)
            )

        realized = np.concatenate(realized)
        predictions = np.concatenate(predictions)
        benchmarks = np.concatenate(benchmarks)
        row = {
            "model_family": family,
            "parameters": str(params),
            **params,
            **evaluate_predictions(realized, predictions, benchmarks),
            "cv_folds": len(folds),
            "cv_mse_mean": np.mean(fold_mse),
            "cv_mse_standard_error": (
                np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))
            ),
            "validation_start": folds[0]["validation_start"],
            "validation_end": folds[-1]["validation_end"],
            "elapsed_seconds": perf_counter() - start,
        }
        if fold_oob:
            row["oob_r2_mean"] = np.mean(fold_oob)
        rows.append(row)
    return rows


def best_by_family(results):
    """Apply the one-standard-error rule within each model family."""
    selected = []
    for family, group in results.groupby("model_family"):
        minimum = group.loc[group["cv_mse_mean"].idxmin()]
        eligible = group[
            group["cv_mse_mean"]
            <= minimum["cv_mse_mean"] + minimum["cv_mse_standard_error"]
        ].copy()
        eligible["complexity"] = eligible.apply(_complexity, axis=1)
        winner = eligible.sort_values(
            ["complexity", "cv_mse_mean"], ascending=[True, True]
        ).iloc[0].copy()
        winner["selection_rule"] = "one_standard_error_time_series_cv"
        selected.append(winner)
    return pd.DataFrame(selected).drop(columns="complexity", errors="ignore")


def _complexity(row):
    """Lower values represent simpler specifications."""
    family = row["model_family"]
    if family in {"pcr", "pls"}:
        return float(row["n_components"])
    if family in {"ridge", "lasso", "elastic_net"}:
        return -float(row["alpha"])
    value = lambda name, default: (
        default if pd.isna(row.get(name, np.nan)) else float(row[name])
    )
    depth = value("max_depth", 10_000)
    leaves = -value("min_samples_leaf", 0)
    estimators = value("n_estimators", 1)
    pruning = -value("ccp_alpha", 0)
    return depth * 1e9 + leaves * 1e6 + estimators * 1e3 + pruning


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
        row.model_family: literal_eval(
            re.sub(r"np\.(?:float|int)\d+\(([^()]*)\)", r"\1", row.parameters)
        )
        for row in pd.read_csv(path).itertuples()
    }
    missing = set(required_families) - parameters.keys()
    if missing:
        raise ValueError(f"Missing tuned parameters for: {sorted(missing)}")
    return parameters
