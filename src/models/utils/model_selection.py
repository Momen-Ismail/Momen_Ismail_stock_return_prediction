"""Definitions and validation for the official expanding-window selection."""

from __future__ import annotations

from itertools import product
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TARGET
from src.models.utils.evaluation import monthly_mse


RANDOM_STATE = 42
VALIDATION_YEARS = list(range(2005, 2020))

FIXED_CONFIGURATIONS = {
    "pls": {"n_components": 20},
    "elastic_net": {"alpha": 1e-4, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 100},
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 2,
    },
}

CANDIDATE_CONFIGURATIONS = {
    "pls": [{"n_components": value} for value in [1, 2, 3, 5, 20]],
    "elastic_net": [
        {"alpha": alpha, "l1_ratio": ratio}
        for alpha, ratio in product(
            [0.01, 0.015, 0.02],
            [0.75, 0.85, 0.90],
        )
    ] + [FIXED_CONFIGURATIONS["elastic_net"]],
    "random_forest": [
        {"n_estimators": value} for value in [100, 200, 300]
    ],
    "gradient_boosting": [
        {
            "n_estimators": trees,
            "learning_rate": rate,
            "max_depth": depth,
        }
        for trees, rate, depth in product(
            [100, 200, 300],
            [0.01, 0.3],
            [1, 2],
        )
    ],
}

CONSTANT_PARAMETERS = {
    "pls": {"standard_scaler": True, "pls_scale": False},
    "elastic_net": {
        "standard_scaler": True,
        "max_iter": 20_000,
        "tol": 1e-4,
    },
    "random_forest": {
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "gradient_boosting": {"random_state": RANDOM_STATE},
}


def normalize_parameters(parameters):
    """Convert NumPy scalars to plain Python values."""
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in parameters.items()
    }


def parameters_equal(left, right):
    """Compare two candidate dictionaries."""
    return normalize_parameters(left) == normalize_parameters(right)


def complete_parameters(family, parameters):
    """Add invariant estimator settings to a tuned configuration."""
    return {
        **normalize_parameters(parameters),
        **CONSTANT_PARAMETERS[family],
    }


def validate_grids():
    """Check candidate counts, uniqueness, and fixed-grid inclusion."""
    expected = {
        "pls": 5,
        "elastic_net": 10,
        "random_forest": 3,
        "gradient_boosting": 12,
    }
    for family, candidates in CANDIDATE_CONFIGURATIONS.items():
        if len(candidates) != expected[family]:
            raise ValueError(f"Unexpected {family} candidate count.")
        if len({repr(item) for item in candidates}) != len(candidates):
            raise ValueError(f"Duplicate {family} candidates.")
        if not any(
            parameters_equal(item, FIXED_CONFIGURATIONS[family])
            for item in candidates
        ):
            raise ValueError(f"Fixed {family} configuration is absent.")


def official_folds(development):
    """Return the required 15 annual expanding-window folds."""
    if development.empty:
        raise ValueError("Development sample is empty.")
    if development["month"].min() != pd.Timestamp("1990-02-28"):
        raise ValueError("Development sample must begin in February 1990.")
    if development["month"].max() != pd.Timestamp("2019-12-31"):
        raise ValueError("Development sample must end in December 2019.")

    folds = []
    years = development["month"].dt.year.to_numpy()
    for number, validation_year in enumerate(VALIDATION_YEARS, start=1):
        train_index = np.flatnonzero(years < validation_year)
        validation_index = np.flatnonzero(years == validation_year)
        if not len(train_index) or not len(validation_index):
            raise ValueError(f"Empty fold for validation year {validation_year}.")
        if development.iloc[train_index]["month"].max().year >= validation_year:
            raise ValueError(f"Leakage in validation year {validation_year}.")
        folds.append({
            "fold_number": number,
            "validation_year": validation_year,
            "train_index": train_index,
            "validation_index": validation_index,
        })

    if len(folds) != 15:
        raise ValueError("Expected exactly 15 annual folds.")
    return folds


def make_candidate_model(family, parameters):
    """Create a fresh estimator; linear preprocessing is fold-local."""
    parameters = normalize_parameters(parameters)
    if family == "pls":
        return make_pipeline(
            StandardScaler(),
            PLSRegression(
                n_components=int(parameters["n_components"]),
                scale=False,
            ),
        )
    if family == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=float(parameters["alpha"]),
                l1_ratio=float(parameters["l1_ratio"]),
                max_iter=20_000,
                tol=1e-4,
            ),
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_features="sqrt",
            min_samples_leaf=20,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model family: {family}")


def tune_family(family, development, predictors, folds):
    """Evaluate every candidate and retain fold diagnostics."""
    fold_rows = []
    for number, parameters in enumerate(
        CANDIDATE_CONFIGURATIONS[family], start=1
    ):
        parameters = normalize_parameters(parameters)
        complete = complete_parameters(family, parameters)
        candidate_id = f"{family}_{number:03d}"
        print(f"\n{candidate_id}: {parameters}")

        for fold in folds:
            train = development.iloc[fold["train_index"]]
            validation = development.iloc[fold["validation_index"]]
            model = make_candidate_model(family, parameters)
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(
                    train[predictors].to_numpy(dtype=np.float32),
                    train[TARGET].to_numpy(dtype=np.float32),
                )
                prediction = np.asarray(model.predict(
                    validation[predictors].to_numpy(dtype=np.float32)
                )).reshape(-1)
            elapsed = perf_counter() - started
            categories = sorted({item.category.__name__ for item in caught})
            messages = " | ".join(
                dict.fromkeys(str(item.message) for item in caught)
            )
            convergence = any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            fold_rows.append({
                "model_family": family,
                "candidate_id": candidate_id,
                "parameters": repr(parameters),
                "model_parameters": repr(complete),
                "is_original_fixed_configuration": parameters_equal(
                    parameters, FIXED_CONFIGURATIONS[family]
                ),
                "fold_number": fold["fold_number"],
                "validation_year": fold["validation_year"],
                "train_start": train["month"].min(),
                "train_end": train["month"].max(),
                "validation_start": validation["month"].min(),
                "validation_end": validation["month"].max(),
                "train_observations": len(train),
                "validation_observations": len(validation),
                "monthly_mse": monthly_mse(
                    validation[TARGET], prediction, validation["month"]
                ),
                "elapsed_seconds": elapsed,
                "warning_count": len(caught),
                "warning_categories": ";".join(categories),
                "warning_messages": messages,
                "convergence_status": "warning" if convergence else "ok",
                **complete,
            })
            print(
                f"  {fold['validation_year']}: "
                f"MSE={fold_rows[-1]['monthly_mse']:.10f}, {elapsed:.1f}s"
            )

    fold_results = pd.DataFrame(fold_rows)
    parameter_columns = sorted(complete_parameters(
        family, CANDIDATE_CONFIGURATIONS[family][0]
    ))
    grouping = [
        "model_family", "candidate_id", "parameters", "model_parameters",
        "is_original_fixed_configuration", *parameter_columns,
    ]
    results = fold_results.groupby(
        grouping, dropna=False, as_index=False
    ).agg(
        average_monthly_mse=("monthly_mse", "mean"),
        fold_mse_std=("monthly_mse", "std"),
        fold_mse_se=("monthly_mse", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        completed_folds=("monthly_mse", "count"),
        elapsed_seconds=("elapsed_seconds", "sum"),
        warning_count=("warning_count", "sum"),
    )
    if results["completed_folds"].ne(15).any():
        raise ValueError(f"Incomplete {family} tuning results.")
    return fold_results, results
