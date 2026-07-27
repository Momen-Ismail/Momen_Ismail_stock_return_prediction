"""Fit final models on development data and evaluate the test set once."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET
from src.models.utils.data import arrays, load_model_data, ols3_predictors
from src.models.utils.estimation import load_best_parameters
from src.models.utils.evaluation import (
    evaluate_model,
    monthly_mse,
    rank_models,
)


TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "test"
INTERPRETATION_DIR = OUTPUT_DIR / "interpretation_inputs"

RANDOM_STATE = 42
COEF_TOL = 1e-10


def build_models(predictors):
    """Create the final specifications selected using validation results."""
    pls_params = load_best_parameters(
        TUNING_DIR / "pls_best_parameters.csv"
    )

    en_params = load_best_parameters(
        TUNING_DIR / "elastic_net_best_parameters.csv"
    )

    # Validation selected optimized linear models but fixed tree models.
    rf_params = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 20,
        "bootstrap": True,
    }

    gb_params = {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 2,
    }

    models = {
        "ols_3": (
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
            ols3_predictors(),
        ),
        "pls": (
            make_pipeline(
                StandardScaler(),
                PLSRegression(
                    n_components=int(pls_params["n_components"]),
                    scale=False,
                ),
            ),
            predictors,
        ),
        "elastic_net": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=float(en_params["alpha"]),
                    l1_ratio=float(en_params["l1_ratio"]),
                    max_iter=20_000,
                    tol=1e-4,
                ),
            ),
            predictors,
        ),
        "random_forest": (
            RandomForestRegressor(
                **rf_params,
                oob_score=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(
                **gb_params,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
    }

    specifications = pd.DataFrame([
        {
            "model": "historical_mean",
            "parameters": str({
                "benchmark": "development_target_mean",
            }),
        },
        {
            "model": "ols_3",
            "parameters": str({
                "predictors": ols3_predictors(),
            }),
        },
        {
            "model": "pls",
            "parameters": str({
                **pls_params,
                "selected_version": "optimized",
            }),
        },
        {
            "model": "elastic_net",
            "parameters": str({
                **en_params,
                "selected_version": "optimized",
            }),
        },
        {
            "model": "random_forest",
            "parameters": str({
                **rf_params,
                "selected_version": "fixed",
            }),
        },
        {
            "model": "gradient_boosting",
            "parameters": str({
                **gb_params,
                "selected_version": "fixed",
            }),
        },
    ])

    return models, specifications


def coefficient_table(model_name, model, predictors):
    """Return standardized coefficients for a fitted linear pipeline."""
    coefficients = np.asarray(model[-1].coef_).reshape(-1)

    if len(coefficients) != len(predictors):
        raise ValueError(
            f"{model_name}: coefficient and predictor counts differ."
        )

    table = pd.DataFrame({
        "model": model_name,
        "predictor": predictors,
        "standardized_coefficient": coefficients,
    })

    table["absolute_standardized_coefficient"] = (
        table["standardized_coefficient"].abs()
    )

    table["nonzero"] = (
        table["absolute_standardized_coefficient"] > COEF_TOL
    )

    table["rank"] = (
        table["absolute_standardized_coefficient"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return table.sort_values("rank").reset_index(drop=True)


def importance_table(model_name, model, predictors):
    """Return impurity importance for a fitted tree ensemble."""
    importance = np.asarray(model.feature_importances_)

    if len(importance) != len(predictors):
        raise ValueError(
            f"{model_name}: importance and predictor counts differ."
        )

    table = pd.DataFrame({
        "model": model_name,
        "predictor": predictors,
        "impurity_importance": importance,
    })

    table["rank"] = (
        table["impurity_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return table.sort_values("rank").reset_index(drop=True)


def pls_components_table(model):
    """Summarize variation in fitted PLS component scores."""
    scores = np.asarray(model[-1].x_scores_)
    variance = np.var(scores, axis=0, ddof=1)
    total = variance.sum()

    shares = (
        variance / total
        if total > 0
        else np.full(len(variance), np.nan)
    )

    return pd.DataFrame({
        "component": np.arange(1, len(variance) + 1),
        "x_score_variance": variance,
        "x_score_variance_share": shares,
        "cumulative_x_score_variance_share": np.cumsum(shares),
    })


def complexity_row(model_name, model, predictors):
    """Return one compact model-complexity summary."""
    row = {
        "model": model_name,
        "number_of_predictors": len(predictors),
    }

    if model_name in {"ols_3", "elastic_net"}:
        coefficients = np.asarray(model[-1].coef_).reshape(-1)
        nonzero = int(np.sum(np.abs(coefficients) > COEF_TOL))

        row["nonzero_coefficients"] = nonzero
        row["nonzero_share"] = nonzero / len(predictors)

    elif model_name == "pls":
        row["pls_components"] = int(model[-1].n_components)

    elif model_name == "random_forest":
        depths = np.array([
            tree.get_depth()
            for tree in model.estimators_
        ])

        leaves = np.array([
            tree.get_n_leaves()
            for tree in model.estimators_
        ])

        row.update({
            "forest_trees": len(model.estimators_),
            "average_tree_depth": depths.mean(),
            "maximum_tree_depth": depths.max(),
            "average_tree_leaves": leaves.mean(),
            "oob_r2_development_diagnostic": model.oob_score_,
        })

    elif model_name == "gradient_boosting":
        trees = np.asarray(model.estimators_).reshape(-1)

        depths = np.array([
            tree.get_depth()
            for tree in trees
        ])

        leaves = np.array([
            tree.get_n_leaves()
            for tree in trees
        ])

        row.update({
            "boosting_trees": model.n_estimators,
            "learning_rate": model.learning_rate,
            "maximum_depth_setting": model.max_depth,
            "average_tree_depth": depths.mean(),
            "average_tree_leaves": leaves.mean(),
        })

    return row


def main():
    samples, predictors = load_model_data(
        ("development", "test")
    )

    development = samples["development"]
    test = samples["test"]
    development_mean = development[TARGET].mean()

    test_sample = {"test": test}

    metrics_list = []
    predictions_list = []
    coefficient_tables = []
    importance_tables = []
    complexity_rows = []
    pls_components = None

    # Historical-mean benchmark
    benchmark_prediction = np.full(
        len(test),
        development_mean,
        dtype=np.float32,
    )

    metrics, predictions = evaluate_model(
        "historical_mean",
        test_sample,
        {"test": benchmark_prediction},
        TARGET,
        development_mean,
    )

    metrics_list.append(metrics)
    predictions_list.append(predictions)

    models, specifications = build_models(predictors)

    for model_name, (model, model_predictors) in models.items():
        print(
            f"Final test model: {model_name} "
            f"({len(model_predictors)} predictors)"
        )

        model_arrays = arrays(
            samples,
            model_predictors,
        )

        X_development, y_development = model_arrays["development"]
        X_test, _ = model_arrays["test"]

        model.fit(X_development, y_development)
        test_prediction = np.asarray(model.predict(X_test)).reshape(-1)

        metrics, predictions = evaluate_model(
            model_name,
            test_sample,
            {"test": test_prediction},
            TARGET,
            development_mean,
        )

        # Development-sample OOB diagnostic for Random Forest
        if model_name == "random_forest":
            oob_prediction = np.asarray(
                model.oob_prediction_
            ).reshape(-1)

            valid = np.isfinite(oob_prediction)

            metrics["oob_pooled_mse_development"] = mean_squared_error(
                y_development[valid],
                oob_prediction[valid],
            )

            metrics["oob_monthly_mse_development"] = monthly_mse(
                y_development[valid],
                oob_prediction[valid],
                development.loc[valid, "month"],
            )

            metrics["oob_r2_development_diagnostic"] = model.oob_score_

        metrics_list.append(metrics)
        predictions_list.append(predictions)

        if model_name in {"ols_3", "pls", "elastic_net"}:
            coefficient_tables.append(
                coefficient_table(
                    model_name,
                    model,
                    model_predictors,
                )
            )

        if model_name in {"random_forest", "gradient_boosting"}:
            importance_tables.append(
                importance_table(
                    model_name,
                    model,
                    model_predictors,
                )
            )

        if model_name == "pls":
            pls_components = pls_components_table(model)

        complexity_rows.append(
            complexity_row(
                model_name,
                model,
                model_predictors,
            )
        )

    metrics = pd.concat(metrics_list, ignore_index=True)
    predictions = pd.concat(predictions_list, ignore_index=True)
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    importances = pd.concat(importance_tables, ignore_index=True)
    complexity = pd.DataFrame(complexity_rows)

    if predictions["prediction"].isna().any():
        raise ValueError("Final predictions contain missing values.")

    if np.isinf(predictions["prediction"]).any():
        raise ValueError("Final predictions contain infinite values.")

    duplicates = predictions.duplicated([
        "model",
        "sample",
        "ticker",
        "month",
    ]).sum()

    if duplicates:
        raise ValueError(
            f"Duplicate final predictions: {duplicates}"
        )

    ranking = rank_models(
        metrics,
        sample="test",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERPRETATION_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False,
    )

    ranking.to_csv(
        OUTPUT_DIR / "final_test_model_comparison.csv",
        index=False,
    )

    predictions.to_parquet(
        OUTPUT_DIR / "final_test_predictions.parquet",
        index=False,
    )

    specifications.to_csv(
        INTERPRETATION_DIR / "final_model_specifications.csv",
        index=False,
    )

    coefficients.to_csv(
        INTERPRETATION_DIR / "linear_model_coefficients.csv",
        index=False,
    )

    importances.to_csv(
        INTERPRETATION_DIR / "tree_feature_importance.csv",
        index=False,
    )

    complexity.to_csv(
        INTERPRETATION_DIR / "model_complexity.csv",
        index=False,
    )

    if pls_components is not None:
        pls_components.to_csv(
            INTERPRETATION_DIR / "pls_components.csv",
            index=False,
        )

    columns = [
        "rank",
        "model",
        "observations",
        "months",
        "monthly_mse",
        "monthly_rmse",
        "monthly_oos_r2",
        "pooled_rmse",
        "pooled_mae",
        "oos_r2",
        "prediction_target_correlation",
    ]

    print("\nFinal test ranking:")
    print(
        ranking[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved final test outputs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()