"""Fit selected models on development data and evaluate the test set once."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import (  # noqa: E402
    arrays,
    load_model_data,
    ols3_predictors,
)
from src.models.utils.estimation import (  # noqa: E402
    load_best_parameters,
)
from src.models.utils.evaluation import (  # noqa: E402
    evaluate_model,
    rank_models,
)

TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "test"
INTERPRETATION_INPUT_DIR = OUTPUT_DIR / "interpretation_inputs"

RANDOM_STATE = 42
COEFFICIENT_TOLERANCE = 1e-10


def model_specifications(predictors):
    """Return the final models selected using validation results."""
    pls = load_best_parameters(
        TUNING_DIR / "pls_best_parameters.csv"
    )
    elastic_net = load_best_parameters(
        TUNING_DIR / "elastic_net_best_parameters.csv"
    )
    tree = load_best_parameters(
        TUNING_DIR / "decision_tree_best_parameters.csv"
    )
    forest = load_best_parameters(
        TUNING_DIR / "random_forest_best_parameters.csv"
    )

    return {
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
                    n_components=int(pls["n_components"]),
                    scale=False,
                ),
            ),
            predictors,
        ),
        "elastic_net": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=float(elastic_net["alpha"]),
                    l1_ratio=float(elastic_net["l1_ratio"]),
                    max_iter=20_000,
                    tol=1e-4,
                ),
            ),
            predictors,
        ),
        "decision_tree": (
            DecisionTreeRegressor(
                max_depth=(
                    None
                    if tree["max_depth"] is None
                    else int(tree["max_depth"])
                ),
                min_samples_leaf=int(
                    tree["min_samples_leaf"]
                ),
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=int(
                    forest["n_estimators"]
                ),
                max_depth=(
                    None
                    if forest["max_depth"] is None
                    else int(forest["max_depth"])
                ),
                min_samples_leaf=int(
                    forest["min_samples_leaf"]
                ),
                max_features=forest["max_features"],
                n_jobs=-1,
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
    }


def linear_coefficient_table(
    model_name,
    model,
    predictors,
):
    """Extract coefficients from a fitted linear-model pipeline."""
    estimator = model[-1]

    coefficients = np.asarray(
        estimator.coef_
    ).reshape(-1)

    if len(coefficients) != len(predictors):
        raise ValueError(
            f"{model_name} has {len(coefficients)} coefficients "
            f"but {len(predictors)} predictors."
        )

    table = pd.DataFrame({
        "model": model_name,
        "predictor": predictors,
        "coefficient": coefficients,
    })

    table["absolute_coefficient"] = (
        table["coefficient"].abs()
    )

    table["nonzero"] = (
        table["absolute_coefficient"]
        > COEFFICIENT_TOLERANCE
    )

    table["rank"] = (
        table["absolute_coefficient"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return table.sort_values(
        ["model", "rank"]
    ).reset_index(drop=True)


def tree_feature_importance_table(
    model_name,
    model,
    predictors,
):
    """Extract impurity-based importance from a fitted tree model."""
    importance = np.asarray(
        model.feature_importances_
    ).reshape(-1)

    if len(importance) != len(predictors):
        raise ValueError(
            f"{model_name} has {len(importance)} importance values "
            f"but {len(predictors)} predictors."
        )

    table = pd.DataFrame({
        "model": model_name,
        "predictor": predictors,
        "impurity_importance": importance,
    })

    table["rank"] = (
        table["impurity_importance"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return table.sort_values(
        ["model", "rank"]
    ).reset_index(drop=True)


def pls_component_table(model):
    """Summarize fitted PLS component-score variation."""
    estimator = model[-1]

    score_variance = np.var(
        estimator.x_scores_,
        axis=0,
        ddof=1,
    )

    total_variance = score_variance.sum()

    if total_variance > 0:
        score_variance_share = (
            score_variance / total_variance
        )
    else:
        score_variance_share = np.full(
            len(score_variance),
            np.nan,
        )

    return pd.DataFrame({
        "component": np.arange(
            1,
            len(score_variance) + 1,
        ),
        "x_score_variance": score_variance,
        "x_score_variance_share": (
            score_variance_share
        ),
        "cumulative_x_score_variance_share": (
            np.cumsum(score_variance_share)
        ),
    })


def model_complexity_row(
    model_name,
    model,
    predictors,
):
    """Create one model-complexity summary row."""
    row = {
        "model": model_name,
        "number_of_predictors": len(predictors),
    }

    if model_name == "ols_3":
        row["nonzero_coefficients"] = int(
            np.sum(
                np.abs(
                    np.asarray(
                        model[-1].coef_
                    ).reshape(-1)
                )
                > COEFFICIENT_TOLERANCE
            )
        )

    elif model_name == "pls":
        row["pls_components"] = int(
            model[-1].n_components
        )

    elif model_name == "elastic_net":
        coefficients = np.asarray(
            model[-1].coef_
        ).reshape(-1)

        nonzero = int(
            np.sum(
                np.abs(coefficients)
                > COEFFICIENT_TOLERANCE
            )
        )

        row["nonzero_coefficients"] = nonzero
        row["nonzero_share"] = (
            nonzero / len(predictors)
        )
        row["coefficient_tolerance"] = (
            COEFFICIENT_TOLERANCE
        )

    elif model_name == "decision_tree":
        row["tree_depth"] = model.get_depth()
        row["tree_leaves"] = model.get_n_leaves()

    elif model_name == "random_forest":
        depths = np.array([
            tree.get_depth()
            for tree in model.estimators_
        ])

        leaves = np.array([
            tree.get_n_leaves()
            for tree in model.estimators_
        ])

        row["forest_trees"] = len(
            model.estimators_
        )
        row["forest_max_depth_setting"] = (
            model.max_depth
        )
        row["average_tree_depth"] = (
            depths.mean()
        )
        row["average_tree_leaves"] = (
            leaves.mean()
        )
        row["oob_r2_development_diagnostic"] = (
            model.oob_score_
        )

    return row


def main():
    samples, predictors = load_model_data(
        ("development", "test")
    )

    evaluation_samples = {
        "development": samples["development"],
        "test": samples["test"],
    }

    development_mean = (
        samples["development"][TARGET].mean()
    )

    metrics = []
    predictions = []

    linear_coefficients = []
    tree_importances = []
    model_complexity = []
    pls_components = None

    mean_predictions = {
        sample: np.full(
            len(data),
            development_mean,
            dtype=np.float32,
        )
        for sample, data
        in evaluation_samples.items()
    }

    model_metrics, model_predictions = evaluate_model(
        "historical_mean",
        evaluation_samples,
        mean_predictions,
        TARGET,
        development_mean,
    )

    metrics.append(
        model_metrics[
            model_metrics["sample"].eq("test")
        ]
    )

    predictions.append(
        model_predictions[
            model_predictions["sample"].eq("test")
        ]
    )

    for name, (
        model,
        model_predictors,
    ) in model_specifications(
        predictors
    ).items():

        print(f"Final test model: {name}")

        model_arrays = arrays(
            evaluation_samples,
            model_predictors,
        )

        X_development, y_development = (
            model_arrays["development"]
        )

        model.fit(
            X_development,
            y_development,
        )

        predictions_by_sample = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _)
            in model_arrays.items()
        }

        model_metrics, prediction_frame = (
            evaluate_model(
                name,
                evaluation_samples,
                predictions_by_sample,
                TARGET,
                development_mean,
            )
        )

        if hasattr(model, "oob_score_"):
            model_metrics[
                "oob_r2_development_diagnostic"
            ] = model.oob_score_

        metrics.append(
            model_metrics[
                model_metrics["sample"].eq("test")
            ]
        )

        predictions.append(
            prediction_frame[
                prediction_frame["sample"].eq("test")
            ]
        )

        if name in {
            "ols_3",
            "pls",
            "elastic_net",
        }:
            linear_coefficients.append(
                linear_coefficient_table(
                    name,
                    model,
                    model_predictors,
                )
            )

        if name in {
            "decision_tree",
            "random_forest",
        }:
            tree_importances.append(
                tree_feature_importance_table(
                    name,
                    model,
                    model_predictors,
                )
            )

        if name == "pls":
            pls_components = (
                pls_component_table(model)
            )

        model_complexity.append(
            model_complexity_row(
                name,
                model,
                model_predictors,
            )
        )

    metrics = pd.concat(
        metrics,
        ignore_index=True,
    )

    predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    linear_coefficients = pd.concat(
        linear_coefficients,
        ignore_index=True,
    )

    tree_importances = pd.concat(
        tree_importances,
        ignore_index=True,
    )

    model_complexity = pd.DataFrame(
        model_complexity
    )

    ranking = rank_models(
        metrics,
        "test",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERPRETATION_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False,
    )

    ranking.to_csv(
        OUTPUT_DIR
        / "final_test_model_comparison.csv",
        index=False,
    )

    predictions.to_parquet(
        OUTPUT_DIR
        / "final_test_predictions.parquet",
        index=False,
    )

    linear_coefficients.to_csv(
        INTERPRETATION_INPUT_DIR
        / "linear_model_coefficients.csv",
        index=False,
    )

    tree_importances.to_csv(
        INTERPRETATION_INPUT_DIR
        / "tree_feature_importance.csv",
        index=False,
    )

    model_complexity.to_csv(
        INTERPRETATION_INPUT_DIR
        / "model_complexity.csv",
        index=False,
    )

    pls_components.to_csv(
        INTERPRETATION_INPUT_DIR
        / "pls_components.csv",
        index=False,
    )

    print(
        ranking[
            [
                "rank",
                "model",
                "observations",
                "months",
                "monthly_mse",
                "pooled_rmse",
                "pooled_mae",
                "oos_r2",
                "prediction_target_correlation",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation inputs saved:")
    print(
        INTERPRETATION_INPUT_DIR
        / "linear_model_coefficients.csv"
    )
    print(
        INTERPRETATION_INPUT_DIR
        / "tree_feature_importance.csv"
    )
    print(
        INTERPRETATION_INPUT_DIR
        / "model_complexity.csv"
    )
    print(
        INTERPRETATION_INPUT_DIR
        / "pls_components.csv"
    )


if __name__ == "__main__":
    main()