"""Shared inputs and calculations for Step 6 interpretation scripts."""

import numpy as np
import pandas as pd

from src.config import (
    INTERPRETATION_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
)
from src.models.utils.evaluation import monthly_mse


TEST_DIR = MODEL_OUTPUT_DIR / "test"

TEST_INTERPRETATION_DIR = (
    TEST_DIR / "interpretation_inputs"
)

FIGURE_DIR = (
    INTERPRETATION_OUTPUT_DIR / "figures"
)


ACTIVE_MODELS = [
    "historical_mean",
    "ols_3",
    "pls",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
]


DISPLAY_NAMES = {
    "historical_mean": "Historical Mean",
    "ols_3": "OLS-3",
    "pls": "PLS",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


MARKET_PREDICTORS = {
    "market_ret_1m",
    "market_vol_1m",
    "vix_avg_1m",
    "vix_change_1m",
}


MACRO_PREDICTORS = {
    "wg_dp",
    "wg_ep",
    "wg_bm",
    "wg_ntis",
    "wg_tbl",
    "wg_tms",
    "wg_dfy",
    "wg_svar",
}


def require_file(path, prior_step):
    """Return a required path or explain which prior step must run."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}. "
            f"Run {prior_step} first."
        )

    return path


def require_columns(data, columns, path):
    """Check that one input contains all required columns."""
    missing = set(columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )


def read_csv(path, prior_step, columns=()):
    """Read one required CSV file and validate its columns."""
    data = pd.read_csv(
        require_file(
            path,
            prior_step,
        )
    )

    require_columns(
        data,
        columns,
        path,
    )

    return data


def read_parquet(path, prior_step, columns=()):
    """Read one required Parquet file and validate its columns."""
    data = pd.read_parquet(
        require_file(
            path,
            prior_step,
        )
    )

    require_columns(
        data,
        columns,
        path,
    )

    return data


def ensure_unique(data, columns, label):
    """Reject duplicated report keys."""
    duplicates = data.duplicated(
        columns
    ).sum()

    if duplicates:
        raise ValueError(
            f"{label} contains {duplicates} duplicate "
            f"rows for {columns}."
        )


def save_csv(data, filename):
    """Write one Step 6 CSV output."""
    INTERPRETATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        INTERPRETATION_OUTPUT_DIR / filename,
        index=False,
    )


def predictor_group(predictor):
    """Assign one predictor to its economic predictor family."""
    if predictor in MARKET_PREDICTORS:
        return "market_vix"

    if predictor in MACRO_PREDICTORS:
        return "welch_goyal"

    if "_x_wg_" in predictor:
        return "characteristic_macro_interaction"

    if predictor.startswith("sic2_"):
        return "sic2"

    return "stock_characteristic"


def load_test_outputs():
    """Load final test metrics, ranking, and predictions."""
    metrics_path = (
        TEST_DIR / "final_test_metrics.csv"
    )

    ranking_path = (
        TEST_DIR
        / "final_test_model_comparison.csv"
    )

    predictions_path = (
        TEST_DIR
        / "final_test_predictions.parquet"
    )

    metrics = read_csv(
        metrics_path,
        "Step 5 final test evaluation",
        [
            "model",
            "sample",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "pooled_mse",
            "pooled_rmse",
            "pooled_mae",
            "oos_r2",
            "prediction_target_correlation",
            "prediction_mean",
            "prediction_std",
        ],
    )

    ranking = read_csv(
        ranking_path,
        "Step 5 final test evaluation",
        [
            "rank",
            "model",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "pooled_rmse",
            "pooled_mae",
            "oos_r2",
        ],
    )

    predictions = read_parquet(
        predictions_path,
        "Step 5 final test evaluation",
        [
            "ticker",
            "month",
            "realized_target",
            "prediction",
            "model",
            "sample",
        ],
    )

    metrics = metrics[
        metrics["sample"].eq("test")
        & metrics["model"].isin(ACTIVE_MODELS)
    ].copy()

    ranking = ranking[
        ranking["model"].isin(ACTIVE_MODELS)
    ].copy()

    predictions = predictions[
        predictions["sample"].eq("test")
        & predictions["model"].isin(ACTIVE_MODELS)
    ].copy()

    predictions["month"] = pd.to_datetime(
        predictions["month"]
    )

    ensure_unique(
        metrics,
        ["model", "sample"],
        "Final test metrics",
    )

    ensure_unique(
        ranking,
        ["model"],
        "Final test ranking",
    )

    ensure_unique(
        predictions,
        [
            "ticker",
            "month",
            "model",
        ],
        "Final test predictions",
    )

    missing_metrics = (
        set(ACTIVE_MODELS)
        - set(metrics["model"])
    )

    missing_ranking = (
        set(ACTIVE_MODELS)
        - set(ranking["model"])
    )

    missing_predictions = (
        set(ACTIVE_MODELS)
        - set(predictions["model"])
    )

    if missing_metrics:
        raise ValueError(
            "Final test metrics are missing models: "
            f"{sorted(missing_metrics)}"
        )

    if missing_ranking:
        raise ValueError(
            "Final test ranking is missing models: "
            f"{sorted(missing_ranking)}"
        )

    if missing_predictions:
        raise ValueError(
            "Final test predictions are missing models: "
            f"{sorted(missing_predictions)}"
        )

    if predictions[
        [
            "realized_target",
            "prediction",
        ]
    ].isna().any().any():
        raise ValueError(
            "Final test predictions contain missing values."
        )

    if np.isinf(
        predictions[
            [
                "realized_target",
                "prediction",
            ]
        ].to_numpy()
    ).any():
        raise ValueError(
            "Final test predictions contain infinite values."
        )

    ranking = ranking.sort_values(
        "rank"
    ).reset_index(drop=True)

    return metrics, ranking, predictions


def load_interpretation_inputs():
    """Load model information saved during Step 5 fitting."""
    specifications_path = (
        TEST_INTERPRETATION_DIR
        / "final_model_specifications.csv"
    )

    coefficient_path = (
        TEST_INTERPRETATION_DIR
        / "linear_model_coefficients.csv"
    )

    tree_path = (
        TEST_INTERPRETATION_DIR
        / "tree_feature_importance.csv"
    )

    complexity_path = (
        TEST_INTERPRETATION_DIR
        / "model_complexity.csv"
    )

    pls_component_path = (
        TEST_INTERPRETATION_DIR
        / "pls_components.csv"
    )

    specifications = read_csv(
        specifications_path,
        "Step 5 final test evaluation",
        [
            "model",
            "parameters",
        ],
    )

    coefficients = read_csv(
        coefficient_path,
        "Step 5 final test evaluation",
        [
            "model",
            "predictor",
            "standardized_coefficient",
            "absolute_standardized_coefficient",
            "nonzero",
            "rank",
        ],
    )

    tree_importance = read_csv(
        tree_path,
        "Step 5 final test evaluation",
        [
            "model",
            "predictor",
            "impurity_importance",
            "rank",
        ],
    )

    complexity = read_csv(
        complexity_path,
        "Step 5 final test evaluation",
        [
            "model",
            "number_of_predictors",
        ],
    )

    pls_components = read_csv(
        pls_component_path,
        "Step 5 final test evaluation",
        [
            "component",
            "x_score_variance",
            "x_score_variance_share",
            "cumulative_x_score_variance_share",
        ],
    )

    coefficients = coefficients[
        coefficients["model"].isin(
            [
                "ols_3",
                "pls",
                "elastic_net",
            ]
        )
    ].copy()

    tree_importance = tree_importance[
        tree_importance["model"].isin(
            [
                "random_forest",
                "gradient_boosting",
            ]
        )
    ].copy()

    complexity = complexity[
        complexity["model"].isin(
            [
                "ols_3",
                "pls",
                "elastic_net",
                "random_forest",
                "gradient_boosting",
            ]
        )
    ].copy()

    specifications = specifications[
        specifications["model"].isin(
            ACTIVE_MODELS
        )
    ].copy()

    ensure_unique(
        specifications,
        ["model"],
        "Final model specifications",
    )

    ensure_unique(
        coefficients,
        [
            "model",
            "predictor",
        ],
        "Linear model coefficients",
    )

    ensure_unique(
        tree_importance,
        [
            "model",
            "predictor",
        ],
        "Tree feature importance",
    )

    ensure_unique(
        complexity,
        ["model"],
        "Model complexity",
    )

    ensure_unique(
        pls_components,
        ["component"],
        "PLS components",
    )

    expected_coefficient_models = {
        "ols_3",
        "pls",
        "elastic_net",
    }

    expected_tree_models = {
        "random_forest",
        "gradient_boosting",
    }

    expected_complexity_models = {
        "ols_3",
        "pls",
        "elastic_net",
        "random_forest",
        "gradient_boosting",
    }

    missing_coefficient_models = (
        expected_coefficient_models
        - set(coefficients["model"])
    )

    missing_tree_models = (
        expected_tree_models
        - set(tree_importance["model"])
    )

    missing_complexity_models = (
        expected_complexity_models
        - set(complexity["model"])
    )

    if missing_coefficient_models:
        raise ValueError(
            "Coefficient inputs are missing models: "
            f"{sorted(missing_coefficient_models)}"
        )

    if missing_tree_models:
        raise ValueError(
            "Tree-importance inputs are missing models: "
            f"{sorted(missing_tree_models)}"
        )

    if missing_complexity_models:
        raise ValueError(
            "Complexity inputs are missing models: "
            f"{sorted(missing_complexity_models)}"
        )

    coefficients = coefficients.sort_values(
        [
            "model",
            "rank",
        ]
    ).reset_index(drop=True)

    tree_importance = tree_importance.sort_values(
        [
            "model",
            "rank",
        ]
    ).reset_index(drop=True)

    pls_components = pls_components.sort_values(
        "component"
    ).reset_index(drop=True)

    return {
        "specifications": specifications,
        "coefficients": coefficients,
        "tree_importance": tree_importance,
        "complexity": complexity,
        "pls_components": pls_components,
    }


def yearly_prediction_metrics(predictions):
    """Calculate test performance separately by model and year."""
    benchmark = (
        predictions[
            predictions["model"].eq(
                "historical_mean"
            )
        ][
            [
                "ticker",
                "month",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction": "benchmark"
            }
        )
    )

    data = predictions.merge(
        benchmark,
        on=[
            "ticker",
            "month",
        ],
        how="left",
        validate="many_to_one",
    )

    if data["benchmark"].isna().any():
        raise ValueError(
            "Historical-mean benchmark predictions "
            "are incomplete."
        )

    data["year"] = data["month"].dt.year

    rows = []

    for (
        model,
        year,
    ), group in data.groupby(
        [
            "model",
            "year",
        ]
    ):
        realized = group[
            "realized_target"
        ].to_numpy()

        prediction = group[
            "prediction"
        ].to_numpy()

        benchmark_values = group[
            "benchmark"
        ].to_numpy()

        squared_error = (
            realized - prediction
        ) ** 2

        model_sse = squared_error.sum()

        benchmark_sse = np.sum(
            (
                realized
                - benchmark_values
            ) ** 2
        )

        month_model_mse = monthly_mse(
            realized,
            prediction,
            group["month"],
        )

        month_benchmark_mse = monthly_mse(
            realized,
            benchmark_values,
            group["month"],
        )

        prediction_std = np.std(
            prediction
        )

        correlation = (
            np.corrcoef(
                realized,
                prediction,
            )[0, 1]
            if prediction_std > 1e-8
            else np.nan
        )

        pooled_oos_r2 = (
            1 - model_sse / benchmark_sse
            if benchmark_sse > 0
            else np.nan
        )

        monthly_oos_r2 = (
            1
            - month_model_mse
            / month_benchmark_mse
            if month_benchmark_mse > 0
            else np.nan
        )

        rows.append({
            "year": int(year),
            "model": model,
            "observations": len(group),
            "months": group["month"].nunique(),
            "monthly_mse": month_model_mse,
            "monthly_rmse": np.sqrt(
                month_model_mse
            ),
            "pooled_mse": np.mean(
                squared_error
            ),
            "rmse": np.sqrt(
                np.mean(
                    squared_error
                )
            ),
            "mae": np.mean(
                np.abs(
                    realized
                    - prediction
                )
            ),
            "monthly_oos_r2": monthly_oos_r2,
            "oos_r2": pooled_oos_r2,
            "correlation": correlation,
        })

    yearly = pd.DataFrame(
        rows
    ).sort_values(
        [
            "year",
            "monthly_mse",
        ]
    )

    ensure_unique(
        yearly,
        [
            "model",
            "year",
        ],
        "Yearly prediction results",
    )

    return yearly.reset_index(
        drop=True
    )


def add_display_names(data):
    """Add readable model names to an output table."""
    result = data.copy()

    result["display_name"] = (
        result["model"]
        .map(DISPLAY_NAMES)
        .fillna(
            result["model"]
        )
    )

    return result