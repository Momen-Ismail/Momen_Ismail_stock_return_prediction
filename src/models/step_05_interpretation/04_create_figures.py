"""Create report figures from saved final-test interpretation outputs."""

from importlib import import_module
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".matplotlib"),
)

sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


utils = import_module(
    "src.models.step_05_interpretation.00_utils"
)


TOP_FEATURES = 15
SCATTER_SAMPLE_SIZE = 5_000
RANDOM_STATE = 42


def save_figure(filename):
    """Save the current figure and close it."""
    utils.FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        utils.FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def model_names(models):
    """Convert internal model identifiers into readable names."""
    return [
        utils.DISPLAY_NAMES.get(
            model,
            model,
        )
        for model in models
    ]


def parse_boolean(series):
    """Convert saved CSV boolean values to Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
        .astype(bool)
    )


def model_bar_chart(
    data,
    value_column,
    filename,
    title,
    ylabel,
    ascending=True,
    zero_line=False,
):
    """Create one model-comparison bar chart."""
    plot = (
        data.dropna(
            subset=[value_column]
        )
        .sort_values(
            value_column,
            ascending=ascending,
        )
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        model_names(plot["model"]),
        plot[value_column],
    )

    if zero_line:
        plt.axhline(
            0,
            linewidth=1,
        )

    plt.title(title)
    plt.ylabel(ylabel)

    plt.xticks(
        rotation=30,
        ha="right",
    )

    save_figure(filename)


def yearly_performance_chart(yearly):
    """Plot yearly monthly OOS R-squared for every model."""
    plt.figure(figsize=(9, 5))

    for model, group in yearly.groupby("model"):
        group = group.sort_values("year")

        plt.plot(
            group["year"],
            group["monthly_oos_r2"],
            marker="o",
            label=utils.DISPLAY_NAMES.get(
                model,
                model,
            ),
        )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title(
        "Yearly Final-Test Monthly OOS R-squared"
    )

    plt.xlabel("Year")
    plt.ylabel("Monthly OOS R-squared")
    plt.legend()

    save_figure(
        "yearly_monthly_oos_r2_by_model.png"
    )


def best_model_scatter(results, predictions):
    """Plot realized against predicted returns for the best fitted model."""
    fitted_results = results[
        ~results["model"].eq(
            "historical_mean"
        )
    ].sort_values("rank")

    best_model = fitted_results.iloc[0][
        "model"
    ]

    plot = predictions[
        predictions["model"].eq(
            best_model
        )
    ].copy()

    sample_size = min(
        SCATTER_SAMPLE_SIZE,
        len(plot),
    )

    plot = plot.sample(
        sample_size,
        random_state=RANDOM_STATE,
    )

    lower = min(
        plot["realized_target"].min(),
        plot["prediction"].min(),
    )

    upper = max(
        plot["realized_target"].max(),
        plot["prediction"].max(),
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        plot["realized_target"],
        plot["prediction"],
        s=8,
        alpha=0.35,
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linewidth=1,
    )

    plt.title(
        "Predicted versus Realized Returns: "
        f"{utils.DISPLAY_NAMES.get(best_model, best_model)}"
    )

    plt.xlabel("Realized excess return")
    plt.ylabel("Predicted excess return")

    save_figure(
        "best_model_prediction_vs_realized.png"
    )


def tree_importance_chart(
    filename,
    model_name,
    output_filename,
):
    """Plot the leading predictors from one tree ensemble."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / filename
    )

    data = utils.read_csv(
        path,
        "Step 6 file 02",
        [
            "predictor",
            "impurity_importance",
            "rank",
        ],
    )

    plot = (
        data.nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "impurity_importance"
        )
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        plot["predictor"],
        plot["impurity_importance"],
    )

    plt.title(
        f"Top {model_name} Predictors"
    )

    plt.xlabel(
        "Impurity-based importance"
    )

    save_figure(output_filename)


def coefficient_chart(
    filename,
    model_name,
    output_filename,
    nonzero_only=False,
):
    """Plot leading standardized coefficients from one linear model."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / filename
    )

    required_columns = [
        "predictor",
        "standardized_coefficient",
        "absolute_standardized_coefficient",
        "rank",
    ]

    if nonzero_only:
        required_columns.append(
            "nonzero"
        )

    data = utils.read_csv(
        path,
        "Step 6 file 02",
        required_columns,
    )

    if nonzero_only:
        data["nonzero"] = parse_boolean(
            data["nonzero"]
        )

        data = data[
            data["nonzero"]
        ].copy()

    if data.empty:
        print(
            f"No nonzero coefficients found for {model_name}; "
            "coefficient figure skipped."
        )
        return

    plot = (
        data.nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "standardized_coefficient"
        )
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        plot["predictor"],
        plot["standardized_coefficient"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        f"Top {model_name} Standardized Coefficients"
    )

    plt.xlabel(
        "Standardized coefficient"
    )

    save_figure(output_filename)


def predictor_group_charts():
    """Plot predictor-family importance for the tree models."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "final_predictor_group_summary.csv"
    )

    data = utils.read_csv(
        path,
        "Step 6 file 03",
        [
            "model",
            "predictor_group",
            "total_importance",
        ],
    )

    tree_models = [
        "random_forest",
        "gradient_boosting",
    ]

    for model in tree_models:
        plot = data[
            data["model"].eq(model)
        ].sort_values(
            "total_importance"
        )

        if plot.empty:
            continue

        plt.figure(figsize=(8, 5))

        plt.barh(
            plot["predictor_group"],
            plot["total_importance"],
        )

        plt.title(
            "Predictor-Group Importance: "
            f"{utils.DISPLAY_NAMES[model]}"
        )

        plt.xlabel("Total importance")

        save_figure(
            f"{model}_predictor_group_importance.png"
        )


def main():
    """Create the final set of report figures."""
    results = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "final_prediction_results.csv"
        ),
        "Step 6 file 03",
        [
            "rank",
            "model",
            "monthly_mse",
            "monthly_oos_r2",
            "correlation",
        ],
    )

    yearly = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "yearly_prediction_results.csv"
        ),
        "Step 6 file 01",
        [
            "year",
            "model",
            "monthly_oos_r2",
        ],
    )

    _, _, predictions = (
        utils.load_test_outputs()
    )

    model_bar_chart(
        results,
        value_column="monthly_mse",
        filename="test_monthly_mse_by_model.png",
        title="Final-Test Monthly MSE",
        ylabel="Monthly MSE",
        ascending=True,
    )

    model_bar_chart(
        results,
        value_column="monthly_oos_r2",
        filename="test_monthly_oos_r2_by_model.png",
        title="Final-Test Monthly OOS R-squared",
        ylabel="Monthly OOS R-squared",
        ascending=False,
        zero_line=True,
    )

    model_bar_chart(
        results,
        value_column="correlation",
        filename="test_prediction_correlation_by_model.png",
        title="Final-Test Prediction-Target Correlation",
        ylabel="Correlation",
        ascending=False,
        zero_line=True,
    )

    yearly_performance_chart(
        yearly
    )

    best_model_scatter(
        results,
        predictions,
    )

    tree_importance_chart(
        filename="random_forest_feature_importance.csv",
        model_name="Random Forest",
        output_filename="top_random_forest_features.png",
    )

    tree_importance_chart(
        filename="gradient_boosting_feature_importance.csv",
        model_name="Gradient Boosting",
        output_filename="top_gradient_boosting_features.png",
    )

    coefficient_chart(
        filename="pls_coefficients.csv",
        model_name="PLS",
        output_filename="top_pls_coefficients.png",
    )

    coefficient_chart(
        filename="elastic_net_coefficients.csv",
        model_name="Elastic Net",
        output_filename="top_elastic_net_coefficients.png",
        nonzero_only=True,
    )

    predictor_group_charts()

    print("\nSaved interpretation figures to:")
    print(utils.FIGURE_DIR)


if __name__ == "__main__":
    main()