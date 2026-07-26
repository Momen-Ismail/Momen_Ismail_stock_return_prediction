"""Create reproducible figures from saved interpretation outputs."""

from importlib import import_module
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".matplotlib"),
)

sys.path.append(str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


utils = import_module(
    "src.models.step_06_interpretation.00_utils"
)


RANDOM_STATE = 42
SCATTER_SAMPLE_SIZE = 5_000
TOP_FEATURES = 20


def save_figure(filename):
    """Save and close the current Matplotlib figure."""
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


def display_names(models):
    """Return readable names for a sequence of model identifiers."""
    return [
        utils.DISPLAY_NAMES.get(
            model,
            model,
        )
        for model in models
    ]


def safe_filename(value):
    """Convert a model name into a safe filename component."""
    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(value),
    ).strip("_").lower()


def parse_boolean(series):
    """Convert CSV boolean values reliably to True and False."""
    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
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


def bar_figure(
    data,
    value,
    filename,
    title,
    ascending=True,
    zero_line=False,
):
    """Create one sorted model-comparison bar chart."""
    plot = (
        data
        .dropna(subset=[value])
        .sort_values(
            value,
            ascending=ascending,
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        display_names(plot["model"]),
        plot[value],
    )

    if zero_line:
        plt.axhline(
            0,
            linewidth=1,
        )

    plt.title(title)

    plt.ylabel(
        value
        .replace("_", " ")
        .title()
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    save_figure(filename)


def yearly_lines(
    yearly,
    value,
    filename,
    title,
    zero_line=False,
):
    """Create one yearly model-performance line chart."""
    plt.figure(
        figsize=(9, 5)
    )

    for model, group in yearly.groupby(
        "model"
    ):
        group = group.sort_values(
            "year"
        )

        plt.plot(
            group["year"],
            group[value],
            marker="o",
            label=utils.DISPLAY_NAMES.get(
                model,
                model,
            ),
        )

    if zero_line:
        plt.axhline(
            0,
            linewidth=1,
        )

    plt.title(title)
    plt.xlabel("Year")

    plt.ylabel(
        value
        .replace("_", " ")
        .title()
    )

    plt.legend()

    save_figure(filename)


def prediction_scatter(
    results,
    predictions,
):
    """Create realized-versus-predicted scatter for the best model."""
    best_model = (
        results
        .sort_values("rank")
        .iloc[0]["model"]
    )

    scatter = predictions[
        predictions["model"].eq(
            best_model
        )
    ].copy()

    sample_size = min(
        SCATTER_SAMPLE_SIZE,
        len(scatter),
    )

    scatter = scatter.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    lower_limit = min(
        scatter["realized_target"].min(),
        scatter["prediction"].min(),
    )

    upper_limit = max(
        scatter["realized_target"].max(),
        scatter["prediction"].max(),
    )

    limits = [
        lower_limit,
        upper_limit,
    ]

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        scatter["realized_target"],
        scatter["prediction"],
        s=8,
        alpha=0.35,
    )

    plt.plot(
        limits,
        limits,
        linewidth=1,
    )

    plt.title(
        "Prediction vs. Realized: "
        f"{utils.DISPLAY_NAMES.get(best_model, best_model)} "
        f"(n={sample_size:,})"
    )

    plt.xlabel(
        "Realized excess return"
    )

    plt.ylabel(
        "Predicted excess return"
    )

    filename = (
        "prediction_vs_realized_"
        f"{safe_filename(best_model)}.png"
    )

    save_figure(filename)


def prediction_distributions(predictions):
    """Compare the prediction distributions of fitted models."""
    distribution = predictions[
        ~predictions["model"].eq(
            "historical_mean"
        )
    ].copy()

    lower = distribution[
        "prediction"
    ].quantile(0.005)

    upper = distribution[
        "prediction"
    ].quantile(0.995)

    bins = np.linspace(
        lower,
        upper,
        50,
    )

    plt.figure(
        figsize=(9, 5)
    )

    for model, group in distribution.groupby(
        "model"
    ):
        plt.hist(
            group["prediction"],
            bins=bins,
            alpha=0.4,
            label=utils.DISPLAY_NAMES.get(
                model,
                model,
            ),
        )

    plt.title(
        "Final-Test Prediction Distributions"
    )

    plt.xlabel(
        "Predicted excess return"
    )

    plt.ylabel(
        "Observations"
    )

    plt.legend()

    save_figure(
        "prediction_distribution_by_model.png"
    )


def random_forest_importance_figure():
    """Plot the most important Random Forest predictors."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "random_forest_feature_importance.csv"
    )

    forest = utils.read_csv(
        path,
        "Step 6 file 02",
        [
            "predictor",
            "impurity_importance",
            "rank",
        ],
    )

    forest = (
        forest
        .nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "impurity_importance"
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.barh(
        forest["predictor"],
        forest["impurity_importance"],
    )

    plt.title(
        "Top Random Forest Features"
    )

    plt.xlabel(
        "Impurity-based importance"
    )

    save_figure(
        "top_random_forest_features.png"
    )


def decision_tree_importance_figure():
    """Plot the most important Decision Tree predictors."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "decision_tree_feature_importance.csv"
    )

    tree = utils.read_csv(
        path,
        "Step 6 file 02",
        [
            "predictor",
            "impurity_importance",
            "rank",
        ],
    )

    tree = (
        tree
        .nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "impurity_importance"
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.barh(
        tree["predictor"],
        tree["impurity_importance"],
    )

    plt.title(
        "Top Decision Tree Features"
    )

    plt.xlabel(
        "Impurity-based importance"
    )

    save_figure(
        "top_decision_tree_features.png"
    )


def pls_coefficient_figure():
    """Plot the largest PLS coefficients by absolute magnitude."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "pls_coefficients.csv"
    )

    pls = utils.read_csv(
        path,
        "Step 6 file 02",
        [
            "predictor",
            "coefficient",
            "rank",
        ],
    )

    pls = (
        pls
        .nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.barh(
        pls["predictor"],
        pls["coefficient"],
    )

    plt.title(
        "Top PLS Coefficients by Absolute Magnitude"
    )

    plt.xlabel(
        "Scaled coefficient"
    )

    save_figure(
        "top_pls_coefficients.png"
    )


def elastic_net_coefficient_figure():
    """Plot nonzero Elastic Net coefficients or save an explanatory note."""
    path = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "elastic_net_coefficients.csv"
    )

    elastic_net = utils.read_csv(
        path,
        "Step 6 file 02",
        [
            "predictor",
            "coefficient",
            "nonzero",
            "rank",
        ],
    )

    elastic_net["nonzero"] = parse_boolean(
        elastic_net["nonzero"]
    )

    nonzero = elastic_net[
        elastic_net["nonzero"]
    ].copy()

    utils.FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        utils.FIGURE_DIR
        / "top_elastic_net_coefficients.png"
    )

    note_path = (
        utils.FIGURE_DIR
        / "top_elastic_net_coefficients_skipped.txt"
    )

    if nonzero.empty:
        plot_path.unlink(
            missing_ok=True
        )

        note_path.write_text(
            "The optimized Elastic Net has no coefficients "
            "above the documented coefficient tolerance. "
            "It therefore produces constant or near-constant "
            "predictions, and no coefficient chart was created.\n",
            encoding="utf-8",
        )

        return

    note_path.unlink(
        missing_ok=True
    )

    plot = (
        nonzero
        .nsmallest(
            TOP_FEATURES,
            "rank",
        )
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.barh(
        plot["predictor"],
        plot["coefficient"],
    )

    plt.title(
        "Top Nonzero Elastic Net Coefficients"
    )

    plt.xlabel(
        "Scaled coefficient"
    )

    save_figure(
        "top_elastic_net_coefficients.png"
    )


def main():
    """Create all report figures from saved Step 6 outputs."""
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
            "oos_r2",
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
            "monthly_mse",
            "oos_r2",
        ],
    )

    _, _, predictions = (
        utils.load_test_outputs()
    )

    bar_figure(
        data=results,
        value="monthly_mse",
        filename="test_monthly_mse_by_model.png",
        title="Final-Test Monthly MSE",
        ascending=True,
    )

    bar_figure(
        data=results,
        value="oos_r2",
        filename="test_oos_r2_by_model.png",
        title="Final-Test Out-of-Sample R-squared",
        ascending=False,
        zero_line=True,
    )

    bar_figure(
        data=results,
        value="correlation",
        filename="test_correlation_by_model.png",
        title="Final-Test Prediction-Target Correlation",
        ascending=False,
    )

    yearly_lines(
        yearly=yearly,
        value="monthly_mse",
        filename="yearly_mse_by_model.png",
        title="Yearly Final-Test Monthly MSE",
    )

    yearly_lines(
        yearly=yearly,
        value="oos_r2",
        filename="yearly_oos_r2_by_model.png",
        title="Yearly Final-Test Out-of-Sample R-squared",
        zero_line=True,
    )

    prediction_scatter(
        results,
        predictions,
    )

    prediction_distributions(
        predictions
    )

    random_forest_importance_figure()

    decision_tree_importance_figure()

    pls_coefficient_figure()

    elastic_net_coefficient_figure()

    print(
        "Saved figures to:"
    )

    print(
        utils.FIGURE_DIR
    )


if __name__ == "__main__":
    main()