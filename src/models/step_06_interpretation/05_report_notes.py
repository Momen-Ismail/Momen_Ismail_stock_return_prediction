"""Generate data-driven draft notes for the written report."""

from importlib import import_module
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

utils = import_module(
    "src.models.step_06_interpretation.00_utils"
)


def metric(value):
    """Format a numerical report value."""
    if pd.isna(value):
        return "not defined"

    return f"{value:.6f}"


def integer_metric(value):
    """Format a count without decimal places."""
    if pd.isna(value):
        return "not available"

    return str(int(value))


def parse_boolean(series):
    """Convert CSV boolean values reliably."""
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


def display_name(model):
    """Return the readable name of one model."""
    return utils.DISPLAY_NAMES.get(
        model,
        model,
    )


def main():
    """Create automatically generated interpretation notes."""
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
            "rmse",
            "mae",
            "oos_r2",
            "correlation",
        ],
    )

    interpretation = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "model_interpretation_metrics.csv"
        ),
        "Step 6 file 01",
        [
            "rank",
            "model",
            "prediction_std",
            "prediction_bias",
            "best_test_year",
            "best_test_year_mse",
            "best_test_year_oos_r2",
            "worst_test_year",
            "worst_test_year_mse",
            "worst_test_year_oos_r2",
            "yearly_mse_std",
            "yearly_oos_r2_std",
            "positive_oos_r2_year_share",
            "number_of_years_evaluated",
        ],
    )

    flags = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "model_behavior_flags.csv"
        ),
        "Step 6 file 01",
        [
            "model",
            "beats_historical_mean",
            "positive_oos_r2",
            "constant_or_near_constant_predictions",
            "positive_in_most_test_years",
            "best_prediction_model",
            "worst_prediction_model",
        ],
    )

    complexity = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "final_model_complexity.csv"
        ),
        "Step 6 file 03",
        [
            "model",
            "number_of_predictors",
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

    features = utils.read_csv(
        (
            utils.INTERPRETATION_OUTPUT_DIR
            / "feature_importance_top_variables.csv"
        ),
        "Step 6 file 02",
        [
            "model",
            "importance_type",
            "predictor",
            "signed_value",
            "absolute_value",
            "rank",
        ],
    )

    boolean_columns = [
        "beats_historical_mean",
        "positive_oos_r2",
        "constant_or_near_constant_predictions",
        "positive_in_most_test_years",
        "best_prediction_model",
        "worst_prediction_model",
    ]

    for column in boolean_columns:
        flags[column] = parse_boolean(
            flags[column]
        )

    results = results.sort_values(
        "rank"
    ).reset_index(drop=True)

    best_model = results.iloc[0]

    best_r2 = (
        results
        .sort_values(
            "oos_r2",
            ascending=False,
        )
        .iloc[0]
    )

    benchmark = (
        results[
            results["model"].eq(
                "historical_mean"
            )
        ]
        .iloc[0]
    )

    notes = [
        "# Main Prediction Results",
        "",
        (
            f"- The best final-test model is "
            f"{display_name(best_model['model'])}, ranked first "
            f"with monthly MSE {metric(best_model['monthly_mse'])}, "
            f"RMSE {metric(best_model['rmse'])}, and OOS R-squared "
            f"{metric(best_model['oos_r2'])}."
        ),
        (
            f"- The highest final-test OOS R-squared is produced by "
            f"{display_name(best_r2['model'])}: "
            f"{metric(best_r2['oos_r2'])}."
        ),
        (
            f"- The historical-mean benchmark has monthly MSE "
            f"{metric(benchmark['monthly_mse'])}, RMSE "
            f"{metric(benchmark['rmse'])}, and OOS R-squared "
            f"{metric(benchmark['oos_r2'])}."
        ),
    ]

    for row in results.itertuples():
        if row.model == "historical_mean":
            continue

        if row.monthly_mse < benchmark.monthly_mse:
            comparison = "lower"
        elif row.monthly_mse > benchmark.monthly_mse:
            comparison = "higher"
        else:
            comparison = "approximately equal"

        notes.append(
            f"- {display_name(row.model)} has {comparison} monthly MSE "
            f"than the historical-mean benchmark. Its monthly MSE is "
            f"{metric(row.monthly_mse)}, OOS R-squared is "
            f"{metric(row.oos_r2)}, and prediction-target correlation is "
            f"{metric(row.correlation)}."
        )

    notes.extend([
        "",
        "# Model Interpretation",
        "",
    ])

    merged = (
        interpretation
        .merge(
            flags,
            on="model",
            how="left",
            validate="one_to_one",
        )
        .merge(
            complexity,
            on="model",
            how="left",
            validate="one_to_one",
        )
        .sort_values("rank")
    )

    for row in merged.itertuples():
        name = display_name(row.model)

        if row.model == "historical_mean":
            notes.append(
                f"- {name} is the constant benchmark used to calculate "
                "out-of-sample R-squared."
            )

        elif row.model == "ols_3":
            notes.append(
                f"- {name} uses three economically motivated predictors. "
                f"It achieves OOS R-squared {metric(row.oos_r2)} and has "
                f"prediction standard deviation {metric(row.prediction_std)}."
            )

        elif row.model == "pls":
            notes.append(
                f"- {name} reduces the high-dimensional predictor set to "
                f"{integer_metric(getattr(row, 'pls_components', float('nan')))} "
                f"components. It achieves OOS R-squared "
                f"{metric(row.oos_r2)}. Its coefficients describe predictive "
                "associations rather than causal effects."
            )

        elif row.model == "elastic_net":
            nonzero = getattr(
                row,
                "nonzero_coefficients",
                float("nan"),
            )

            notes.append(
                f"- {name} retains "
                f"{integer_metric(nonzero)} coefficients above the documented "
                f"tolerance. Its prediction standard deviation is "
                f"{metric(row.prediction_std)}."
            )

            if row.constant_or_near_constant_predictions:
                notes.append(
                    f"- {name} is flagged as producing constant or "
                    "near-constant predictions. This explains why its "
                    "prediction-target correlation is not defined."
                )

        elif row.model == "decision_tree":
            depth = getattr(
                row,
                "tree_depth",
                float("nan"),
            )

            leaves = getattr(
                row,
                "tree_leaves",
                float("nan"),
            )

            notes.append(
                f"- {name} has depth {integer_metric(depth)} and "
                f"{integer_metric(leaves)} terminal leaves. Its negative "
                f"OOS R-squared of {metric(row.oos_r2)} indicates that it "
                "performs worse than the historical-mean benchmark."
            )

        elif row.model == "random_forest":
            trees = getattr(
                row,
                "forest_trees",
                float("nan"),
            )

            average_depth = getattr(
                row,
                "average_tree_depth",
                float("nan"),
            )

            notes.append(
                f"- {name} contains {integer_metric(trees)} trees with "
                f"average depth {metric(average_depth)}. It can represent "
                "nonlinearities and interactions, but its feature importance "
                "does not identify causal mechanisms."
            )

    notes.extend([
        "",
        "# Stability Over Time",
        "",
    ])

    for row in interpretation.sort_values(
        "rank"
    ).itertuples():

        notes.append(
            f"- {display_name(row.model)} performs best in "
            f"{int(row.best_test_year)}, when its monthly MSE is "
            f"{metric(row.best_test_year_mse)} and yearly OOS R-squared is "
            f"{metric(row.best_test_year_oos_r2)}."
        )

        notes.append(
            f"- {display_name(row.model)} performs worst in "
            f"{int(row.worst_test_year)}, when its monthly MSE is "
            f"{metric(row.worst_test_year_mse)} and yearly OOS R-squared is "
            f"{metric(row.worst_test_year_oos_r2)}."
        )

        notes.append(
            f"- Across {int(row.number_of_years_evaluated)} test years, "
            f"{display_name(row.model)} has yearly MSE standard deviation "
            f"{metric(row.yearly_mse_std)} and positive yearly OOS R-squared "
            f"in {row.positive_oos_r2_year_share:.1%} of years."
        )

    yearly_winners = (
        yearly.loc[
            yearly
            .groupby("year")["monthly_mse"]
            .idxmin(),
            "model",
        ]
        .value_counts()
    )

    winner_text = ", ".join(
        f"{display_name(model)}: {count} year(s)"
        for model, count in yearly_winners.items()
    )

    notes.append(
        f"- The yearly MSE winners are distributed as follows: "
        f"{winner_text}."
    )

    notes.extend([
        "",
        "# Feature Interpretation",
        "",
    ])

    for (
        model,
        importance_type,
    ), group in features.groupby(
        [
            "model",
            "importance_type",
        ]
    ):
        predictors = ", ".join(
            group
            .nsmallest(
                5,
                "rank",
            )["predictor"]
        )

        readable_type = (
            importance_type
            .replace("_", " ")
        )

        notes.append(
            f"- The top {readable_type} variables for "
            f"{display_name(model)} include: {predictors}."
        )

    notes.append(
        "- Linear-model coefficients describe conditional predictive "
        "associations after preprocessing and scaling. Their magnitudes "
        "should not be interpreted as causal effects."
    )

    notes.append(
        "- Tree impurity importance indicates how often and how effectively "
        "a predictor reduces squared error inside the fitted trees. It may "
        "favor variables with many possible split points and does not measure "
        "causal importance."
    )

    notes.extend([
        "",
        "# Suggested Report Tables and Figures",
        "",
        "- `final_report_results.xlsx` contains the main report-ready tables.",
        "- `final_prediction_results.csv` presents the final test ranking.",
        "- `fixed_vs_optimized_results.csv` compares fixed and optimized "
        "validation performance.",
        "- `yearly_prediction_results.csv` presents performance by test year.",
        "- `test_monthly_mse_by_model.png` and "
        "`test_oos_r2_by_model.png` summarize overall predictive performance.",
        "- `yearly_mse_by_model.png` and "
        "`yearly_oos_r2_by_model.png` show stability over time.",
        "- The prediction-versus-realized figure illustrates the limited "
        "dispersion and predictive strength of the best model.",
        "- `top_random_forest_features.png`, "
        "`top_decision_tree_features.png`, and "
        "`top_pls_coefficients.png` summarize model interpretation.",
        "",
        "# Limitations",
        "",
        "- Monthly stock returns are noisy and inherently difficult to predict.",
        "- The positive OOS R-squared values are economically and statistically "
        "small and should not be overstated.",
        "- Predictor-return relationships may change across market regimes.",
        "- The final test period covers only 2020–2025 and includes unusual "
        "market conditions such as the COVID-19 shock and subsequent inflation "
        "and interest-rate changes.",
        "- The stock universe is based on the available S&P 500 constituent "
        "history and may retain survivorship or membership-history limitations.",
        "- Yahoo Finance, Compustat, FRED, Fama–French, and Welch–Goyal inputs "
        "may contain coverage, measurement, timing, or revision limitations.",
        "- Impurity importance can be biased toward predictors with more "
        "possible split points.",
        "- Coefficients and feature importances describe prediction, not causality.",
        "- The test sample must remain untouched after this final evaluation; "
        "the reported results should not be used to retune the models.",
    ])

    output = (
        utils.INTERPRETATION_OUTPUT_DIR
        / "report_notes.md"
    )

    output.write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved {output}"
    )


if __name__ == "__main__":
    main()