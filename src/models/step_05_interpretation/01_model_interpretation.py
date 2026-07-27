"""Summarize final-test model behavior from saved Step 5 outputs."""

from importlib import import_module
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

utils = import_module(
    "src.models.step_05_interpretation.00_utils"
)


NEAR_CONSTANT_STD = 1e-8
OOS_R2_TOLERANCE = 1e-6
POSITIVE_YEAR_SHARE_THRESHOLD = 0.50


def main():
    """Create model-level and yearly final-test summaries."""
    metrics, ranking, predictions = (
        utils.load_test_outputs()
    )

    yearly = utils.yearly_prediction_metrics(
        predictions
    )

    realized_summary = (
        predictions
        .groupby("model")[
            "realized_target"
        ]
        .agg(
            realized_mean="mean",
            realized_std="std",
        )
        .reset_index()
    )

    yearly_summary = (
        yearly
        .groupby("model")
        .agg(
            yearly_mean_mse=(
                "monthly_mse",
                "mean",
            ),
            yearly_mse_std=(
                "monthly_mse",
                "std",
            ),
            yearly_mean_monthly_oos_r2=(
                "monthly_oos_r2",
                "mean",
            ),
            yearly_monthly_oos_r2_std=(
                "monthly_oos_r2",
                "std",
            ),
            yearly_mean_pooled_oos_r2=(
                "oos_r2",
                "mean",
            ),
            positive_monthly_oos_r2_year_share=(
                "monthly_oos_r2",
                lambda values: (
                    values
                    > OOS_R2_TOLERANCE
                ).mean(),
            ),
            number_of_years_evaluated=(
                "year",
                "count",
            ),
        )
        .reset_index()
    )

    best_year = (
        yearly.loc[
            yearly
            .groupby("model")[
                "monthly_mse"
            ]
            .idxmin(),
            [
                "model",
                "year",
                "monthly_mse",
                "monthly_oos_r2",
                "oos_r2",
            ],
        ]
        .rename(
            columns={
                "year": "best_test_year",
                "monthly_mse": (
                    "best_test_year_mse"
                ),
                "monthly_oos_r2": (
                    "best_test_year_monthly_oos_r2"
                ),
                "oos_r2": (
                    "best_test_year_pooled_oos_r2"
                ),
            }
        )
    )

    worst_year = (
        yearly.loc[
            yearly
            .groupby("model")[
                "monthly_mse"
            ]
            .idxmax(),
            [
                "model",
                "year",
                "monthly_mse",
                "monthly_oos_r2",
                "oos_r2",
            ],
        ]
        .rename(
            columns={
                "year": "worst_test_year",
                "monthly_mse": (
                    "worst_test_year_mse"
                ),
                "monthly_oos_r2": (
                    "worst_test_year_monthly_oos_r2"
                ),
                "oos_r2": (
                    "worst_test_year_pooled_oos_r2"
                ),
            }
        )
    )

    interpretation = (
        metrics[
            [
                "model",
                "sample",
                "monthly_mse",
                "monthly_rmse",
                "monthly_oos_r2",
                "pooled_rmse",
                "pooled_mae",
                "oos_r2",
                "prediction_target_correlation",
                "prediction_mean",
                "prediction_std",
            ]
        ]
        .rename(
            columns={
                "pooled_rmse": "rmse",
                "pooled_mae": "mae",
                "prediction_target_correlation": (
                    "correlation"
                ),
            }
        )
        .merge(
            ranking[
                [
                    "rank",
                    "model",
                ]
            ],
            on="model",
            how="left",
            validate="one_to_one",
        )
        .merge(
            realized_summary,
            on="model",
            how="left",
            validate="one_to_one",
        )
        .merge(
            yearly_summary,
            on="model",
            how="left",
            validate="one_to_one",
        )
        .merge(
            best_year,
            on="model",
            how="left",
            validate="one_to_one",
        )
        .merge(
            worst_year,
            on="model",
            how="left",
            validate="one_to_one",
        )
    )

    interpretation["prediction_bias"] = (
        interpretation["prediction_mean"]
        - interpretation["realized_mean"]
    )

    interpretation = interpretation[
        [
            "rank",
            "model",
            "sample",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "rmse",
            "mae",
            "oos_r2",
            "correlation",
            "prediction_mean",
            "prediction_std",
            "realized_mean",
            "realized_std",
            "prediction_bias",
            "yearly_mean_mse",
            "yearly_mse_std",
            "yearly_mean_monthly_oos_r2",
            "yearly_monthly_oos_r2_std",
            "yearly_mean_pooled_oos_r2",
            "positive_monthly_oos_r2_year_share",
            "best_test_year",
            "best_test_year_mse",
            "best_test_year_monthly_oos_r2",
            "best_test_year_pooled_oos_r2",
            "worst_test_year",
            "worst_test_year_mse",
            "worst_test_year_monthly_oos_r2",
            "worst_test_year_pooled_oos_r2",
            "number_of_years_evaluated",
        ]
    ].sort_values(
        "rank"
    ).reset_index(drop=True)

    benchmark_mse = interpretation.loc[
        interpretation["model"].eq(
            "historical_mean"
        ),
        "monthly_mse",
    ].iloc[0]

    best_model = (
        interpretation
        .sort_values("rank")
        .iloc[0]["model"]
    )

    worst_model = (
        interpretation
        .sort_values("rank")
        .iloc[-1]["model"]
    )

    flags = pd.DataFrame({
        "model": interpretation["model"],

        "beats_historical_mean": (
            interpretation["monthly_mse"]
            < benchmark_mse
        ),

        "positive_monthly_oos_r2": (
            interpretation["monthly_oos_r2"]
            > OOS_R2_TOLERANCE
        ),

        "positive_pooled_oos_r2": (
            interpretation["oos_r2"]
            > OOS_R2_TOLERANCE
        ),

        "constant_or_near_constant_predictions": (
            interpretation["prediction_std"]
            <= NEAR_CONSTANT_STD
        ),

        "positive_in_most_test_years": (
            interpretation[
                "positive_monthly_oos_r2_year_share"
            ]
            >= POSITIVE_YEAR_SHARE_THRESHOLD
        ),

        "best_prediction_model": (
            interpretation["model"].eq(
                best_model
            )
        ),

        "worst_prediction_model": (
            interpretation["model"].eq(
                worst_model
            )
        ),

        "near_constant_std_threshold": (
            NEAR_CONSTANT_STD
        ),

        "oos_r2_tolerance": (
            OOS_R2_TOLERANCE
        ),

        "positive_year_share_threshold": (
            POSITIVE_YEAR_SHARE_THRESHOLD
        ),
    })

    yearly = yearly.merge(
        ranking[
            [
                "rank",
                "model",
            ]
        ],
        on="model",
        how="left",
        validate="many_to_one",
    )

    yearly = yearly[
        [
            "rank",
            "year",
            "model",
            "observations",
            "months",
            "monthly_mse",
            "monthly_rmse",
            "monthly_oos_r2",
            "rmse",
            "mae",
            "oos_r2",
            "correlation",
        ]
    ].sort_values(
        [
            "year",
            "rank",
        ]
    ).reset_index(drop=True)

    utils.ensure_unique(
        interpretation,
        ["model"],
        "Model interpretation",
    )

    utils.ensure_unique(
        flags,
        ["model"],
        "Model behavior flags",
    )

    utils.ensure_unique(
        yearly,
        [
            "model",
            "year",
        ],
        "Yearly prediction results",
    )

    utils.save_csv(
        interpretation,
        "model_interpretation_metrics.csv",
    )

    utils.save_csv(
        flags,
        "model_behavior_flags.csv",
    )

    utils.save_csv(
        yearly,
        "yearly_prediction_results.csv",
    )

    print("\nFinal-test model interpretation:")

    print(
        interpretation[
            [
                "rank",
                "model",
                "monthly_mse",
                "monthly_oos_r2",
                "oos_r2",
                "correlation",
                "positive_monthly_oos_r2_year_share",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()