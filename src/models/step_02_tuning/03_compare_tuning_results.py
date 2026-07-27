"""Combine the saved tuning results and selected parameters."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402


OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"

FAMILIES = [
    "pls",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
]


def main():
    results = []
    selected = []

    for family in FAMILIES:
        results_file = (
            OUTPUT_DIR
            / f"{family}_tuning_results.csv"
        )

        parameters_file = (
            OUTPUT_DIR
            / f"{family}_best_parameters.csv"
        )

        if not results_file.exists():
            raise FileNotFoundError(
                f"Missing tuning results: {results_file}"
            )

        if not parameters_file.exists():
            raise FileNotFoundError(
                f"Missing selected parameters: {parameters_file}"
            )

        family_results = pd.read_csv(
            results_file
        )

        family_selected = pd.read_csv(
            parameters_file
        )

        if (
            "sample" in family_results.columns
            and family_results["sample"].eq("test").any()
        ):
            raise ValueError(
                f"{family} tuning results contain test rows."
            )

        if (
            set(family_results["model_family"].unique())
            != {family}
        ):
            raise ValueError(
                f"Unexpected model family in {results_file}."
            )

        if (
            set(family_selected["model_family"].unique())
            != {family}
        ):
            raise ValueError(
                f"Unexpected model family in {parameters_file}."
            )

        results.append(
            family_results
        )

        selected.append(
            family_selected
        )

    all_results = pd.concat(
        results,
        ignore_index=True,
    )

    all_results = all_results.sort_values(
        [
            "model_family",
            "cv_monthly_mse",
        ]
    ).reset_index(drop=True)

    selected_parameters = pd.concat(
        selected,
        ignore_index=True,
    )

    summary = selected_parameters.merge(
        all_results,
        on=[
            "model_family",
            "parameters",
        ],
        how="left",
        validate="one_to_one",
    )

    if summary["cv_monthly_mse"].isna().any():
        missing = summary.loc[
            summary["cv_monthly_mse"].isna(),
            "model_family",
        ].tolist()

        raise ValueError(
            "Selected parameters were not found in the "
            f"tuning results for: {missing}"
        )

    all_results.to_csv(
        OUTPUT_DIR / "tuning_all_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "tuning_summary.csv",
        index=False,
    )

    display_columns = [
        "model_family",
        "parameters",
        "cv_monthly_mse",
    ]

    for optional_column in [
        "cv_monthly_mse_std",
        "cv_monthly_mse_se",
        "cv_folds",
    ]:
        if optional_column in summary.columns:
            display_columns.append(
                optional_column
            )

    print("\nSelected tuning parameters:")
    print(
        summary[
            display_columns
        ].sort_values(
            "cv_monthly_mse"
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.12f}",
        )
    )

    print(
        "\nSaved combined tuning results:"
    )
    print(
        OUTPUT_DIR / "tuning_all_results.csv"
    )
    print(
        OUTPUT_DIR / "tuning_summary.csv"
    )


if __name__ == "__main__":
    main()