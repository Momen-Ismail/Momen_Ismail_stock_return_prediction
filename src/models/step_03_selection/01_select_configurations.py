"""Select each family's exact minimum 15-fold monthly MSE candidate."""

from ast import literal_eval
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLEAN_PREDICTOR_FILE,
    SELECTION_OUTPUT_DIR,
    TUNING_OUTPUT_DIR,
)
from src.models.utils.model_selection import (  # noqa: E402
    CANDIDATE_CONFIGURATIONS,
    FIXED_CONFIGURATIONS,
    VALIDATION_YEARS,
    complete_parameters,
    normalize_parameters,
    parameters_equal,
)


def parse_parameters(value):
    """Parse a saved candidate dictionary."""
    return normalize_parameters(literal_eval(value))


def main():
    """Validate tuning completeness, select exact minima, and save metadata."""
    results_file = TUNING_OUTPUT_DIR / "tuning_all_results.csv"
    folds_file = TUNING_OUTPUT_DIR / "tuning_fold_results.csv"
    if not results_file.exists() or not folds_file.exists():
        raise FileNotFoundError("Run all official Step 02 scripts first.")

    results = pd.read_csv(results_file)
    folds = pd.read_csv(folds_file)
    selected_rows = []
    comparison_rows = []

    for family, candidates in CANDIDATE_CONFIGURATIONS.items():
        family_results = results[results["model_family"].eq(family)].copy()
        family_folds = folds[folds["model_family"].eq(family)].copy()
        if len(family_results) != len(candidates):
            raise ValueError(f"{family}: expected {len(candidates)} candidates.")
        expected_ids = {
            f"{family}_{number:03d}"
            for number in range(1, len(candidates) + 1)
        }
        if set(family_results["candidate_id"]) != expected_ids:
            raise ValueError(f"{family}: candidate IDs are incomplete.")
        if set(family_folds["candidate_id"]) != expected_ids:
            raise ValueError(f"{family}: fold candidate IDs are incomplete.")
        coverage = family_folds.groupby("candidate_id")["validation_year"].agg(list)
        if any(sorted(years) != VALIDATION_YEARS for years in coverage):
            raise ValueError(f"{family}: folds are not exactly 2005--2019.")
        if (pd.to_datetime(family_folds["train_end"]).dt.year
                >= family_folds["validation_year"]).any():
            raise ValueError(f"{family}: validation observations enter training.")

        family_results = family_results.sort_values("candidate_id")
        minimum = family_results["average_monthly_mse"].min()
        winner = family_results[
            family_results["average_monthly_mse"].eq(minimum)
        ].iloc[0].copy()
        if not np.isclose(
            winner["average_monthly_mse"], minimum, rtol=0, atol=0
        ):
            raise ValueError(f"{family}: selected value is not the exact minimum.")
        winner["selection_rule"] = (
            "minimum unrounded average monthly MSE across 15 folds"
        )
        selected_rows.append(winner)

        fixed = FIXED_CONFIGURATIONS[family]
        fixed_rows = family_results[
            family_results["parameters"].map(
                lambda value: parameters_equal(parse_parameters(value), fixed)
            )
        ]
        if len(fixed_rows) != 1:
            raise ValueError(f"{family}: fixed configuration is not unique.")
        fixed_row = fixed_rows.iloc[0]
        selected_parameters = parse_parameters(winner["parameters"])
        comparison_rows.append({
            "model_family": family,
            "fixed_parameters": repr(complete_parameters(family, fixed)),
            "fixed_average_monthly_mse": fixed_row["average_monthly_mse"],
            "selected_parameters": repr(
                complete_parameters(family, selected_parameters)
            ),
            "selected_average_monthly_mse": winner["average_monthly_mse"],
            "selected_is_original_fixed": parameters_equal(
                selected_parameters, fixed
            ),
            "mse_improvement_over_fixed": (
                fixed_row["average_monthly_mse"]
                - winner["average_monthly_mse"]
            ),
            "comparison_role": "descriptive_only",
        })

    selected = pd.DataFrame(selected_rows)
    summary = results.copy()
    summary["selected"] = summary["candidate_id"].isin(selected["candidate_id"])
    summary["within_family_rank"] = summary.groupby("model_family")[
        "average_monthly_mse"
    ].rank(method="first").astype(int)
    comparison = pd.DataFrame(comparison_rows)

    SELECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        SELECTION_OUTPUT_DIR / "selected_parameters.csv", index=False
    )
    summary.sort_values(["model_family", "within_family_rank"]).to_csv(
        SELECTION_OUTPUT_DIR / "complete_tuning_summary.csv", index=False
    )
    comparison.to_csv(
        SELECTION_OUTPUT_DIR / "fixed_vs_selected_summary.csv", index=False
    )

    predictor_count = len(pd.read_csv(CLEAN_PREDICTOR_FILE))
    if predictor_count != 484:
        raise ValueError(f"Expected 484 predictors; found {predictor_count}.")
    metadata = {
        "selection_rule": "exact minimum unrounded average monthly MSE",
        "tie_break": "deterministic candidate grid order for exact ties only",
        "development_period": ["1990-02-28", "2019-12-31"],
        "validation_years": VALIDATION_YEARS,
        "fold_count": 15,
        "predictor_count": predictor_count,
        "test_used_in_selection": False,
        "selected": {
            row.model_family: parse_parameters(row.parameters)
            for row in selected.itertuples()
        },
    }
    (SELECTION_OUTPUT_DIR / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(selected[[
        "model_family", "parameters", "average_monthly_mse", "completed_folds"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
