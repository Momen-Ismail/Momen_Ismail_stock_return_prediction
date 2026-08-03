"""Validate and combine the four official tuning checkpoints."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TUNING_OUTPUT_DIR  # noqa: E402
from src.models.utils.model_selection import (  # noqa: E402
    CANDIDATE_CONFIGURATIONS,
    VALIDATION_YEARS,
)


def main():
    """Combine results only after candidate and fold checks pass."""
    all_folds = []
    all_results = []
    for family, candidates in CANDIDATE_CONFIGURATIONS.items():
        fold_file = TUNING_OUTPUT_DIR / f"{family}_tuning_fold_results.csv"
        result_file = TUNING_OUTPUT_DIR / f"{family}_tuning_results.csv"
        if not fold_file.exists() or not result_file.exists():
            raise FileNotFoundError(f"Missing official tuning output for {family}.")
        folds = pd.read_csv(fold_file)
        results = pd.read_csv(result_file)
        if len(results) != len(candidates):
            raise ValueError(f"Unexpected candidate count for {family}.")
        counts = folds.groupby("candidate_id")["validation_year"].agg(
            ["count", "nunique", "min", "max"]
        )
        expected_ids = {
            f"{family}_{number:03d}"
            for number in range(1, len(candidates) + 1)
        }
        if (
            set(results["candidate_id"]) != expected_ids
            or set(counts.index) != expected_ids
        ):
            raise ValueError(f"Unexpected candidate IDs for {family}.")
        if (counts["count"].ne(15).any()
                or counts["nunique"].ne(15).any()
                or counts["min"].ne(VALIDATION_YEARS[0]).any()
                or counts["max"].ne(VALIDATION_YEARS[-1]).any()):
            raise ValueError(f"Invalid fold coverage for {family}.")
        if (pd.to_datetime(folds["train_end"]).dt.year
                >= folds["validation_year"]).any():
            raise ValueError(f"Training/validation leakage for {family}.")
        all_folds.append(folds)
        all_results.append(results)

    pd.concat(all_folds, ignore_index=True).to_csv(
        TUNING_OUTPUT_DIR / "tuning_fold_results.csv", index=False
    )
    pd.concat(all_results, ignore_index=True).to_csv(
        TUNING_OUTPUT_DIR / "tuning_all_results.csv", index=False
    )
    print("Official tuning outputs passed all completeness and leakage checks.")


if __name__ == "__main__":
    main()
