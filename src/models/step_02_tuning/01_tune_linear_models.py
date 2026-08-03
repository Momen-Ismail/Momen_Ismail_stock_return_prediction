"""Tune PLS and Elastic Net on the 1990--2019 development sample."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TUNING_OUTPUT_DIR  # noqa: E402
from src.models.utils.data import load_model_data  # noqa: E402
from src.models.utils.model_selection import (  # noqa: E402
    official_folds,
    tune_family,
    validate_grids,
)


def main():
    """Run all official linear-family candidates and save checkpoints."""
    validate_grids()
    samples, predictors = load_model_data(("development",))
    if len(predictors) != 484:
        raise ValueError(f"Expected 484 predictors; found {len(predictors)}.")
    development = samples["development"]
    folds = official_folds(development)
    TUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for family in ["pls", "elastic_net"]:
        fold_results, results = tune_family(
            family, development, predictors, folds
        )
        fold_results.to_csv(
            TUNING_OUTPUT_DIR / f"{family}_tuning_fold_results.csv",
            index=False,
        )
        results.to_csv(
            TUNING_OUTPUT_DIR / f"{family}_tuning_results.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
