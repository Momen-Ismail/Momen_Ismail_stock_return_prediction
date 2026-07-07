"""Tune Random Forest and GBRT models using validation data only."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import expanding_month_folds, load_model_data  # noqa: E402
from src.models.utils.estimation import (  # noqa: E402
    best_by_family, save_results, tune_grid,
)

OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"
RANDOM_STATE = 42
TRAIN_SAMPLE_FRAC = 0.20
GRIDS = {
    "decision_tree": {
        "max_depth": [3, 6, None],
        "min_samples_leaf": [20, 100],
        "ccp_alpha": [0.0, 1e-7, 1e-6, 1e-5],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [1, 3, 6],
        "max_features": ["sqrt"],
    },
    "gbrt": {
        "n_estimators": [50, 100, 300],
        "max_depth": [1, 2],
        "learning_rate": [0.03, 0.1],
    },
}


def make_model(family, params):
    """Construct one candidate tree model."""
    if family == "decision_tree":
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    if family == "random_forest":
        return RandomForestRegressor(
            **params, n_jobs=-1, oob_score=True, random_state=RANDOM_STATE
        )
    if family == "gbrt":
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model family: {family}")


def main():
    samples, predictors = load_model_data(("train",))
    train = samples["train"]
    folds = expanding_month_folds(train)
    results_file = OUTPUT_DIR / "tree_tuning_results.csv"
    parameters_file = OUTPUT_DIR / "tree_best_parameters.csv"
    rows = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for family, grid in GRIDS.items():
        rows.extend(tune_grid(
            family, grid, make_model, train, predictors, TARGET, folds,
            train_sample_fraction=TRAIN_SAMPLE_FRAC,
            random_state=RANDOM_STATE,
        ))
        save_results(rows, results_file, parameters_file)

    print(best_by_family(pd.DataFrame(rows)).to_string(index=False))


if __name__ == "__main__":
    main()
