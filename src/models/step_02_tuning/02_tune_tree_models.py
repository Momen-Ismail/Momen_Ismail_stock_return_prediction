"""Tune tree models using annual expanding-window validation."""

from pathlib import Path
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import (  # noqa: E402
    expanding_year_folds,
    load_model_data,
)
from src.models.utils.estimation import (  # noqa: E402
    save_results,
    tune_grid,
)


OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"
RANDOM_STATE = 42


GRIDS = {
    "decision_tree": {
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [100, 250, 500, 1000],
    },

    # "decision_tree": {
    # "max_depth": [2, 3],
    # "min_samples_leaf": [500, 1000, 1500, 2000],
    # }

    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 5],
        "min_samples_leaf": [100],# here we leave just one because
        # we don't need to optimizing on this parameter, we just want to see if the model is better than the decision tree
        
        "max_features": ["sqrt"],
    },

    # "random_forest": {
    # "n_estimators": [200, 300],
    # "max_depth": [1, 2, 3],
    # "min_samples_leaf": [100, 250, 500],
    # "max_features": ["sqrt"],
    # }

}


def make_model(family, params):
    """Construct one tree-model candidate."""
    if family == "decision_tree":
        return DecisionTreeRegressor(
            **params,
            random_state=RANDOM_STATE,
        )

    if family == "random_forest":
        return RandomForestRegressor(
            **params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

def main():
    samples, predictors = load_model_data(("train",))
    train = samples["train"]

    folds = expanding_year_folds(
        train,
        first_validation_year=2005,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for family, grid in GRIDS.items():
        results, best_params = tune_grid(
            family=family,
            grid=grid,
            make_model=make_model,
            data=train,
            predictors=predictors,
            target=TARGET,
            folds=folds,
        )

        save_results(
            family=family,
            results=results,
            best_params=best_params,
            results_file=(
                OUTPUT_DIR
                / f"{family}_tuning_results.csv"
            ),
            parameters_file=(
                OUTPUT_DIR
                / f"{family}_best_parameters.csv"
            ),
        )

        print(f"\nBest {family} parameters:")
        print(best_params)


if __name__ == "__main__":
    main()
