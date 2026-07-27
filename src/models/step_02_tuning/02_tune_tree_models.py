"""Tune tree-ensemble models using annual expanding-window validation."""

from pathlib import Path
import sys

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)

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
    "random_forest": {
        "n_estimators": [100, 200 , 300],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.3],
        "max_depth": [1, 2],
    },
}


def make_model(family, params):
    """Construct one tree-ensemble candidate."""
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_features="sqrt",
            min_samples_leaf=20,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown model family: {family}"
    )


def main():
    samples, predictors = load_model_data(
        ("train",)
    )

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