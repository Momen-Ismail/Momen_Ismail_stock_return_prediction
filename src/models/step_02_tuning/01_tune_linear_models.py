"""Tune PLS and Elastic Net using annual expanding-window validation."""

from pathlib import Path
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


GRIDS = {
    "pls": {
        "n_components": list(range(1, 11)),
    },
    "elastic_net": {
    "alpha": [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02],
    "l1_ratio": [0.85, 0.9, 0.95],
    },
}


def make_model(family, params):
    """Construct one linear-model candidate."""
    if family == "pls":
        return make_pipeline(
            StandardScaler(),
            PLSRegression(
                n_components=int(params["n_components"]),
                scale=False,
            ),
        )

    if family == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]),
                max_iter=20_000,
                tol=1e-3,
                random_state=42,
            ),
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