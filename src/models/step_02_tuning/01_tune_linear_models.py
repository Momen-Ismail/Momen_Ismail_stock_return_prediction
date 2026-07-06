"""Tune linear and dimension-reduction models using validation data only."""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.estimation import (  # noqa: E402
    best_by_family, save_results, tune_grid,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"
RANDOM_STATE = 42
GRIDS = {
    "pcr": {"n_components": [5, 10, 20, 40, 60]},
    "pls": {"n_components": [5, 10, 20, 40, 60]},
    "lasso": {"alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
    "elastic_net": {
        "alpha": [1e-4, 1e-3, 1e-2],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
}


def make_model(family, params):
    """Construct one candidate model."""
    if family == "pcr":
        return make_pipeline(
            PCA(params["n_components"], random_state=RANDOM_STATE),
            LinearRegression(),
        )
    if family == "pls":
        return PLSRegression(params["n_components"])

    common = dict(
        max_iter=5_000,
        tol=1e-3,
        selection="random",
        random_state=RANDOM_STATE,
    )
    if family == "lasso":
        return Lasso(alpha=params["alpha"], **common)
    if family == "elastic_net":
        return ElasticNet(
            alpha=params["alpha"], l1_ratio=params["l1_ratio"], **common
        )
    raise ValueError(f"Unknown model family: {family}")


def main():
    samples, predictors = load_model_data(("train", "validation"))
    model_arrays = arrays(samples, predictors)
    benchmark = np.full(
        len(samples["validation"]), samples["train"][TARGET].mean(), dtype=np.float32
    )
    results_file = OUTPUT_DIR / "linear_tuning_results.csv"
    parameters_file = OUTPUT_DIR / "linear_best_parameters.csv"
    rows = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for family, grid in GRIDS.items():
        rows.extend(tune_grid(
            family, grid, make_model,
            model_arrays["train"], model_arrays["validation"], benchmark,
        ))
        save_results(rows, results_file, parameters_file)

    print(best_by_family(pd.DataFrame(rows)).to_string(index=False))


if __name__ == "__main__":
    main()
