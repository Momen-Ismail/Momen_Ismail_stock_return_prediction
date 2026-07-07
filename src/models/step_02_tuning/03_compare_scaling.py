"""Compare rank-normalized predictors with and without train-fitted scaling."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.estimation import load_best_parameters  # noqa: E402
from src.models.utils.evaluation import evaluate_predictions  # noqa: E402

TUNING_FILE = MODEL_OUTPUT_DIR / "tuning" / "linear_best_parameters.csv"
OUTPUT_FILE = MODEL_OUTPUT_DIR / "tuning" / "scaling_comparison.csv"
RANDOM_STATE = 42


def estimator(family, parameters):
    """Construct the unscaled estimator for one tuned family."""
    params = parameters[family]
    if family == "pcr":
        return make_pipeline(
            PCA(int(params["n_components"]), random_state=RANDOM_STATE),
            LinearRegression(),
        )
    if family == "pls":
        return PLSRegression(int(params["n_components"]))
    common = dict(
        max_iter=5_000, tol=1e-3, selection="random", random_state=RANDOM_STATE
    )
    if family == "lasso":
        return Lasso(alpha=float(params["alpha"]), **common)
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    return ElasticNet(
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        **common,
    )


def main():
    samples, predictors = load_model_data(("train", "validation"))
    model_arrays = arrays(samples, predictors)
    parameters = load_best_parameters(
        TUNING_FILE, ["pcr", "pls", "ridge", "lasso", "elastic_net"]
    )
    benchmark = np.full(
        len(samples["validation"]),
        samples["train"][TARGET].mean(),
        dtype=np.float32,
    )
    rows = []

    for family in parameters:
        for scaled in (False, True):
            model = estimator(family, parameters)
            if scaled:
                model = make_pipeline(StandardScaler(), model)
            model.fit(*model_arrays["train"])
            prediction = np.asarray(
                model.predict(model_arrays["validation"][0])
            ).reshape(-1)
            rows.append({
                "model_family": family,
                "scaled_after_rank_normalization": scaled,
                **evaluate_predictions(
                    model_arrays["validation"][1], prediction, benchmark
                ),
            })

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()


