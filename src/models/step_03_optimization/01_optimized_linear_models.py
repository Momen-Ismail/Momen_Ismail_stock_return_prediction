"""Refit linear models with validation-selected hyperparameters."""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import load_model_data  # noqa: E402
from src.models.utils.estimation import fit_models, load_best_parameters  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TUNING_FILE = MODEL_OUTPUT_DIR / "tuning" / "linear_best_parameters.csv"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"
RANDOM_STATE = 42


def optimized_models(parameters):
    """Construct linear models from validation-selected parameters."""
    common = dict(
        max_iter=5_000,
        tol=1e-3,
        selection="random",
        random_state=RANDOM_STATE,
    )
    return {
        "pcr_optimized": make_pipeline(
            StandardScaler(),
            PCA(int(parameters["pcr"]["n_components"]), random_state=RANDOM_STATE),
            LinearRegression(),
        ),
        "pls_optimized": make_pipeline(
            StandardScaler(),
            PLSRegression(int(parameters["pls"]["n_components"])),
        ),
        "ridge_optimized": make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(parameters["ridge"]["alpha"])),
        ),
        "lasso_optimized": make_pipeline(
            StandardScaler(),
            Lasso(alpha=float(parameters["lasso"]["alpha"]), **common),
        ),
        "enet_optimized": make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=float(parameters["elastic_net"]["alpha"]),
                l1_ratio=float(parameters["elastic_net"]["l1_ratio"]),
                **common,
            ),
        ),
    }


def save_interpretations(models, predictors):
    """Save PCA loadings, PLS weights, and Lasso sparsity diagnostics."""
    pca = models["pcr_optimized"].named_steps["pca"]
    components = [f"PC{i + 1}" for i in range(pca.n_components_)]
    pd.DataFrame({
        "component": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(OUTPUT_DIR / "optimized_pca_explained_variance.csv", index=False)
    pd.DataFrame(
        pca.components_.T, index=predictors, columns=components
    ).rename_axis("predictor").reset_index().to_csv(
        OUTPUT_DIR / "optimized_pca_loadings.csv", index=False
    )

    pls = models["pls_optimized"].named_steps["plsregression"]
    pls_components = [f"PLS{i + 1}" for i in range(pls.x_weights_.shape[1])]
    pd.DataFrame(
        pls.x_weights_, index=predictors, columns=pls_components
    ).rename_axis("predictor").reset_index().to_csv(
        OUTPUT_DIR / "optimized_pls_weights.csv", index=False
    )

    lasso = models["lasso_optimized"].named_steps["lasso"]
    pd.DataFrame([{
        "predictors": len(predictors),
        "nonzero_coefficients": int(np.count_nonzero(lasso.coef_)),
        "zero_coefficients": int(np.count_nonzero(lasso.coef_ == 0)),
    }]).to_csv(OUTPUT_DIR / "optimized_lasso_sparsity.csv", index=False)


def main():
    samples, predictors = load_model_data()
    parameters = load_best_parameters(
        TUNING_FILE, ["pcr", "pls", "ridge", "lasso", "elastic_net"]
    )
    models = optimized_models(parameters)
    metrics, predictions, coefficients = fit_models(
        models,
        samples,
        predictors,
        TARGET,
        effect=("coef_", "coefficient"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "optimized_linear_model_metrics.csv", index=False)
    predictions.to_parquet(
        OUTPUT_DIR / "optimized_linear_model_predictions.parquet", index=False
    )
    if not coefficients.empty:
        coefficients.to_csv(
            OUTPUT_DIR / "optimized_linear_model_coefficients.csv", index=False
        )
    save_interpretations(models, predictors)


if __name__ == "__main__":
    main()
