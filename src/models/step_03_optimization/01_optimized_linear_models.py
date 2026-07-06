"""Refit linear models with validation-selected hyperparameters."""

from pathlib import Path
import sys
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline

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
            PCA(int(parameters["pcr"]["n_components"]), random_state=RANDOM_STATE),
            LinearRegression(),
        ),
        "pls_optimized": PLSRegression(
            int(parameters["pls"]["n_components"])
        ),
        "lasso_optimized": Lasso(
            alpha=float(parameters["lasso"]["alpha"]), **common
        ),
        "enet_optimized": ElasticNet(
            alpha=float(parameters["elastic_net"]["alpha"]),
            l1_ratio=float(parameters["elastic_net"]["l1_ratio"]),
            **common,
        ),
    }


def main():
    samples, predictors = load_model_data()
    parameters = load_best_parameters(
        TUNING_FILE, ["pcr", "pls", "lasso", "elastic_net"]
    )
    metrics, predictions, coefficients = fit_models(
        optimized_models(parameters),
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


if __name__ == "__main__":
    main()
