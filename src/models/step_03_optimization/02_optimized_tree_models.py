"""Refit tree models with validation-selected hyperparameters."""

from pathlib import Path
import sys

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import load_model_data  # noqa: E402
from src.models.utils.estimation import fit_models, load_best_parameters  # noqa: E402

TUNING_FILE = MODEL_OUTPUT_DIR / "tuning" / "tree_best_parameters.csv"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"
RANDOM_STATE = 42


def optimized_models(parameters):
    """Construct tree models from validation-selected parameters."""
    rf, gbrt = parameters["random_forest"], parameters["gbrt"]
    return {
        "rf_optimized": RandomForestRegressor(
            n_estimators=int(rf["n_estimators"]),
            max_depth=int(rf["max_depth"]),
            max_features=rf["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gbrt_optimized": GradientBoostingRegressor(
            n_estimators=int(gbrt["n_estimators"]),
            max_depth=int(gbrt["max_depth"]),
            learning_rate=float(gbrt["learning_rate"]),
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data()
    parameters = load_best_parameters(
        TUNING_FILE, ["random_forest", "gbrt"]
    )
    metrics, predictions, importances = fit_models(
        optimized_models(parameters),
        samples,
        predictors,
        TARGET,
        effect=("feature_importances_", "importance"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "optimized_tree_model_metrics.csv", index=False)
    predictions.to_parquet(
        OUTPUT_DIR / "optimized_tree_model_predictions.parquet", index=False
    )
    importances.to_csv(
        OUTPUT_DIR / "optimized_tree_model_feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
