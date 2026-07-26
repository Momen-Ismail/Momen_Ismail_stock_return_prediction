"""Fit optimized tree models on training and evaluate validation only."""

from pathlib import Path
import sys

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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
    tree = parameters["decision_tree"]
    rf = parameters["random_forest"]
    gb = parameters["gradient_boosting"]
    return {
        "decision_tree_optimized": DecisionTreeRegressor(
            max_depth=None if tree["max_depth"] is None else int(tree["max_depth"]),
            min_samples_leaf=int(tree["min_samples_leaf"]),
            ccp_alpha=float(tree["ccp_alpha"]),
            random_state=RANDOM_STATE,
        ),
        "random_forest_optimized": RandomForestRegressor(
            n_estimators=int(rf["n_estimators"]),
            max_depth=None if rf["max_depth"] is None else int(rf["max_depth"]),
            min_samples_leaf=int(rf["min_samples_leaf"]),
            max_features=rf["max_features"],
            n_jobs=-1,
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting_optimized": GradientBoostingRegressor(
            n_estimators=int(gb["n_estimators"]),
            learning_rate=float(gb["learning_rate"]),
            max_depth=int(gb["max_depth"]),
            min_samples_leaf=int(gb["min_samples_leaf"]),
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data(("train", "validation"))
    parameters = load_best_parameters(
        TUNING_FILE, ["decision_tree", "random_forest", "gradient_boosting"]
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
