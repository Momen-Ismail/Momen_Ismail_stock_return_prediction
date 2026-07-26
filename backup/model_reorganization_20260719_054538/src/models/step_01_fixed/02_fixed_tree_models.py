"""Estimate optimized tree models on train and validation."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.estimation import load_best_parameters  # noqa: E402
from src.models.utils.evaluation import evaluate_model  # noqa: E402


TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"
RANDOM_STATE = 42


def optimized_models():
    """Define the optimized tree-model specifications."""
    tree_parameters = load_best_parameters(
        TUNING_DIR / "decision_tree_best_parameters.csv"
    )

    random_forest_parameters = load_best_parameters(
        TUNING_DIR / "random_forest_best_parameters.csv"
    )

    gradient_boosting_parameters = load_best_parameters(
        TUNING_DIR / "gradient_boosting_best_parameters.csv"
    )

    return {
        "decision_tree_optimized": DecisionTreeRegressor(
            max_depth=int(
                tree_parameters["max_depth"]
            ),
            min_samples_leaf=int(
                tree_parameters["min_samples_leaf"]
            ),
            random_state=RANDOM_STATE,
        ),

        "random_forest_optimized": RandomForestRegressor(
            n_estimators=int(
                random_forest_parameters["n_estimators"]
            ),
            max_depth=int(
                random_forest_parameters["max_depth"]
            ),
            min_samples_leaf=int(
                random_forest_parameters["min_samples_leaf"]
            ),
            max_features=random_forest_parameters["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "gradient_boosting_optimized": GradientBoostingRegressor(
            n_estimators=int(
                gradient_boosting_parameters["n_estimators"]
            ),
            learning_rate=float(
                gradient_boosting_parameters["learning_rate"]
            ),
            max_depth=int(
                gradient_boosting_parameters["max_depth"]
            ),
            min_samples_leaf=int(
                gradient_boosting_parameters["min_samples_leaf"]
            ),
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data(
        ("train", "validation")
    )

    model_arrays = arrays(samples, predictors)

    X_train, y_train = model_arrays["train"]
    train_mean = samples["train"][TARGET].mean()

    all_metrics = []
    all_predictions = []
    all_importances = []

    for name, model in optimized_models().items():
        print(f"Estimating {name} ({len(predictors)} predictors)")

        model.fit(X_train, y_train)

        predictions = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }

        metrics, prediction_frame = evaluate_model(
            name,
            samples,
            predictions,
            TARGET,
            train_mean,
        )

        importance_frame = pd.DataFrame({
            "model": name,
            "predictor": predictors,
            "importance": model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        )

        all_metrics.append(metrics)
        all_predictions.append(prediction_frame)
        all_importances.append(importance_frame)

        print(metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat(
        all_metrics,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "optimized_tree_model_metrics.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_parquet(
        OUTPUT_DIR / "optimized_tree_model_predictions.parquet",
        index=False,
    )

    pd.concat(
        all_importances,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "optimized_tree_model_feature_importance.csv",
        index=False,
    )


if __name__ == "__main__":
    main()