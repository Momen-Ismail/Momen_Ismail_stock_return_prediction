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
    """Define tree models from the saved tuning results."""
    tree = load_best_parameters(TUNING_DIR / "decision_tree_best_parameters.csv")
    forest = load_best_parameters(TUNING_DIR / "random_forest_best_parameters.csv")
    boosting = load_best_parameters(
        TUNING_DIR / "gradient_boosting_best_parameters.csv"
    )

    return {
        "decision_tree_optimized": DecisionTreeRegressor(
            max_depth=None if tree["max_depth"] is None else int(tree["max_depth"]),
            min_samples_leaf=int(tree["min_samples_leaf"]),
            random_state=RANDOM_STATE,
        ),
        "random_forest_optimized": RandomForestRegressor(
            n_estimators=int(forest["n_estimators"]),
            max_depth=(
                None
                if forest["max_depth"] is None
                else int(forest["max_depth"])
            ),
            min_samples_leaf=int(forest["min_samples_leaf"]),
            max_features=forest["max_features"],
            n_jobs=-1,
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting_optimized": GradientBoostingRegressor(
            n_estimators=int(boosting["n_estimators"]),
            learning_rate=float(boosting["learning_rate"]),
            max_depth=int(boosting["max_depth"]),
            min_samples_leaf=int(boosting["min_samples_leaf"]),
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data(("train", "validation"))
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
            name, samples, predictions, TARGET, train_mean
        )
        if hasattr(model, "oob_score_"):
            metrics["oob_r2_train"] = model.oob_score_

        all_metrics.append(metrics)
        all_predictions.append(prediction_frame)
        all_importances.append(
            pd.DataFrame({
                "model": name,
                "predictor": predictors,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
        )
        print(metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_metrics, ignore_index=True).to_csv(
        OUTPUT_DIR / "optimized_tree_model_metrics.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        OUTPUT_DIR / "optimized_tree_model_predictions.parquet", index=False
    )
    pd.concat(all_importances, ignore_index=True).to_csv(
        OUTPUT_DIR / "optimized_tree_model_feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
