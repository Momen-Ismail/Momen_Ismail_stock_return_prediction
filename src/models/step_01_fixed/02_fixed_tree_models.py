"""Estimate fixed tree models on train and validation."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.evaluation import evaluate_model  # noqa: E402

OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"
RANDOM_STATE = 42


def fixed_models():
    """Define the fixed tree-model specifications."""
    return {
        "decision_tree_fixed": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=500,
            random_state=RANDOM_STATE,
        ),
        "random_forest_fixed": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=100,
            max_features="sqrt",
            n_jobs=-1,
            oob_score=True,
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

    for name, model in fixed_models().items():
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
        OUTPUT_DIR / "fixed_tree_model_metrics.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        OUTPUT_DIR / "fixed_tree_model_predictions.parquet", index=False
    )
    pd.concat(all_importances, ignore_index=True).to_csv(
        OUTPUT_DIR / "fixed_tree_model_feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
