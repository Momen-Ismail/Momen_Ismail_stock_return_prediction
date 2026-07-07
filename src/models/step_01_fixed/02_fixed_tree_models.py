"""Estimate fixed decision-tree, Random Forest, and GBRT benchmarks."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.evaluation import (  # noqa: E402
    evaluate_model,
    ranked_effects,
)

OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"
RANDOM_STATE = 42


def fixed_models():
    """Use the baseline parameters documented in the helper repository."""
    return {
        "decision_tree_fixed": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        "rf_fixed": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            max_features="sqrt",
            n_jobs=-1,
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
        "gbrt_fixed": GradientBoostingRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data()
    model_arrays = arrays(samples, predictors)
    metrics, prediction_frames, importances = [], [], []

    for name, model in fixed_models().items():
        print(f"Estimating {name} ({len(predictors)} predictors)")
        model.fit(*model_arrays["train"])
        predictions = {
            sample: model.predict(X)
            for sample, (X, _) in model_arrays.items()
        }

        model_metrics, model_predictions = evaluate_model(
            name, samples, predictions, TARGET
        )
        if hasattr(model, "oob_score_"):
            model_metrics["oob_r2_train"] = model.oob_score_
        metrics.append(model_metrics)
        prediction_frames.append(model_predictions)
        importances.append(
            ranked_effects(
                name, predictors, model.feature_importances_, "importance"
            )
        )
        print(model_metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(metrics).to_csv(OUTPUT_DIR / "fixed_tree_model_metrics.csv", index=False)
    pd.concat(prediction_frames).to_parquet(
        OUTPUT_DIR / "fixed_tree_model_predictions.parquet", index=False
    )
    pd.concat(importances).to_csv(
        OUTPUT_DIR / "fixed_tree_model_feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
