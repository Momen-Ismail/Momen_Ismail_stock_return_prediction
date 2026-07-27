"""Estimate fixed tree-ensemble models on train and validation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data  # noqa: E402
from src.models.utils.evaluation import (  # noqa: E402
    evaluate_model,
    monthly_mse,
)


OUTPUT_DIR = MODEL_OUTPUT_DIR / "fixed"
RANDOM_STATE = 42


def fixed_models():
    """Define fixed professor-aligned tree-ensemble specifications."""
    return {
        "random_forest_fixed": RandomForestRegressor(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=20,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting_fixed": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def main():
    samples, predictors = load_model_data(
        ("train", "validation")
    )

    model_arrays = arrays(
        samples,
        predictors,
    )

    X_train, y_train = model_arrays["train"]
    train_months = samples["train"]["month"].to_numpy()
    train_mean = samples["train"][TARGET].mean()

    all_metrics = []
    all_predictions = []
    all_importances = []

    for name, model in fixed_models().items():
        print(
            f"Estimating {name} "
            f"({len(predictors)} predictors)"
        )

        model.fit(
            X_train,
            y_train,
        )

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

        metrics["oob_pooled_mse_train"] = np.nan
        metrics["oob_monthly_mse_train"] = np.nan
        metrics["oob_r2_train"] = np.nan

        if hasattr(model, "oob_prediction_"):
            oob_prediction = np.asarray(
                model.oob_prediction_
            ).reshape(-1)

            valid_oob = np.isfinite(
                oob_prediction
            )

            oob_pooled_mse = mean_squared_error(
                y_train[valid_oob],
                oob_prediction[valid_oob],
            )

            oob_monthly_mse = monthly_mse(
                y_train[valid_oob],
                oob_prediction[valid_oob],
                train_months[valid_oob],
            )

            metrics.loc[
                metrics["sample"].eq("train"),
                "oob_pooled_mse_train",
            ] = oob_pooled_mse

            metrics.loc[
                metrics["sample"].eq("train"),
                "oob_monthly_mse_train",
            ] = oob_monthly_mse

            metrics.loc[
                metrics["sample"].eq("train"),
                "oob_r2_train",
            ] = model.oob_score_

        all_metrics.append(metrics)
        all_predictions.append(prediction_frame)

        importance = pd.DataFrame({
            "model": name,
            "predictor": predictors,
            "importance": model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        )

        all_importances.append(importance)

        print(metrics.to_string(index=False))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.concat(
        all_metrics,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "fixed_tree_model_metrics.csv",
        index=False,
    )

    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_parquet(
        OUTPUT_DIR / "fixed_tree_model_predictions.parquet",
        index=False,
    )

    pd.concat(
        all_importances,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "fixed_tree_model_feature_importance.csv",
        index=False,
    )

    print("\nSaved fixed tree-model outputs")
    print(
        OUTPUT_DIR
        / "fixed_tree_model_metrics.csv"
    )
    print(
        OUTPUT_DIR
        / "fixed_tree_model_predictions.parquet"
    )
    print(
        OUTPUT_DIR
        / "fixed_tree_model_feature_importance.csv"
    )


if __name__ == "__main__":
    main()