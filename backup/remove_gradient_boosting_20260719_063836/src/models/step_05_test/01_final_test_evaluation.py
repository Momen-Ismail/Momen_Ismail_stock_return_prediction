"""Fit fixed and optimized models on development and evaluate test once."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import arrays, load_model_data, ols3_predictors  # noqa: E402
from src.models.utils.estimation import load_best_parameters  # noqa: E402
from src.models.utils.evaluation import evaluate_model, rank_models  # noqa: E402

TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "test"
RANDOM_STATE = 42


def model_specifications(predictors):
    """Return the prespecified fixed and optimized models."""
    pls = load_best_parameters(TUNING_DIR / "pls_best_parameters.csv")
    elastic_net = load_best_parameters(
        TUNING_DIR / "elastic_net_best_parameters.csv"
    )
    tree = load_best_parameters(TUNING_DIR / "decision_tree_best_parameters.csv")
    forest = load_best_parameters(TUNING_DIR / "random_forest_best_parameters.csv")
    boosting = load_best_parameters(
        TUNING_DIR / "gradient_boosting_best_parameters.csv"
    )

    return {
        "ols_3": (
            make_pipeline(StandardScaler(), LinearRegression()),
            ols3_predictors(),
        ),
        "pls_fixed": (
            make_pipeline(
                StandardScaler(), PLSRegression(n_components=20, scale=False)
            ),
            predictors,
        ),
        "pls_optimized": (
            make_pipeline(
                StandardScaler(),
                PLSRegression(n_components=int(pls["n_components"]), scale=False),
            ),
            predictors,
        ),
        "elastic_net_fixed": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=20_000, tol=1e-4),
            ),
            predictors,
        ),
        "elastic_net_optimized": (
            make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=float(elastic_net["alpha"]),
                    l1_ratio=float(elastic_net["l1_ratio"]),
                    max_iter=20_000,
                    tol=1e-4,
                ),
            ),
            predictors,
        ),
        "decision_tree_fixed": (
            DecisionTreeRegressor(
                max_depth=3,
                min_samples_leaf=500,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "decision_tree_optimized": (
            DecisionTreeRegressor(
                max_depth=(
                    None if tree["max_depth"] is None else int(tree["max_depth"])
                ),
                min_samples_leaf=int(tree["min_samples_leaf"]),
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "random_forest_fixed": (
            RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                max_features="sqrt",
                n_jobs=-1,
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "random_forest_optimized": (
            RandomForestRegressor(
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
            predictors,
        ),
        "gradient_boosting_fixed": (
            GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
        "gradient_boosting_optimized": (
            GradientBoostingRegressor(
                n_estimators=int(boosting["n_estimators"]),
                learning_rate=float(boosting["learning_rate"]),
                max_depth=int(boosting["max_depth"]),
                min_samples_leaf=int(boosting["min_samples_leaf"]),
                random_state=RANDOM_STATE,
            ),
            predictors,
        ),
    }


def main():
    samples, predictors = load_model_data(("development", "test"))
    evaluation_samples = {
        "train": samples["development"],
        "test": samples["test"],
    }
    train_mean = samples["development"][TARGET].mean()
    metrics = []
    predictions = []

    mean_predictions = {
        sample: np.full(len(data), train_mean, dtype=np.float32)
        for sample, data in evaluation_samples.items()
    }
    model_metrics, model_predictions = evaluate_model(
        "historical_mean",
        evaluation_samples,
        mean_predictions,
        TARGET,
        train_mean,
    )
    metrics.append(model_metrics[model_metrics["sample"].eq("test")])
    predictions.append(model_predictions[model_predictions["sample"].eq("test")])

    for name, (model, model_predictors) in model_specifications(predictors).items():
        print(f"Final test model: {name}")
        model_arrays = arrays(evaluation_samples, model_predictors)
        model.fit(*model_arrays["train"])
        model_predictions = {
            sample: model.predict(X).reshape(-1)
            for sample, (X, _) in model_arrays.items()
        }
        model_metrics, prediction_frame = evaluate_model(
            name,
            evaluation_samples,
            model_predictions,
            TARGET,
            train_mean,
        )
        if hasattr(model, "oob_score_"):
            model_metrics["oob_r2_development_diagnostic"] = model.oob_score_
        metrics.append(model_metrics[model_metrics["sample"].eq("test")])
        predictions.append(prediction_frame[prediction_frame["sample"].eq("test")])

    metrics = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    ranking = rank_models(metrics, "test")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "final_test_model_comparison.csv", index=False)
    predictions.to_parquet(
        OUTPUT_DIR / "final_test_predictions.parquet", index=False
    )
    print(ranking[[
        "rank",
        "model",
        "observations",
        "months",
        "monthly_mse",
        "pooled_rmse",
        "pooled_mae",
        "oos_r2",
        "prediction_target_correlation",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
