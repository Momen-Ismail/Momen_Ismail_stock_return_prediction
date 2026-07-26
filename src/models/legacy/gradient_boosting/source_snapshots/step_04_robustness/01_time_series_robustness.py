"""Run expanding- and rolling-window validation robustness checks."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR, TARGET  # noqa: E402
from src.models.utils.data import load_model_data  # noqa: E402
from src.models.utils.estimation import load_best_parameters  # noqa: E402
from src.models.utils.evaluation import evaluate_predictions  # noqa: E402

TUNING_DIR = MODEL_OUTPUT_DIR / "tuning"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "robustness"
RANDOM_STATE = 42
TRAIN_SAMPLE_FRACTION = 0.20
ROLLING_MONTHS = 120


def model_factories():
    """Return fresh optimized tree models for each annual refit."""
    forest = load_best_parameters(TUNING_DIR / "random_forest_best_parameters.csv")
    boosting = load_best_parameters(
        TUNING_DIR / "gradient_boosting_best_parameters.csv"
    )
    forest = {
        **forest,
        "n_estimators": int(forest["n_estimators"]),
        "max_depth": (
            None if forest["max_depth"] is None else int(forest["max_depth"])
        ),
        "min_samples_leaf": int(forest["min_samples_leaf"]),
    }
    boosting = {
        **boosting,
        "n_estimators": int(boosting["n_estimators"]),
        "learning_rate": float(boosting["learning_rate"]),
        "max_depth": int(boosting["max_depth"]),
        "min_samples_leaf": int(boosting["min_samples_leaf"]),
    }
    return {
        "random_forest_optimized": lambda: RandomForestRegressor(
            **forest, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting_optimized": lambda: GradientBoostingRegressor(
            **boosting, random_state=RANDOM_STATE
        ),
    }


def walk_forward(data, predictors, model_name, make_model, window):
    """Generate annual validation forecasts using past observations only."""
    frames = []
    for year in range(2015, 2020):
        forecast_start = pd.Timestamp(year, 1, 31)
        train = data[data["month"] < forecast_start]
        if window:
            first_month = train["month"].drop_duplicates().sort_values().iloc[-window]
            train = train[train["month"] >= first_month]
        train = train.sample(
            frac=TRAIN_SAMPLE_FRACTION, random_state=RANDOM_STATE + year
        )
        validation = data[data["month"].dt.year.eq(year)]

        model = make_model()
        model.fit(
            train[predictors].to_numpy(dtype=np.float32),
            train[TARGET].to_numpy(dtype=np.float32),
        )
        frame = validation[["ticker", "month", TARGET]].copy()
        frame["prediction"] = model.predict(
            validation[predictors].to_numpy(dtype=np.float32)
        )
        frame["benchmark"] = train[TARGET].mean()
        frame["model"] = model_name
        frame["window"] = "rolling_120m" if window else "expanding"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    samples, predictors = load_model_data(("train", "validation"))
    data = pd.concat(samples.values(), ignore_index=True)
    forecasts = []

    for model_name, factory in model_factories().items():
        forecasts.extend([
            walk_forward(data, predictors, model_name, factory, None),
            walk_forward(data, predictors, model_name, factory, ROLLING_MONTHS),
        ])

    predictions = pd.concat(forecasts, ignore_index=True)
    metrics = []
    for (model, window), group in predictions.groupby(["model", "window"]):
        metrics.append({
            "model": model,
            "window": window,
            "sample": "validation_2015_2019",
            "observations": len(group),
            **evaluate_predictions(
                group[TARGET],
                group["prediction"],
                group["benchmark"],
                group["month"],
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(
        OUTPUT_DIR / "time_series_robustness_metrics.csv", index=False
    )
    predictions.to_parquet(
        OUTPUT_DIR / "time_series_robustness_predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
