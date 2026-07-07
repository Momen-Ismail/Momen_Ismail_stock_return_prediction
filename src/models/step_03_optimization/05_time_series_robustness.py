"""Annual expanding- and rolling-window robustness forecasts.

This optional analysis re-estimates the two validation-selected tree families
each year from 2015 through 2019. The test period remains untouched.
"""

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

TUNING_FILE = MODEL_OUTPUT_DIR / "tuning" / "tree_best_parameters.csv"
OUTPUT_DIR = MODEL_OUTPUT_DIR / "optimization"
RANDOM_STATE = 42
TRAIN_SAMPLE_FRACTION = 0.20
ROLLING_MONTHS = 120


def model_factories(parameters):
    """Return fresh selected models for each annual refit."""
    rf, gbrt = parameters["random_forest"], parameters["gbrt"]
    return {
        "rf_optimized": lambda: RandomForestRegressor(
            **rf, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gbrt_optimized": lambda: GradientBoostingRegressor(
            **gbrt, random_state=RANDOM_STATE
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
    parameters = load_best_parameters(
        TUNING_FILE, ["random_forest", "gbrt"]
    )
    forecasts = []
    for model_name, factory in model_factories(parameters).items():
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
            "rows": len(group),
            **evaluate_predictions(
                group[TARGET], group["prediction"], group["benchmark"]
            ),
        })

    pd.DataFrame(metrics).to_csv(
        OUTPUT_DIR / "time_series_robustness_metrics.csv", index=False
    )
    predictions.to_parquet(
        OUTPUT_DIR / "time_series_robustness_predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
