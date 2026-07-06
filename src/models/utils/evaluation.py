"""Prediction evaluation and model-comparison helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true, y_pred, benchmark):
    """Return predictive metrics relative to the training-mean benchmark."""
    mse = mean_squared_error(y_true, y_pred)
    benchmark_mse = mean_squared_error(y_true, benchmark)
    correlation = (
        np.corrcoef(y_true, y_pred)[0, 1]
        if np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else np.nan
    )
    return {
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "benchmark_mse": benchmark_mse,
        "oos_r2_vs_train_mean": (
            1.0 - mse / benchmark_mse if benchmark_mse else np.nan
        ),
        "prediction_target_correlation": correlation,
        "prediction_mean": np.mean(y_pred),
        "prediction_std": np.std(y_pred),
        "prediction_min": np.min(y_pred),
        "prediction_max": np.max(y_pred),
    }


def evaluate_model(model_name, samples, predictions, target):
    """Evaluate one model and build its standardized prediction table."""
    train_mean = np.float32(samples["train"][target].mean())
    metrics, frames = [], []
    for sample, data in samples.items():
        y_true = data[target].to_numpy(dtype=np.float32)
        y_pred = np.asarray(predictions[sample]).reshape(-1)
        benchmark = np.full(len(data), train_mean, dtype=np.float32)
        metrics.append({
            "model": model_name,
            "sample": sample,
            "rows": len(data),
            **evaluate_predictions(y_true, y_pred, benchmark),
        })
        frame = data[["ticker", "month", target]].copy()
        frame["sample"], frame["model"], frame["prediction"] = (
            sample, model_name, y_pred
        )
        frames.append(frame)
    return pd.DataFrame(metrics), pd.concat(frames, ignore_index=True)


def ranked_effects(model_name, predictors, values, value_name):
    """Sort coefficients or feature importances by absolute magnitude."""
    return (
        pd.DataFrame({
            "model": model_name,
            "predictor": predictors,
            value_name: np.asarray(values).reshape(-1),
        })
        .sort_values(value_name, key=abs, ascending=False)
        .reset_index(drop=True)
    )


def rank_models(metrics, sample):
    """Rank a sample by OOS R-squared, then RMSE and MAE."""
    ranking = (
        metrics[metrics["sample"].eq(sample)]
        .sort_values(
            ["oos_r2_vs_train_mean", "rmse", "mae"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank_by_oos_r2", ranking.index + 1)
    return ranking


def wide_summary(metrics, sort_sample="validation"):
    """Place validation and test metrics side by side."""
    values = [
        "rmse", "mae", "oos_r2_vs_train_mean",
        "prediction_target_correlation", "prediction_std",
    ]
    wide = metrics[metrics["sample"].isin(["validation", "test"])].pivot(
        index=["model", "model_group"], columns="sample", values=values
    )
    wide.columns = [f"{metric}_{sample}" for metric, sample in wide.columns]
    return wide.reset_index().sort_values(
        f"oos_r2_vs_train_mean_{sort_sample}", ascending=False
    )
