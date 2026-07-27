"""Prediction evaluation and model comparison helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def monthly_mse(y_true, y_pred, months):
    """Calculate MSE for each month and average equally across months."""
    errors = pd.DataFrame({
        "month": months,
        "squared_error": (np.asarray(y_true) - np.asarray(y_pred)) ** 2,
    })

    return errors.groupby("month")["squared_error"].mean().mean()


def evaluate_predictions(y_true, y_pred, benchmark, months):
    """Calculate prediction metrics."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    benchmark = np.asarray(benchmark).reshape(-1)

    pooled_mse = mean_squared_error(y_true, y_pred)
    month_mse = monthly_mse(y_true, y_pred, months)

    benchmark_month_mse = monthly_mse(
        y_true,
        benchmark,
        months,
    )

    model_sse = np.sum((y_true - y_pred) ** 2)
    benchmark_sse = np.sum((y_true - benchmark) ** 2)

    pooled_oos_r2 = (
        1 - model_sse / benchmark_sse
        if benchmark_sse > 0
        else np.nan
    )

    monthly_oos_r2 = (
        1 - month_mse / benchmark_month_mse
        if benchmark_month_mse > 0
        else np.nan
    )

    prediction_std = np.std(y_pred)

    correlation = (
        np.corrcoef(y_true, y_pred)[0, 1]
        if prediction_std > 1e-8
        else np.nan
    )

    return {
        "pooled_mse": pooled_mse,
        "pooled_rmse": np.sqrt(pooled_mse),
        "pooled_mae": mean_absolute_error(y_true, y_pred),
        "monthly_mse": month_mse,
        "monthly_rmse": np.sqrt(month_mse),
        "oos_r2": pooled_oos_r2,
        "monthly_oos_r2": monthly_oos_r2,
        "prediction_target_correlation": correlation,
        "prediction_mean": np.mean(y_pred),
        "prediction_std": prediction_std,
    }


def evaluate_model(
    model_name,
    samples,
    predictions,
    target,
    benchmark_mean,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model and create its prediction table."""
    all_metrics = []
    all_predictions = []

    for sample, data in samples.items():
        y_true = data[target].to_numpy()
        y_pred = predictions[sample]

        benchmark = np.full(len(data), benchmark_mean)

        metrics = evaluate_predictions(
            y_true,
            y_pred,
            benchmark,
            data["month"],
        )

        metrics["model"] = model_name
        metrics["sample"] = sample
        metrics["observations"] = len(data)
        metrics["months"] = data["month"].nunique()

        all_metrics.append(metrics)

        prediction_frame = data[["ticker", "month"]].copy()
        prediction_frame["realized_target"] = y_true
        prediction_frame["prediction"] = y_pred
        prediction_frame["model"] = model_name
        prediction_frame["sample"] = sample

        all_predictions.append(prediction_frame)

    return (
        pd.DataFrame(all_metrics),
        pd.concat(all_predictions, ignore_index=True),
    )


def rank_models(metrics, sample="test"):
    """Rank models by monthly MSE."""
    ranking = (
        metrics[metrics["sample"] == sample]
        .sort_values(["monthly_mse", "pooled_rmse", "pooled_mae"])
        .reset_index(drop=True)
    )

    ranking.insert(0, "rank", ranking.index + 1)

    return ranking