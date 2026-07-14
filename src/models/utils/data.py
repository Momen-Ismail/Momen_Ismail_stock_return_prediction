"""Load modeling samples and construct predictor arrays."""

import numpy as np
import pandas as pd

from src.config import (
    TARGET,
    TRAIN_END,
    VALIDATION_END,
    CLEAN_FULL_FILE,
    CLEAN_PREDICTOR_FILE,
)


def load_model_data(sample_names=("train", "validation", "test")):
    """Return requested chronological samples and their common predictors."""
    full = pd.read_parquet(CLEAN_FULL_FILE)
    full["month"] = pd.to_datetime(full["month"])

    all_samples = {
        "train": full[full["month"] <= TRAIN_END].copy(),
        "validation": full[
            (full["month"] > TRAIN_END)
            & (full["month"] <= VALIDATION_END)
        ].copy(),
        "test": full[full["month"] > VALIDATION_END].copy(),
    }

    samples = {name: all_samples[name] for name in sample_names}
    predictors = pd.read_csv(CLEAN_PREDICTOR_FILE)["predictor"].astype(str).tolist()
    predictors = [name for name in predictors if name in next(iter(samples.values()))]

    if not predictors or any(TARGET not in data for data in samples.values()):
        raise ValueError("Modeling data are missing the target or predictors.")

    return samples, predictors


def arrays(samples, predictors, target=TARGET):
    """Return float32 predictor and target arrays for each sample."""
    return {
        name: (
            data[predictors].to_numpy(dtype=np.float32),
            data[target].to_numpy(dtype=np.float32),
        )
        for name, data in samples.items()
    }


def expanding_month_folds(data, n_splits=3, min_train_fraction=0.50):
    """Create expanding folds without splitting a calendar month across samples."""
    months = np.array(sorted(data["month"].unique()))
    first_validation = int(len(months) * min_train_fraction)
    validation_blocks = np.array_split(months[first_validation:], n_splits)
    folds = []

    for block in validation_blocks:
        train_mask = data["month"] < block[0]
        validation_mask = data["month"].isin(block)
        folds.append({
            "train_index": data.index[train_mask],
            "validation_index": data.index[validation_mask],
            "train_end": pd.Timestamp(block[0]) - pd.offsets.MonthEnd(1),
            "validation_start": pd.Timestamp(block[0]),
            "validation_end": pd.Timestamp(block[-1]),
        })
    return folds


def ols3_predictors(predictors):
    """Select size, book-to-market, momentum, and their macro interactions."""
    candidates = [
        ("avg_log_dolvol_1m", "log_comp_market_equity"),
        ("be_me",),
        ("mom12m", "mom6m"),
    ]
    base = [next((name for name in group if name in predictors), None) for group in candidates]
    base = [name for name in base if name]
    if len(base) != 3:
        raise ValueError("Could not identify the three OLS-3 characteristics.")
    return base + [
        name for name in predictors
        if any(name.startswith(f"{characteristic}_x_") for characteristic in base)
    ]
