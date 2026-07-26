"""Load model samples and define chronological validation folds."""

import numpy as np
import pandas as pd

from src.config import (
    TARGET,
    VALIDATION_END,
    CLEAN_FULL_FILE,
    CLEAN_PREDICTOR_FILE,
    TRAIN_END,
)




def load_model_data(sample_names=("train", "validation")):
    """Load the model data and return the requested samples."""
    full = pd.read_parquet(CLEAN_FULL_FILE)

    predictors = (
        pd.read_csv(CLEAN_PREDICTOR_FILE)["predictor"]
        .astype(str)
        .tolist()
    )

    samples = {
        "train": full[full["month"] <= TRAIN_END],
        "validation": full[
            (full["month"] > TRAIN_END)
            & (full["month"] <= VALIDATION_END)
        ],
        "development": full[full["month"] <= VALIDATION_END],
        "test": full[full["month"] > VALIDATION_END],
    }

    samples = {
        name: samples[name].reset_index(drop=True)
        for name in sample_names
    }

    return samples, predictors



def arrays(samples, predictors, target=TARGET):
    """Convert each sample into predictor and target arrays."""
    return {
        name: (
            data[predictors].to_numpy(dtype=np.float32),
            data[target].to_numpy(dtype=np.float32),
        )
        for name, data in samples.items()
    }


def expanding_year_folds(data, first_validation_year=2005):
    """Create annual expanding-window validation folds."""
    years = data["month"].dt.year
    folds = []

    for year in sorted(years[years >= first_validation_year].unique()):
        train_mask = years < year
        validation_mask = years == year

        folds.append({
            "train_index": np.flatnonzero(train_mask),
            "validation_index": np.flatnonzero(validation_mask),
        })

    return folds



def ols3_predictors():
    """Return size, book-to-market, and momentum predictors."""
    return [
        "log_comp_market_equity",
        "be_me",
        "mom12m",
    ]
    

    
