"""Create the raw Kelly-style predictor dataset.

The file combines:
- stock characteristics;
- market and VIX variables;
- SIC2 industry dummies;
- stock-characteristic × Welch-Goyal interactions.

Final cleaning and normalization are done in file 05.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    PANEL_WITH_FUNDAMENTALS_FILE,
    WELCH_GOYAL_INPUT_FILE,
    RAW_KELLY_FILE,
    RAW_PREDICTOR_FILE,
    RAW_KELLY_SUMMARY_FILE,
)
from src.feature_definitions import (  # noqa: E402
    LOCKED_CHARACTERISTIC_COLUMNS,
    LOCKED_MARKET_COLUMNS,
    LOCKED_MACRO_COLUMNS,
)


# ---------------------------------------------------------------------
# 2) Add industry and macro variables
# ---------------------------------------------------------------------
def add_industry_dummies(data):
    """Create SIC2 industry dummies."""
    sic2 = pd.get_dummies(
        pd.to_numeric(data["sic2"], errors="coerce")
        .fillna(-1)
        .astype(int),
        prefix="sic2",
        dtype=int,
    )

    sic2 = sic2.rename(columns={"sic2_-1": "sic2_missing"})

    return pd.concat([data, sic2], axis=1), sic2.columns.tolist()


def add_welch_goyal_macro(data):
    """Merge permanent Welch-Goyal monthly variables."""
    macro = pd.read_csv(
        WELCH_GOYAL_INPUT_FILE,
        usecols=["month"] + LOCKED_MACRO_COLUMNS,
    )

    macro["month"] = pd.to_datetime(macro["month"])
    macro = (
        macro.dropna(subset=["month"])
        .drop_duplicates("month")
        .sort_values("month")
    )

    return data.merge(macro, on="month", how="left")


# ---------------------------------------------------------------------
# 3) Validate predictors and create interactions
# ---------------------------------------------------------------------
def validate_locked_columns(data, requested, label):
    """Return locked columns or fail if the locked design is unavailable."""
    missing = [name for name in requested if name not in data.columns]

    if missing:
        raise ValueError(f"Missing locked {label} columns: {missing}")

    nonnumeric = [
        name for name in requested
        if not pd.api.types.is_numeric_dtype(data[name])
    ]

    if nonnumeric:
        raise ValueError(f"Non-numeric locked {label} columns: {nonnumeric}")

    empty = [name for name in requested if data[name].isna().all()]

    if empty:
        raise ValueError(f"Completely empty locked {label} columns: {empty}")

    return list(requested)


def add_interactions(data, characteristics):
    """Create stock-characteristic × macro-state interactions."""
    blocks = []

    for macro in LOCKED_MACRO_COLUMNS:
        macro_values = pd.to_numeric(
            data[macro],
            errors="coerce",
        ).astype("float32")

        block = pd.DataFrame(
            {
                f"{characteristic}_x_{macro}": (
                    pd.to_numeric(
                        data[characteristic],
                        errors="coerce",
                    ).astype("float32")
                    * macro_values
                )
                for characteristic in characteristics
            },
            index=data.index,
        )

        blocks.append(block)

    interactions = (
        pd.concat(blocks, axis=1)
        .replace([np.inf, -np.inf], np.nan)
    )

    return (
        pd.concat([data, interactions], axis=1),
        interactions.columns.tolist(),
    )


# ---------------------------------------------------------------------
# 4) Save the raw model dataset
# ---------------------------------------------------------------------
def prepare_model_data(
    data,
    characteristics,
    market_columns,
    interactions,
    sic2_dummies,
):
    """Save the raw predictor panel and predictor list."""
    predictors = (
        characteristics
        + market_columns
        + interactions
        + sic2_dummies
    )

    predictors = list(dict.fromkeys(predictors))
    expected_interactions = len(characteristics) * len(LOCKED_MACRO_COLUMNS)

    if len(interactions) != expected_interactions:
        raise ValueError(
            "Unexpected interaction count: "
            f"created {len(interactions)}, expected {expected_interactions}"
        )

    full = (
        data[["ticker", "month", TARGET] + predictors]
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )

    duplicates = int(
        full.duplicated(["ticker", "month"]).sum()
    )

    if duplicates:
        raise ValueError(
            f"Duplicate ticker-month rows found: {duplicates}"
        )

    missing_targets = int(full[TARGET].isna().sum())

    full.to_csv(RAW_KELLY_FILE, index=False)

    pd.DataFrame(
        {"predictor": predictors}
    ).to_csv(
        RAW_PREDICTOR_FILE,
        index=False,
    )

    summary = {
        "raw_rows": len(full),
        "raw_columns": full.shape[1],
        "raw_predictors": len(predictors),
        "stock_characteristics": len(characteristics),
        "market_variables": len(market_columns),
        "macro_variables": len(LOCKED_MACRO_COLUMNS),
        "interactions": len(interactions),
        "expected_interactions": expected_interactions,
        "sic2_dummies": len(sic2_dummies),
        "missing_targets": missing_targets,
        "duplicate_ticker_months": duplicates,
        "first_month": full["month"].min(),
        "last_month": full["month"].max(),
    }

    pd.DataFrame(
        summary.items(),
        columns=["item", "value"],
    ).to_csv(
        RAW_KELLY_SUMMARY_FILE,
        index=False,
    )

    return full, predictors, summary


# ---------------------------------------------------------------------
# 5) Run pipeline
# ---------------------------------------------------------------------
def main():
    """Build and save the raw Kelly-style dataset."""
    data = pd.read_csv(
        PANEL_WITH_FUNDAMENTALS_FILE,
        low_memory=False,
    )

    data["month"] = pd.to_datetime(data["month"])
    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data, sic2_dummies = add_industry_dummies(data)
    data = add_welch_goyal_macro(data)

    characteristics = validate_locked_columns(
        data,
        LOCKED_CHARACTERISTIC_COLUMNS,
        "stock-characteristic",
    )

    market_columns = validate_locked_columns(
        data,
        LOCKED_MARKET_COLUMNS,
        "market",
    )

    validate_locked_columns(
        data,
        LOCKED_MACRO_COLUMNS,
        "macro",
    )

    data, interactions = add_interactions(
        data,
        characteristics,
    )

    full, predictors, summary = prepare_model_data(
        data=data,
        characteristics=characteristics,
        market_columns=market_columns,
        interactions=interactions,
        sic2_dummies=sic2_dummies,
    )

    print(f"Saved: {RAW_KELLY_FILE}")
    print(f"Shape: {full.shape}")
    print(f"Predictors: {len(predictors)}")
    print(
        f"Stock characteristics: "
        f"{summary['stock_characteristics']}"
    )
    print(
        f"Market variables: "
        f"{summary['market_variables']}"
    )
    print(f"Macro variables: {summary['macro_variables']}")
    print(f"Interactions: {summary['interactions']}")
    print(f"Expected interactions: {summary['expected_interactions']}")
    print(f"SIC2 dummies: {summary['sic2_dummies']}")
    print(f"Missing targets: {summary['missing_targets']}")
    print(
        f"Duplicate ticker-months: "
        f"{summary['duplicate_ticker_months']}"
    )
    print(
        f"Date range: {full['month'].min()} "
        f"to {full['month'].max()}"
    )


if __name__ == "__main__":
    main()
