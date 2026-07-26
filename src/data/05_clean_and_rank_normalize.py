"""Create the final winsorized Kelly-style model dataset.

Continuous firm characteristics are imputed with same-month medians
and winsorized within each month. Original units are preserved.
Binary characteristics, market variables, macro variables, and SIC2
dummies remain unchanged. Interactions are created between continuous
firm characteristics and lagged macro variables.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    TARGET,
    PREDICTOR_WINSOR_LOWER,
    PREDICTOR_WINSOR_UPPER,
    RAW_KELLY_FILE,
    RAW_PREDICTOR_FILE,
    CLEAN_FULL_FILE,
    CLEAN_PREDICTOR_FILE,
    CLEANING_SUMMARY_FILE,
)
from src.feature_definitions import (  # noqa: E402
    LOCKED_BINARY_CHARACTERISTIC_COLUMNS,
    LOCKED_CONTINUOUS_CHARACTERISTIC_COLUMNS,
    LOCKED_INTERACTION_CHARACTERISTIC_COLUMNS,
    LOCKED_MARKET_COLUMNS,
    LOCKED_MACRO_COLUMNS,
)


# ---------------------------------------------------------------------
# 1) Impute and winsorize continuous firm characteristics
# ---------------------------------------------------------------------
def clean_continuous_characteristics(
    data,
    characteristics,
):
    monthly_medians = (
        data.groupby("month")[characteristics]
        .transform("median")
    )

    data[characteristics] = (
        data[characteristics]
        .fillna(monthly_medians)
        .fillna(0.0)
        .astype("float64")
    )

    lower = (
        data.groupby("month")[characteristics]
        .transform(
            "quantile",
            q=PREDICTOR_WINSOR_LOWER,
        )
    )

    upper = (
        data.groupby("month")[characteristics]
        .transform(
            "quantile",
            q=PREDICTOR_WINSOR_UPPER,
        )
    )

    values = data[characteristics]

    data[characteristics] = (
        values
        .mask(values < lower, lower)
        .mask(values > upper, upper)
    )

    return data


# ---------------------------------------------------------------------
# 2) Create continuous-characteristic macro interactions
# ---------------------------------------------------------------------
def create_interactions(
    data,
    characteristics,
    macro_variables,
):
    blocks = []

    for macro in macro_variables:
        block = pd.DataFrame(
            {
                f"{characteristic}_x_{macro}": (
                    data[characteristic]
                    * data[macro]
                )
                for characteristic in characteristics
            },
            index=data.index,
        )

        blocks.append(block)

    interactions = pd.concat(
        blocks,
        axis=1,
    )

    data = pd.concat(
        [data, interactions],
        axis=1,
    )

    return data, interactions.columns.tolist()


# ---------------------------------------------------------------------
# 3) Build and save the final model dataset
# ---------------------------------------------------------------------
def main():
    data = pd.read_parquet(
        RAW_KELLY_FILE
    )

    data["month"] = pd.to_datetime(
        data["month"]
    )

    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    base_predictors = (
        pd.read_csv(
            RAW_PREDICTOR_FILE
        )["predictor"]
        .astype(str)
        .tolist()
    )

    continuous = list(
        LOCKED_CONTINUOUS_CHARACTERISTIC_COLUMNS
    )

    binary = list(
        LOCKED_BINARY_CHARACTERISTIC_COLUMNS
    )

    interaction_characteristics = list(
        LOCKED_INTERACTION_CHARACTERISTIC_COLUMNS
    )

    market_variables = list(
        LOCKED_MARKET_COLUMNS
    )

    macro_variables = list(
        LOCKED_MACRO_COLUMNS
    )

    sic2_dummies = [
        name
        for name in base_predictors
        if name.startswith("sic2_")
    ]

    raw_rows = len(data)
    first_raw_month = data["month"].min()

    data = (
        data.loc[
            data["month"] > first_raw_month
        ]
        .copy()
    )

    rows_dropped = (
        raw_rows - len(data)
    )

    data[binary] = (
        data[binary]
        .fillna(0)
        .astype("int8")
    )

    data[sic2_dummies] = (
        data[sic2_dummies]
        .astype("int8")
    )

    data = clean_continuous_characteristics(
        data,
        continuous,
    )

    data, interactions = create_interactions(
        data,
        interaction_characteristics,
        macro_variables,
    )

    predictors = (
        base_predictors
        + interactions
    )

    clean = (
        data[
            [
                "ticker",
                "month",
                TARGET,
            ]
            + predictors
        ]
        .sort_values(
            ["ticker", "month"]
        )
        .reset_index(drop=True)
    )

    missing_values = int(
        clean.isna().sum().sum()
    )

    infinite_values = int(
        np.isinf(
            clean.select_dtypes(
                include=np.number
            )
        ).sum().sum()
    )

    duplicate_rows = int(
        clean.duplicated(
            ["ticker", "month"]
        ).sum()
    )

    if (
        missing_values
        or infinite_values
        or duplicate_rows
    ):
        raise ValueError(
            "Final dataset failed checks: "
            f"missing={missing_values}, "
            f"infinite={infinite_values}, "
            f"duplicates={duplicate_rows}"
        )

    CLEAN_FULL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLEANING_SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_parquet(
        CLEAN_FULL_FILE,
        index=False,
    )

    pd.DataFrame({
        "predictor": predictors,
    }).to_csv(
        CLEAN_PREDICTOR_FILE,
        index=False,
    )

    summary = {
        "raw_rows": raw_rows,
        "rows_dropped_first_month": rows_dropped,
        "final_rows": len(clean),
        "final_columns": clean.shape[1],
        "base_predictors": len(base_predictors),
        "continuous_characteristics": len(continuous),
        "binary_characteristics": len(binary),
        "market_variables": len(market_variables),
        "macro_variables": len(macro_variables),
        "sic2_dummies": len(sic2_dummies),
        "interactions": len(interactions),
        "final_predictors": len(predictors),
        "unique_tickers": clean["ticker"].nunique(),
        "first_month": clean["month"].min(),
        "last_month": clean["month"].max(),
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "duplicate_ticker_months": duplicate_rows,
        "continuous_missing_treatment": (
            "same_month_median"
        ),
        "continuous_outlier_treatment": (
            "same_month_1_99_percent_winsorization"
        ),
        "continuous_scaling": "none",
        "market_macro_scaling": "none",
        "target_treatment": "unchanged",
        "interaction_definition": (
            "winsorized_continuous_characteristic_"
            "times_previous_month_macro"
        ),
    }

    pd.DataFrame(
        summary.items(),
        columns=[
            "item",
            "value",
        ],
    ).to_csv(
        CLEANING_SUMMARY_FILE,
        index=False,
    )

    print("\nFinal summary")
    print(f"Raw rows: {raw_rows:,}")
    print(
        "Rows dropped from first month: "
        f"{rows_dropped:,}"
    )
    print(f"Final rows: {len(clean):,}")
    print(f"Final columns: {clean.shape[1]}")
    print(
        f"Final tickers: "
        f"{clean['ticker'].nunique()}"
    )
    print(
        f"Base predictors: "
        f"{len(base_predictors)}"
    )
    print(
        "Continuous characteristics: "
        f"{len(continuous)}"
    )
    print(
        f"Binary characteristics: "
        f"{len(binary)}"
    )
    print(
        f"Market variables: "
        f"{len(market_variables)}"
    )
    print(
        f"Macro variables: "
        f"{len(macro_variables)}"
    )
    print(
        f"SIC2 dummies: "
        f"{len(sic2_dummies)}"
    )
    print(
        f"Interactions: "
        f"{len(interactions)}"
    )
    print(
        f"Final predictors: "
        f"{len(predictors)}"
    )
    print(
        f"Missing values: "
        f"{missing_values}"
    )
    print(
        f"Infinite values: "
        f"{infinite_values}"
    )
    print(
        "Duplicate ticker-months: "
        f"{duplicate_rows}"
    )
    print(
        f"Date range: "
        f"{clean['month'].min()} "
        f"to {clean['month'].max()}"
    )
    print(f"Saved: {CLEAN_FULL_FILE}")
    print(f"Saved: {CLEAN_PREDICTOR_FILE}")
    print(f"Saved: {CLEANING_SUMMARY_FILE}")


if __name__ == "__main__":
    main()