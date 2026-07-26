"""Create the raw Kelly-style predictor dataset.

The file combines:
- stock characteristics;
- market and VIX variables;
- one-month-lagged Welch-Goyal variables;
- SIC2 industry dummies.

File 05 performs imputation, monthly winsorization,
and interaction construction.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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


MAX_EXACT_MACRO_REPEAT_MONTHS = 12


# ---------------------------------------------------------------------
# SIC2 industry dummies
# ---------------------------------------------------------------------
def add_industry_dummies(data):
    sic2 = (
        pd.to_numeric(
            data["sic2"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    dummies = pd.get_dummies(
        sic2,
        prefix="sic2",
        dtype=np.int8,
    ).rename(
        columns={
            "sic2_-1": "sic2_missing",
        }
    )

    data = pd.concat(
        [
            data.reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return data, dummies.columns.tolist()


# ---------------------------------------------------------------------
# Find longest consecutive run of an unchanged value
# ---------------------------------------------------------------------
def longest_constant_run(data, variable):
    changed = data[variable].ne(
        data[variable].shift(1)
    )

    block = changed.cumsum()

    runs = (
        data.assign(block=block)
        .groupby("block")
        .agg(
            start_month=("source_month", "min"),
            end_month=("source_month", "max"),
            value=(variable, "first"),
            months=(variable, "size"),
        )
        .sort_values(
            "months",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return runs.iloc[0]


# ---------------------------------------------------------------------
# Load and validate Welch-Goyal data
# ---------------------------------------------------------------------
def load_welch_goyal():
    macro_columns = list(
        LOCKED_MACRO_COLUMNS
    )

    macro = pd.read_csv(
        WELCH_GOYAL_INPUT_FILE,
        usecols=["month"] + macro_columns,
    )

    macro["source_month"] = (
        pd.to_datetime(
            macro["month"],
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )

    if macro["source_month"].isna().any():
        raise ValueError(
            "Invalid month values in the Welch-Goyal file."
        )

    if macro["source_month"].duplicated().any():
        raise ValueError(
            "Duplicate months in the Welch-Goyal file."
        )

    for name in macro_columns:
        macro[name] = pd.to_numeric(
            macro[name],
            errors="coerce",
        )

    macro[macro_columns] = (
        macro[macro_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    macro = (
        macro.sort_values("source_month")
        .reset_index(drop=True)
    )

    expected_months = pd.date_range(
        start=macro["source_month"].min(),
        end=macro["source_month"].max(),
        freq="ME",
    )

    if not macro["source_month"].equals(
        pd.Series(expected_months)
    ):
        raise ValueError(
            "The Welch-Goyal file does not contain "
            "a continuous monthly sequence."
        )

    missing_by_variable = (
        macro[macro_columns]
        .isna()
        .sum()
    )

    missing_by_variable = (
        missing_by_variable[
            missing_by_variable.gt(0)
        ]
    )

    if not missing_by_variable.empty:
        raise ValueError(
            "Missing Welch-Goyal values:\n"
            f"{missing_by_variable.to_string()}"
        )

    # Dividend-price and earnings-price ratios should vary monthly.
    for name in ["wg_dp", "wg_ep"]:
        longest_run = longest_constant_run(
            macro,
            name,
        )

        if (
            longest_run["months"]
            > MAX_EXACT_MACRO_REPEAT_MONTHS
        ):
            raise ValueError(
                f"{name} is unchanged for "
                f"{int(longest_run['months'])} consecutive months: "
                f"{longest_run['start_month'].date()} to "
                f"{longest_run['end_month'].date()}. "
                "Correct WELCH_GOYAL_INPUT_FILE before continuing."
            )

    return macro[
        ["source_month"] + macro_columns
    ]


# ---------------------------------------------------------------------
# Add one-month-lagged Welch-Goyal variables
# ---------------------------------------------------------------------
def add_welch_goyal_macro(data):
    macro_columns = list(
        LOCKED_MACRO_COLUMNS
    )

    macro = load_welch_goyal()

    macro["month"] = (
        macro["source_month"]
        + pd.offsets.MonthEnd(1)
    )

    macro = macro[
        ["month"] + macro_columns
    ]

    merged = data.merge(
        macro,
        on="month",
        how="left",
        validate="many_to_one",
    )

    first_month = merged["month"].min()

    unexpected_missing = (
        merged["month"].gt(first_month)
        & merged[macro_columns]
        .isna()
        .any(axis=1)
    )

    if unexpected_missing.any():
        affected_months = (
            merged.loc[
                unexpected_missing,
                "month",
            ]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise ValueError(
            "Missing lagged Welch-Goyal values for: "
            f"{affected_months}"
        )

    return merged


# ---------------------------------------------------------------------
# Confirm required columns
# ---------------------------------------------------------------------
def validate_columns(data, columns, label):
    columns = list(columns)

    missing = [
        name
        for name in columns
        if name not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {label} columns: {missing}"
        )

    return columns


# ---------------------------------------------------------------------
# Select and save raw predictor dataset
# ---------------------------------------------------------------------
def prepare_raw_data(
    data,
    characteristics,
    market_variables,
    macro_variables,
    sic2_dummies,
):
    predictors = (
        characteristics
        + market_variables
        + macro_variables
        + sic2_dummies
    )

    if len(predictors) != len(set(predictors)):
        duplicated = sorted(
            name
            for name in set(predictors)
            if predictors.count(name) > 1
        )

        raise ValueError(
            "Duplicated predictors: "
            f"{duplicated}"
        )

    full = (
        data[
            ["ticker", "month", TARGET]
            + predictors
        ]
        .sort_values(
            ["ticker", "month"]
        )
        .reset_index(drop=True)
    )

    duplicate_rows = int(
        full.duplicated(
            ["ticker", "month"]
        ).sum()
    )

    missing_targets = int(
        full[TARGET].isna().sum()
    )

    infinite_values = int(
        np.isinf(
            full.select_dtypes(
                include=np.number
            )
            .to_numpy(dtype=np.float64)
        ).sum()
    )

    if duplicate_rows:
        raise ValueError(
            "Duplicate ticker-month observations: "
            f"{duplicate_rows}"
        )

    if missing_targets:
        raise ValueError(
            f"Missing targets: {missing_targets}"
        )

    if infinite_values:
        raise ValueError(
            "Infinite numeric values: "
            f"{infinite_values}"
        )

    macro_any_missing = (
        full[macro_variables]
        .isna()
        .any(axis=1)
    )

    macro_all_missing = (
        full[macro_variables]
        .isna()
        .all(axis=1)
    )

    macro_missing_months = (
        full.loc[
            macro_any_missing,
            "month",
        ]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for output_file in [
        RAW_KELLY_FILE,
        RAW_PREDICTOR_FILE,
        RAW_KELLY_SUMMARY_FILE,
    ]:
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    full.to_parquet(
        RAW_KELLY_FILE,
        index=False,
    )

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
        "market_variables": len(market_variables),
        "macro_variables": len(macro_variables),
        "sic2_dummies": len(sic2_dummies),
        "rows_without_compustat": int(
            full["has_compustat_annual"]
            .eq(0)
            .sum()
        ),
        "rows_with_any_macro_missing": int(
            macro_any_missing.sum()
        ),
        "rows_with_all_macro_missing": int(
            macro_all_missing.sum()
        ),
        "macro_missing_months": (
            ", ".join(macro_missing_months)
            if macro_missing_months
            else "none"
        ),
        "infinite_numeric_values": infinite_values,
        "missing_targets": missing_targets,
        "duplicate_ticker_months": duplicate_rows,
        "first_month": full["month"].min(),
        "last_month": full["month"].max(),
        "interactions": (
            "constructed_in_file_05"
        ),
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
# Main
# ---------------------------------------------------------------------
def main():
    data = pd.read_parquet(
        PANEL_WITH_FUNDAMENTALS_FILE
    )

    data["month"] = (
        pd.to_datetime(
            data["month"],
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )

    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data, sic2_dummies = (
        add_industry_dummies(data)
    )

    data = add_welch_goyal_macro(data)

    characteristics = validate_columns(
        data,
        LOCKED_CHARACTERISTIC_COLUMNS,
        "stock-characteristic",
    )

    market_variables = validate_columns(
        data,
        LOCKED_MARKET_COLUMNS,
        "market",
    )

    macro_variables = validate_columns(
        data,
        LOCKED_MACRO_COLUMNS,
        "macro",
    )

    full, predictors, summary = prepare_raw_data(
        data=data,
        characteristics=characteristics,
        market_variables=market_variables,
        macro_variables=macro_variables,
        sic2_dummies=sic2_dummies,
    )

    print("\nFinal summary")
    print(f"Rows: {len(full):,}")
    print(f"Columns: {full.shape[1]:,}")
    print(
        f"Tickers: "
        f"{full['ticker'].nunique():,}"
    )
    print(
        f"Predictors: "
        f"{len(predictors):,}"
    )
    print(
        "Stock characteristics: "
        f"{summary['stock_characteristics']}"
    )
    print(
        "Market variables: "
        f"{summary['market_variables']}"
    )
    print(
        "Macro variables: "
        f"{summary['macro_variables']}"
    )
    print(
        "SIC2 dummies: "
        f"{summary['sic2_dummies']}"
    )
    print(
        "Rows without Compustat: "
        f"{summary['rows_without_compustat']:,}"
    )
    print(
        "Rows with macro missing: "
        f"{summary['rows_with_any_macro_missing']:,}"
    )
    print(
        "Macro-missing months: "
        f"{summary['macro_missing_months']}"
    )
    print(
        "Missing targets: "
        f"{summary['missing_targets']:,}"
    )
    print(
        "Infinite values: "
        f"{summary['infinite_numeric_values']:,}"
    )
    print(
        "Duplicate ticker-months: "
        f"{summary['duplicate_ticker_months']:,}"
    )
    print(
        f"Date range: "
        f"{full['month'].min()} to "
        f"{full['month'].max()}"
    )
    print(f"Saved: {RAW_KELLY_FILE}")
    print(f"Saved: {RAW_PREDICTOR_FILE}")
    print(f"Saved: {RAW_KELLY_SUMMARY_FILE}")


if __name__ == "__main__":
    main()