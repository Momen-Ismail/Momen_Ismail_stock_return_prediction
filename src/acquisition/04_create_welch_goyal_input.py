"""Create and validate the cleaned Welch-Goyal monthly input."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.manifest import update_input_manifest  # noqa: E402
from src.config import (  # noqa: E402
    INPUT_MANIFEST_FILE,
    WELCH_GOYAL_INPUT_FILE,
)
from src.feature_definitions import (  # noqa: E402
    LOCKED_MACRO_COLUMNS,
)


WELCH_GOYAL_RAW_FILE = (
    PROJECT_ROOT
    / "input"
    / "raw"
    / "PredictorData2025.xlsx"
)

START_MONTH = pd.Timestamp("1990-01-31")
END_MONTH = pd.Timestamp("2025-12-31")


# ---------------------------------------------------------------------
# Longest consecutive run of an unchanged value
# ---------------------------------------------------------------------
def longest_constant_run(data, variable):
    blocks = (
        data[variable]
        .ne(data[variable].shift(1))
        .cumsum()
    )

    runs = (
        data.assign(block=blocks)
        .groupby("block", as_index=False)
        .agg(
            start_month=("month", "min"),
            end_month=("month", "max"),
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
# Create cleaned Welch-Goyal predictors
# ---------------------------------------------------------------------
def create_welch_goyal_input():
    source_columns = [
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "svar",
    ]

    raw = pd.read_excel(
        WELCH_GOYAL_RAW_FILE,
        sheet_name="Monthly",
        usecols=source_columns,
    )

    yyyymm = pd.to_numeric(
        raw["yyyymm"],
        errors="coerce",
    ).astype("Int64")

    raw["month"] = (
        pd.to_datetime(
            yyyymm.astype(str),
            format="%Y%m",
            errors="coerce",
        )
        + pd.offsets.MonthEnd(0)
    )

    numeric_columns = [
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "svar",
    ]

    for name in numeric_columns:
        raw[name] = pd.to_numeric(
            raw[name],
            errors="coerce",
        )

    raw = (
        raw.loc[
            raw["month"].between(
                START_MONTH,
                END_MONTH,
            )
        ]
        .sort_values("month")
        .reset_index(drop=True)
    )

    macro = pd.DataFrame(
        {
            "month": raw["month"],
            "wg_dp": np.log(
                raw["D12"] / raw["Index"]
            ),
            "wg_ep": np.log(
                raw["E12"] / raw["Index"]
            ),
            "wg_bm": raw["b/m"],
            "wg_ntis": raw["ntis"],
            "wg_tbl": raw["tbl"],
            "wg_tms": raw["lty"] - raw["tbl"],
            "wg_dfy": raw["BAA"] - raw["AAA"],
            "wg_svar": raw["svar"],
        }
    )

    macro = macro[
        ["month"] + list(LOCKED_MACRO_COLUMNS)
    ]

    macro[list(LOCKED_MACRO_COLUMNS)] = (
        macro[list(LOCKED_MACRO_COLUMNS)]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return macro


# ---------------------------------------------------------------------
# Validate cleaned data
# ---------------------------------------------------------------------
def validate_welch_goyal(macro):
    expected_months = pd.Series(
        pd.date_range(
            START_MONTH,
            END_MONTH,
            freq="ME",
        ),
        name="month",
    )

    if not macro["month"].equals(expected_months):
        raise ValueError(
            "Welch-Goyal file does not contain a continuous "
            "January 1990–December 2025 monthly sequence."
        )

    missing = (
        macro.isna()
        .sum()
    )

    missing = missing[
        missing.gt(0)
    ]

    if not missing.empty:
        raise ValueError(
            "Missing Welch-Goyal values:\n"
            f"{missing.to_string()}"
        )

    if macro["month"].duplicated().any():
        raise ValueError(
            "Duplicate Welch-Goyal months."
        )

    for name in ["wg_dp", "wg_ep"]:
        longest = longest_constant_run(
            macro,
            name,
        )

        if longest["months"] > 12:
            raise ValueError(
                f"{name} is unchanged for "
                f"{int(longest['months'])} months: "
                f"{longest['start_month'].date()} to "
                f"{longest['end_month'].date()}."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not WELCH_GOYAL_RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing raw Welch-Goyal file: "
            f"{WELCH_GOYAL_RAW_FILE}"
        )

    macro = create_welch_goyal_input()

    validate_welch_goyal(macro)

    WELCH_GOYAL_INPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    macro.to_csv(
        WELCH_GOYAL_INPUT_FILE,
        index=False,
    )

    update_input_manifest(
        manifest_file=INPUT_MANIFEST_FILE,
        input_file=WELCH_GOYAL_INPUT_FILE,
        source=(
            "Amit Goyal PredictorData2025.xlsx, "
            "Monthly sheet"
        ),
        coverage_start=macro["month"].min().date(),
        coverage_end=macro["month"].max().date(),
        notes=(
            "wg_dp=log(D12/Index); "
            "wg_ep=log(E12/Index); "
            "wg_tms=lty-tbl; "
            "wg_dfy=BAA-AAA. "
            "No forward filling."
        ),
    )

    print("\nWelch-Goyal input created")
    print(f"Rows: {len(macro):,}")
    print(f"Columns: {macro.shape[1]:,}")
    print(
        f"Date range: "
        f"{macro['month'].min()} to "
        f"{macro['month'].max()}"
    )
    print(
        f"Unique wg_dp: "
        f"{macro['wg_dp'].nunique():,}"
    )
    print(
        f"Unique wg_ep: "
        f"{macro['wg_ep'].nunique():,}"
    )
    print(f"Saved: {WELCH_GOYAL_INPUT_FILE}")
    print(f"Updated: {INPUT_MANIFEST_FILE}")


if __name__ == "__main__":
    main()