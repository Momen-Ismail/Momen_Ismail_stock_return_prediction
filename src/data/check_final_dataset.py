"""Validate the final winsorized model dataset."""

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
    LOCKED_CHARACTERISTIC_COLUMNS,
    LOCKED_BINARY_CHARACTERISTIC_COLUMNS,
    LOCKED_CONTINUOUS_CHARACTERISTIC_COLUMNS,
    LOCKED_INTERACTION_CHARACTERISTIC_COLUMNS,
    LOCKED_MARKET_COLUMNS,
    LOCKED_MACRO_COLUMNS,
)


# ---------------------------------------------------------------------
# Check helper
# ---------------------------------------------------------------------
results = []


def check(name, condition):
    condition = bool(condition)
    results.append(condition)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")


# ---------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------
raw = pd.read_parquet(RAW_KELLY_FILE)
clean = pd.read_parquet(CLEAN_FULL_FILE)

raw["month"] = pd.to_datetime(raw["month"])
clean["month"] = pd.to_datetime(clean["month"])

raw = (
    raw.sort_values(["ticker", "month"])
    .reset_index(drop=True)
)

clean = (
    clean.sort_values(["ticker", "month"])
    .reset_index(drop=True)
)

base_predictors = (
    pd.read_csv(RAW_PREDICTOR_FILE)["predictor"]
    .astype(str)
    .tolist()
)

final_predictors = (
    pd.read_csv(CLEAN_PREDICTOR_FILE)["predictor"]
    .astype(str)
    .tolist()
)

summary = (
    pd.read_csv(CLEANING_SUMMARY_FILE)
    .set_index("item")["value"]
    .to_dict()
)

characteristics = list(
    LOCKED_CHARACTERISTIC_COLUMNS
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

market = list(
    LOCKED_MARKET_COLUMNS
)

macro = list(
    LOCKED_MACRO_COLUMNS
)

sic2 = [
    name
    for name in base_predictors
    if name.startswith("sic2_")
]

interaction_names = [
    f"{characteristic}_x_{macro_name}"
    for macro_name in macro
    for characteristic in interaction_characteristics
]


# ---------------------------------------------------------------------
# 1) Output names
# ---------------------------------------------------------------------
print("\n1. Output names")

check(
    "final dataset filename says winsorized",
    "winsorized" in CLEAN_FULL_FILE.name,
)

check(
    "predictor filename says winsorized",
    "winsorized" in CLEAN_PREDICTOR_FILE.name,
)


# ---------------------------------------------------------------------
# 2) Dataset structure
# ---------------------------------------------------------------------
print("\n2. Dataset structure")

check(
    "raw dataset has 211,341 rows",
    len(raw) == 211_341,
)

check(
    "final dataset has 211,059 rows",
    len(clean) == 211_059,
)

check(
    "final dataset has 487 columns",
    clean.shape[1] == 487,
)

check(
    "final dataset has 656 tickers",
    clean["ticker"].nunique() == 656,
)

check(
    "final sample starts in February 1990",
    clean["month"].min()
    == pd.Timestamp("1990-02-28"),
)

check(
    "final sample ends in December 2025",
    clean["month"].max()
    == pd.Timestamp("2025-12-31"),
)

check(
    "ticker-month rows are unique",
    clean.duplicated(
        ["ticker", "month"]
    ).sum() == 0,
)

check(
    "dataset is sorted by ticker and month",
    clean.equals(
        clean.sort_values(
            ["ticker", "month"]
        ).reset_index(drop=True)
    ),
)


# ---------------------------------------------------------------------
# 3) Predictor groups and names
# ---------------------------------------------------------------------
print("\n3. Predictor definitions")

check(
    "48 firm characteristics",
    len(characteristics) == 48,
)

check(
    "45 continuous characteristics",
    len(continuous) == 45,
)

check(
    "3 binary characteristics",
    len(binary) == 3,
)

check(
    "45 interaction characteristics",
    len(interaction_characteristics) == 45,
)

check(
    "4 market variables",
    len(market) == 4,
)

check(
    "8 macro variables",
    len(macro) == 8,
)

check(
    "64 SIC2 dummies",
    len(sic2) == 64,
)

check(
    "124 base predictors",
    len(base_predictors) == 124,
)

check(
    "360 interactions",
    len(interaction_names) == 360,
)

check(
    "484 final predictors",
    len(final_predictors) == 484,
)

check(
    "base predictor names are unique",
    len(base_predictors)
    == len(set(base_predictors)),
)

check(
    "final predictor names are unique",
    len(final_predictors)
    == len(set(final_predictors)),
)

check(
    "binary and continuous groups reproduce all characteristics",
    set(continuous + binary)
    == set(characteristics),
)

check(
    "only continuous characteristics enter interactions",
    interaction_characteristics
    == continuous,
)

check(
    "base predictor ordering is correct",
    base_predictors
    == characteristics + market + macro + sic2,
)

check(
    "final predictor ordering is correct",
    final_predictors
    == base_predictors + interaction_names,
)

check(
    "final columns match predictor file",
    clean.columns.tolist()
    == ["ticker", "month", TARGET]
    + final_predictors,
)


# ---------------------------------------------------------------------
# 4) January 1990 removal
# ---------------------------------------------------------------------
print("\n4. January 1990 removal")

first_month = raw["month"].min()

removed = raw.loc[
    raw["month"].eq(first_month)
]

retained = (
    raw.loc[
        raw["month"].gt(first_month)
    ]
    .sort_values(["ticker", "month"])
    .reset_index(drop=True)
)

check(
    "first raw month is January 1990",
    first_month
    == pd.Timestamp("1990-01-31"),
)

check(
    "exactly 282 rows were removed",
    len(removed) == 282,
)

check(
    "only January 1990 was removed",
    removed["month"]
    .eq(pd.Timestamp("1990-01-31"))
    .all(),
)

check(
    "retained raw rows equal final rows",
    len(retained) == len(clean),
)


# ---------------------------------------------------------------------
# 5) Identifiers and target
# ---------------------------------------------------------------------
print("\n5. Identifiers and target")

check(
    "tickers are preserved",
    clean["ticker"].equals(
        retained["ticker"]
    ),
)

check(
    "months are preserved",
    clean["month"].equals(
        retained["month"]
    ),
)

check(
    "target values are unchanged",
    np.array_equal(
        clean[TARGET].to_numpy(),
        retained[TARGET].to_numpy(),
    ),
)


# ---------------------------------------------------------------------
# 6) Continuous predictor cleaning
# ---------------------------------------------------------------------
print("\n6. Continuous predictor cleaning")

raw_continuous = (
    retained[continuous]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .astype("float64")
)

monthly_medians = (
    raw_continuous
    .groupby(retained["month"])
    .transform("median")
)

imputed = (
    raw_continuous
    .fillna(monthly_medians)
    .fillna(0.0)
)

lower = (
    imputed
    .groupby(retained["month"])
    .transform(
        "quantile",
        q=PREDICTOR_WINSOR_LOWER,
    )
)

upper = (
    imputed
    .groupby(retained["month"])
    .transform(
        "quantile",
        q=PREDICTOR_WINSOR_UPPER,
    )
)

expected_continuous = (
    imputed
    .mask(imputed < lower, lower)
    .mask(imputed > upper, upper)
)

continuous_error = float(
    np.max(
        np.abs(
            clean[continuous]
            .to_numpy(dtype=np.float64)
            - expected_continuous
            .to_numpy(dtype=np.float64)
        )
    )
)

check(
    "continuous variables match median imputation and winsorization",
    continuous_error <= 1e-10,
)

print(
    "Maximum continuous-value error: "
    f"{continuous_error:.12g}"
)

observed = raw_continuous.notna()

inside_bounds = (
    observed
    & raw_continuous.ge(lower)
    & raw_continuous.le(upper)
)

unchanged = np.isclose(
    clean[continuous].to_numpy(
        dtype=np.float64
    ),
    raw_continuous.to_numpy(
        dtype=np.float64
    ),
    rtol=0,
    atol=1e-10,
)

check(
    "ordinary non-outlier values remain unchanged",
    unchanged[
        inside_bounds.to_numpy()
    ].all(),
)

check(
    "existing logged dollar-volume variable is included",
    "avg_log_dolvol_1m"
    in continuous,
)

check(
    "existing logged firm-size variable is included",
    "log_comp_market_equity"
    in continuous,
)


# ---------------------------------------------------------------------
# 7) Binary predictors
# ---------------------------------------------------------------------
print("\n7. Binary predictors")

for name in binary:
    expected = (
        pd.to_numeric(
            retained[name],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )

    actual = (
        pd.to_numeric(
            clean[name],
            errors="coerce",
        )
        .astype("int8")
    )

    check(
        f"{name} follows the missing-to-zero rule",
        actual.equals(expected),
    )

    check(
        f"{name} contains only zero and one",
        set(actual.unique())
        .issubset({0, 1}),
    )


# ---------------------------------------------------------------------
# 8) Market and macro variables
# ---------------------------------------------------------------------
print("\n8. Market and macro variables")

for name in market + macro:
    error = float(
        np.max(
            np.abs(
                clean[name].to_numpy(
                    dtype=np.float64
                )
                - retained[name].to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    check(
        f"{name} remains unchanged",
        error <= 1e-12,
    )


# ---------------------------------------------------------------------
# 9) SIC2 dummies
# ---------------------------------------------------------------------
print("\n9. SIC2 dummies")

sic2_error = float(
    np.max(
        np.abs(
            clean[sic2].to_numpy(
                dtype=np.float64
            )
            - retained[sic2].to_numpy(
                dtype=np.float64
            )
        )
    )
)

check(
    "SIC2 values remain unchanged",
    sic2_error == 0,
)

check(
    "all SIC2 values are zero or one",
    all(
        set(clean[name].unique())
        .issubset({0, 1})
        for name in sic2
    ),
)

check(
    "every observation belongs to one SIC2 group",
    clean[sic2]
    .sum(axis=1)
    .eq(1)
    .all(),
)


# ---------------------------------------------------------------------
# 10) Interactions
# ---------------------------------------------------------------------
print("\n10. Interactions")

interaction_checks = []

for macro_name in macro:
    names = [
        f"{characteristic}_x_{macro_name}"
        for characteristic
        in interaction_characteristics
    ]

    expected = (
        clean[
            interaction_characteristics
        ].to_numpy(dtype=np.float64)
        * clean[
            macro_name
        ].to_numpy(dtype=np.float64)[:, None]
    )

    actual = clean[
        names
    ].to_numpy(dtype=np.float64)

    error = float(
        np.max(
            np.abs(actual - expected)
        )
    )

    passed = error <= 1e-5
    interaction_checks.append(passed)

    check(
        f"interactions with {macro_name} are correct",
        passed,
    )

    print(
        f"Maximum interaction error: "
        f"{error:.12g}"
    )

check(
    "all 360 interactions are correct",
    all(interaction_checks),
)


# ---------------------------------------------------------------------
# 11) Final integrity
# ---------------------------------------------------------------------
print("\n11. Final integrity")

check(
    "no missing values",
    clean.isna().sum().sum() == 0,
)

check(
    "no infinite values",
    np.isfinite(
        clean.select_dtypes(
            include=np.number
        ).to_numpy(dtype=np.float64)
    ).all(),
)

check(
    "no duplicate ticker-months",
    clean.duplicated(
        ["ticker", "month"]
    ).sum() == 0,
)

check(
    "target was not winsorized in File 05",
    np.array_equal(
        clean[TARGET].to_numpy(),
        retained[TARGET].to_numpy(),
    ),
)


# ---------------------------------------------------------------------
# 12) Summary file
# ---------------------------------------------------------------------
print("\n12. Summary file")


def summary_number(name):
    return int(
        float(summary[name])
    )


check(
    "summary raw rows are correct",
    summary_number("raw_rows")
    == len(raw),
)

check(
    "summary removed rows are correct",
    summary_number(
        "rows_dropped_first_month"
    ) == len(removed),
)

check(
    "summary final rows are correct",
    summary_number("final_rows")
    == len(clean),
)

check(
    "summary final columns are correct",
    summary_number("final_columns")
    == clean.shape[1],
)

check(
    "summary final predictors are correct",
    summary_number("final_predictors")
    == len(final_predictors),
)

check(
    "summary interactions are correct",
    summary_number("interactions")
    == len(interaction_names),
)

check(
    "summary first month is correct",
    pd.to_datetime(
        summary["first_month"]
    ) == clean["month"].min(),
)

check(
    "summary last month is correct",
    pd.to_datetime(
        summary["last_month"]
    ) == clean["month"].max(),
)


# ---------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------
print("\n13. Final result")

if all(results):
    print(
        "PASS: the final dataset is correctly constructed. "
        "January 1990 is removed, identifiers and target are "
        "preserved, continuous predictors are median-imputed "
        "and winsorized, ordinary values remain unchanged, "
        "binary/state/SIC2 predictors are preserved, all 360 "
        "interactions are correct, and the output contains no "
        "missing, infinite, or duplicate observations."
    )
else:
    print(
        "FAIL: at least one final dataset check failed."
    )