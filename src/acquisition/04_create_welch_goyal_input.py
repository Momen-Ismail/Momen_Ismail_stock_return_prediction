"""Validate and register the permanent cleaned Welch-Goyal input.

The cleaned Welch-Goyal file is already treated as an external local input.
This script does not download data; it validates the permanent CSV and records
it in the input manifest.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.acquisition.manifest import update_input_manifest  # noqa: E402
from src.config import INPUT_MANIFEST_FILE, WELCH_GOYAL_INPUT_FILE  # noqa: E402
from src.feature_definitions import LOCKED_MACRO_COLUMNS  # noqa: E402


def load_and_validate_welch_goyal():
    """Validate the local cleaned Welch-Goyal monthly input."""
    if not WELCH_GOYAL_INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing Welch-Goyal input: {WELCH_GOYAL_INPUT_FILE}")

    macro = pd.read_csv(WELCH_GOYAL_INPUT_FILE)
    required_columns = ["month"] + LOCKED_MACRO_COLUMNS
    missing = [column for column in required_columns if column not in macro.columns]

    if missing:
        raise ValueError(f"Missing Welch-Goyal columns: {missing}")

    if macro.empty:
        raise ValueError("Welch-Goyal input is empty.")

    macro = macro[required_columns].copy()
    macro["month"] = pd.to_datetime(macro["month"], errors="coerce")

    if macro["month"].isna().any():
        raise ValueError("Welch-Goyal input has invalid month values.")

    if macro["month"].duplicated().any():
        raise ValueError("Welch-Goyal input contains duplicate month rows.")

    return macro.sort_values("month")


def main():
    macro = load_and_validate_welch_goyal()

    update_input_manifest(
        manifest_file=INPUT_MANIFEST_FILE,
        input_file=WELCH_GOYAL_INPUT_FILE,
        source="Cleaned Welch-Goyal macro input prepared outside the normal pipeline",
        coverage_start=macro["month"].min().date(),
        coverage_end=macro["month"].max().date(),
        notes="Permanent cleaned monthly Welch-Goyal macro predictors.",
    )

    print(f"Validated {WELCH_GOYAL_INPUT_FILE}: {macro.shape}")
    print(f"Date range: {macro['month'].min()} to {macro['month'].max()}")


if __name__ == "__main__":
    main()
