"""Combine the saved tuning results and selected parameters."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_OUTPUT_DIR  # noqa: E402

OUTPUT_DIR = MODEL_OUTPUT_DIR / "tuning"
FAMILIES = [
    "pls",
    "elastic_net",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]


def main():
    results = []
    selected = []

    for family in FAMILIES:
        family_results = pd.read_csv(
            OUTPUT_DIR / f"{family}_tuning_results.csv"
        )
        if "sample" in family_results and family_results["sample"].eq("test").any():
            raise ValueError("Tuning results must not contain test rows.")
        results.append(family_results)
        selected.append(
            pd.read_csv(OUTPUT_DIR / f"{family}_best_parameters.csv")
        )

    all_results = pd.concat(results, ignore_index=True)
    summary = pd.concat(selected, ignore_index=True).merge(
        all_results,
        on=["model_family", "parameters"],
        how="left",
    )

    all_results.to_csv(OUTPUT_DIR / "tuning_all_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
