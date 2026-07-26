"""Create feature-importance tables from saved Step 5 model outputs."""

from importlib import import_module
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

utils = import_module(
    "src.models.step_06_interpretation.00_utils"
)


TOP_FEATURES = 20


def top_coefficient_table(
    coefficients,
    model_name,
):
    """Create the unified top-coefficient format for one model."""
    model_data = coefficients[
        coefficients["model"].eq(model_name)
    ].copy()

    if model_name == "elastic_net":
        model_data = model_data[
            model_data["nonzero"]
        ].copy()

    if model_data.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "importance_type",
                "predictor",
                "signed_value",
                "absolute_value",
                "rank",
            ]
        )

    top = (
        model_data
        .sort_values("rank")
        .head(TOP_FEATURES)
        .copy()
    )

    return pd.DataFrame({
        "model": top["model"],
        "importance_type": (
            "scaled_coefficient"
        ),
        "predictor": top["predictor"],
        "signed_value": top["coefficient"],
        "absolute_value": (
            top["absolute_coefficient"]
        ),
        "rank": top["rank"],
    })


def top_tree_importance_table(
    tree_importance,
    model_name,
):
    """Create the unified top tree-importance format."""
    model_data = tree_importance[
        tree_importance["model"].eq(
            model_name
        )
    ].copy()

    top = (
        model_data
        .sort_values("rank")
        .head(TOP_FEATURES)
        .copy()
    )

    return pd.DataFrame({
        "model": top["model"],
        "importance_type": (
            "impurity_importance"
        ),
        "predictor": top["predictor"],
        "signed_value": (
            top["impurity_importance"]
        ),
        "absolute_value": (
            top["impurity_importance"].abs()
        ),
        "rank": top["rank"],
    })


def main():
    """Load Step 5 interpretation inputs and create report tables."""
    inputs = utils.load_interpretation_inputs()

    coefficients = inputs["coefficients"]
    tree_importance = inputs["tree_importance"]
    complexity = inputs["complexity"]
    pls_components = inputs["pls_components"]

    ols_coefficients = (
        coefficients[
            coefficients["model"].eq("ols_3")
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    pls_coefficients = (
        coefficients[
            coefficients["model"].eq("pls")
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    elastic_net_coefficients = (
        coefficients[
            coefficients["model"].eq(
                "elastic_net"
            )
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    decision_tree_importance = (
        tree_importance[
            tree_importance["model"].eq(
                "decision_tree"
            )
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    random_forest_importance = (
        tree_importance[
            tree_importance["model"].eq(
                "random_forest"
            )
        ]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    utils.ensure_unique(
        ols_coefficients,
        ["predictor"],
        "OLS-3 coefficients",
    )

    utils.ensure_unique(
        pls_coefficients,
        ["predictor"],
        "PLS coefficients",
    )

    utils.ensure_unique(
        elastic_net_coefficients,
        ["predictor"],
        "Elastic Net coefficients",
    )

    utils.ensure_unique(
        decision_tree_importance,
        ["predictor"],
        "Decision Tree feature importance",
    )

    utils.ensure_unique(
        random_forest_importance,
        ["predictor"],
        "Random Forest feature importance",
    )

    utils.save_csv(
        ols_coefficients,
        "ols_3_coefficients.csv",
    )

    utils.save_csv(
        pls_coefficients,
        "pls_coefficients.csv",
    )

    utils.save_csv(
        elastic_net_coefficients,
        "elastic_net_coefficients.csv",
    )

    utils.save_csv(
        pls_components,
        "pls_components.csv",
    )

    utils.save_csv(
        decision_tree_importance,
        "decision_tree_feature_importance.csv",
    )

    utils.save_csv(
        random_forest_importance,
        "random_forest_feature_importance.csv",
    )

    utils.save_csv(
        complexity,
        "model_complexity_summary.csv",
    )

    top_tables = [
        top_coefficient_table(
            coefficients,
            "ols_3",
        ),
        top_coefficient_table(
            coefficients,
            "pls",
        ),
        top_coefficient_table(
            coefficients,
            "elastic_net",
        ),
        top_tree_importance_table(
            tree_importance,
            "decision_tree",
        ),
        top_tree_importance_table(
            tree_importance,
            "random_forest",
        ),
    ]

    top_tables = [
        table
        for table in top_tables
        if not table.empty
    ]

    top = pd.concat(
        top_tables,
        ignore_index=True,
    )

    utils.ensure_unique(
        top,
        [
            "model",
            "importance_type",
            "rank",
        ],
        "Top feature table",
    )

    top = top.sort_values(
        [
            "model",
            "importance_type",
            "rank",
        ]
    ).reset_index(drop=True)

    utils.save_csv(
        top,
        "feature_importance_top_variables.csv",
    )

    print("\nSaved coefficient tables:")

    print(
        f"OLS-3: {len(ols_coefficients)} predictors"
    )

    print(
        f"PLS: {len(pls_coefficients)} predictors"
    )

    elastic_nonzero = int(
        elastic_net_coefficients[
            "nonzero"
        ].sum()
    )

    print(
        "Elastic Net: "
        f"{elastic_nonzero} nonzero coefficients "
        f"out of {len(elastic_net_coefficients)}"
    )

    print("\nSaved tree importance tables:")

    print(
        "Decision Tree: "
        f"{len(decision_tree_importance)} predictors"
    )

    print(
        "Random Forest: "
        f"{len(random_forest_importance)} predictors"
    )

    print(
        "\nUnified top-variable rows: "
        f"{len(top)}"
    )


if __name__ == "__main__":
    main()