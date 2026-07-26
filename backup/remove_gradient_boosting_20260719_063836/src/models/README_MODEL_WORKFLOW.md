# Model workflow

The modeling pipeline has six chronological stages. Model selection uses only
the training and validation periods. The test period is opened once, after all
specifications and hyperparameters have been fixed.

## 1. Fixed models

Scripts: `step_01_fixed/01_fixed_linear_models.py`,
`02_fixed_tree_models.py`, and `03_compare_fixed_models.py`.

The models are fitted on `train` and evaluated on `train` and `validation`.
Outputs are written to `output/models/fixed/`. The optional diagnostic script
`04_diagnose_fixed_models.py` remains in this folder.

## 2. Tuning

Scripts: `step_02_tuning/01_tune_linear_models.py`,
`02_tune_tree_models.py`, and `03_compare_tuning_results.py`.

Tuning uses the training sample only and annual expanding-window folds. It
writes family-level candidate results, selected parameters,
`tuning_all_results.csv`, and `tuning_summary.csv` to `output/models/tuning/`.
The earlier scaling diagnostic is preserved as `03_compare_scaling.py`.

## 3. Optimization

Scripts: `step_03_optimization/01_optimized_linear_models.py`,
`02_optimized_tree_models.py`, `03_compare_optimized_models.py`, and
`04_compare_fixed_vs_optimized.py`.

Optimized models load the saved tuning parameters, fit on `train`, and evaluate
`train` and `validation`. Validation selects the preferred specification.
Outputs are written to `output/models/optimization/`. The comparison scripts
reject metric files containing test rows.

## 4. Robustness

Script: `step_04_robustness/01_time_series_robustness.py`.

This optional stage performs annual expanding- and rolling-window checks using
data through validation only. Outputs are written to
`output/models/robustness/`.

## 5. Final test

Script: `step_05_test/01_final_test_evaluation.py`.

After validation has fixed the specifications, models are refitted on the
development sample and evaluated once on `test`. Outputs are written to
`output/models/test/`.

## 6. Portfolio analysis

Scripts: `step_06_portfolio/01_portfolio_sorts.py`,
`02_compare_portfolio_results.py`, and
`03_final_model_and_portfolio_summary.py`.

These scripts use final-test predictions only and write to
`output/models/portfolio/`.

## Naming conventions

- Benchmarks: `historical_mean`, `ols_3`
- Fixed: `pls_fixed`, `elastic_net_fixed`, `decision_tree_fixed`,
  `random_forest_fixed`, `gradient_boosting_fixed`
- Optimized: `pls_optimized`, `elastic_net_optimized`,
  `decision_tree_optimized`, `random_forest_optimized`,
  `gradient_boosting_optimized`

Ranked model tables use the column name `rank`. Test observations must never
enter tuning, optimization, robustness, or validation-comparison outputs.

The older robustness, final-test, and portfolio scripts remain in their
original locations for preservation. The organized stage 04--06 copies above
are the canonical paths for future runs.
