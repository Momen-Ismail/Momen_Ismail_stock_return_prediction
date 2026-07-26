# Step 6: Model interpretation and reporting

Step 6 converts the latest model outputs into reproducible interpretation
tables, figures, an Excel workbook, and draft report notes. It does not tune
models, select parameters, or use test outcomes for fitting.

## Required prior outputs

- `output/models/test/final_test_metrics.csv`
- `output/models/test/final_test_predictions.parquet`
- `output/models/robustness/time_series_robustness_yearly_metrics.csv`
- `output/models/optimization/fixed_vs_optimized_all_metrics.csv`
- one `*_best_parameters.csv` file per tuned model under
  `output/models/tuning/`
- the final ranked dataset and predictor-name file configured in `src/config.py`

No fitted estimators are currently saved by Steps 3 or 5. Therefore file 02
uses the exact Step 5 constructors and selected parameters to refit OLS-3,
PLS, Elastic Net, Decision Tree, and Random Forest on the development sample.
Test observations are used only to calculate permutation importance.

`00_utils.py` is named first for visibility. It is a shared imported helper,
not a separate manual run step. Python module names cannot begin with a digit
in a normal `from ... import` statement, so the scripts load it through
`import_module`.

## Run order

```bash
python src/models/step_06_interpretation/01_model_interpretation.py
python src/models/step_06_interpretation/02_feature_importance.py
python src/models/step_06_interpretation/03_create_result_tables.py
python src/models/step_06_interpretation/04_create_figures.py
python src/models/step_06_interpretation/05_report_notes.py
```

Or run all five sequentially:

```bash
python src/models/step_06_interpretation/run_all.py
```

All generated files are written to `output/models/interpretation/`. Each run
reads current Steps 1--5 files and overwrites the corresponding Step 6 output.
No model result or conclusion is manually embedded in the scripts.

Coefficient magnitudes and feature importance describe predictive
associations. They should not be interpreted as causal effects. Permutation
importance uses a deterministic test subsample for predictive evaluation only.
