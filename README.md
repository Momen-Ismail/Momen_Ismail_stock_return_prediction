# Stock Return Prediction with Machine Learning

**Author:** Momen Ismail  
**Matriculation number:** 50307564  
**Institutional email:** s73misma@uni-bonn.de  
**Course:** Machine Learning for Finance  
**University:** University of Bonn  
**Instructor:** Prof. Ivan Gufler  

**Repository:**  
https://github.com/Momen-Ismail/stock-return-ml-project

---

## Project overview

This repository contains the code, documentation, and empirical workflow for my
Machine Learning for Finance course project.

The project compares six models for predicting next-month stock excess returns:

1. Historical Mean
2. OLS-3
3. Partial Least Squares
4. Elastic Net
5. Random Forest
6. Gradient Boosting

For stock \(i\) in month \(t\), the information available by the end of month
\(t\) is used to predict:

\[
y_{i,t+1}=r_{i,t+1}-RF_{t+1},
\]

where \(r_{i,t+1}\) is the stock return in the immediately following calendar
month and \(RF_{t+1}\) is the corresponding monthly risk-free rate.

The final dataset contains:

| Item | Value |
|---|---:|
| Stock-month observations | 211,059 |
| Tickers | 656 |
| Predictors | 484 |
| Sample period | February 1990–December 2025 |
| Missing predictor values | 0 |
| Infinite predictor values | 0 |
| Duplicate ticker-month rows | 0 |

---

# Complete replication order

All commands below must be run from the repository root.

The repository root is the directory containing:

```text
README.md
requirements.txt
src/
input/
output/
documentation/
```

The complete empirical workflow is:

```text
Input preparation
        ↓
Data construction: Files 01–05
        ↓
Fixed model estimation
        ↓
Expanding-window hyperparameter tuning
        ↓
Validation-based specification selection
        ↓
Final model refitting and test evaluation
        ↓
Interpretation tables and figures
        ↓
PDF documentation
```

## 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Momen-Ismail/stock-return-ml-project.git
cd stock-return-ml-project
python3 -m pip install -r requirements.txt
```

## 2. Provide the restricted Compustat input

A complete reconstruction from raw data requires the annual Compustat file:

```text
input/raw/compustat_annual_1980_2025.csv
```

The Compustat data were obtained through WRDS and are subject to licensing
restrictions. The raw file is therefore not intended for public redistribution.

The Compustat path defined in:

```text
src/config.py
```

is authoritative. The required file must be placed at that location before
running Data File 03.

## 3. Rebuild the Welch–Goyal input when necessary

The cleaned Welch–Goyal file is already a permanent project input. To rebuild it
from the original workbook, run:

```bash
python3 src/acquisition/04_create_welch_goyal_input.py
```

This command reads:

```text
input/raw/PredictorData2025.xlsx
```

and creates:

```text
input/external/welch_goyal_macro_1990_2025.csv
```

## 4. Run the data-construction pipeline

Run the following scripts in numerical order:

```bash
python3 src/data/01_build_clean_yahoo_daily.py
python3 src/data/02_build_monthly_stock_features.py
python3 src/data/03_add_fundamentals_and_macro.py
python3 src/data/04_build_raw_kelly_dataset.py
python3 src/data/05_clean_and_rank_normalize.py
```

The script name `05_clean_and_rank_normalize.py` is retained from an earlier
version of the project. The final implementation does **not** rank-normalize the
predictors. It applies monthly median imputation and monthly 1st/99th percentile
winsorization to the continuous stock-characteristic block.

The direct execution of the five numbered scripts is the authoritative
data-construction procedure.

## 5. Run the fixed models

```bash
python3 src/models/step_01_fixed/01_fixed_linear_models.py
python3 src/models/step_01_fixed/02_fixed_tree_models.py
python3 src/models/step_01_fixed/03_compare_fixed_models.py
```

An additional diagnostic script is available:

```bash
python3 src/models/step_01_fixed/04_diagnose_fixed_models.py
```

This stage estimates the initial fixed specifications using the training sample
and evaluates them on the 2015–2019 validation period.

## 6. Run expanding-window hyperparameter tuning

```bash
python3 src/models/step_02_tuning/01_tune_linear_models.py
python3 src/models/step_02_tuning/02_tune_tree_models.py
python3 src/models/step_02_tuning/03_compare_tuning_results.py
```

Tuning uses ten annual expanding-window folds:

```text
Train through 2004 → validate on 2005
Train through 2005 → validate on 2006
...
Train through 2013 → validate on 2014
```

The main tuning criterion is average fold-level monthly mean squared error.

The parameters considered are:

- PLS: number of components;
- Elastic Net: `alpha` and `l1_ratio`;
- Random Forest: number of trees;
- Gradient Boosting: number of trees, learning rate, and tree depth.

## 7. Run the optimized models and validation comparison

```bash
python3 src/models/step_03_optimization/01_optimized_linear_models.py
python3 src/models/step_03_optimization/02_optimized_tree_models.py
python3 src/models/step_03_optimization/03_compare_optimized_models.py
python3 src/models/step_03_optimization/04_compare_fixed_vs_optimized.py
```

This stage estimates the tuned candidates on the complete training sample and
compares them with the fixed versions on the separate 2015–2019 validation
period.

The final validation-based choices are:

| Model family | Final specification |
|---|---|
| PLS | Optimized, 2 components |
| Elastic Net | Optimized, `alpha = 0.015`, `l1_ratio = 0.85` |
| Random Forest | Fixed, 100 trees |
| Gradient Boosting | Fixed, 100 trees, learning rate 0.01, depth 2 |

The tuned 300-tree Random Forest and Gradient Boosting specifications were not
used in the final test because they did not improve validation monthly MSE
relative to the fixed 100-tree versions.

## 8. Run the official final-test evaluation

```bash
python3 src/models/step_04_test/01_final_test_evaluation.py
```

This script:

1. loads the locked final specifications;
2. combines the training and validation periods into the development sample;
3. refits all six models using February 1990–December 2019;
4. predicts the untouched January 2020–December 2025 test period;
5. saves the official test metrics and predictions.

The test sample is not used for tuning or specification selection.

Re-running this script with the locked specifications is a replication exercise.
The test results should not be used to change the model specifications.

## 9. Generate the interpretation outputs

```bash
python3 src/models/step_05_interpretation/run_all.py
```

This stage reads the saved official test outputs and generates:

- final model-comparison tables;
- yearly performance results;
- prediction summaries;
- standardized linear-model coefficients;
- Elastic Net selection summaries;
- PLS summaries;
- Random Forest feature importance;
- Gradient Boosting feature importance;
- predictor-group importance;
- report-ready CSV files;
- a formatted Excel workbook;
- final figures.

This stage does not tune, select, or refit the models.

## 10. Build the documentation

```bash
python3 src/documentation/build_documents.py
```

The main compiled documents are written to:

```text
documentation/pdf/
```

---

# Important documentation

## Final paper

**PDF:**

[Open the final paper](documentation/pdf/Momen%20Ismail.pdf)

```text
documentation/pdf/Momen Ismail.pdf
```

**LaTeX source:**

```text
documentation/documents/thesis/paper.tex
```

The paper contains the research question, data summary, methodology, results,
critical reflection, and conclusion.

## Data Construction Guide

**PDF:**

[Open the Data Construction Guide](documentation/pdf/Data_Construction_Guide.pdf)

```text
documentation/pdf/Data_Construction_Guide.pdf
```

**LaTeX source:**

```text
documentation/documents/data_construction/Data_Construction_Guide.tex
```

The Data Construction Guide provides the detailed technical documentation of:

- the five data-construction scripts;
- permanent and restricted inputs;
- stock-return and predictor formulas;
- daily and monthly quality filters;
- target construction;
- the exact-next-calendar-month condition;
- the six-month Compustat availability lag;
- the one-month Welch–Goyal lag;
- monthly median imputation;
- monthly 1st/99th percentile winsorization;
- interaction construction;
- final dataset validation;
- remaining data limitations.

The paper should be used for the empirical conclusions. The Data Construction
Guide should be used for detailed questions about variables, timing rules,
merges, cleaning decisions, and the construction of the final panel.

---

# Data-construction stages

## File 01: Daily Yahoo Finance data

```text
src/data/01_build_clean_yahoo_daily.py
```

This script:

- reads the locked stock universe;
- downloads or loads daily Yahoo Finance data;
- checks OHLCV consistency;
- removes invalid observations;
- removes unusable ticker histories;
- saves the cleaned daily stock panel and quality reports.

## File 02: Monthly stock characteristics and target

```text
src/data/02_build_monthly_stock_features.py
```

This script:

- aggregates daily observations to stock-month observations;
- calculates monthly returns;
- calculates momentum variables;
- constructs volatility and liquidity variables;
- adds market and VIX information;
- calculates beta and idiosyncratic volatility;
- constructs the next-month excess-return target.

The target is retained only when the next observation for the same ticker is the
immediately following calendar month.

## File 03: Compustat fundamentals

```text
src/data/03_add_fundamentals_and_macro.py
```

This script:

- cleans annual Compustat observations;
- creates accounting characteristics;
- assigns an availability date six months after fiscal-year end;
- performs a backward as-of merge;
- prevents future reports from entering earlier stock months;
- invalidates accounting observations that have become too stale.

## File 04: Kelly-style base dataset

```text
src/data/04_build_raw_kelly_dataset.py
```

This script:

- adds SIC2 industry indicators;
- merges the eight Welch–Goyal variables;
- applies the one-month Welch–Goyal lag;
- freezes the 124 base predictors.

## File 05: Final preprocessing and interactions

```text
src/data/05_clean_and_rank_normalize.py
```

This script:

- removes the initial month without lagged macro information;
- imputes continuous characteristics using same-month medians;
- winsorizes continuous characteristics at the monthly 1st and 99th percentiles;
- preserves binary variables as binary;
- creates 360 characteristic–macro interactions;
- saves the final model-ready dataset;
- saves the authoritative predictor list;
- performs the final quality checks.

---

# Final dataset and predictor design

The final 484 predictors consist of:

| Predictor block | Number |
|---|---:|
| Continuous stock characteristics | 45 |
| Binary stock characteristics | 3 |
| Market and VIX variables | 4 |
| Welch–Goyal variables | 8 |
| SIC2 industry indicators | 64 |
| Characteristic–macro interactions | 360 |
| **Total** | **484** |

The 360 interactions are created as:

\[
45\text{ continuous characteristics}
\times
8\text{ macro states}
=
360.
\]

Binary characteristics, market variables, macro variables, and industry
indicators are not interacted.

The final model-ready dataset is:

```text
output/data/final/006_model_dataset_kelly_winsorized_1990_2025.parquet
```

The authoritative predictor list is:

```text
output/data/final/006_predictor_columns_kelly_winsorized.csv
```

The final cleaning summary is:

```text
output/quality/006_cleaning_summary.csv
```

---

# Chronological sample design

| Sample | Period | Observations | Role |
|---|---|---:|---|
| Training | February 1990–December 2014 | 131,061 | Fixed estimation and tuning |
| Validation | January 2015–December 2019 | 35,386 | Fixed-versus-optimized comparison |
| Development | February 1990–December 2019 | 166,447 | Final model refitting |
| Test | January 2020–December 2025 | 44,612 | Official final evaluation |

Random train-test splitting is not used.

January 2026 price information may be used only to realize the next-month target
for December 2025 predictor observations. The final predictor sample ends in
December 2025.

---

# Main outputs

## Final-test outputs

```text
output/models/test/final_test_metrics.csv
output/models/test/final_test_model_comparison.csv
output/models/test/final_test_predictions.parquet
```

The main final ranking is stored in:

```text
output/models/test/final_test_model_comparison.csv
```

## Interpretation outputs

```text
output/models/interpretation/final_prediction_results.csv
output/models/interpretation/fixed_vs_optimized_results.csv
output/models/interpretation/best_hyperparameters.csv
output/models/interpretation/final_predictor_group_summary.csv
output/models/interpretation/final_report_results.xlsx
output/models/interpretation/figures/
```

## Report figures

```text
documentation/figures/
```

---

# Main empirical result

| Rank | Model | Monthly out-of-sample \(R^2\) |
|---:|---|---:|
| 1 | Random Forest | 1.1427% |
| 2 | OLS-3 | 0.0301% |
| 3 | Elastic Net | approximately 0.0000% |
| 4 | Historical Mean | 0.0000% |
| 5 | Partial Least Squares | -0.6363% |
| 6 | Gradient Boosting | -0.9192% |

Random Forest produces the lowest average monthly MSE in the 2020–2025 test
sample and reduces squared forecast error by approximately 1.14% relative to the
historical-mean benchmark.

The improvement is modest and is not stable across every test year. The project
therefore interprets the result as limited statistical predictability rather
than evidence of a profitable trading strategy.

---

# Timing and leakage controls

The main timing rules are:

- a month-\(t\) row predicts the stock's return in month \(t+1\);
- the target requires the exact next calendar month;
- momentum variables use lagged returns;
- Compustat information becomes available six months after fiscal-year end;
- accounting data are merged backward using only reports already considered
  available;
- Welch–Goyal variables are lagged by one month;
- imputation and winsorization use only the same month's cross-section;
- model standardization is fitted only on the relevant training sample;
- hyperparameter tuning uses chronological expanding windows;
- the 2020–2025 test sample is excluded from all model-selection decisions.

---

# Repository structure

```text
stock-return-ml-project/
├── input/
│   ├── raw/
│   ├── external/
│   └── stock_universe_locked.csv
│
├── output/
│   ├── data/
│   │   ├── staging/
│   │   └── final/
│   ├── quality/
│   └── models/
│       ├── fixed/
│       ├── tuning/
│       ├── optimization/
│       ├── test/
│       └── interpretation/
│
├── documentation/
│   ├── documents/
│   │   ├── thesis/
│   │   └── data_construction/
│   ├── figures/
│   ├── tables/
│   └── pdf/
│
├── src/
│   ├── acquisition/
│   ├── data/
│   ├── models/
│   │   ├── step_01_fixed/
│   │   ├── step_02_tuning/
│   │   ├── step_03_optimization/
│   │   ├── step_04_test/
│   │   └── step_05_interpretation/
│   └── documentation/
│
├── README.md
├── requirements.txt
└── run_pipeline.sh
```

---

# Reproducibility note

The complete raw-data reconstruction requires access to the licensed Compustat
input.

The project separates:

- permanent inputs;
- restricted inputs;
- intermediate datasets;
- final model-ready data;
- model-selection outputs;
- official test outputs;
- interpretation outputs;
- documentation.

The authoritative empirical order is:

```text
Data Files 01–05
→ Fixed models
→ Expanding-window tuning
→ Validation comparison
→ Locked final specifications
→ Development-sample refitting
→ One-time test evaluation
→ Interpretation
→ Documentation
```

The predictor list saved beside the final dataset is authoritative and should
not be reconstructed manually.

The official test results must not be used to retune the models or select
alternative specifications.