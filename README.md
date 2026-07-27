# Stock Return Prediction with Machine Learning

This project builds a monthly stock-level dataset and compares machine-learning models for predicting next-month stock excess returns.

For each stock \(i\) and month \(t\), information available by the end of month \(t\) is used to predict:

\[
r_{i,t+1} - RF_{t+1},
\]

where \(r_{i,t+1}\) is the stock return in the following month and \(RF_{t+1}\) is the corresponding risk-free rate.

The project covers:

- data acquisition and cleaning;
- predictor construction;
- leakage-safe preprocessing;
- fixed model estimation;
- hyperparameter tuning;
- validation-based model selection;
- final out-of-sample testing;
- model interpretation and reporting.

---

## 1. Project workflow

The project consists of two main stages:

1. Data construction
2. Model estimation and evaluation

---

## 2. Data construction

The production data pipeline is located under:

```text
src/data/
```

### Step 1: Build clean Yahoo Finance daily data

```text
01_build_clean_yahoo_daily.py
```

This step:

- loads the locked stock universe;
- downloads or reads daily Yahoo Finance stock data;
- applies transparent daily price-quality filters;
- removes unusable stock histories;
- saves the clean daily stock-price panel.

### Step 2: Build monthly stock characteristics

```text
02_build_monthly_stock_features.py
```

This step:

- aggregates daily prices to monthly observations;
- constructs returns, momentum, volatility, liquidity, beta, and trend variables;
- adds market and VIX information;
- constructs the next-month excess-return target;
- removes repeated poor-quality monthly histories.

### Step 3: Add Compustat fundamentals and macroeconomic data

```text
03_add_fundamentals_and_macro.py
```

This step:

- cleans annual Compustat observations;
- constructs accounting characteristics;
- applies a conservative six-month reporting lag;
- uses a backward as-of merge to assign the latest available accounting report to each stock-month;
- invalidates accounting observations that have become too stale;
- adds lagged macroeconomic and Fama–French variables.

### Step 4: Build the raw Kelly-style dataset

```text
04_build_raw_kelly_dataset.py
```

This step:

- adds SIC2 industry controls;
- adds eight Welch–Goyal aggregate state variables;
- creates stock-characteristic × macro-state interactions;
- combines stock, market, macroeconomic, and accounting predictors.

### Step 5: Clean and rank-normalize the final dataset

```text
05_clean_and_rank_normalize.py
```

This step:

- drops observations without a valid target;
- documents target outliers;
- imputes missing predictors using monthly cross-sectional medians;
- rank-normalizes continuous predictors within each month;
- saves the final model-ready panel;
- saves the authoritative predictor list and cleaning summary.

---

## 3. Timing and leakage controls

The project uses several controls to prevent look-ahead bias.

### Target timing

A row dated month \(t\) contains information available by the end of month \(t\) and predicts the excess return in month \(t+1\).

### Accounting information

Compustat `datadate` is the fiscal-year-end date, not the date on which the information became public.

Annual Compustat observations are therefore treated as available only after a six-month reporting lag.

For each stock-month, the merge uses the most recent report whose availability date is not later than the stock month.

### Macroeconomic information

Welch–Goyal and other macroeconomic variables are lagged before entering the predictor set.

### Cross-sectional preprocessing

Monthly imputation and rank normalization use only firms observed in the same month.

No future month is used when preprocessing an observation.

### Chronological samples

The modeling workflow uses the following chronological samples:

```text
Training:     February 1990 – December 2014
Validation:   January 2015 – December 2019
Development:  February 1990 – December 2019
Test:         January 2020 – December 2025
```

The test period is not used for model selection or hyperparameter tuning.

January 2026 price information may be used only to realize the December 2025 next-month target. Final predictor rows end in December 2025.

---

## 4. Model workflow

The modeling code is located under:

```text
src/models/
```

### Step 1: Fixed models

```text
src/models/step_01_fixed/
```

This stage estimates the initial fixed model specifications:

- Historical Mean
- OLS-3
- Partial Least Squares
- Elastic Net
- Random Forest
- Gradient Boosting

These specifications provide baseline results before tuning.

### Step 2: Hyperparameter tuning

```text
src/models/step_02_tuning/
```

Hyperparameters are selected using annual expanding-window validation folds inside the training period.

The primary tuning criterion is average monthly mean squared error.

The tuned parameters are:

- PLS: number of components;
- Elastic Net: `alpha` and `l1_ratio`;
- Random Forest: number of trees;
- Gradient Boosting: number of trees, learning rate, and maximum depth.

### Step 3: Optimized-model comparison

```text
src/models/step_03_optimization/
```

This stage:

- estimates tuned models on the training sample;
- evaluates them on the 2015–2019 validation period;
- ranks models using validation monthly MSE;
- compares fixed and optimized specifications;
- locks the final model specifications before the test sample is evaluated.

### Step 4: Final test evaluation

```text
src/models/step_04_test/
```

The final model specifications are refitted on the complete development sample from February 1990 through December 2019.

They are then evaluated once on the untouched test period from January 2020 through December 2025.

The main evaluation measures are:

- monthly MSE;
- monthly RMSE;
- monthly out-of-sample \(R^2\);
- pooled MSE;
- pooled RMSE;
- pooled MAE;
- pooled out-of-sample \(R^2\);
- prediction-target correlation.

The historical mean is the benchmark used to calculate out-of-sample \(R^2\).

### Step 5: Interpretation and reporting

```text
src/models/step_05_interpretation/
```

This stage reads the saved final-test outputs and creates:

- model-performance summaries;
- yearly test results;
- prediction-bias and prediction-dispersion measures;
- standardized linear-model coefficients;
- Elastic Net variable-selection summaries;
- PLS component summaries;
- Random Forest feature importance;
- Gradient Boosting feature importance;
- predictor-family importance;
- report-ready CSV tables;
- a formatted Excel workbook;
- final figures.

This stage does not tune, select, or refit models.

Run the complete interpretation stage with:

```bash
python3 src/models/step_05_interpretation/run_all.py
```

---

## 5. Required inputs

The main permanent input files include:

```text
input/raw/compustat_annual_1980_2025.csv
input/raw/PredictorData2025.xlsx
input/stock_universe_locked.csv
input/fama_french_rf_monthly.csv
input/market_gspc_daily.csv
input/market_vix_daily.csv
input/external/welch_goyal_macro_1990_2025.csv
```

One-time acquisition scripts are located under:

```text
src/acquisition/
```

Normal production runs use the locked local inputs and do not repeatedly download Wikipedia, Fama–French, GSPC, VIX, or Welch–Goyal data.

---

## 6. Installation

Install the required Python packages from the project root:

```bash
python3 -m pip install -r requirements.txt
```

---

## 7. Running the data pipeline

Run the complete data-construction pipeline from the project root:

```bash
./run_pipeline.sh
```

The script executes the production data steps in the correct order.

---

## 8. Main data outputs

The final model-ready dataset is:

```text
output/data/final/006_model_dataset_kelly_winsorized_1990_2025.parquet
```

The authoritative predictor list is:

```text
output/data/final/006_predictor_columns_kelly_winsorized.csv
```

The cleaning summary is:

```text
output/quality/006_cleaning_summary.csv
```

The current final dataset contains:

- 211,059 stock-month observations;
- 656 tickers;
- 484 predictors;
- observations from February 1990 through December 2025;
- no missing predictor values;
- no infinite values;
- no duplicated stock-month rows.

The predictor set includes:

- stock characteristics;
- accounting characteristics;
- market and VIX variables;
- Welch–Goyal state variables;
- SIC2 industry indicators;
- stock-characteristic × macro-state interactions.

---

## 9. Main model outputs

Model outputs are written under:

```text
output/models/
```

Important subfolders include:

```text
output/models/fixed/
output/models/tuning/
output/models/optimization/
output/models/test/
output/models/interpretation/
```

The final test outputs include:

```text
output/models/test/final_test_metrics.csv
output/models/test/final_test_model_comparison.csv
output/models/test/final_test_predictions.parquet
```

The final interpretation outputs include:

```text
output/models/interpretation/final_prediction_results.csv
output/models/interpretation/fixed_vs_optimized_results.csv
output/models/interpretation/best_hyperparameters.csv
output/models/interpretation/final_predictor_group_summary.csv
output/models/interpretation/final_report_results.xlsx
output/models/interpretation/figures/
```

---

## 10. Documentation

Editable documentation sources are stored under:

```text
documentation/documents/
```

The main documents are:

```text
documentation/documents/thesis/paper.tex
documentation/documents/thesis/references.bib
documentation/documents/data_construction/Data_Construction_Guide.tex
documentation/documents/audit/full_project_audit.tex
documentation/documents/audit/full_project_audit.md
```

Generated document assets are stored under:

```text
documentation/figures/
documentation/tables/
```

Final compiled PDFs are written under:

```text
documentation/pdf/
```

Build all standalone LaTeX documents with:

```bash
python3 src/documentation/build_documents.py
```

The separate pipeline-overview PDF is not retained.

A concise operational overview is included in this README. Detailed technical construction decisions are documented in the Data Construction Guide, while the academic methodology and results are presented in the thesis.

---

## 11. Project structure

```text
stock_return_ml_project_clean/
├── input/
├── output/
├── documentation/
│   ├── documents/
│   │   ├── audit/
│   │   ├── data_construction/
│   │   └── thesis/
│   ├── figures/
│   ├── tables/
│   └── pdf/
├── src/
│   ├── acquisition/
│   ├── data/
│   ├── documentation/
│   └── models/
├── README.md
├── requirements.txt
└── run_pipeline.sh
```

---

## 12. Reproducibility

The production pipeline separates:

- permanent raw inputs;
- intermediate data;
- final model-ready data;
- model outputs;
- interpretation outputs;
- documentation sources;
- generated reports.

The scripts contain data-construction, estimation, evaluation, and reporting logic.

Manual exploratory checks and obsolete scripts are excluded from the production workflow.

The final test period must remain untouched after the official final evaluation. Test results must not be used to retune models or select alternative specifications.

---

## 13. Interpretation caution

The project is designed for prediction rather than causal inference.

Linear-model coefficients describe conditional predictive associations after predictor standardization.

PLS coefficients summarize predictive relationships through latent components and should be interpreted cautiously.

Tree impurity importance measures how variables contribute to reductions in squared error within fitted trees. It may favor predictors with more possible split points.

Neither coefficients nor feature-importance measures should be interpreted as causal effects.

Monthly stock returns are noisy, predictor relationships may vary across market regimes, and small positive out-of-sample improvements should not be overstated.