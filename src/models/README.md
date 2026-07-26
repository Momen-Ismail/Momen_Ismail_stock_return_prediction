# Model pipeline

This folder implements the stock-return prediction design used in the project.
The target is next-month excess stock return.  Predictors are the rank-normalized
stock characteristics and macro variables produced by the data pipeline.

## Final sample split

- Development sample: January 1990 to December 2019.
- Final test sample: January 2020 to December 2025.

The final test sample is not used for tuning.  Hyperparameters are selected
inside the development sample only.

## Models

- Historical mean benchmark.
- OLS with three predictors: size, book-to-market, and momentum.
- Elastic Net, fixed and tuned.
- Partial Least Squares, fixed and tuned.
- Decision Tree, fixed and tuned.
- Random Forest, fixed and tuned.

Gradient Boosting was explored during model development and archived. It is
not part of the final specification, which is kept parsimonious for clearer
comparison and interpretation.

The project intentionally excludes full OLS, Ridge-only, Lasso-only, PCR/PCA
regression, KNN, neural networks, XGBoost, bagging-only models, and
classification models.

## Tuning design

Tuning uses annual expanding-window validation inside 1990-2019:

- 1990-2004 -> 2005
- 1990-2005 -> 2006
- ...
- 1990-2018 -> 2019

The primary tuning criterion is equal-weighted monthly MSE.  Pooled MSE, RMSE,
MAE, monthly RMSE, and out-of-sample R-squared are saved as diagnostics.

The one-standard-error rule selects the simplest candidate whose mean monthly
MSE is within one standard error of the minimum.

## Execution order

1. Fixed benchmark diagnostics using the preliminary 1990-2014 / 2015-2019
   split.
2. Hyperparameter tuning inside the full 1990-2019 development sample.
3. Fit all fixed and tuned models on 1990-2019.
4. Predict and compare on 2020-2025.
5. Run portfolio sorts using final-test predictions only.
