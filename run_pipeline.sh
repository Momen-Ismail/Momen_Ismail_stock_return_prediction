#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

echo "01 - Build and clean Yahoo daily data"
python src/data/01_build_clean_yahoo_daily.py

echo "02 - Build monthly stock features"
python src/data/02_build_monthly_stock_features.py

echo "03 - Add lagged Compustat fundamentals"
python src/data/03_add_fundamentals_and_macro.py

echo "04 - Build raw Kelly-style dataset"
python src/data/04_build_raw_kelly_dataset.py

echo "05 - Clean and rank-normalize dataset"
python src/data/05_clean_and_rank_normalize.py

echo "Model stage 01 - fixed train/validation benchmarks"
python src/models/step_01_fixed/01_fixed_linear_models.py
python src/models/step_01_fixed/02_fixed_tree_models.py
python src/models/step_01_fixed/03_compare_fixed_models.py
python src/models/step_01_fixed/04_diagnose_fixed_models.py

echo "Model stage 02 - annual development-sample tuning"
python src/models/step_02_tuning/01_tune_linear_models.py
python src/models/step_02_tuning/02_tune_tree_models.py
python src/models/step_02_tuning/03_compare_scaling.py

echo "Model stage 03 - final 2020-2025 test comparison"
python src/models/step_03_optimization/06_final_test_evaluation.py

echo "Model stage 04 - final-test portfolio analysis"
python src/models/step_04_portfolio/01_portfolio_sorts.py
python src/models/step_04_portfolio/02_compare_portfolio_results.py
python src/models/step_04_portfolio/03_final_model_and_portfolio_summary.py

echo "Pipeline completed."
