#!/usr/bin/env bash
set -euo pipefail

python3 src/models/step_01_fixed/01_fixed_linear_models.py
python3 src/models/step_01_fixed/02_fixed_tree_models.py
python3 src/models/step_01_fixed/03_compare_fixed_models.py

python3 src/models/step_02_tuning/01_tune_linear_models.py
python3 src/models/step_02_tuning/02_tune_tree_models.py
python3 src/models/step_02_tuning/03_compare_tuning_results.py

python3 src/models/step_03_selection/01_select_configurations.py
python3 src/models/step_04_test/01_final_test_evaluation.py
python3 src/models/step_05_interpretation/run_all.py
