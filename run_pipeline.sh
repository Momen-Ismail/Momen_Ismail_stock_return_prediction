#!/bin/bash

set -e

echo "01 - Build and clean Yahoo daily data"
python src/data/01_build_clean_yahoo_daily.py

echo "02 - Build monthly stock features"
python src/data/02_build_monthly_stock_features.py

echo "03 - Add fundamentals and macro variables"
python src/data/03_add_fundamentals_and_macro.py

echo "04 - Build raw Kelly-style dataset"
python src/data/04_build_raw_kelly_dataset.py

echo "05 - Clean and rank-normalize dataset"
python src/data/05_clean_and_rank_normalize.py

echo "Pipeline completed."
