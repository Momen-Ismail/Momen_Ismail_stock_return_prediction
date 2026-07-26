"""Run all interpretation scripts in order and stop on the first failure."""

from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "01_model_interpretation.py",
    "02_feature_importance.py",
    "03_create_result_tables.py",
    "04_create_figures.py",
    "05_report_notes.py",
]


def main():
    folder = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = folder / script
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
