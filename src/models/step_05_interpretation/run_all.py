"""Run all interpretation scripts in order and stop on the first failure."""

from pathlib import Path
import subprocess
import sys


SCRIPTS = [
    "01_model_interpretation.py",
    "02_feature_importance.py",
    "03_create_result_tables.py",
    "04_create_figures.py",
]


def main():
    """Run the complete interpretation stage."""
    folder = Path(__file__).resolve().parent

    for script in SCRIPTS:
        path = folder / script

        if not path.exists():
            raise FileNotFoundError(
                f"Missing interpretation script: {path}"
            )

        print(
            f"\nRunning {script}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(path),
            ],
            check=True,
        )

    print(
        "\nAll interpretation scripts completed successfully."
    )


if __name__ == "__main__":
    main()