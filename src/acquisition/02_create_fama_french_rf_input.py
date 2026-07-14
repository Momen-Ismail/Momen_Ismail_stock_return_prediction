"""Create the permanent monthly Fama-French risk-free-rate input."""

from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.acquisition.manifest import update_input_manifest  # noqa: E402
from src.config import FF_URL, FAMA_FRENCH_RF_FILE, INPUT_MANIFEST_FILE  # noqa: E402


def download_fama_french_factors():
    """Download and parse monthly Fama-French three-factor data."""
    response = requests.get(FF_URL, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        lines = archive.read(name).decode("latin1").splitlines()

    header_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith(",Mkt-RF,SMB,HML,RF")
    )

    data_lines = []
    for line in lines[header_index + 1:]:
        first_value = line.split(",", 1)[0].strip()
        if not first_value.isdigit():
            break
        data_lines.append(line)

    factors = pd.DataFrame(
        [line.split(",") for line in data_lines],
        columns=["month", "Mkt_RF", "SMB", "HML", "RF"],
    )

    factors["month"] = pd.to_datetime(factors["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    factors["RF"] = pd.to_numeric(factors["RF"], errors="coerce") / 100.0

    return factors[["month", "RF"]].dropna().sort_values("month")


def main():
    rf = download_fama_french_factors()

    if rf.empty:
        raise ValueError("Downloaded Fama-French RF data is empty.")

    FAMA_FRENCH_RF_FILE.parent.mkdir(parents=True, exist_ok=True)
    rf.to_csv(FAMA_FRENCH_RF_FILE, index=False)

    update_input_manifest(
        manifest_file=INPUT_MANIFEST_FILE,
        input_file=FAMA_FRENCH_RF_FILE,
        source=FF_URL,
        coverage_start=rf["month"].min().date(),
        coverage_end=rf["month"].max().date(),
        notes="Monthly Fama-French RF in decimal form; columns: month, RF.",
    )

    print(f"Saved {FAMA_FRENCH_RF_FILE}: {rf.shape}")
    print(f"Date range: {rf['month'].min()} to {rf['month'].max()}")


if __name__ == "__main__":
    main()
