"""Helpers for maintaining the permanent input manifest."""

from datetime import date

import pandas as pd


MANIFEST_COLUMNS = [
    "file",
    "source",
    "created_on",
    "coverage_start",
    "coverage_end",
    "notes",
]


def update_input_manifest(manifest_file, input_file, source, coverage_start, coverage_end, notes):
    """Insert or replace one manifest row for a permanent input file."""
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    input_file = input_file.resolve()

    if manifest_file.exists():
        manifest = pd.read_csv(manifest_file)
    else:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)

    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""

    row = {
        "file": str(input_file),
        "source": source,
        "created_on": date.today().isoformat(),
        "coverage_start": str(coverage_start),
        "coverage_end": str(coverage_end),
        "notes": notes,
    }

    manifest = manifest[manifest["file"].ne(row["file"])]
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest = manifest[MANIFEST_COLUMNS].sort_values("file")
    manifest.to_csv(manifest_file, index=False)
