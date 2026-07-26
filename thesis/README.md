# Machine Learning for Finance paper

This directory contains a simplified, single-file LaTeX reporting layer for
the frozen Machine Learning for Finance project. It does not execute or modify
the Python pipeline.

## Files

- `paper.tex`: complete paper, including packages, required sections, tables,
  figures, and appendix.
- `references.bib`: bibliography database.
- `Makefile`: reproducible compilation commands.
- `paper.pdf`: compiled paper.

## Compile

From the repository root:

```bash
make -C thesis
```

Or from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The command runs LaTeX and Biber as required and produces `thesis/paper.pdf`.
Use `make -C thesis clean` to remove auxiliary files while retaining the PDF,
or `make -C thesis distclean` to remove all build artifacts including the PDF.

## Frozen result inputs

The paper reads report tables directly from
`../output/models/interpretation/`:

- `final_prediction_results.csv`
- `best_hyperparameters.csv`
- `final_model_complexity.csv`
- `fixed_vs_optimized_results.csv`
- `yearly_prediction_results.csv`
- `feature_importance_top_variables.csv`

Figures are included directly from
`../output/models/interpretation/figures/`. No table values or figures are
copied into this directory.

If separately approved output files are replaced, recompiling the paper reads
their current values automatically. File names and CSV column names must remain
unchanged.

## Manual completion

Search `paper.tex` for `TODO:`. These visible markers identify author metadata,
financial interpretation, institutional details, and conclusions requiring
manual verification. They should not be replaced with invented explanations.
