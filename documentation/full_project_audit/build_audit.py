"""Generate the repository audit Markdown and LaTeX sources.

This documentation-only helper reads the repository without changing it and
writes only files in documentation/full_project_audit.
"""

from __future__ import annotations

import ast
from collections import Counter
from datetime import date
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MD = OUT / "full_project_audit.md"
TEX = OUT / "full_project_audit.tex"
PDF = OUT / "full_project_audit.pdf"
SNAPSHOT_DATE = date(2026, 7, 20)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_records() -> list[dict]:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or ".git" in path.parts or OUT in path.parents or path == OUT:
            continue
        rel = relative(path)
        suffix = path.suffix.lower() or path.name
        if "__pycache__" in rel or suffix in {".pyc", ".DS_Store"}:
            category = "cache or temporary"
        elif rel.startswith("backup/"):
            category = "backup or possibly obsolete"
        elif rel.startswith("src/") and suffix == ".py":
            category = "source code"
        elif rel.startswith("input/"):
            category = "data input"
        elif rel.startswith("output/data/intermediate/"):
            category = "intermediate data"
        elif rel.startswith("output/data/final/"):
            category = "final data"
        elif rel.startswith("output/models/"):
            category = "model output"
        elif suffix in {".png", ".jpg", ".jpeg"}:
            category = "figure"
        elif suffix in {".md", ".tex", ".pdf", ".bib"} or rel.startswith(("docs/", "thesis/")):
            category = "documentation"
        elif path.name in {"requirements.txt", ".gitignore", "Makefile"} or suffix in {".sh", ".toml", ".yaml", ".yml"}:
            category = "configuration or environment"
        else:
            category = "other"
        records.append({
            "path": rel,
            "name": path.name,
            "type": suffix,
            "size": path.stat().st_size,
            "category": category,
        })
    return records


def python_inventory(records: list[dict]) -> tuple[list[dict], list[dict]]:
    files, functions = [], []
    for record in records:
        if record["type"] != ".py":
            continue
        path = ROOT / record["path"]
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except Exception as error:
            files.append({**record, "lines": 0, "imports": [], "functions": [], "main": False, "error": str(error)})
            continue
        imports, local_functions, constants = [], [], []
        main = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                item = {
                    "file": record["path"],
                    "name": node.name,
                    "args": args,
                    "line": node.lineno,
                    "doc": (ast.get_docstring(node) or "No function docstring.").splitlines()[0],
                }
                functions.append(item)
                local_functions.append(item)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append((node.module or "") + ":" + ",".join(alias.name for alias in node.names))
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.isupper():
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        value = ast.unparse(node.value)
                    constants.append((name, str(value)[:160], node.lineno))
            elif isinstance(node, ast.If):
                try:
                    expression = ast.unparse(node.test)
                    main = main or ("__name__" in expression and "__main__" in expression)
                except Exception:
                    pass
        files.append({
            **record,
            "lines": len(source.splitlines()),
            "imports": sorted(set(imports)),
            "functions": local_functions,
            "constants": constants,
            "main": main,
        })
    return files, functions


def human_size(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{value} B"


def mpath(path: str) -> str:
    return "\\path{" + path + "}"


def escape_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(escape_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def role_for(path: str, category: str) -> str:
    name = Path(path).name
    if path == "src/config.py": return "Central paths, dates, target, and sample boundaries."
    if path == "src/feature_definitions.py": return "Locked predictor-family definitions."
    if path == "run_pipeline.sh": return "Intended end-to-end execution wrapper; currently stale after model reorganization."
    if path.startswith("src/acquisition/"): return "One-time acquisition or input-registration logic."
    if path.startswith("src/data/"): return "Active data-construction stage."
    if path.startswith("src/models/utils/"): return "Shared model loading, estimation, or evaluation utility."
    if path.startswith("src/models/step_01_fixed/"): return "Fixed train/validation benchmark stage."
    if path.startswith("src/models/step_02_tuning/"): return "Annual expanding-window tuning stage."
    if path.startswith("src/models/step_03_optimization/"): return "Optimized train/validation comparison stage."
    if path.startswith("src/models/step_04_robustness/"): return "Time-series robustness stage."
    if path.startswith("src/models/step_05_test/"): return "One-time final test estimation and output stage."
    if path.startswith("src/models/step_06_interpretation/"): return "Frozen-output interpretation and report-artifact stage."
    if "/legacy/" in path or path.startswith("backup/"): return "Archived snapshot; not part of the active workflow."
    if path.startswith("input/"): return "Permanent local input or input metadata."
    if path.startswith("output/data/intermediate/"): return "Saved intermediate data product."
    if path.startswith("output/data/final/"): return "Saved final or model-ready data product."
    if path.startswith("output/models/legacy/"): return "Archived model output from removed Gradient Boosting workflow."
    if path.startswith("output/models/"): return "Frozen model, diagnostic, portfolio, or interpretation output."
    if path.startswith("thesis/"): return "Current course-paper source, build artifact, or bibliography."
    if path.startswith("docs/") or path.startswith("src/documentation/"): return "Earlier data-construction or pipeline documentation."
    if category == "cache or temporary": return "Machine-generated cache or operating-system metadata."
    if name == "requirements.txt": return "Unpinned Python dependency list."
    return "Repository support file; exact role follows from its category and name."


def status_for(path: str, category: str) -> tuple[str, str, str]:
    if category == "cache or temporary": return "No", "Regenerable", "Cache/temporary"
    if path.startswith("backup/") or "/legacy/" in path: return "No for active run", "Archival copy", "Possibly obsolete"
    if path.startswith("output/models/legacy/"): return "No for active run", "From archived code", "Legacy"
    if path.startswith("input/"): return "Yes if referenced", "External acquisition required", "Input"
    if path.startswith("output/"): return "Yes if consumed", "Reproducible only if inputs/code match", "Frozen output"
    if path.startswith("src/") or path in {"run_pipeline.sh", "requirements.txt"}: return "Usually", "Source-controlled logic", "Active or support"
    return "Context dependent", "Unclear or documented", "Review"


def lineage_for(path: str) -> tuple[str, str]:
    mappings = [
        ("stock_universe_locked.csv", "src/acquisition/01_create_locked_stock_universe.py", "src/data/01_build_clean_yahoo_daily.py"),
        ("fama_french_rf_monthly.csv", "src/acquisition/02_create_fama_french_rf_input.py", "src/data/02_build_monthly_stock_features.py"),
        ("market_gspc_daily.csv", "src/acquisition/03_download_market_inputs.py", "src/data/02_build_monthly_stock_features.py"),
        ("market_vix_daily.csv", "src/acquisition/03_download_market_inputs.py", "src/data/02_build_monthly_stock_features.py"),
        ("welch_goyal_macro_1990_2025.csv", "external preparation; validator 04_create_welch_goyal_input.py", "src/data/04_build_raw_kelly_dataset.py"),
        ("compustat_annual_1980_2025.csv", "WRDS/Compustat extraction; not scripted", "src/data/03_add_fundamentals_and_macro.py"),
        ("daily_prices_raw_1987_2026.csv", "src/data/01_build_clean_yahoo_daily.py", "same script"),
        ("daily_prices_clean_1987_2026.csv", "src/data/01_build_clean_yahoo_daily.py", "src/data/02_build_monthly_stock_features.py"),
        ("monthly_stock_panel_with_targets_1990_2025.csv", "src/data/02_build_monthly_stock_features.py", "src/data/03_add_fundamentals_and_macro.py"),
        ("monthly_panel_with_compustat_macro_1990_2025.csv", "src/data/03_add_fundamentals_and_macro.py", "src/data/04_build_raw_kelly_dataset.py"),
        ("model_dataset_kelly_raw_full_1990_2025.csv", "src/data/04_build_raw_kelly_dataset.py", "src/data/05_clean_and_rank_normalize.py"),
        ("model_dataset_kelly_ranked_full_1990_2025.parquet", "src/data/05_clean_and_rank_normalize.py", "src/models/utils/data.py"),
        ("final_test_predictions.parquet", "src/models/step_05_test/01_final_test_evaluation.py", "src/models/step_06_interpretation/*"),
        ("final_prediction_results.csv", "src/models/step_06_interpretation/03_create_result_tables.py", "thesis/paper.tex and interpretation figures"),
    ]
    for token, producer, consumer in mappings:
        if token in path:
            return producer, consumer
    if path.startswith("output/models/interpretation/figures/"):
        return "src/models/step_06_interpretation/04_create_figures.py", "thesis/paper.tex"
    if path.startswith("output/models/"):
        return "Model-stage script matching directory", "Later comparison/interpretation scripts"
    if path.startswith("backup/") or "/legacy/" in path:
        return "Historical reorganization/removal process", "No active consumer found"
    return "Not verifiable from the current repository.", "Not verifiable from the current repository."


DATASETS = [
    ["Locked universe", "input/stock_universe_locked.csv", "867 x 1", "n/a", "867", "0", "0", "Permanent Wikipedia-derived current-plus-historical union"],
    ["Fama-French RF", "input/fama_french_rf_monthly.csv", "1,199 x 2", "1926-07 to 2026-05", "n/a", "0", "0 months", "RF decimal, month-end"],
    ["GSPC daily", "input/market_gspc_daily.csv", "9,845 x 7", "1987-01-02 to 2026-01-30", "1 series", "0", "0 dates", "auto_adjust=False"],
    ["VIX daily", "input/market_vix_daily.csv", "9,087 x 7", "1990-01-02 to 2026-01-30", "1 series", "0", "0 dates", "auto_adjust=False"],
    ["Raw Compustat", "input/raw/compustat_annual_1980_2025.csv", "21,575 x 84", "1980-01-31 to 2026-01-31", "649 tickers", "148,161", "0 ticker-date", "Unsorted input; annual statements"],
    ["Welch-Goyal input", "input/external/welch_goyal_macro_1990_2025.csv", "432 x 10", "1990-01 to 2025-12", "n/a", "0", "0 months", "Nine values plus month; only eight are used"],
    ["Clean daily prices", "output/data/final/daily_prices_clean_1987_2026.csv", "4,701,144 x 8", "1987-01-02 to 2026-01-30", "670", "0", "0 ticker-date", "Sorted"],
    ["Monthly stock panel", "output/data/final/monthly_stock_panel_with_targets_1990_2025.csv", "212,794 x 30", "1990-01 to 2025-12", "663", "43,229", "0 ticker-month", "54 infinities before final cleaning"],
    ["Fundamental panel", "output/data/intermediate/monthly_panel_with_compustat_macro_1990_2025.csv", "212,794 x 61", "1990-01 to 2025-12", "663", "497,813", "0 ticker-month", "207,739 matched rows; no lag violation"],
    ["Raw Kelly panel", "output/data/final/model_dataset_kelly_raw_full_1990_2025.csv", "212,794 x 126", "1990-01 to 2025-12", "663", "482,648", "0 ticker-month", "123 base predictors"],
    ["Ranked model panel", "output/data/final/model_dataset_kelly_ranked_full_1990_2025.parquet", "212,794 x 502", "1990-01 to 2025-12", "663", "0", "0 ticker-month", "499 predictors; no infinities"],
    ["Final test predictions", "output/models/test/final_test_predictions.parquet", "268,968 x 6", "2020-01 to 2025-12", "645", "0", "0 ticker-month-model", "44,828 rows per six models"],
]


DECISIONS = [
    ["D01", "Locked union of current and recorded historical S&P 500 tickers", "src/acquisition/01_create_locked_stock_universe.py:64-105", "Broadens coverage beyond current constituents", "Wikipedia history and ticker identifiers remain incomplete", "Verify"],
    ["D02", "Yahoo download uses adjusted close with auto_adjust=False", "src/data/01_build_clean_yahoo_daily.py:71-103", "Preserves raw OHLC and adjusted close", "Vendor revisions and delisting gaps", "Retain with disclosure"],
    ["D03", "Daily ticker removal if bad-row share exceeds 10%", "src/config.py:69; data file 01:183-194", "Drops severely corrupted histories", "Threshold is judgmental; failed downloads are not saved", "Verify"],
    ["D04", "Daily returns outside [-0.95, 3.0] are invalid", "data file 01:146-149", "Flags extreme likely data errors", "Could remove genuine events or fail to catch split errors", "Reconsider"],
    ["D05", "Ticker dropped after two implausible monthly returns", "data file 02:85-126", "Targets repeated corruption", "Whole-history deletion and gap creation", "Reconsider"],
    ["D06", "Monthly return is month-end adjusted-close percentage change", "data file 02:54-77", "Standard simple return", "Next-observation shift is unsafe across missing months", "Retain formula; verify continuity"],
    ["D07", "Momentum excludes current return via shift(1)", "data file 02:154-169", "Prevents overlap with current-month return", "Requires complete histories", "Retain"],
    ["D08", "Six-, 12-, and 36-month compound windows require full windows", "data file 02:159-169", "Transparent history requirement", "Missing periods are observation windows, not guaranteed calendar windows", "Verify"],
    ["D09", "Beta and idiosyncratic volatility use 252 daily observations, minimum 126", "data file 02:286-335", "Uses approximately one year of data", "Residual-variance shortcut is not a full rolling regression", "Verify"],
    ["D10", "Target uses group shift(-1) then subtracts shifted RF", "data file 02:365-385", "Creates next-observation excess return", "26 retained transitions skip calendar months", "Reconsider urgently"],
    ["D11", "Sample ends 2025-12; price download includes 2026-01", "src/config.py:54-58", "Makes December 2025 target possible", "Depends on stable January 2026 input", "Retain"],
    ["D12", "Compustat keeps consolidated industrial standard USD records", "data file 03:82-91", "Improves comparability", "May exclude financial-format observations and create coverage patterns", "Retain with disclosure"],
    ["D13", "Accounting availability equals fiscal date plus six months at month-end", "data file 03:318-323", "Conservative look-ahead protection", "Ignores actual filing dates", "Retain"],
    ["D14", "Backward as-of merge by ticker", "data file 03:374-402", "Prevents future statements entering past months", "Ticker matching can break across symbol changes", "Retain and verify matches"],
    ["D15", "SIC2 missing is an explicit dummy", "data file 04:37-59", "Preserves unmatched observations", "Many sparse indicators increase dimensionality", "Retain"],
    ["D16", "Welch-Goyal variables merge on same month", "data file 04:62-84", "Simple alignment", "Contradicts intended one-month macro lag and may use unavailable information", "Reconsider urgently"],
    ["D17", "Eight macro series are used; inflation column is ignored", "src/feature_definitions.py:83-92", "Locks intended state variables", "Input contains an unused ninth series", "Verify"],
    ["D18", "Missing predictors use monthly cross-sectional medians, then zero", "data file 05:71-105", "No time-series look-ahead; retains rows", "Zero fallback for all-missing month is arbitrary", "Retain with sensitivity check"],
    ["D19", "Forty-seven characteristics rank to [-1,1] each month", "data file 05:108-131", "Robust to outliers and comparable scales", "Removes magnitude information", "Retain"],
    ["D20", "376 ranked-characteristic x macro interactions", "data file 05:134-166", "Allows state dependence", "Dominates 499-dimensional design and raises overfitting risk", "Reconsider"],
    ["D21", "Raw target is not winsorized", "data file 05:377-417", "Preserves economic outcomes", "Squared loss is dominated by extremes", "Sensitivity check"],
    ["D22", "Train ends 2014; validation ends 2019; test begins 2020", "src/config.py:60-63; models/utils/data.py:27-35", "Chronological separation", "Test has only 72 months", "Retain"],
    ["D23", "Tuning is executed on train only, annual folds beginning 2005", "tuning files and models/utils/data.py:57-71", "Prevents 2015-2019 optimization leakage", "Contradicts README claim of folds through 2019", "Clarify"],
    ["D24", "Candidate selected by minimum average monthly MSE", "models/utils/estimation.py:60-100", "Directly optimizes stated loss", "No uncertainty-aware one-standard-error rule despite documentation", "Clarify or implement later"],
    ["D25", "Linear estimators use StandardScaler inside sklearn pipelines", "fixed/tuning/test linear scripts", "Scaler fits only on model training data", "Cross-sectional ranks are scaled again", "Retain"],
    ["D26", "Tree models use unscaled 499 predictors", "tree scripts", "Scaling unnecessary for splits", "Sparse/high-dimensional importance is unstable", "Retain with caution"],
    ["D27", "Random seed 42 for stochastic estimators", "tree, robustness, and test scripts", "Reproducible conditional on versions/threading", "Dependencies are unpinned", "Retain and lock environment"],
    ["D28", "Monthly MSE averages firm MSE equally over months", "models/utils/evaluation.py:8-15", "Each month has equal weight", "Differs slightly from pooled MSE when counts vary", "Retain"],
    ["D29", "Historical mean for final test is the development target mean", "step_05_test:324-374", "Allowable constant benchmark", "Not recursively updated during test", "Retain and describe"],
    ["D30", "Final models refit on 1990-2019 then predict 2020-2025 once", "step_05_test:324-524", "Clean final evaluation", "Whether researchers avoided viewing test during development is not verifiable", "Process control"],
]


ISSUES = [
    ["I01", "High", "Validity", "26 targets cross gaps and are not next-calendar-month returns", "Monthly panel plus data file 02 shift(-1)", "Require the next observed month to equal month+1 before assigning target"],
    ["I02", "High", "Validity", "Same-month macro merge contradicts intended one-month lag", "data file 04:62-84", "Verify publication timing and lag macro inputs explicitly"],
    ["I03", "High", "Reproducibility", "run_pipeline.sh calls five missing scripts", "Direct existence check", "Update wrapper after deciding canonical stages"],
    ["I04", "High", "Reproducibility", "README tuning period and one-standard-error claim do not match code", "models README versus tuning/estimation code", "Correct documentation or implementation"],
    ["I05", "Medium", "Reproducibility", "Failed Yahoo downloads are printed but not persisted", "data file 01 main", "Save an acquisition failure report"],
    ["I06", "Medium", "Validity", "Wikipedia union cannot guarantee point-in-time membership or delisting returns", "Universe acquisition logic", "Use CRSP permanent identifiers for a future version"],
    ["I07", "Medium", "Validity", "Ticker-based Compustat-Yahoo merge risks symbol-change mismatches", "data file 03", "Audit permanent identifier mapping"],
    ["I08", "Medium", "Interpretation", "499 predictors include 376 interactions", "Cleaning summary", "Report regularization burden and run lower-dimensional sensitivity"],
    ["I09", "Medium", "Interpretation", "Unwinsorized target contains 91 observations above 100% absolute return", "extreme_target_counts.csv", "Report robustness to robust loss or documented winsorization"],
    ["I10", "Medium", "Empirical credibility", "No formal forecast-loss comparison test", "No active test file found", "Add Diebold-Mariano or panel-aware resampling in future"],
    ["I11", "Medium", "Economic interpretation", "Portfolio outputs exist but current paper focuses mostly on prediction", "output/models/portfolio versus thesis", "Decide whether portfolio evidence is in scope and discuss costs"],
    ["I12", "Medium", "Reproducibility", "requirements are unpinned and Python version is unspecified", "requirements.txt", "Lock package and Python versions"],
    ["I13", "Medium", "Reproducibility", "Input manifest stores machine-specific absolute paths", "input/input_manifest.csv", "Store repository-relative paths"],
    ["I14", "Medium", "Staleness", "Source caches remain for deleted scripts", "orphan pyc inventory", "Remove caches from submissions; do not treat them as source"],
    ["I15", "Low", "Organization", "Identical Fama-French RF files exist in input/ and input/external/", "Matching SHA-1", "Keep one canonical copy after confirming consumers"],
    ["I16", "Low", "Organization", "Large legacy and backup trees duplicate old model logic and outputs", "backup/ and legacy/", "Exclude from submission archive or label clearly"],
    ["I17", "Medium", "Reproducibility", "No automated unit-test suite is present", "Repository tree", "Add tests for target continuity, lags, folds, and metrics"],
    ["I18", "Low", "Presentation", "Existing thesis build auxiliaries and OS metadata are present", "thesis/*.aux etc. and .DS_Store", "Clean submission artifact directories"],
    ["I19", "Medium", "Data quality", "54 infinities exist before final cleaning", "Dataset scan", "Document affected columns and assert replacement before imputation"],
    ["I20", "Medium", "Methodology", "Beta/idio-vol formula is a rolling variance identity, not an explicit regression with intercept", "data file 02:306-321", "Clarify definition or compare with rolling OLS"],
    ["I21", "Low", "Presentation", "Existing bibliography contains an unresolved course-reference comment", "thesis/references.bib", "Verify required course citations before submission"],
    ["I22", "Medium", "Process", "Untouched-test discipline cannot be proven from code after results exist", "Repository state", "Describe the procedural claim cautiously"],
]


VERIFICATION = [
    [1, "Next-month stock excess-return target", "Partially verified", "Formula is stock next-observation return minus next RF; 26 nonconsecutive transitions."],
    [2, "Time-aware shifting", "Partially verified", "Sorted group shift(-1), but no calendar-continuity guard."],
    [3, "Sort before group shifts", "Verified", "Data file 02 sorts ticker/month before shifts."],
    [4, "Predictor timing", "Partially verified", "Stock and Compustat timing mostly sound; macro lag is unresolved/inconsistent."],
    [5, "Month-end dates", "Verified", "All audited monthly inputs and panels are month-end."],
    [6, "Decimal returns", "Verified", "RF ranges 0 to 0.0069 and return magnitudes are decimal simple returns."],
    [7, "Daily returns for volatility", "Verified", "Daily adjusted-close percentage changes feed monthly and rolling volatility."],
    [8, "Monthly adjusted-close return", "Verified", "Last monthly adjusted close percentage change."],
    [9, "Rolling features historical only", "Partially verified", "Momentum is lagged; contemporaneous month-end variables are used for t+1, as intended."],
    [10, "36-month history", "Partially verified", "min_periods=36, but the window counts observations and missing months are not explicitly checked."],
    [11, "Six-month Compustat lag", "Verified", "207,739 matched rows and zero saved timing violations."],
    [12, "Backward Compustat as-of merge", "Verified", "direction='backward' by ticker."],
    [13, "No future Compustat reports", "Verified", "Zero comp_available_month > stock month violations."],
    [14, "Macro variables appropriately lagged", "Inconsistent", "Active code merges same-month values; intended one-month lag absent."],
    [15, "Fama-French alignment", "Verified", "Month-end RF shifted one row; monthly input is unique and decimal."],
    [16, "Market/VIX are market-wide", "Verified", "At most one value per month across tickers for all four variables."],
    [17, "Predictors lagged where required", "Partially verified", "Target and momentum timing explicit; macro publication timing unresolved."],
    [18, "Training-only standardization", "Verified", "StandardScaler is inside each fitted sklearn pipeline."],
    [19, "No validation/test scaler fit", "Verified", "Pipeline fit occurs on fold training, train, or development arrays only."],
    [20, "Chronological rather than random CV", "Verified", "Annual expanding masks use years < validation year."],
    [21, "Expanding folds use earlier data", "Verified", "1990-2004 -> 2005 through 1990-2013 -> 2014 in actual tuning."],
    [22, "Hyperparameters exclude test", "Verified", "Tuning loads train only; final script only reads saved parameters."],
    [23, "Test not used for tuning", "Verified in code", "Human viewing history is not verifiable from current repository."],
    [24, "Final test untouched until evaluation", "Not verifiable", "Code separates it, but procedural history cannot be proven."],
    [25, "Predictor file matches model columns", "Verified", "499 unique names exactly equal parquet columns after identifiers/target."],
    [26, "Identifiers and target excluded", "Verified", "Predictor list begins after ticker, month, target."],
    [27, "Missing handling consistent", "Verified for final panel", "Monthly median then zero; final missing count is zero."],
    [28, "No duplicate ticker-month rows", "Verified", "Zero in monthly, raw, and ranked panels."],
    [29, "Documented dataset dimensions", "Verified for generated summaries", "212,794 x 502 and 499 predictors match output metadata."],
    [30, "Result tables match predictions", "Verified", "Recomputed discrepancies below 7.5e-11."],
    [31, "Figures match frozen results", "Verified by lineage/timestamps", "All figures are newer than source result tables."],
    [32, "Paper values consistent", "Verified for direct-linked tables", "Current paper reads frozen CSV/PNG outputs and is newer than figures."],
    [33, "Random seeds consistent", "Partially verified", "42 is used for stochastic active models; PLS/OLS deterministic."],
    [34, "Fixed/optimized naming", "Verified in outputs", "Suffixes are consistent; final test removes suffixes deliberately."],
    [35, "Historical mean uses allowable data", "Verified", "Final constant equals development mean up to float32 precision."],
]


RESULTS = [
    [1, "PLS", "0.0137866", "0.1171849", "0.0801883", "0.0017914", "0.0438813"],
    [2, "Random Forest", "0.0138037", "0.1172563", "0.0800818", "0.0005754", "0.0214511"],
    [3, "OLS-3", "0.0138049", "0.1172615", "0.0801398", "0.0004857", "0.0256182"],
    [4, "Elastic Net", "0.0138120", "0.1172900", "0.0800774", "approximately 0", "undefined"],
    [5, "Historical Mean", "0.0138120", "0.1172900", "0.0800774", "approximately 0", "undefined"],
    [6, "Decision Tree", "0.0156624", "0.1248378", "0.0848455", "-0.1328434", "-0.1453403"],
]


def active_file_decisions(path: str) -> str:
    mapping = {
        "src/config.py": "Centralizes 1990-2025 sample, 1987-2026 price window, target name, train end 2014, validation end 2019, and all canonical paths. Importing it creates directories as a side effect.",
        "src/data/01_build_clean_yahoo_daily.py": "Uses a locked universe; downloads each ticker serially; flags OHLC errors, missing core fields, nonpositive prices, extreme returns, and duplicate dates; removes rows and tickers above a 10 percent bad-row share.",
        "src/data/02_build_monthly_stock_features.py": "Uses month-end adjusted-close returns, full rolling windows, daily 252/126 risk windows, current-month market/VIX, shifted next-observation return, and shifted RF.",
        "src/data/03_add_fundamentals_and_macro.py": "Filters standard consolidated industrial USD records, builds ratios with zero-denominator protection, delays annual data six months, and uses a backward ticker-level as-of merge.",
        "src/data/04_build_raw_kelly_dataset.py": "Creates SIC2 dummies, merges same-month Welch-Goyal variables, locks 47 characteristics, four market variables, and eight macro variables.",
        "src/data/05_clean_and_rank_normalize.py": "Drops missing targets, median-imputes within month, zero-fills all-missing monthly predictors, ranks 47 characteristics to [-1,1], creates 376 interactions, and leaves target unwinsorized.",
        "src/models/utils/data.py": "Defines train through 2014, validation 2015-2019, development through 2019, test 2020-2025, and annual expanding folds.",
        "src/models/utils/estimation.py": "Grid search chooses the minimum mean monthly MSE; no one-standard-error rule is implemented.",
        "src/models/utils/evaluation.py": "Primary loss equally weights months; OOS R-squared uses a constant historical-mean benchmark.",
        "src/models/step_05_test/01_final_test_evaluation.py": "Refits frozen specifications on development data and saves only test metrics/predictions plus interpretation inputs.",
        "run_pipeline.sh": "Uses fail-fast shell options but references five obsolete paths, so it cannot currently complete.",
    }
    if path in mapping:
        return mapping[path]
    if path.startswith("src/models/legacy/") or path.startswith("backup/"):
        return "Archival decisions may differ from the active workflow and must not be used to interpret current frozen results."
    if path.startswith("src/models/step_02_tuning/"):
        return "Loads train only, uses annual expanding folds beginning in 2005, searches an explicit grid, and saves the minimum-MSE parameters."
    if path.startswith("src/models/step_01_fixed/"):
        return "Fits fixed specifications on train and reports train/validation metrics; test rows are not used."
    if path.startswith("src/models/step_03_optimization/"):
        return "Loads tuned parameters, fits train, compares on validation, and rejects test rows in comparison files."
    if path.startswith("src/models/step_04_robustness/"):
        return "Compares expanding with 120-month rolling Random Forest refits over validation years 2015-2019."
    if path.startswith("src/models/step_06_interpretation/"):
        return "Reads frozen outputs, validates schemas/model membership, and produces tables, figures, workbook, or narrative notes without refitting models."
    return "No separate consequential live decision identified beyond the file's documented role."


def quality_for(path: str) -> str:
    if path.startswith("backup/") or "/legacy/" in path:
        return "Clear archival value, but high stale-output and duplicated-logic risk if confused with the active pipeline."
    if "__pycache__" in path or path.endswith(".pyc"):
        return "Regenerable cache; source correspondence is not guaranteed and several caches are orphaned."
    if path.startswith("src/data/"):
        return "Generally explicit and chronological. Main risks are next-observation target gaps, same-month macro timing, vendor/ticker quality, and large CSV memory use."
    if path.startswith("src/models/"):
        return "Chronological separation and pipelines are strong. Main risks are documentation drift, dimensionality, no formal loss-comparison tests, and unpinned dependencies."
    if path.startswith("output/"):
        return "Frozen evidence. Reproducibility depends on matching code/input versions; no provenance hashes are stored."
    if path.startswith("input/"):
        return "Permanent local input. External acquisition provenance varies; Compustat extraction is not scripted."
    return "No material leakage risk by itself; review status and staleness according to category."


def source_like(record: dict) -> bool:
    return record["type"] in {".py", ".sh", ".tex", ".md", ".bib", ".txt", ".gitignore"} or record["name"] in {"Makefile", "requirements.txt"}


def build_markdown(records: list[dict], python_files: list[dict], functions: list[dict]) -> str:
    py_by_path = {item["path"]: item for item in python_files}
    lines: list[str] = []
    add = lines.append
    add("---")
    add('title: "Full Project Technical Audit"')
    add('subtitle: "Machine Learning for Cross-Sectional Stock Return Prediction"')
    add('author: "Repository evidence review"')
    add(f'date: "{SNAPSHOT_DATE.isoformat()}"')
    add("documentclass: article")
    add("classoption: [a4paper]")
    add("geometry: [top=2.1cm,bottom=2.1cm,left=2.0cm,right=2.0cm]")
    add("fontsize: 11pt")
    add("header-includes:")
    add("  - |")
    add("    \\usepackage{fancyhdr}")
    add("    \\usepackage{booktabs,longtable,pdflscape,xurl,graphicx,float,tikz,enumitem,fvextra}")
    add("    \\usetikzlibrary{arrows.meta,positioning,shapes.geometric}")
    add("    \\pagestyle{fancy}")
    add("    \\fancyhf{}")
    add("    \\fancyhead[L]{Full Project Technical Audit}")
    add("    \\fancyhead[R]{Stock Return ML Project}")
    add("    \\fancyfoot[C]{\\thepage}")
    add("    \\setlength{\\headheight}{14pt}")
    add("    \\setlength{\\emergencystretch}{3em}")
    add("    \\setcounter{tocdepth}{2}")
    add("---\n")
    add("\\pagenumbering{gobble}\n\\begin{titlepage}\n\\thispagestyle{empty}\n\\centering\n\\vspace*{2.2cm}\n{\\Huge\\bfseries Full Project Technical Audit\\par}\n\\vspace{0.6cm}\n{\\Large Machine Learning for Cross-Sectional Stock Return Prediction\\par}\n\\vspace{1.5cm}\n{\\large Complete data, code, modeling, output, consistency, and reproducibility review\\par}\n\\vfill\n{\\large Evidence snapshot: 20 July 2026\\par}\n\\vspace{0.4cm}\n{\\normalsize Repository: \\texttt{stock\\_return\\_ml\\_project\\_clean}\\par}\n\\vspace{1.0cm}\n{\\small This report is documentation only. No existing project file was changed or executed for expensive acquisition or estimation.\\par}\n\\end{titlepage}\n")
    add("\\pagenumbering{roman}\n\\tableofcontents\n\\clearpage\n\\pagenumbering{arabic}\n")
    add("# Executive Summary {#sec-executive}\n")
    add("**Audit convention.** Statements marked **Verified fact** were checked directly against current code or outputs. **Interpretation** explains implications. **Potential problem** identifies evidence-supported risk. **Recommendation** proposes a future check but does not alter the project. Where evidence is absent, this report states: **Not verifiable from the current repository.**\n")
    add("The project predicts next-month stock excess returns in a monthly ticker panel using market, technical, accounting, industry, and macroeconomic information. The final model-ready file contains 212,794 observations, 663 tickers, and 499 predictors from January 1990 through December 2025. The development period ends in December 2019 and the frozen final test covers January 2020 through December 2025. Six final specifications are compared: a historical mean, OLS-3, PLS, Elastic Net, Decision Tree, and Random Forest.\n")
    add("**Verified finding.** PLS ranks first on final monthly MSE, but its OOS R-squared is only 0.00179. Random Forest and OLS-3 also achieve small positive OOS R-squared values. Elastic Net collapses to zero coefficients, and the single Decision Tree materially underperforms the historical mean. These results support weak statistical predictability, not strong economic profitability.\n")
    add("**Most urgent checks.** Enforce next-calendar-month target continuity, resolve macro publication lags, repair the stale shell pipeline, reconcile tuning documentation with actual 1990-2014 tuning and minimum-MSE selection, persist failed Yahoo tickers, and lock software versions.\n")
    add("## Project at a glance {#sec-at-glance}\n")
    add(table(["Item", "Verified repository value"], [
        ["Research question", "Can stock, market, accounting, industry, and macro predictors forecast next-month stock excess returns out of sample?"],
        ["Unit", "Ticker-month"], ["Target", "target_excess_return_next_1m"],
        ["Coverage", "1990-01-31 to 2025-12-31"], ["Universe", "Locked 867-ticker current-plus-historical Wikipedia union; 663 in final panel"],
        ["Main sources", "Yahoo Finance, WRDS/Compustat, Ken French RF, cleaned Welch-Goyal input, Wikipedia"],
        ["Predictors", "47 ranked characteristics, 4 market/VIX, 8 macro, 64 SIC2 dummies, 376 interactions"],
        ["Development/test", "1990-2019 / 2020-2025"], ["Actual tuning folds", "Annual expanding folds within 1990-2014, validation years 2005-2014"],
        ["Primary metric", "Equal-weighted monthly MSE"], ["Best final model", "PLS: monthly MSE 0.0137866, OOS R2 0.0017914"],
        ["Main strength", "Strong chronological separation and verified six-month backward Compustat timing"],
        ["Main limitation", "Weak gains plus timing/continuity and documentation inconsistencies"],
    ]))

    sections = [
        ("Research Objective and Project Overview", "The active repository implements a regression problem, not classification. Predictors dated at month t are intended to forecast stock return in t+1 less the t+1 risk-free rate. The design follows the empirical asset-pricing machine-learning tradition: combine many firm characteristics with macro states, compare flexible methods against transparent benchmarks, and reserve a chronologically later test period."),
        ("Complete Chronological Project Story", "The workflow begins with one-time local inputs, then downloads individual stock histories, cleans daily observations, aggregates month-end features, constructs the target, attaches conservatively lagged accounting statements, merges macro states and industries, performs cross-sectional imputation/ranking, estimates fixed and tuned models, runs robustness checks, opens a final test once, and creates frozen interpretation and paper artifacts. The current shell wrapper no longer represents the actual model-stage paths."),
        ("Repository Architecture", f"The pre-audit snapshot contains {len(records)} files outside .git and the new audit folder. Active logic lives under src; permanent inputs under input; frozen data and model results under output; archival code/results under backup and legacy; and the current course paper under thesis."),
        ("Complete File Inventory", "Every file is listed in Appendix B with type, category, role, producer, consumer, essential status, and reproducibility assessment. Cache files, OS metadata, build auxiliaries, backups, and legacy Gradient Boosting artifacts are deliberately included rather than silently omitted."),
        ("File-by-File Technical Documentation", "Appendix C gives a dedicated subsection for each Python, shell, LaTeX, Markdown, bibliography, text, Makefile, dependency, and ignore file. Active files receive decision and quality notes; archival files are explicitly separated from current evidence."),
        ("Data Sources and Data Lineage", "The permanent-input design successfully prevents small online datasets from changing during a normal build. Individual stock prices remain an online acquisition inside data file 01. Raw Compustat extraction credentials and query are not stored; therefore exact reconstruction of that input is not verifiable from the current repository. No active FRED ingestion exists."),
        ("Data Construction and Timing Decisions", "Daily adjusted prices produce simple returns and monthly technical characteristics. Current month-end information predicts the subsequent target. Momentum explicitly lags current return. Compustat is delayed six months and merged backward. Same-month Welch-Goyal values are merged without the intended one-month shift, and target shifting does not enforce calendar continuity."),
        ("Final Dataset Audit", "The final parquet is sorted by ticker and month, has unique column names, zero missing values, zero infinities, and zero duplicate ticker-month rows. Its 499 predictors exactly match the predictor inventory. Near-duplicate predictors were not exhaustively tested because a complete 499-by-499 dependence audit is computationally substantial; this is not verifiable from the current repository outputs."),
        ("Modeling Framework", "Linear models use StandardScaler within sklearn pipelines. Trees use raw transformed predictors. Tuning minimizes annual-fold average monthly MSE. Models are compared against a historical mean, and final models are refit on all 1990-2019 development observations before 2020-2025 prediction."),
        ("Model-by-Model Documentation", "The historical mean is the development target average. OLS-3 uses size, book-to-market, and 12-month momentum. PLS uses one selected component. Elastic Net uses alpha 0.01 and l1 ratio 0.9 and yields zero coefficients. The selected Decision Tree has depth 3 and minimum leaf 1,000; the selected Random Forest has 200 depth-2 trees, minimum leaf 100, and square-root feature sampling. Gradient Boosting exists only in legacy archives."),
        ("Validation and Test Design", "The code uses train 1990-2014, preliminary validation 2015-2019, development 1990-2019, and test 2020-2025. Hyperparameter grids run only on train with validation years 2005-2014. Optimized models are then checked on 2015-2019. This is leakage-resistant but differs from the active README claim that tuning folds extend through 2019."),
        ("Results", "PLS is numerically best by final monthly MSE and OOS R-squared. The improvement over the historical mean is very small. Random Forest has the lowest MAE among the six. Decision Tree is decisively worst. Validation optimization helps PLS, Elastic Net, and Decision Tree relative to their fixed forms, but makes Random Forest slightly worse."),
        ("Interpretation", "Small positive OOS R-squared means a tiny reduction in squared forecast error relative to the constant benchmark, not that return variation is well explained. Negative OOS R-squared means the benchmark wins. Elastic Net's zero solution indicates the chosen penalty finds no stable incremental signal. Tree impurity importance is predictive and model-specific, not causal. PLS coefficients describe a supervised latent projection after standardization and should not be interpreted as structural premia."),
        ("What Was Done Well", "Verified strengths include chronological sorting, month-end normalization, time-separated samples, training-only pipeline scaling, explicit historical-mean comparison, six-month accounting delay, backward as-of merging with zero violations, permanent local small inputs, saved predictor inventories, integrity checks, multiple model families, frozen interpretation artifacts, and honest negative findings."),
        ("Limitations and Questionable Choices", "Material limitations include the 26 target continuity failures, unlagged macro merge, public ticker-based universe and matching, Yahoo revision/delisting limitations, 499-dimensional design, unwinsorized squared-loss sensitivity, weak and unstable gains, a poor single tree, zero Elastic Net solution, unpinned dependencies, stale execution wrapper, duplicate/legacy artifacts, no formal loss test, and no fully specified transaction-cost implementation in the main paper."),
        ("Leakage and Look-Ahead-Bias Audit", "No direct test leakage was found in scaler fitting, hyperparameter selection, Compustat merging, or final model training. The principal timing concern is same-month macro availability. The target is future by construction, but gap transitions sometimes extend beyond one month. Human adherence to an untouched-test protocol cannot be reconstructed from files after the test outputs exist."),
        ("Reproducibility Audit", "Data steps are mostly explicit and canonical paths are centralized. Reproducibility is weakened by online per-ticker Yahoo downloads, an unscripted Compustat extraction, absolute manifest paths, unpinned packages, no lock file, missing shell targets, absent tests, and no recorded runtime/provenance hashes. Existing outputs permit result reproduction without retraining, but not proof that they came from the current exact source state."),
        ("Consistency and Staleness Audit", "Final metrics recomputed from predictions agree within floating precision, the ranking order is correct, figures are newer than source result tables, and the paper is newer than figures. Inconsistencies are concentrated in workflow documentation, shell paths, duplicated input/legacy files, orphan bytecode, and intended versus actual macro/tuning logic."),
        ("Required Checks Before Submission", "The priority is to verify target calendar continuity and macro publication timing, repair the run instructions, align the methodological description with actual folds and selection rule, confirm required citations, state the universe/ticker limitations, and decide whether the portfolio evidence is in scope. These checks should be completed before claiming a fully reproducible and leakage-safe submission."),
        ("Recommended Future Improvements", "Use point-in-time CRSP/Compustat identifiers and delisting returns, actual filing dates, explicit macro release lags, a calendar-grid target, locked environments, automated tests, provenance hashes, nested or repeated chronological validation, robust-loss sensitivity, formal forecast-comparison inference, and a prespecified net-of-cost portfolio design."),
        ("Final Overall Assessment", "The project is methodologically thoughtful and unusually transparent for a course project, but it requires important checks before submission. The core result - weak predictability with several model failures - is credible as a negative finding. Strong claims about a strict one-month target, macro timing, complete reproducibility, or profitability are not yet justified."),
        ("Reproduction Guide", "Three reproducibility levels are possible: rebuild all data (requires network access and the supplied Compustat CSV), rerun models from the final parquet, or regenerate interpretation and paper artifacts only. Because run_pipeline.sh is stale, the canonical manual order in this report should be followed until the wrapper is corrected."),
        ("Glossary", "Key terms: adjusted close (split/dividend-adjusted price); OOS R2 (benchmark-relative squared-error improvement); monthly MSE (equal-weighted mean of within-month MSE); PLS (partial least squares); RF (risk-free rate or, when named as a model, Random Forest); SIC2 (two-digit industry code); expanding window (all prior years train the next year); development sample (all pre-test observations)."),
    ]
    for number, (title, text) in enumerate(sections, 1):
        add(f"# {title} {{#sec-{number:02d}}}\n")
        add(text + "\n")
        if title == "Complete Chronological Project Story":
            add("## Stage-by-stage narrative {#sec-story-stages}\n")
            stages = [
                [1, "Create permanent inputs", "src/acquisition/01-04", "Wikipedia, Ken French, Yahoo market series, external Welch-Goyal, Compustat supplied separately", "input/*", "Check manifest and coverage"],
                [2, "Build daily stock file", "src/data/01_build_clean_yahoo_daily.py", "locked universe", "raw/clean daily CSV and quality reports", "Check row/ticker counts and removals"],
                [3, "Build monthly stock panel", "src/data/02_build_monthly_stock_features.py", "clean daily, GSPC, VIX, RF", "monthly panel and target", "Check month-end, duplicates, continuity"],
                [4, "Add fundamentals", "src/data/03_add_fundamentals_and_macro.py", "monthly panel, raw Compustat", "clean Compustat and merged panel", "Check available_month <= month"],
                [5, "Build raw Kelly panel", "src/data/04_build_raw_kelly_dataset.py", "fundamental panel, Welch-Goyal", "123-predictor raw panel", "Check macro lag"],
                [6, "Clean and rank", "src/data/05_clean_and_rank_normalize.py", "raw Kelly panel", "499-predictor parquet", "Check missing, finite, duplicate, rank ranges"],
                [7, "Fixed models", "src/models/step_01_fixed/*", "final parquet", "fixed metrics/predictions", "No test rows"],
                [8, "Tune", "src/models/step_02_tuning/01-03", "train through 2014", "grids and best parameters", "Annual chronological folds"],
                [9, "Validate optimized", "src/models/step_03_optimization/*", "train/validation", "optimized comparisons", "No test rows"],
                [10, "Robustness", "src/models/step_04_robustness/*", "train/validation", "expanding vs rolling metrics", "Check 2015-2019 only"],
                [11, "Final test", "src/models/step_05_test/*", "development and saved parameters", "test predictions/metrics", "One-time process not historically provable"],
                [12, "Interpret/report", "src/models/step_06_interpretation/* and thesis/paper.tex", "frozen outputs", "tables, figures, notes, paper", "No model refit"],
            ]
            stage_rows = [[row[0], row[1], mpath(row[2]), *row[3:]] for row in stages]
            add(table(["Step", "Action", "Script", "Inputs", "Outputs", "Verification"], stage_rows))
        if title == "Data Sources and Data Lineage":
            add("## End-to-end flowchart {#sec-lineage-flow}\n")
            add(r"""\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=7mm and 10mm, every node/.style={font=\scriptsize}, box/.style={draw,rounded corners,align=center,minimum height=8mm,text width=27mm}, arr/.style={-{Latex[length=2mm]},thick}]
\node[box] (sources) {Wikipedia, Yahoo, Ken French, Compustat, Welch-Goyal};
\node[box,right=of sources] (inputs) {Permanent inputs and daily stock acquisition};
\node[box,right=of inputs] (monthly) {Monthly stock panel and t+1 excess target};
\node[box,below=of monthly] (fund) {Six-month-lagged Compustat merge};
\node[box,left=of fund] (kelly) {Macro, SIC2, imputation, ranks, interactions};
\node[box,left=of kelly] (models) {Fixed, tuned, robust, and final-test models};
\node[box,below=of models] (report) {Predictions, metrics, figures, portfolio, paper};
\draw[arr] (sources)--(inputs); \draw[arr] (inputs)--(monthly); \draw[arr] (monthly)--(fund); \draw[arr] (fund)--(kelly); \draw[arr] (kelly)--(models); \draw[arr] (models)--(report);
\end{tikzpicture}
\caption{Verified high-level data and output lineage.}
\end{figure}
""")
        if title == "Final Dataset Audit":
            add("## Major dataset validation table {#sec-dataset-table}\n")
            dataset_rows = [[row[0], mpath(row[1]), *row[2:]] for row in DATASETS]
            add(table(["Dataset", "Path", "Shape", "Coverage", "Entities", "Missing", "Duplicates", "Notes"], dataset_rows))
            add("The final target has mean 0.0129067, standard deviation 0.109243, median 0.0108016, 1st percentile -0.262064, 99th percentile 0.325020, minimum -0.935113, and maximum 2.920526. The target is intentionally not winsorized.\n")
        if title == "Results":
            add("## Frozen final-test comparison {#sec-final-results}\n")
            add(table(["Rank", "Model", "Monthly MSE", "RMSE", "MAE", "OOS R2", "Correlation"], RESULTS))
            add("Source: " + mpath("output/models/test/final_test_model_comparison.csv") + ", reproduced in " + mpath("output/models/interpretation/final_prediction_results.csv") + ". Metrics recomputed from " + mpath("output/models/test/final_test_predictions.parquet") + " differ by no more than 7.5e-11.\n")
            add("![Final-test monthly MSE by model. Existing frozen figure; not regenerated for this audit.](../../output/models/interpretation/figures/test_monthly_mse_by_model.png){width=85%}\n")
            add("![Final-test OOS R-squared by model. Existing frozen figure; not regenerated for this audit.](../../output/models/interpretation/figures/test_oos_r2_by_model.png){width=85%}\n")
        if title == "Validation and Test Design":
            add("## Split counts {#sec-split-counts}\n")
            add(table(["Sample", "Rows", "Months", "Tickers", "Dates"], [
                ["Train", "132,336", "300", "578", "1990-01 to 2014-12"], ["Validation", "35,630", "60", "618", "2015-01 to 2019-12"], ["Development", "167,966", "360", "620", "1990-01 to 2019-12"], ["Test", "44,828", "72", "645", "2020-01 to 2025-12"],
            ]))
        if title == "Leakage and Look-Ahead-Bias Audit":
            add("## Verification matrix {#sec-verification-matrix}\n")
            add(table(["No.", "Design point", "Status", "Evidence"], VERIFICATION))
        if title == "Consistency and Staleness Audit":
            add("## Evidence-supported inconsistency register {#sec-issue-register}\n")
            add(table(["ID", "Severity", "Dimension", "Issue", "Evidence", "Recommended check"], ISSUES))
        if title == "Final Overall Assessment":
            add("## Scorecard {#sec-scorecard}\n")
            add(table(["Category", "Score / 10", "Reason"], [
                ["Methodological quality", 7, "Strong chronology and benchmarks; target continuity and macro lag require correction"],
                ["Data quality", 6, "Large transparent panel; public-vendor and ticker-history limitations"],
                ["Leakage protection", 8, "Scaler, test, and Compustat protections verified; macro timing unresolved"],
                ["Reproducibility", 5, "Frozen outputs and central config help; wrapper, versions, and Compustat extraction hinder"],
                ["Code organization", 7, "Clear stages/utilities; archives, stale paths, and caches create ambiguity"],
                ["Documentation", 6, "Substantial guides and paper; tuning claims drift from code"],
                ["Empirical credibility", 7, "Negative findings are honest; no formal loss test and gains are tiny"],
                ["Submission readiness", 6, "Requires important checks before submission"],
            ]))
            add("**Final verdict: Requires important checks.**\n")
            add("### Top ten strongest aspects\n")
            add("1. Chronological train/validation/development/test separation.\n2. Training-only pipeline standardization.\n3. Verified six-month Compustat lag.\n4. Verified backward as-of merge.\n5. Permanent local small external inputs.\n6. Exact predictor inventory.\n7. Multiple transparent benchmarks and model families.\n8. Frozen predictions and recomputable metrics.\n9. Honest reporting of weak and negative results.\n10. Interpretation, diagnostics, robustness, and portfolio artifacts.\n")
            add("### Top ten weaknesses\n")
            add("1. Twenty-six non-calendar target transitions.\n2. Same-month macro merge.\n3. Stale shell workflow.\n4. Tuning documentation mismatch.\n5. Unpinned environment.\n6. Public ticker universe and matching.\n7. Unscripted Compustat extraction.\n8. High interaction dimensionality.\n9. No formal forecast-loss test.\n10. No proof of historical untouched-test discipline.\n")
        if title == "Reproduction Guide":
            add("## Required environment {#sec-environment}\n")
            add("The repository lists pandas, numpy, scikit-learn, statsmodels, requests, yfinance, pyarrow, matplotlib, openpyxl. The inspected environment is Python 3.13.2 with pandas 2.3.3, numpy 2.2.3, scikit-learn 1.8.0, statsmodels 0.14.5, requests 2.32.3, yfinance 1.4.1, pyarrow 19.0.1, matplotlib 3.10.1, and openpyxl 3.1.5. These are observations, not locked requirements. WRDS credentials are required only to recreate the Compustat extraction, whose query is absent. Wikipedia/Yahoo/Ken French access is required for acquisition. No FRED API is used.\n")
            add("## Three run levels {#sec-run-levels}\n")
            add("**Raw rebuild:** run acquisition scripts as needed, provide raw Compustat, then data files 01-05. This downloads large stock histories and was not executed during this audit.\n\n**Model-only rebuild:** with the final parquet and predictor CSV present, run fixed, tuning, optimized comparison, robustness, final test, then interpretation using the active paths documented above. Do not rely on the current shell wrapper.\n\n**Reporting-only rebuild:** run `python src/models/step_06_interpretation/run_all.py` only if frozen test/validation inputs are already present, then compile `thesis/paper.tex` with its Makefile.\n")

    add("\\appendix\n")
    add("# Full Directory Tree {#app-tree}\n")
    add("The tree is a complete pre-audit snapshot excluding `.git` and the newly created audit directory.\n")
    add("```text")
    for record in records:
        add(record["path"])
    add("```\n")

    add("# Complete File Inventory {#app-inventory}\n")
    identity_rows=[]
    lineage_rows=[]
    for record in records:
        essential, reproducible, state = status_for(record["path"], record["category"])
        producer, consumer = lineage_for(record["path"])
        identity_rows.append([mpath(record["path"]), record["type"], human_size(record["size"]), record["category"], role_for(record["path"], record["category"]), state])
        lineage_rows.append([mpath(record["path"]), producer, consumer, essential, reproducible])
    add("## Identity and classification inventory {#app-inventory-identity}\n")
    add(table(["Path", "Type", "Size", "Class", "Role", "Status"], identity_rows))
    add("## Producer, consumer, and reproducibility inventory {#app-inventory-lineage}\n")
    add(table(["Path", "Producer", "Consumer", "Essential", "Reproducibility"], lineage_rows))

    add("# Complete File-by-File and Function Documentation {#app-files}\n")
    for file_number, record in enumerate([item for item in records if source_like(item)], start=1):
        path=record["path"]; py=py_by_path.get(path)
        readable_name = record["name"].replace("_", " ")
        add(f"## File {file_number}: {readable_name} {{#file-{abs(hash(path))}}}\n")
        producer, consumer=lineage_for(path)
        add(f"**Identity.** {mpath(path)}. **Role.** {role_for(path, record['category'])} Type `{record['type']}`, size {human_size(record['size'])}. Producer/input relationship: {producer}. Consumer/dependent relationship: {consumer}.\n")
        if py:
            imports = ", ".join(mpath(item) for item in py["imports"]) if py["imports"] else "none"
            add(f"**Execution.** {py['lines']} lines; direct main block: {'yes' if py['main'] else 'no'}; imports: {imports}.\n")
            if py["functions"]:
                rows=[]
                for function in py["functions"]:
                    side="Writes/fits only if stated by the function or enclosing main workflow." if function["name"]=="main" else "See docstring; no additional side effect verified."
                    args = ", ".join(mpath(item) for item in function["args"]) if function["args"] else "none"
                    rows.append([mpath(function["name"]), args, function["doc"], side, function["line"]])
                add(table(["Function", "Inputs", "Return/output meaning", "Side effects", "Line"], rows))
            else:
                add("No functions are defined.\n")
        else:
            add("No Python function inventory applies. Its execution or compilation behavior follows the file type.\n")
        add(f"**Critical decisions.** {active_file_decisions(path)}\n")
        add(f"**Quality assessment.** {quality_for(path)}\n")

    add("# Complete Function Inventory {#app-functions}\n")
    add(f"The AST scan found {len(functions)} function definitions across active, legacy, and backup Python files; {sum(item['file'].startswith('src/') and '/legacy/' not in item['file'] for item in functions)} belong to active non-legacy source. Nested helpers are included.\n")
    function_rows = []
    for function in functions:
        args = ", ".join(mpath(item) for item in function["args"]) if function["args"] else "none"
        function_rows.append([mpath(function["file"]), mpath(function["name"]), args, function["line"], function["doc"]])
    add(table(["File", "Function", "Arguments", "Line", "Purpose"], function_rows))

    add("# Dataset Schemas {#app-schemas}\n")
    schemas = {
        "Monthly stock panel": "ticker, month, last adjusted close, 20 stock/technical fields, four market/VIX fields, beta/beta-squared/idio-vol, next stock return, next RF, next excess return",
        "Fundamental panel": "all monthly fields plus Compustat date/availability, SIC2, 27 accounting characteristics, R&D missing and match indicator",
        "Raw Kelly panel": "ticker, month, target, 47 stock characteristics, four market variables, eight Welch-Goyal variables, 64 SIC2 dummies",
        "Ranked model panel": "ticker, month, target, 123 base predictors plus 376 characteristic-macro interactions = 499 predictors",
        "Test predictions": "ticker, month, realized_target, prediction, model, sample",
    }
    add(table(["Dataset", "Compact schema"], [[k,v] for k,v in schemas.items()]))
    add("The full 502-column schema is recoverable from the parquet metadata and predictor CSV. It is not duplicated verbatim here because the complete predictor inventory already exists at `output/data/final/predictor_columns_kelly_ranked.csv`.\n")

    add("# Critical Decision Register {#app-decisions}\n")
    add(table(["ID", "Decision", "Evidence", "Benefit", "Risk", "Disposition"], DECISIONS))
    add("# Risk Register {#app-risks}\n")
    add(table(["ID", "Severity", "Dimension", "Risk", "Evidence", "Response"], ISSUES))
    add("# Hard-Coded Parameter and Seed Inventory {#app-parameters}\n")
    add(table(["Parameter", "Value", "Location", "Comment"], [
        ["Sample", "1990-01 to 2025-12", "src/config.py", "Price begins 1987; Jan 2026 enables final target"],
        ["Train/validation end", "2014-12 / 2019-12", "src/config.py", "Defines development/test"],
        ["Daily bad-row threshold", "0.10", "src/config.py", "Ticker-level removal"],
        ["Daily/monthly extreme rules", ">3 or <-0.95", "data files 01/02", "Data-quality flags"],
        ["Risk window", "252; minimum 126 daily", "data file 02", "Beta and idio-vol"],
        ["Momentum windows", "6, 12, 36 monthly observations", "data file 02", "Momentum excludes current return"],
        ["Compustat lag", "6 months", "data file 03", "Availability proxy"],
        ["Rank interval", "[-1,1]", "data file 05", "Monthly characteristic rank"],
        ["Tuning start year", "2005", "tuning scripts", "Actual folds end 2014"],
        ["Random seed", "42", "active stochastic model scripts", "Random Forest, Decision Tree, Elastic Net tuning"],
        ["Rolling robustness", "120 months", "step_04_robustness", "Compared with expanding"],
        ["Coefficient tolerance", "1e-10", "step_05_test", "Nonzero count"],
    ]))

    add("# Output Lineage {#app-output-lineage}\n")
    add(table(["Source", "Transformation", "Output", "Downstream use"], [
        ["Locked tickers", "Yahoo download and quality cleaning", "daily raw/clean and reports", "monthly features"],
        ["Daily stock + GSPC + VIX + RF", "month aggregation, rolling features, shift target", "monthly stock panel", "Compustat merge"],
        ["Raw Compustat", "filter, ratios, six-month lag", "clean Compustat", "backward as-of panel"],
        ["Fundamental panel + Welch-Goyal", "SIC2 and selected variables", "raw Kelly panel", "imputation/ranking"],
        ["Raw Kelly", "monthly median, ranks, interactions", "ranked parquet + predictor CSV", "all active models"],
        ["Fixed/tuning outputs", "validation comparisons", "saved parameter files", "final test specifications"],
        ["Ranked parquet + parameters", "development fit, test prediction", "test metrics/predictions", "interpretation and portfolio"],
        ["Frozen metrics/predictions", "tables/figures/notes", "interpretation directory", "thesis paper"],
    ]))

    add("# Unused, Duplicate, Orphan, or Obsolete Inventory {#app-obsolete}\n")
    add(table(["Item", "Evidence", "Status"], [
        ["input/external/fama_french_rf_monthly.csv", "SHA-1 identical to canonical input/fama_french_rf_monthly.csv", "Duplicate, inactive"],
        ["backup/model_reorganization_*", "Copies of pre-reorganization model files", "Archival/possibly obsolete"],
        ["backup/remove_gradient_boosting_*", "Removal snapshots and former portfolio scripts", "Archival/possibly obsolete"],
        ["src/models/legacy/gradient_boosting", "Old Gradient Boosting source snapshots", "Explicit legacy"],
        ["output/models/legacy/gradient_boosting", "Old Gradient Boosting results", "Explicit legacy"],
        ["orphan .pyc files", "Caches exist for compare_scaling, portfolio, old robustness/test, and inspection sources that are absent", "Temporary/stale"],
        ["wg_infl", "Present in Welch-Goyal input but absent from locked macro columns", "Unused input column"],
        ["notebooks/", "Empty directory", "Nonessential"],
        ["Thesis build auxiliaries", "Standard LaTeX auxiliary files", "Regenerable temporary files"],
    ]))

    add("# Final Submission Checklist {#app-checklist}\n")
    checklist = [
        ["A - must", "Target continuity", "data file 02 and monthly panel", "Confirm or rebuild only when next month is exactly t+1", "Code change required", "Moderate"],
        ["A - must", "Macro lag", "data file 04 and Welch-Goyal source timing", "Verify release availability and apply documented lag", "Likely code change", "Moderate"],
        ["A - must", "Canonical run order", "run_pipeline.sh", "Replace missing paths after deciding portfolio scope", "Code/doc change", "Easy"],
        ["A - must", "Tuning description", "models README and estimation code", "Choose actual minimum-MSE train-only description or revise method", "Documentation or code", "Easy"],
        ["A - must", "Course references", "thesis/references.bib", "Verify professor-required citations", "Documentation only", "Easy"],
        ["B - should", "Yahoo failures", "data file 01", "Recover failed list from logs or redownload metadata only", "Potential code change", "Moderate"],
        ["B - should", "Ticker/Compustat mapping", "input and merged panel", "Sample unmatched/symbol-change cases", "No immediate code change", "Difficult"],
        ["B - should", "Environment lock", "requirements.txt", "Record tested versions and platform", "Config change", "Easy"],
        ["B - should", "Portfolio scope", "output/models/portfolio and paper", "Decide inclusion; disclose no costs", "Documentation only", "Easy"],
        ["C - future", "Point-in-time universe", "universe/price data", "Use CRSP identifiers and delisting returns", "Major rebuild", "Difficult"],
        ["C - future", "Formal forecast tests", "prediction output", "Add time/panel-aware loss comparison", "New analysis", "Moderate"],
        ["C - future", "Robust target/loss", "cleaning and modeling", "Compare clipping, Huber loss, or robust evaluation", "New analysis", "Moderate"],
        ["D - optional", "Alternative models", "model stages", "Add only under prespecified chronological validation", "New analysis", "Difficult"],
    ]
    add(table(["Priority", "Issue", "File", "How to verify", "Change required", "Difficulty"], checklist))

    add("# Assumption and Readiness Register {#app-assumptions}\n")
    add("1. Yahoo adjusted prices are treated as usable but are not independently validated against CRSP.\n2. The supplied raw Compustat file is assumed to be the intended extraction; query provenance is absent.\n3. A six-month reporting lag is a conservative proxy, not an actual filing date.\n4. Same-month market variables are assumed observable at month-end for next-month prediction.\n5. Welch-Goyal availability timing is unresolved.\n6. The 2020-2025 test is treated as final, but viewing history is not verifiable.\n7. Frozen outputs are treated as the current run because timestamps and internal values align; source-to-output hashes are absent.\n")
    add("**Readiness conclusion:** Requires important checks. The report identifies 22 evidence-supported issues, of which four are high severity.\n")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = file_records()
    python_files, functions = python_inventory(records)
    markdown = build_markdown(records, python_files, functions)
    if "TODO" in markdown:
        raise ValueError("Raw placeholder token found in generated report.")
    MD.write_text(markdown)
    subprocess.run([
        "pandoc", MD.name,
        "--from=markdown+raw_tex+pipe_tables+fenced_code_blocks+yaml_metadata_block",
        "--to=latex", "--standalone",
        "--number-sections", "--top-level-division=section", "--output", TEX.name,
    ], cwd=OUT, check=True)
    latex = TEX.read_text().replace("\\maketitle\n\n", "", 1)
    latex = latex.replace(
        "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\\\\{\\}}",
        "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\\\\{\\},breaklines=true,breakanywhere=true,fontsize=\\scriptsize}",
    )
    latex = latex.replace("\\_", "\\_\\allowbreak{}")
    tables = []
    cursor = 0
    while True:
        start = latex.find("\\begin{longtable}", cursor)
        if start < 0:
            break
        end = latex.find("\\end{longtable}", start) + len("\\end{longtable}")
        header_end = latex.find("\\midrule", start, end)
        columns = latex[start:header_end].count("\\begin{minipage}")
        if columns >= 5:
            tables.append((start, end))
        cursor = end
    for start, end in reversed(tables):
        latex = (
            latex[:start]
            + "\\begin{landscape}\n\\scriptsize\n"
            + latex[start:end]
            + "\n\\end{landscape}\n\\normalsize"
            + latex[end:]
        )
    TEX.write_text(latex)
    summary = {
        "files_documented": len(records),
        "functions_documented": len(functions),
        "critical_decisions": len(DECISIONS),
        "potential_issues": len(ISSUES),
        "markdown": str(MD),
        "latex": str(TEX),
        "pdf": str(PDF),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
