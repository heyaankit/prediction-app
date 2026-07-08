"""
Stage 1 of the DVC pipeline for the prediction-app project.

This file mirrors EXACTLY the steps the original notebook
(`notebooks/Demand_Forecasting.ipynb`) performs BEFORE the
train/val/test split. Anything past the split (feature scaling, encoding,
cross-validation, model training) belongs in downstream DVC stages
(`feature_engineering.py`, `train.py`, etc.) so this stage stays
single-responsibility: raw CSV → validated, lightly cleaned dataframe.

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging — print to stdout so DVC captures the stage output in `dvc repro`.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_ingestion")


# ---------------------------------------------------------------------------
# Schema definition — the columns your RAW CSV must contain.
# Lifted directly from notebook cell 1 + cell 11 (before any drops).
# NOTE the spaces (not underscores): 'Competitor Pricing', 'Weather Condition',
# 'Store ID', etc. The notebook's raw CSV uses spaces — keep them or the
# downstream ColumnTransformer in the training stage will break.
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS: dict[str, str] = {
    "Store ID": "categorical",         # dropped later, but must exist in raw
    "Product ID": "categorical",       # dropped later
    "Date": "datetime",                # parsed + later used for Time_Step
    "Region": "categorical",
    "Weather Condition": "categorical",
    "Category": "categorical",
    "Inventory Level": "numeric",      # dropped later
    "Units Sold": "numeric",           # dropped later
    "Units Ordered": "numeric",        # dropped later
    "Seasonality": "categorical",      # dropped later (notebook drops it)
    "Price": "numeric",
    "Discount": "numeric",
    "Competitor Pricing": "numeric",   # NOTE: space, not underscore
    "Promotion": "numeric",
    "Epidemic": "numeric",
    "Demand": "numeric",               # target
}

# Columns the notebook drops in cell 11 because they were either
# (a) used only to engineer other features (Month, DayOfWeek, Date) or
# (b) leak the target / are not available at inference time
# (Inventory Level, Units Sold, Units Ordered, Seasonality, Store ID, Product ID).
#
# IMPORTANT: in our pipeline, feature engineering (Month_Sin, Time_Step etc.)
# happens in a LATER stage, so we keep `Date` here. We only drop the columns
# the notebook never uses at all.
DROP_COLUMNS: list[str] = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Seasonality",
    "Store ID",
    "Product ID",
]

# Default config path — `params.yaml` is the DVC convention.
DEFAULT_PARAMS_PATH = "params.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    """Load `params.yaml`. DVC reads the same file to track param dependencies."""
    params_path = Path(path)
    if not params_path.exists():
        log.warning(
            "%s not found — falling back to built-in defaults. "
            "Create one so `dvc.yaml` can track params.", path
        )
        return {}
    with params_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(args: argparse.Namespace) -> dict:
    """Merge params.yaml + CLI overrides. CLI wins when provided."""
    params = load_params(args.params)
    section = params.get("data_ingestion", {}) if isinstance(params, dict) else {}

    return {
        "raw_path": args.raw_path
        or section.get("raw_path", "data/raw/demand_forecasting.csv"),
        "output_path": args.output_path
        or section.get("output_path", "data/processed/ingested.csv"),
        "target_column": args.target_column
        or section.get("target_column", "Demand"),
        "drop_columns": section.get("drop_columns", DROP_COLUMNS),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, expected: dict[str, str]) -> None:
    """Ensure the raw dataframe has all expected columns with the right dtype class.

    Raises
    ------
    ValueError
        If any expected column is missing or has an obviously wrong dtype.
    """
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed — missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    for col, kind in expected.items():
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' should be numeric but is {df[col].dtype}."
            )
    log.info("Schema OK — all %d expected columns present.", len(expected))


# ---------------------------------------------------------------------------
# Transformations (mirrors notebook cells 5 + 11 ONLY)
# ---------------------------------------------------------------------------
def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 5: `df['Date'] = pd.to_datetime(df['Date'])`.

    We use `errors='coerce'` + drop NaT to be defensive against malformed
    rows in future data refreshes; the notebook's data was clean.
    """
    n_before = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad = df["Date"].isna().sum()
    if n_bad:
        log.warning("Dropping %d rows with unparseable Date.", n_bad)
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
    log.info("Parsed Date column (%d rows retained of %d).", len(df), n_before)
    return df


def drop_unwanted_columns(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    """Drop columns the notebook never feeds to the model.

    NOTE on `Date`, `Month`, `DayOfWeek`:
        The notebook drops them in cell 11 ONLY AFTER using them in cells 5-8
        to engineer `Month_Sin/Cos`, `DayOfWeek_Sin/Cos`, `Time_Step`. In our
        pipeline, that engineering happens in a LATER stage, so we keep `Date`
        here. We never even create `Month`/`DayOfWeek` — they're intermediate
        columns that belonged to the notebook's monolithic flow, not to a
        single-responsibility ingestion stage.
    """
    present = [c for c in drop_columns if c in df.columns]
    absent = [c for c in drop_columns if c not in df.columns]
    if absent:
        log.warning("drop_columns entries not found in data (skipped): %s", absent)
    df = df.drop(columns=present)
    log.info("Dropped %d unwanted columns. Remaining: %s",
             len(present), list(df.columns))
    return df


def drop_rows_missing_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """A row without a target cannot be used for supervised training.

    The notebook's data didn't have missing targets, but if a future refresh
    introduces them, we want to fail-safe here rather than produce NaN
    gradients inside XGBoost.
    """
    n_missing = df[target_column].isna().sum()
    if n_missing:
        log.warning("Dropping %d rows with missing target '%s'.",
                    n_missing, target_column)
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_ingested(df: pd.DataFrame, output_path: str) -> Path:
    """Write the cleaned dataframe to disk. Parent dir is auto-created.

    `index=False` is mandatory — otherwise pandas writes an unnamed index
    column that downstream stages would have to silently handle.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote ingested data → %s (%d rows, %d cols)",
             out_path, len(df), df.shape[1])
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: data ingestion")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH,
                   help="Path to params.yaml (default: params.yaml)")
    p.add_argument("--raw-path", default=None,
                   help="Override raw data path")
    p.add_argument("--output-path", default=None,
                   help="Override ingested output path")
    p.add_argument("--target-column", default=None,
                   help="Override target column name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    raw_path = Path(cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "If it is DVC-tracked, run `dvc pull` first."
        )

    log.info("Reading raw data: %s", raw_path)
    df = pd.read_csv(raw_path)
    log.info("Loaded %d rows x %d columns.", *df.shape)

    validate_schema(df, EXPECTED_COLUMNS)

    # Mirror notebook cell 5
    df = parse_date(df)

    # Drop rows with missing target (defensive; notebook's data was clean)
    df = drop_rows_missing_target(df, cfg["target_column"])

    # Mirror notebook cell 11 (only the truly-unused columns; keep Date)
    df = drop_unwanted_columns(df, cfg["drop_columns"])

    # Summary that you'll see when running `dvc repro`.
    log.info(
        "Target '%s' — count=%d, mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        cfg["target_column"],
        df[cfg["target_column"]].count(),
        df[cfg["target_column"]].mean(),
        df[cfg["target_column"]].std(),
        df[cfg["target_column"]].min(),
        df[cfg["target_column"]].max(),
    )
    log.info("Date range: %s → %s", df["Date"].min(), df["Date"].max())

    save_ingested(df, cfg["output_path"])


if __name__ == "__main__":
    main()
