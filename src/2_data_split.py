"""
Stage 2 of the DVC pipeline for the prediction-app project.

Input:  data/processed/ingested.csv   (produced by data_ingestion.py)
Outputs:
  - data/processed/train.csv
  - data/processed/val.csv
  - data/processed/test.csv

We split on `Date` directly using day-based thresholds that produce
IDENTICAL splits to the notebook's Time_Step approach:
    max_time_step_days = (max(Date) - min(Date)).days
    train_cutoff_days  = int(max_time_step_days * 0.70)
    val_cutoff_days    = int(max_time_step_days * 0.85)
    train_cutoff_date  = min(Date) + timedelta(days=train_cutoff_days)
    val_cutoff_date    = min(Date) + timedelta(days=val_cutoff_days)

"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_split")

DEFAULT_PARAMS_PATH = "params.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning("%s not found — using built-in defaults.", path)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(args: argparse.Namespace) -> dict:
    params = load_params(args.params)
    section = params.get("data_split", {}) if isinstance(params, dict) else {}
    return {
        "input_path": args.input_path
        or section.get("input_path", "data/processed/ingested.csv"),
        "train_output": args.train_output
        or section.get("train_output", "data/processed/train.csv"),
        "val_output": args.val_output
        or section.get("val_output", "data/processed/val.csv"),
        "test_output": args.test_output
        or section.get("test_output", "data/processed/test.csv"),
        "train_ratio": section.get("train_ratio", 0.70),
        "val_ratio": section.get("val_ratio", 0.15),
    }


# ---------------------------------------------------------------------------
# Time-based split (mirrors notebook cell 20, computed on Date)
# ---------------------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by Date thresholds.

    Equivalent to the notebook's Time_Step-based split:
        max_ts = (max(Date) - min(Date)).days
        train_cutoff_ts = int(max_ts * 0.70)
        val_cutoff_ts   = int(max_ts * 0.85)
    but expressed directly on Date so we don't need to engineer Time_Step
    in a separate stage.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    if "Date" not in df.columns:
        raise ValueError(
            "Column 'Date' not found. data_ingestion.py must retain Date."
        )

    # Ensure datetime in case the CSV read it back as strings.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        n_bad = df["Date"].isna().sum()
        raise ValueError(
            f"{n_bad} rows have unparseable Date — fix in data_ingestion.py."
        )

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    total_days = (max_date - min_date).days

    train_cutoff_date = min_date + timedelta(days=int(total_days * train_ratio))
    val_cutoff_date = min_date + timedelta(
        days=int(total_days * (train_ratio + val_ratio))
    )

    train_df = df[df["Date"] <= train_cutoff_date].reset_index(drop=True)
    val_df = df[
        (df["Date"] > train_cutoff_date) & (df["Date"] <= val_cutoff_date)
    ].reset_index(drop=True)
    test_df = df[df["Date"] > val_cutoff_date].reset_index(drop=True)

    log.info("Date range: %s → %s (total %d days)", min_date, max_date, total_days)
    log.info("Train cutoff date: %s — %d rows", train_cutoff_date, len(train_df))
    log.info("Val   cutoff date: %s — %d rows", val_cutoff_date, len(val_df))
    log.info("Test  (> %s)      — %d rows", val_cutoff_date, len(test_df))

    total = len(train_df) + len(val_df) + len(test_df)
    log.info(
        "Split ratios — train=%.1f%%, val=%.1f%%, test=%.1f%% (total=%d)",
        100 * len(train_df) / total,
        100 * len(val_df) / total,
        100 * len(test_df) / total,
        total,
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_split(df: pd.DataFrame, path: str, label: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    log.info("Wrote %s split → %s (%d rows)", label, p, len(df))
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: time-based data split")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--input-path", default=None)
    p.add_argument("--train-output", default=None)
    p.add_argument("--val-output", default=None)
    p.add_argument("--test-output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    in_path = Path(cfg["input_path"])
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input not found: {in_path}. Run data_ingestion.py first."
        )

    log.info("Reading ingested data: %s", in_path)
    df = pd.read_csv(in_path)
    log.info("Loaded %d rows x %d columns.", *df.shape)

    train_df, val_df, test_df = time_based_split(
        df,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
    )

    save_split(train_df, cfg["train_output"], "train")
    save_split(val_df, cfg["val_output"], "val")
    save_split(test_df, cfg["test_output"], "test")


if __name__ == "__main__":
    main()
