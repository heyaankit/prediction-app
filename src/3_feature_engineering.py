"""
Stage 3 of the DVC pipeline for the prediction-app project.


As a DVC stage, it runs `main()` which:
   - Loads ingested.csv
   - Fits the FeatureEngineer on the data
   - Produces `data/processed/features_manifest.json` — a JSON listing ALL
     features that the transformer WILL create. This manifest is consumed
     by `feature_selection.py` (next stage).
   - Produces `reports/features_preview.csv` — a sample of the transformed
     data so you can eyeball the engineered features without running training.

Input:  data/processed/ingested.csv   (produced by data_ingestion.py)
Outputs:
  - data/processed/features_manifest.json
  - reports/features_preview.csv

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_engineering")

DEFAULT_PARAMS_PATH = "params.yaml"

# Columns the FeatureEngineer expects to receive from data_ingestion.py
REQUIRED_INPUT_COLUMNS = [
    "Date",
    "Price",
    "Discount",
    "Competitor Pricing",  # space, not underscore
]

# Features that the FeatureEngineer CREATES (output side)
# Used to build features_manifest.json for the feature_selection stage.
ENGINEERED_FEATURES = [
    "Month_Sin",
    "Month_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Time_Step",
    "Price_Gap",
]

# All features that exist AFTER transformation (engineered + retained raw)
# This is the full menu from which feature_selection picks.
ALL_OUTPUT_FEATURES = [
    "Price",
    "Discount",
    "Competitor Pricing",
    "Month_Sin",
    "Month_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Time_Step",
    "Price_Gap",
    "Region",
    "Weather Condition",
    "Category",
    "Epidemic",
    "Promotion",
]


# ===========================================================================
# FeatureEngineer Transformer (notebook cells 5-8, 12)
# ===========================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that engineers cyclical + ratio features from Date.

    Mirrors notebook cells 5-8 and 12. Baked into the sklearn Pipeline in
    train.py as the FIRST step, so the saved .pkl is self-contained.

    Parameters
    ----------
    drop_date : bool, default True
        Whether to drop the Date column after engineering (notebook cell 11).
        Keep it True for training; set False only for debugging/EDA.

    Attributes
    ----------
    min_date_ : pd.Timestamp
        The minimum Date in the training data. Used as the origin for
        Time_Step. Learned in fit() so there's no data leakage from val/test.
    """

    def __init__(self, drop_date: bool = True):
        self.drop_date = drop_date

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn the Time_Step origin from training data only."""
        if "Date" not in X.columns:
            raise ValueError("FeatureEngineer requires a 'Date' column.")
        dates = pd.to_datetime(X["Date"], errors="coerce")
        self.min_date_ = dates.min()
        log.info("FeatureEngineer.fit: min_date_ = %s", self.min_date_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps (notebook cells 5-8, 12)."""
        df = X.copy()

        # --- Cell 5: parse Date ---
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        month = df["Date"].dt.month
        dayofweek = df["Date"].dt.dayofweek

        # --- Cell 6: cyclical encoding of month ---
        df["Month_Sin"] = np.sin(2 * np.pi * month / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * month / 12)

        # --- Cell 7: cyclical encoding of day of week ---
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * dayofweek / 7)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * dayofweek / 7)

        # --- Cell 8: Time_Step (days since min date in TRAINING data) ---
        df["Time_Step"] = (df["Date"] - self.min_date_).dt.days

        # --- Cell 12: Price_Gap (% diff vs competitor pricing) ---
        # NOTE: notebook does not guard against divide-by-zero. We log a
        # warning if it happens but follow the same formula for parity.
        df["Price_Gap"] = (
            (df["Price"] - df["Competitor Pricing"]) / df["Competitor Pricing"]
        ) * 100
        n_inf = np.isinf(df["Price_Gap"]).sum()
        if n_inf > 0:
            log.warning(
                "%d rows produced +/-inf Price_Gap (Competitor Pricing == 0). "
                "Consider handling upstream.", n_inf
            )

        # --- Cell 11 (partial): drop Date ---
        if self.drop_date:
            df = df.drop(columns=["Date"])

        return df


# ===========================================================================
# DVC stage: produce manifest + preview
# ===========================================================================
def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning("%s not found — using built-in defaults.", path)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(args: argparse.Namespace) -> dict:
    params = load_params(args.params)
    section = params.get("feature_engineering", {}) if isinstance(params, dict) else {}
    return {
        "input_path": args.input_path
        or section.get("input_path", "data/processed/ingested.csv"),
        "manifest_output": args.manifest_output
        or section.get("manifest_output", "data/processed/features_manifest.json"),
        "preview_output": args.preview_output
        or section.get("preview_output", "reports/features_preview.csv"),
    }


def build_manifest() -> dict:
    """Build the features manifest JSON consumed by feature_selection.py.

    Lists every feature that exists AFTER the FeatureEngineer transforms the
    data, along with its type ('numeric', 'categorical', 'binary') and source
    ('raw' = retained from ingestion, 'engineered' = created by transformer).
    """
    type_map = {
        # Raw retained
        "Price": "numeric",
        "Discount": "numeric",
        "Competitor Pricing": "numeric",
        "Region": "categorical",
        "Weather Condition": "categorical",
        "Category": "categorical",
        "Epidemic": "binary",
        "Promotion": "binary",
        # Engineered
        "Month_Sin": "numeric",
        "Month_Cos": "numeric",
        "DayOfWeek_Sin": "numeric",
        "DayOfWeek_Cos": "numeric",
        "Time_Step": "numeric",
        "Price_Gap": "numeric",
    }
    features = []
    for name in ALL_OUTPUT_FEATURES:
        features.append({
            "name": name,
            "type": type_map.get(name, "unknown"),
            "source": "engineered" if name in ENGINEERED_FEATURES else "raw",
            "selected": True,  # default: all selected; feature_selection.py edits this
        })
    return {
        "engineered_by_feature_engineer": ENGINEERED_FEATURES,
        "features": features,
        "target_column": "Demand",
    }


def save_manifest(manifest: dict, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Wrote features manifest → %s (%d features)",
             out, len(manifest["features"]))
    return out


def save_preview(df_in: pd.DataFrame, path: str) -> Path:
    """Apply FeatureEngineer to ingested data and save a preview CSV."""
    fe = FeatureEngineer(drop_date=True)
    fe.fit(df_in)
    df_out = fe.transform(df_in)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    log.info("Wrote features preview → %s (%d rows, %d cols)",
             out, *df_out.shape)
    log.info("Preview columns: %s", list(df_out.columns))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: feature engineering")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--input-path", default=None)
    p.add_argument("--manifest-output", default=None)
    p.add_argument("--preview-output", default=None)
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

    # Validate required input columns
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_engineering input missing columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    # Build + save manifest (consumed by feature_selection.py)
    manifest = build_manifest()
    save_manifest(manifest, cfg["manifest_output"])

    # Generate preview (for human inspection)
    save_preview(df, cfg["preview_output"])

    log.info("Feature engineering stage complete.")
    log.info("The FeatureEngineer transformer class is importable: "
             "from src.feature_engineering import FeatureEngineer")


if __name__ == "__main__":
    main()
