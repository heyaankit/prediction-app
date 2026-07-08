"""
3_feature_engineering.py  (DVC Stage 3 runner)
==============================================
Stage 3 of the DVC pipeline. As a SCRIPT, this file:
  - Loads data/processed/ingested.csv
  - Produces data/processed/features_manifest.json  (consumed by stage 4)
  - Produces reports/features_preview.csv            (human inspection)

The FeatureEngineer CLASS itself lives in src/feature_engineering.py
(the importable module). This split is necessary because:
  - The `3_` prefix gives visual ordering in `ls src/`
  - But Python module names can't start with a digit, so the class
    must live in a normally-named file for pickle/joblib to work
    across multiprocessing boundaries (sklearn workers) and at
    serving time (main.py loading the .pkl).

Input:  data/processed/ingested.csv
Outputs:
  - data/processed/features_manifest.json
  - reports/features_preview.csv

dvc.yaml snippet:
    stages:
      feature_engineering:
        cmd: python src/3_feature_engineering.py
        deps:
          - data/processed/ingested.csv
          - src/3_feature_engineering.py
          - src/feature_engineering.py    # class definition (dependency)
        outs:
          - data/processed/features_manifest.json
          - reports/features_preview.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Make repo root importable, then import the FeatureEngineer class from
# the importable module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import (
    ALL_OUTPUT_FEATURES,
    ENGINEERED_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    FeatureEngineer,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_engineering_stage")

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
    section = params.get("feature_engineering", {}) if isinstance(params, dict) else {}
    return {
        "input_path": args.input_path
        or section.get("input_path", "data/processed/ingested.csv"),
        "manifest_output": args.manifest_output
        or section.get("manifest_output", "data/processed/features_manifest.json"),
        "preview_output": args.preview_output
        or section.get("preview_output", "reports/features_preview.csv"),
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------
def build_manifest() -> dict:
    """Build the features manifest JSON consumed by feature_selection.py."""
    type_map = {
        "Price": "numeric",
        "Discount": "numeric",
        "Competitor Pricing": "numeric",
        "Region": "categorical",
        "Weather Condition": "categorical",
        "Category": "categorical",
        "Epidemic": "binary",
        "Promotion": "binary",
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
            "selected": True,
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

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_engineering input missing columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    manifest = build_manifest()
    save_manifest(manifest, cfg["manifest_output"])
    save_preview(df, cfg["preview_output"])

    log.info("Feature engineering stage complete.")
    log.info("The FeatureEngineer class lives in src/feature_engineering.py "
             "(importable module).")


if __name__ == "__main__":
    main()
