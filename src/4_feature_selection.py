"""
Stage 4 of the DVC pipeline for the prediction-app project.

Reads `features_manifest.json` (produced by feature_engineering.py) and
produces `selected_features.json` — a user-editable config that tells
`train.py` which features to actually feed to the model.

Input:  data/processed/features_manifest.json
Output: data/processed/selected_features.json

How it works
------------
- FIRST RUN: creates `selected_features.json` with ALL features selected.
  This mirrors the notebook (which keeps every engineered feature).
- SUBSEQUENT RUNS: preserves your edits. If you manually edited the JSON
  to deselect features, `dvc repro` will NOT overwrite your selection —
  it only adds new features (from an updated manifest) as selected by
  default and removes features that no longer exist in the manifest.


"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_selection")

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
    section = params.get("feature_selection", {}) if isinstance(params, dict) else {}
    return {
        "manifest_path": args.manifest_path
        or section.get("manifest_path", "data/processed/features_manifest.json"),
        "output_path": args.output_path
        or section.get("output_path", "data/processed/selected_features.json"),
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Manifest not found: {p}. Run feature_engineering.py first."
        )
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_existing_selection(path: str) -> dict | None:
    """Load existing selected_features.json if it exists (to preserve edits)."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_selection(manifest: dict, existing: dict | None) -> dict:
    """Build selected_features.json from manifest, preserving prior edits.

    Rules:
    - If a feature exists in both manifest and existing: keep existing 'selected' value.
    - If a feature is new in manifest (not in existing): default selected=True.
    - If a feature was in existing but not in manifest: drop it (feature no longer exists).
    """
    manifest_features = {f["name"]: f for f in manifest.get("features", [])}
    existing_map = {}
    if existing:
        for f in existing.get("features", []):
            existing_map[f["name"]] = f.get("selected", True)

    merged = []
    for name, feat in manifest_features.items():
        # Preserve user's previous selection if they edited it; else default True
        selected = existing_map.get(name, True)
        merged.append({
            "name": name,
            "type": feat.get("type", "unknown"),
            "source": feat.get("source", "unknown"),
            "selected": selected,
        })

    return {
        "target_column": manifest.get("target_column", "Demand"),
        "features": merged,
    }


def save_selection(selection: dict, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2)
    return out


def print_summary(selection: dict) -> None:
    features = selection["features"]
    selected = [f["name"] for f in features if f["selected"]]
    deselected = [f["name"] for f in features if not f["selected"]]
    log.info("=" * 60)
    log.info("FEATURE SELECTION SUMMARY")
    log.info("=" * 60)
    log.info("Total features: %d", len(features))
    log.info("Selected (%d): %s", len(selected), selected)
    if deselected:
        log.info("Deselected (%d): %s", len(deselected), deselected)
    else:
        log.info("Deselected (0): none — all features active")
    log.info("=" * 60)
    log.info("To experiment: edit data/processed/selected_features.json, "
             "then run `dvc repro`.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: feature selection")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--output-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    manifest = load_manifest(cfg["manifest_path"])
    existing = load_existing_selection(cfg["output_path"])

    if existing:
        log.info("Found existing %s — preserving your edits.", cfg["output_path"])
    else:
        log.info("No existing selection — creating with ALL features selected "
                 "(matches notebook behavior).")

    selection = merge_selection(manifest, existing)
    save_selection(selection, cfg["output_path"])
    log.info("Wrote selected_features.json → %s", cfg["output_path"])

    print_summary(selection)


if __name__ == "__main__":
    main()
