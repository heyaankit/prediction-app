"""
Stage 7 of the DVC pipeline for the prediction-app project.

Promotes the trained pipeline from the internal `models/` directory to
the PRODUCTION path that the FastAPI app (`main.py`) loads at startup.

Input:  models/trained_pipeline.pkl   (produced by train.py)
Output: demand_forecasting_pipeline.pkl   (repo root — what main.py loads)

"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("save_model")

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
    section = params.get("save_model", {}) if isinstance(params, dict) else {}
    return {
        "model_path": args.model_path
        or section.get("model_path", "models/trained_pipeline.pkl"),
        "evaluation_path": args.evaluation_path
        or section.get("evaluation_path", "reports/evaluation.json"),
        "production_path": args.production_path
        or section.get("production_path", "demand_forecasting_pipeline.pkl"),
        # Optional quality gate: if set, refuse to promote if MAE exceeds this.
        "max_acceptable_mae": section.get("max_acceptable_mae", None),
    }


# ---------------------------------------------------------------------------
# Quality gate (optional)
# ---------------------------------------------------------------------------
def check_quality_gate(evaluation: dict, max_mae: float | None) -> None:
    """Refuse to promote the model if test MAE exceeds the threshold.

    Set `max_acceptable_mae` in params.yaml to enable. Leave it null/absent
    to skip the gate (useful while learning — you want to see every model).
    """
    if max_mae is None:
        log.info("No quality gate configured (max_acceptable_mae=null) — "
                 "promoting model unconditionally.")
        return
    actual_mae = evaluation.get("metrics", {}).get("mae", float("inf"))
    if actual_mae > max_mae:
        raise ValueError(
            f"QUALITY GATE FAILED: test MAE={actual_mae:.4f} exceeds "
            f"max_acceptable_mae={max_mae}. Model NOT promoted. "
            f"Adjust hyperparameters in train.py or relax the gate."
        )
    log.info("Quality gate PASSED: test MAE=%.4f <= %.4f",
             actual_mae, max_mae)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: save/promote model")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--model-path", default=None)
    p.add_argument("--evaluation-path", default=None)
    p.add_argument("--production-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    # Make repo root importable so joblib can unpickle the Pipeline
    # (it contains a reference to src.feature_engineering.FeatureEngineer).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.feature_engineering import FeatureEngineer  # noqa: F401

    # --- Verify trained model exists ---
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run train.py first."
        )

    # --- Load evaluation report (for quality gate + metadata) ---
    eval_path = Path(cfg["evaluation_path"])
    if not eval_path.exists():
        log.warning(
            "Evaluation report not found at %s — skipping quality gate. "
            "Run evaluate.py first for full traceability.", eval_path
        )
        evaluation = {}
    else:
        with eval_path.open("r", encoding="utf-8") as fh:
            evaluation = json.load(fh)

    # --- Quality gate ---
    check_quality_gate(evaluation, cfg["max_acceptable_mae"])

    # --- Load + validate the pipeline can predict on a tiny sample ---
    log.info("Loading model for validation: %s", model_path)
    pipeline = joblib.load(model_path)
    log.info("Pipeline steps: %s", [name for name, _ in pipeline.steps])

    # --- Copy to production path ---
    prod_path = Path(cfg["production_path"])
    prod_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, prod_path)
    log.info("Promoted model → %s (%.2f KB)",
             prod_path, prod_path.stat().st_size / 1024)

    # --- Write a sidecar metadata file (NOT DVC-tracked, just for humans) ---
    # This helps you know WHICH MLflow run produced the production model.
    meta_path = prod_path.with_suffix(".meta.json")
    metadata = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(model_path),
        "production_path": str(prod_path),
        "evaluation": evaluation.get("metrics", {}),
        "pipeline_steps": [name for name, _ in pipeline.steps],
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Wrote model metadata → %s", meta_path)

    log.info("save_model stage complete. FastAPI app will load: %s", prod_path)
    log.info("NOTE: main.py no longer needs manual feature engineering — "
             "the Pipeline's FeatureEngineer step handles it automatically.")


if __name__ == "__main__":
    main()
