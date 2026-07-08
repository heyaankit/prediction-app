"""
Stage 6 of the DVC pipeline for the prediction-app project.

Loads the trained pipeline + test set, computes evaluation metrics, logs
them to MLflow (resuming the same run started by train.py), and writes
a JSON report for human/DVC inspection.

Inputs:
  - models/trained_pipeline.pkl   (produced by train.py)
  - data/processed/test.csv       (produced by data_split.py)
  - mlflow_run_id.txt             (produced by train.py — to resume MLflow run)
Output:
  - reports/evaluation.json       (metrics + metadata, DVC-tracked)

Metrics computed (notebook cell 46 computes MAE only; we add RMSE + R²
because they're standard regression metrics and cost nothing extra):
  - MAE  (mean absolute error)       — notebook cell 46
  - RMSE (root mean squared error)   — penalizes large errors more
  - R²   (coefficient of determination) — variance explained
  - MAPE (mean absolute % error)     — relative error

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

DEFAULT_PARAMS_PATH = "params.yaml"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


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
    section = params.get("evaluate", {}) if isinstance(params, dict) else {}
    return {
        "model_path": args.model_path
        or section.get("model_path", "models/trained_pipeline.pkl"),
        "test_path": args.test_path
        or section.get("test_path", "data/processed/test.csv"),
        "run_id_path": args.run_id_path
        or section.get("run_id_path", "mlflow_run_id.txt"),
        "output_path": args.output_path
        or section.get("output_path", "reports/evaluation.json"),
        "target_column": section.get("target_column", "Demand"),
    }


# ---------------------------------------------------------------------------
# Metrics (notebook cell 46 computes MAE; we add RMSE, R², MAPE)
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE can blow up if y_true contains zeros; guard it.
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except (ZeroDivisionError, ValueError):
        mape = float("nan")
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: evaluate trained model")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--model-path", default=None)
    p.add_argument("--test-path", default=None)
    p.add_argument("--run-id-path", default=None)
    p.add_argument("--output-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    # Make repo root importable so joblib can unpickle the Pipeline
    # (it contains a reference to src.feature_engineering.FeatureEngineer).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # Pre-import FeatureEngineer so pickle finds it in sys.modules.
    from src.feature_engineering import FeatureEngineer  # noqa: F401

    # --- Load trained pipeline ---
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run train.py first."
        )
    log.info("Loading model: %s", model_path)
    pipeline = joblib.load(model_path)
    log.info("Pipeline steps: %s", [name for name, _ in pipeline.steps])

    # --- Load test data (notebook cell 45) ---
    test_path = Path(cfg["test_path"])
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Run data_split.py first."
        )
    log.info("Reading test data: %s", test_path)
    test_df = pd.read_csv(test_path)
    log.info("Loaded %d rows x %d columns.", *test_df.shape)

    target = cfg["target_column"]
    if target not in test_df.columns:
        raise ValueError(
            f"Target column '{target}' not in test data. "
            f"Columns: {list(test_df.columns)}"
        )
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # --- Predict (notebook cell 45) ---
    log.info("Predicting on %d test rows...", len(X_test))
    y_pred = pipeline.predict(X_test)

    # --- Compute metrics ---
    metrics = compute_metrics(y_test.values, y_pred)
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info("MAE:  %.4f", metrics["mae"])
    log.info("RMSE: %.4f", metrics["rmse"])
    log.info("R²:   %.4f", metrics["r2"])
    log.info("MAPE: %.4f", metrics["mape"])
    log.info("=" * 60)

    # --- Build evaluation report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "test_path": str(test_path),
        "n_test_rows": len(X_test),
        "target_column": target,
        "metrics": metrics,
        "y_test_summary": {
            "mean": float(y_test.mean()),
            "std": float(y_test.std()),
            "min": float(y_test.min()),
            "max": float(y_test.max()),
        },
        "y_pred_summary": {
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
        },
    }

    # --- Log to MLflow (resume the run started by train.py) ---
    run_id_path = Path(cfg["run_id_path"])
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("stage", "evaluate")
            mlflow.log_metric("test_mae", metrics["mae"])
            mlflow.log_metric("test_rmse", metrics["rmse"])
            mlflow.log_metric("test_r2", metrics["r2"])
            mlflow.log_metric("test_mape", metrics["mape"])
            # Log the evaluation report as an artifact
            mlflow.log_dict(report, "evaluation_report.json")
            log.info("Logged test metrics to MLflow run: %s", run_id)
    else:
        log.warning(
            "mlflow_run_id.txt not found at %s — skipping MLflow logging. "
            "Metrics are still saved to the JSON report.", run_id_path
        )

    # --- Save evaluation report (DVC-tracked) ---
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Wrote evaluation report → %s", out_path)

    log.info("Evaluation stage complete.")


if __name__ == "__main__":
    main()
