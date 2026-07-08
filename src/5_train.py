"""
Stage 5 of the DVC pipeline for the prediction-app project.

Trains the XGBoost demand-forecasting model using a sklearn Pipeline that
includes the FeatureEngineer transformer (so the saved .pkl is self-contained).

Inputs:
  - data/processed/train.csv              (produced by data_split.py)
  - data/processed/selected_features.json (produced by feature_selection.py)
  - src/feature_engineering.py            (FeatureEngineer class definition)
Outputs:
  - models/trained_pipeline.pkl           (joblib dump of best pipeline)
  - mlflow_run_id.txt                     (MLflow run ID — consumed by evaluate.py)

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from category_encoders import BinaryEncoder
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor

# Import the FeatureEngineer transformer defined in feature_engineering.py.
# This requires the repo root to be on sys.path (DVC runs from repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import FeatureEngineer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

DEFAULT_PARAMS_PATH = "params.yaml"

# ---------------------------------------------------------------------------
# MLflow config — matches the original train.py so existing mlflow.db
# and mlruns/ continue to work and accumulate history.
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "Demand_Forecasting_XGBoost"

# Hyperparameter search space (notebook cell 41)
PARAM_DISTRIBUTIONS = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.3),
}

# Default feature groups (notebook cell 29) — used as fallback if
# selected_features.json doesn't classify features by group.
DEFAULT_NUMERIC = [
    "Price", "Discount", "Competitor Pricing",
    "Time_Step", "Price_Gap",
    "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
]
DEFAULT_CATEGORICAL_LOW = ["Region", "Weather Condition"]
DEFAULT_CATEGORICAL_HIGH = ["Category"]
DEFAULT_BINARIES = ["Epidemic", "Promotion"]


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
    section = params.get("train", {}) if isinstance(params, dict) else {}
    return {
        "train_path": args.train_path
        or section.get("train_path", "data/processed/train.csv"),
        "selected_features_path": args.selected_features_path
        or section.get("selected_features_path",
                       "data/processed/selected_features.json"),
        "model_output": args.model_output
        or section.get("model_output", "models/trained_pipeline.pkl"),
        "run_id_output": args.run_id_output
        or section.get("run_id_output", "mlflow_run_id.txt"),
        "target_column": section.get("target_column", "Demand"),
        "cv_splits": section.get("cv_splits", 10),
        "cv_test_size_ratio": section.get("cv_test_size_ratio", 0.094),
        "n_iter": section.get("n_iter", 10),
        "random_state": section.get("random_state", 42),
    }


# ---------------------------------------------------------------------------
# Load + parse selected_features.json
# ---------------------------------------------------------------------------
def load_selected_features(path: str) -> dict:
    """Load selected_features.json and return the list of SELECTED feature names,
    grouped by type for the ColumnTransformer."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"selected_features.json not found: {p}. Run feature_selection.py first."
        )
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    selected = [f["name"] for f in data["features"] if f.get("selected", True)]
    type_map = {f["name"]: f.get("type", "unknown") for f in data["features"]}

    # Group selected features by type
    numeric = [f for f in selected if type_map.get(f) == "numeric"]
    categorical_low = [f for f in selected if type_map.get(f) == "categorical"
                       and f not in DEFAULT_CATEGORICAL_HIGH]
    categorical_high = [f for f in selected if f in DEFAULT_CATEGORICAL_HIGH]
    binaries = [f for f in selected if type_map.get(f) == "binary"]

    log.info("Selected features → numeric=%s", numeric)
    log.info("Selected features → categorical_low=%s", categorical_low)
    log.info("Selected features → categorical_high=%s", categorical_high)
    log.info("Selected features → binaries=%s", binaries)

    return {
        "numeric": numeric,
        "categorical_low": categorical_low,
        "categorical_high": categorical_high,
        "binaries": binaries,
        "all_selected": selected,
    }


# ---------------------------------------------------------------------------
# Pipeline construction (notebook cells 29-35, with FeatureEngineer prepended)
# ---------------------------------------------------------------------------
def build_pipeline(
    selected: dict,
    random_state: int = 42,
) -> Pipeline:
    """Build the full sklearn Pipeline.

    Step 1: FeatureEngineer (cells 5-8, 12) — engineers cyclical + ratio features.
    Step 2: ColumnTransformer (cells 32-33) — impute + scale/encode.
    Step 3: XGBRegressor (cell 34-35).
    """
    # Cell 32: per-group transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_low_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    categorical_high_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", BinaryEncoder()),
    ])

    # Cell 33: ColumnTransformer — only include SELECTED feature groups.
    # Empty groups are omitted to avoid sklearn errors.
    transformers = []
    if selected["numeric"]:
        transformers.append(("num", numeric_transformer, selected["numeric"]))
    if selected["categorical_low"]:
        transformers.append(("cat_low", categorical_low_transformer,
                             selected["categorical_low"]))
    if selected["categorical_high"]:
        transformers.append(("cat_high", categorical_high_transformer,
                             selected["categorical_high"]))
    if selected["binaries"]:
        transformers.append(("binaries", "passthrough", selected["binaries"]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Full pipeline with FeatureEngineer as step 1
    model_pipeline = Pipeline(steps=[
        ("feature_engineer", FeatureEngineer(drop_date=True)),
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=random_state)),
    ])
    return model_pipeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: train XGBoost pipeline")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--train-path", default=None)
    p.add_argument("--selected-features-path", default=None)
    p.add_argument("--model-output", default=None)
    p.add_argument("--run-id-output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    # --- Load train data (notebook cell 23) ---
    train_path = Path(cfg["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data not found: {train_path}. Run data_split.py first."
        )
    log.info("Reading train data: %s", train_path)
    train_df = pd.read_csv(train_path)
    log.info("Loaded %d rows x %d columns.", *train_df.shape)

    target = cfg["target_column"]
    if target not in train_df.columns:
        raise ValueError(
            f"Target column '{target}' not in train data. "
            f"Columns: {list(train_df.columns)}"
        )
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    log.info("X_train: %s | y_train: %s", X_train.shape, y_train.shape)

    # --- Load selected features ---
    selected = load_selected_features(cfg["selected_features_path"])

    # --- Build pipeline (notebook cells 29-35 + FeatureEngineer) ---
    pipeline = build_pipeline(selected, random_state=cfg["random_state"])
    log.info("Pipeline steps: %s",
             [name for name, _ in pipeline.steps])

    # --- CV splitter (notebook cell 37) ---
    tscv = TimeSeriesSplit(
        n_splits=cfg["cv_splits"],
        test_size=int(len(X_train) * cfg["cv_test_size_ratio"]),
    )
    log.info("TimeSeriesSplit: n_splits=%d, test_size=%d",
             cfg["cv_splits"], int(len(X_train) * cfg["cv_test_size_ratio"]))

    # --- Randomized search (notebook cells 41-43) ---
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=cfg["n_iter"],
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=cfg["random_state"],
        verbose=1,
    )

    # --- MLflow tracking ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"xgb_randsearch_iter{cfg['n_iter']}_seed{cfg['random_state']}"
    with mlflow.start_run(run_name=run_name) as run:
        log.info("MLflow run_id=%s | run_name=%s", run.info.run_id, run_name)
        mlflow.set_tag("stage", "train")
        mlflow.set_tag("pipeline", "dvc-prediction-app")

        # Log config
        mlflow.log_param("cv_splits", cfg["cv_splits"])
        mlflow.log_param("cv_test_size_ratio", cfg["cv_test_size_ratio"])
        mlflow.log_param("n_iter", cfg["n_iter"])
        mlflow.log_param("random_state", cfg["random_state"])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("n_selected_features", len(selected["all_selected"]))
        mlflow.log_param("selected_features", ",".join(selected["all_selected"]))

        log.info("Fitting RandomizedSearchCV (this may take a while)...")
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_cv_mae = -random_search.best_score_
        log.info("Best params: %s", best_params)
        log.info("Best CV MAE: %.4f", best_cv_mae)

        # Log searched hyperparameters + CV metric
        for k, v in best_params.items():
            mlflow.log_param("best_" + k.replace("model__", ""), v)
        mlflow.log_metric("cv_mae", best_cv_mae)

        # Log the refit pipeline as an MLflow model
        best_pipeline = random_search.best_estimator_
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        # Log a train data summary (mirrors original train.py)
        summary = (
            "Train Data Summary:\n"
            f"- Rows: {len(X_train)}\n"
            f"- Target mean: {y_train.mean():.2f}\n"
            f"- Target std:  {y_train.std():.2f}\n"
            f"- Target min:  {y_train.min()}\n"
            f"- Target max:  {y_train.max()}\n"
            f"- Selected features: {selected['all_selected']}\n"
        )
        mlflow.log_text(summary, "train_data_summary.txt")

        # --- Persist artifacts ---
        model_path = Path(cfg["model_output"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        log.info("Saved best pipeline → %s", model_path)

        # Save run_id so evaluate.py can resume this run.
        run_id_path = Path(cfg["run_id_output"])
        run_id_path.write_text(run.info.run_id, encoding="utf-8")
        log.info("Saved MLflow run_id → %s (%s)", run_id_path, run.info.run_id)

    log.info("Training stage complete.")


if __name__ == "__main__":
    main()
