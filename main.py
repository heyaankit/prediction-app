"""
main.py
=======
FastAPI app that serves demand forecasts using the trained Pipeline.

REFACTORED for the new DVC pipeline structure:
----------------------------------------------
Previously this file manually engineered features (Month_Sin, Time_Step,
Price_Gap, etc.) before calling model.predict(). That was necessary
because the OLD .pkl only contained the preprocessor + XGBRegressor.

Now the saved Pipeline INCLUDES the FeatureEngineer transformer as its
FIRST step, so the .pkl is self-contained. main.py just passes the raw
fields and Pipeline does all the feature engineering automatically.

What this file does:
  1. Loads demand_forecasting_pipeline.pkl at startup.
  2. Receives a POST /predict with raw fields (Date, Price, Discount, etc.).
  3. Builds a 1-row DataFrame with the EXACT raw column names the
     FeatureEngineer expects (Date, Price, Discount, Competitor Pricing,
     Region, Weather Condition, Category, Epidemic, Promotion).
  4. Calls pipeline.predict(df) — Pipeline handles feature engineering
     + preprocessing + prediction in one call.
  5. Returns the predicted demand.

Why this is better:
  - No duplicate feature-engineering logic between training and serving.
  - No risk of training/serving skew (e.g., different Time_Step origins).
  - main.py is ~70 lines shorter and much easier to read.

NOTE on the API input schema:
  The Pydantic DemandInput uses underscores (Competitor_Pricing) because
  that's natural for Python clients. Inside the dict we convert to the
  spaced names ('Competitor Pricing', 'Weather Condition') that the raw
  CSV and trained Pipeline expect.

Run locally:
    uvicorn main:app --reload
"""

import importlib.util
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure repo root is on sys.path AND pre-register the FeatureEngineer
# class under the module name joblib will look for when unpickling the
# saved Pipeline.
#
# Why this dance: the FeatureEngineer class lives in src/3_feature_engineering.py,
# but Python module names cannot start with a digit, so we can't write
# `from src.3_feature_engineering import FeatureEngineer`. We load it via
# importlib and register it in sys.modules under the plain name
# "feature_engineering" so pickle can find it.
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root))

_fe_path = _repo_root / "src" / "3_feature_engineering.py"
_spec = importlib.util.spec_from_file_location("feature_engineering", _fe_path)
_fe_module = importlib.util.module_from_spec(_spec)
sys.modules["feature_engineering"] = _fe_module
_spec.loader.exec_module(_fe_module)

app = FastAPI(title="Demand Forecasting API")

# Load the production model once at startup.
# The Pipeline structure: FeatureEngineer → ColumnTransformer → XGBRegressor
MODEL_PATH = "demand_forecasting_pipeline.pkl"
model = joblib.load(MODEL_PATH)
print(f"[startup] Loaded pipeline from {MODEL_PATH}")
print(f"[startup] Pipeline steps: {[name for name, _ in model.steps]}")


# ---------------------------------------------------------------------------
# API input schema — what a user actually sends.
# Uses underscores (Python-friendly). Converted to spaced names below.
# ---------------------------------------------------------------------------
class DemandInput(BaseModel):
    Date: str                # e.g., "2023-10-25"
    Price: float
    Discount: float
    Competitor_Pricing: float
    Region: str
    Weather_Condition: str
    Category: str
    Epidemic: int
    Promotion: int


@app.get("/")
def health():
    """Health check — useful for Docker/k8s probes."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict_demand(data: DemandInput):
    """Predict demand from raw inputs.

    The Pipeline's FeatureEngineer step will automatically compute:
      - Month_Sin/Cos, DayOfWeek_Sin/Cos, Time_Step from Date
      - Price_Gap from Price + Competitor Pricing

    So we just pass the raw fields and let the Pipeline handle the rest.
    """
    # Build a 1-row DataFrame using the EXACT raw column names the
    # FeatureEngineer expects (spaces, not underscores — matches the
    # original CSV the model was trained on).
    input_dict = {
        "Date": data.Date,
        "Price": data.Price,
        "Discount": data.Discount,
        "Competitor Pricing": data.Competitor_Pricing,    # space, not underscore
        "Region": data.Region,
        "Weather Condition": data.Weather_Condition,      # space, not underscore
        "Category": data.Category,
        "Epidemic": data.Epidemic,
        "Promotion": data.Promotion,
    }

    df = pd.DataFrame([input_dict])

    # One call does it all: feature engineering + preprocessing + prediction.
    prediction = model.predict(df)

    return {
        "predicted_demand": float(prediction[0])
    }
