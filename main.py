"""

What this file does:
  1. Loads demand_forecasting_pipeline.pkl at startup.
  2. Receives a POST /predict with raw fields (Date, Price, Discount, etc.).
  3. Builds a 1-row DataFrame with the EXACT raw column names the
     FeatureEngineer expects (Date, Price, Discount, Competitor Pricing,
     Region, Weather Condition, Category, Epidemic, Promotion).
  4. Calls pipeline.predict(df) — Pipeline handles feature engineering
     + preprocessing + prediction in one call.
  5. Returns the predicted demand.


NOTE on the API input schema:
  The Pydantic DemandInput uses underscores (Competitor_Pricing) because
  that's natural for Python clients. Inside the dict we convert to the
  spaced names ('Competitor Pricing', 'Weather Condition') that the raw
  CSV and trained Pipeline expect.

Run locally:
    uvicorn main:app --reload
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure src/ is importable so the FeatureEngineer class can be loaded
# when joblib unpickles the Pipeline. (Without this, you'd get
# "ModuleNotFoundError: No module named 'src'" on startup.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
