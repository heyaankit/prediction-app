"""
feature_engineering.py  (the REAL module — class lives here)
============================================================
Defines the FeatureEngineer sklearn transformer used by the training
Pipeline. This file MUST have a normal Python module name (no leading
digit) because:

  1. pickle/joblib stores class references as `<module>.<ClassName>`
  2. When sklearn's RandomizedSearchCV(n_jobs=-1) spawns worker
     processes via joblib, each worker must re-import the module
     by name to unpickle the Pipeline. A leading digit in the
     filename makes the module un-importable, breaking workers.

  3. When main.py loads the saved .pkl at serving time, the same
     import must succeed.

The companion file `3_feature_engineering.py` is the DVC STAGE RUNNER
(it produces features_manifest.json + features_preview.csv). It imports
FeatureEngineer from THIS file. The `3_` prefix gives visual ordering
in `ls src/` without breaking pickling.

Usage from anywhere in the repo:
    from src.feature_engineering import FeatureEngineer
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger("feature_engineering")

# ---------------------------------------------------------------------------
# Constants used by both the stage runner and train.py
# ---------------------------------------------------------------------------
REQUIRED_INPUT_COLUMNS = [
    "Date",
    "Price",
    "Discount",
    "Competitor Pricing",  # space, not underscore — matches raw CSV
]

ENGINEERED_FEATURES = [
    "Month_Sin",
    "Month_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Time_Step",
    "Price_Gap",
]

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
# FeatureEngineer Transformer (mirrors notebook cells 5-8, 12)
# ===========================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that engineers cyclical + ratio features from Date.

    Baked into the sklearn Pipeline in train.py as the FIRST step, so the
    saved .pkl is self-contained (FastAPI main.py just passes raw fields).

    Parameters
    ----------
    drop_date : bool, default True
        Whether to drop the Date column after engineering (notebook cell 11).

    Attributes
    ----------
    min_date_ : pd.Timestamp
        Minimum Date in training data. Used as Time_Step origin. Learned
        in fit() so there's no data leakage from val/test (this is a
        strict improvement over the notebook, which computed min(Date)
        over the full dataset before splitting).
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


__all__ = [
    "FeatureEngineer",
    "REQUIRED_INPUT_COLUMNS",
    "ENGINEERED_FEATURES",
    "ALL_OUTPUT_FEATURES",
]
