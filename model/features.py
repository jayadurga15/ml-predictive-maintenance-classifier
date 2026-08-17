"""
features.py
------------
Shared feature engineering for the AI4I 2020 Predictive Maintenance project.
Used by BOTH the training script (train_models.py) and the Streamlit app (app.py)
so that raw uploaded CSVs are transformed in exactly the same way the models were trained on.

Raw columns expected in the input CSV (as published in the AI4I 2020 dataset):
    UDI, Product ID, Type, Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure,
    TWF, HDF, PWF, OSF, RNF

Only 'Type', the 5 sensor columns, and 'Machine failure' (the label) are used.
UDI, Product ID and the 5 failure-mode flags (TWF/HDF/PWF/OSF/RNF) are dropped
because they either carry no predictive signal (IDs) or leak the label
(the flags are sub-causes that were used to construct 'Machine failure' itself).
"""

import numpy as np
import pandas as pd

RAW_NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COL = "Machine failure"

# Final feature list produced by build_features(), in a fixed order.
# 5 raw sensor readings + 2 one-hot Type dummies + 5 engineered features = 12 features.
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
    "Power [W]",
    "Temp difference [K]",
    "Torque per wear",
    "Strain",
    "Speed to torque ratio",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a raw AI4I-2020-shaped dataframe and return a dataframe of exactly
    FEATURE_COLUMNS, ready to feed into a trained model.
    """
    df = df.copy()

    # Basic sanity: make sure required raw columns exist
    missing = [c for c in RAW_NUMERIC_COLS + ["Type"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    # One-hot encode Type (L, M, H) -> Type_L, Type_M (H is the baseline / reference level)
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)

    # --- Engineered features ---
    # Mechanical power delivered by the tool, in Watts.
    # Power (W) = Torque (Nm) * Angular speed (rad/s), angular speed = rpm * 2*pi/60
    df["Power [W]"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    # How much hotter the process is running relative to ambient air -- a classic
    # predictive-maintenance signal for heat-dissipation failures.
    df["Temp difference [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Torque relative to accumulated tool wear (avoid divide-by-zero with +1)
    df["Torque per wear"] = df["Torque [Nm]"] / (df["Tool wear [min]"] + 1)

    # "Strain" proxy used in overstrain-failure literature: wear * torque
    df["Strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Speed-to-torque ratio (avoid divide-by-zero with +1)
    df["Speed to torque ratio"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1)

    return df[FEATURE_COLUMNS]


def build_features_and_target(df: pd.DataFrame):
    """Convenience helper: returns (X, y) if the label column is present, else (X, None)."""
    X = build_features(df)
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y
