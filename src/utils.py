from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from src.config import FEATURE_COLS_PATH, MODEL_PATH, THRESHOLD_PATH


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    return model, feature_cols, threshold


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob <= 0.7:
        return "MEDIUM"
    return "HIGH"


def format_shap_explanation(shap_values: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    impacts = list(zip(feature_names, shap_values.tolist(), strict=False))
    sorted_impacts = sorted(impacts, key=lambda item: abs(item[1]), reverse=True)[:3]
    return [{"feature": feature, "impact": float(impact)} for feature, impact in sorted_impacts]


def validate_transaction(data: dict[str, Any]) -> None:
    for key in ["amount", "user_id", "merchant_id", "timestamp"]:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")
    amount = float(data["amount"])
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if not str(data["user_id"]).strip():
        raise ValueError("user_id cannot be empty")
    if not str(data["merchant_id"]).strip():
        raise ValueError("merchant_id cannot be empty")
