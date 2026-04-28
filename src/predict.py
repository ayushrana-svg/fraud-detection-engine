from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import FEATURE_COLS_PATH, SCALER_PATH
from src.features import create_features
from src.utils import get_risk_level, load_artifacts, validate_transaction


MODEL, FEATURE_COLS, THRESHOLD = load_artifacts()
SCALER = joblib.load(SCALER_PATH)
_ = joblib.load(FEATURE_COLS_PATH)


def predict(transaction: dict) -> dict:
    validate_transaction(transaction)

    payload = transaction.copy()
    payload.setdefault("transaction_id", f"txn_{uuid4().hex[:12]}")

    df = pd.DataFrame([payload])
    features = create_features(df)
    X = features.reindex(columns=FEATURE_COLS, fill_value=0)
    X_scaled = SCALER.transform(X)

    fraud_probability = float(MODEL.predict_proba(X_scaled)[0, 1])
    is_fraud = fraud_probability >= THRESHOLD

    return {
        "transaction_id": payload["transaction_id"],
        "fraud_probability": fraud_probability,
        "is_fraud": bool(is_fraud),
        "risk_level": get_risk_level(fraud_probability),
        "threshold_used": float(THRESHOLD),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
