from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api.schemas import FeatureImpact, PredictionOutput, TransactionInput
from src.config import METRICS_PATH, PREDICTION_AUDIT_LOG_PATH, SCALER_PATH
from src.features import create_features
from src.utils import format_shap_explanation, get_risk_level, load_artifacts, validate_transaction

app_state: dict = {
    "model": None,
    "feature_cols": None,
    "threshold": None,
    "scaler": None,
    "explainer": None,
    "metrics": {},
    "model_loaded": False,
}

def _write_prediction_audit(entry: dict) -> None:
    PREDICTION_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_AUDIT_LOG_PATH.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(entry) + "\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model, feature_cols, threshold = load_artifacts()
        scaler = joblib.load(SCALER_PATH)
        explainer = shap.TreeExplainer(model)
        metrics = {}
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

        app_state.update(
            {
                "model": model,
                "feature_cols": feature_cols,
                "threshold": threshold,
                "scaler": scaler,
                "explainer": explainer,
                "metrics": metrics,
                "model_loaded": True,
            }
        )
        print("Model loaded successfully.")
    except Exception as exc:  # pragma: no cover
        app_state["model_loaded"] = False
        print(f"Model loading failed: {exc}")
    yield


app = FastAPI(title="Real-Time Payment Fraud Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app_state["model_loaded"]}


@app.get("/metrics")
def metrics():
    if not app_state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return app_state["metrics"]


@app.post("/predict", response_model=PredictionOutput)
def predict_transaction(payload: TransactionInput):
    if not app_state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        payload_dict = payload.model_dump()
        validate_transaction(payload_dict)

        txn_id = f"txn_{uuid4().hex[:12]}"
        raw_df = pd.DataFrame([{**payload_dict, "transaction_id": txn_id}])
        features_df = create_features(raw_df)

        X = features_df.reindex(columns=app_state["feature_cols"], fill_value=0)
        X_scaled = app_state["scaler"].transform(X)

        fraud_probability = float(app_state["model"].predict_proba(X_scaled)[0, 1])
        threshold = float(app_state["threshold"])
        is_fraud = bool(fraud_probability >= threshold)
        risk_level = get_risk_level(fraud_probability)

        shap_values = app_state["explainer"].shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_vector = np.array(shap_values[1][0])
        else:
            shap_vector = np.array(shap_values[0])

        top_features = [
            FeatureImpact(**item)
            for item in format_shap_explanation(shap_vector, app_state["feature_cols"])
        ]

        response = PredictionOutput(
            transaction_id=txn_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            risk_level=risk_level,
            explanation={"top_features": top_features},
            threshold_used=threshold,
            processed_at=datetime.now(timezone.utc),
        )
        _write_prediction_audit(
            {
                "transaction_id": txn_id,
                "payload": payload_dict,
                "fraud_probability": fraud_probability,
                "is_fraud": is_fraud,
                "risk_level": risk_level,
                "threshold_used": threshold,
                "top_features": [item.model_dump() for item in top_features],
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
