# Real-Time Payment Fraud Detection Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

Production-ready ML system for real-time payment fraud scoring, explainability, and risk-tier routing.

## Problem Statement

Digital payments require instant decisions under extreme class imbalance. This project detects potentially fraudulent transactions in real time using engineered behavioral signals, an imbalanced-learning pipeline, and explainable API responses suitable for analysts and downstream rules engines.

## Architecture

```text
               +--------------------+
Incoming Tx -->|  FastAPI /predict  |---------------------+
               +---------+----------+                     |
                         |                                v
                         |                    +------------------------+
                         +------------------->|  Feature Engineering   |
                                              |  (single source truth) |
                                              +-----------+------------+
                                                          |
                                                          v
                                              +------------------------+
                                              | XGBoost Fraud Model    |
                                              | + threshold + SHAP     |
                                              +-----------+------------+
                                                          |
                                                          v
                                              +------------------------+
                                              | JSON Response          |
                                              | prob, risk, top factors|
                                              +------------------------+
```

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy |
| Features/ML | scikit-learn, imbalanced-learn (SMOTE), xgboost |
| Explainability | shap |
| Serving | fastapi, uvicorn, pydantic |
| Serialization | joblib |
| Testing/HTTP | pytest, httpx |

## Setup

```bash
git clone <your-repo-url>
cd fraud-detection-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload --port 8000
```

## API Usage

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"amount\": 1499.0, \"user_id\": \"user_00012\", \"merchant_id\": \"merchant_0781\", \"timestamp\": \"2026-04-28T21:15:00\"}"
```

Sample request:

```json
{
  "amount": 1499.0,
  "user_id": "user_00012",
  "merchant_id": "merchant_0781",
  "timestamp": "2026-04-28T21:15:00"
}
```

Sample response:

```json
{
  "transaction_id": "txn_a1b2c3d4e5f6",
  "fraud_probability": 0.9182,
  "is_fraud": true,
  "risk_level": "HIGH",
  "explanation": {
    "top_features": [
      {"feature": "amount", "impact": 1.284},
      {"feature": "amount_deviation", "impact": 0.731},
      {"feature": "is_night", "impact": 0.392}
    ]
  },
  "threshold_used": 0.48,
  "processed_at": "2026-04-28T17:12:51.128812"
}
```

## Model Performance

Metrics from latest `python src/train.py` run:

| Metric | Value |
|---|---|
| ROC-AUC | `0.7985` |
| Precision (default 0.5) | `0.0432` |
| Recall (default 0.5) | `0.8450` |
| F1 (default 0.5) | `0.0821` |
| Optimal threshold (PR/F1) | `0.9816` |
| Precision (optimal) | `0.2887` |
| Recall (optimal) | `0.1025` |
| F1 (optimal) | `0.1513` |

## Business Impact

Fraud operations balance false positives (blocking good customers) vs false negatives (missing fraud). The default and tuned thresholds allow you to move along this curve:

- Lower threshold: catches more fraud (higher recall), but increases manual review/customer friction.
- Higher threshold: fewer false alerts (higher precision), but greater fraud leakage risk.

## Threshold Tuning

The training pipeline computes a precision-recall curve and picks the threshold that maximizes F1 on held-out test data. This value is persisted to `models/threshold.pkl` and used directly by the live API for consistent decisioning between offline and online workflows.
