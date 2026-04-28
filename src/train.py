from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    DEFAULT_FRAUD_THRESHOLD,
    DROP_COLUMNS_FOR_TRAINING,
    FEATURE_COLS_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RAW_COLUMNS,
    RAW_TRANSACTIONS_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
    THRESHOLD_PATH,
    XGBOOST_PARAMS,
)
from src.features import create_features


def generate_synthetic_dataset(n_rows: int = 100_000, fraud_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n_users = 4_000
    n_merchants = 900
    start_time = datetime(2025, 1, 1)

    user_ids = np.array([f"user_{i:05d}" for i in range(n_users)])
    merchant_ids = np.array([f"merchant_{i:04d}" for i in range(n_merchants)])

    sampled_users = rng.choice(user_ids, size=n_rows, replace=True)
    sampled_merchants = rng.choice(merchant_ids, size=n_rows, replace=True)
    amounts = rng.lognormal(mean=4.3, sigma=0.85, size=n_rows)

    user_tenure_days = {user: int(rng.integers(1, 365)) for user in user_ids}
    first_seen_merchant_index = {user: set() for user in user_ids}
    user_last_tx: dict[str, datetime] = {}

    rows = []
    risk_scores: list[float] = []

    for i in range(n_rows):
        user = str(sampled_users[i])
        merchant = str(sampled_merchants[i])
        tx_time = start_time + timedelta(minutes=int(rng.integers(0, 60 * 24 * 180)))

        if user in user_last_tx and rng.random() < 0.08:
            tx_time = user_last_tx[user] + timedelta(seconds=int(rng.integers(10, 300)))
        user_last_tx[user] = tx_time

        hour = tx_time.hour
        is_night = 1 if 0 <= hour <= 6 else 0
        amount = float(amounts[i])

        is_new_merchant = merchant not in first_seen_merchant_index[user]
        first_seen_merchant_index[user].add(merchant)
        velocity_burst = 1 if rng.random() < 0.06 else 0

        risk_score = 0.0
        risk_score += 3.2 if amount > 1100 else 0.0
        risk_score += 1.8 if amount > 700 else 0.0
        risk_score += 1.4 * is_night
        risk_score += 1.7 if is_new_merchant else 0.0
        risk_score += 2.1 * velocity_burst
        risk_score += 0.6 if user_tenure_days[user] < 21 else 0.0
        risk_score += rng.normal(0, 0.45)
        risk_scores.append(risk_score)

        rows.append(
            {
                "transaction_id": f"txn_{i+1:07d}",
                "user_id": user,
                "merchant_id": merchant,
                "amount": round(amount, 2),
                "timestamp": tx_time.isoformat(),
                "is_fraud": 0,
            }
        )

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    # Enforce exact fraud rate while preserving pattern signal.
    score_series = pd.Series(risk_scores, index=df.index)
    threshold_score = float(score_series.quantile(1 - fraud_rate))
    df["is_fraud"] = (score_series >= threshold_score).astype(int)

    return df


def train() -> None:
    print("[1/8] Ensuring directories exist...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRANSACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[2/8] Generating synthetic dataset...")
    dataset = generate_synthetic_dataset()
    dataset.to_csv(RAW_TRANSACTIONS_PATH, index=False)
    print(
        f"Saved dataset to {RAW_TRANSACTIONS_PATH} | rows={len(dataset)} | fraud_rate={dataset['is_fraud'].mean():.4f}"
    )

    print("[3/8] Creating engineered features...")
    feature_df = create_features(dataset)
    y = feature_df["is_fraud"].astype(int)
    X = feature_df.drop(columns=[*DROP_COLUMNS_FOR_TRAINING, "is_fraud"])
    X = X[FEATURE_COLUMNS]

    print("[4/8] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[5/8] Applying SMOTE and training XGBoost...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)

    print("[6/8] Evaluating model...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = (y_prob >= DEFAULT_FRAUD_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred_default, zero_division=0)
    recall = recall_score(y_test, y_pred_default, zero_division=0)
    f1 = f1_score(y_test, y_pred_default, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_default).tolist()

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else DEFAULT_FRAUD_THRESHOLD

    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    opt_precision = precision_score(y_test, y_pred_opt, zero_division=0)
    opt_recall = recall_score(y_test, y_pred_opt, zero_division=0)
    opt_f1 = f1_score(y_test, y_pred_opt, zero_division=0)

    print("[7/8] Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURE_COLS_PATH)
    joblib.dump(optimal_threshold, THRESHOLD_PATH)

    metrics = {
        "roc_auc": float(roc_auc),
        "precision_default_threshold": float(precision),
        "recall_default_threshold": float(recall),
        "f1_default_threshold": float(f1),
        "confusion_matrix_default_threshold": cm,
        "optimal_threshold": float(optimal_threshold),
        "precision_optimal_threshold": float(opt_precision),
        "recall_optimal_threshold": float(opt_recall),
        "f1_optimal_threshold": float(opt_f1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[8/8] Training complete. Final summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
