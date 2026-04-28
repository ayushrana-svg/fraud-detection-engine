from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
PREDICTION_AUDIT_LOG_PATH = PROCESSED_DATA_DIR / "prediction_audit.jsonl"

DEFAULT_FRAUD_THRESHOLD = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": 50,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RAW_COLUMNS = [
    "transaction_id",
    "user_id",
    "merchant_id",
    "amount",
    "timestamp",
    "is_fraud",
]

DROP_COLUMNS_FOR_TRAINING = ["transaction_id", "user_id", "merchant_id", "timestamp"]

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "is_night",
    "user_tx_count",
    "user_avg_amount",
    "amount_deviation",
    "merchant_fraud_rate",
    "amount_to_avg_ratio",
]
