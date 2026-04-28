from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.features import create_features


def test_create_features_single_row_no_nan():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "txn_test_1",
                "user_id": "user_1",
                "merchant_id": "merchant_1",
                "amount": 199.99,
                "timestamp": "2026-01-01T02:12:00",
            }
        ]
    )

    result = create_features(df)

    expected_columns = {
        "log_amount",
        "hour",
        "day_of_week",
        "is_night",
        "user_tx_count",
        "user_avg_amount",
        "amount_deviation",
        "merchant_fraud_rate",
        "amount_to_avg_ratio",
    }
    assert expected_columns.issubset(set(result.columns))
    assert result.isna().sum().sum() == 0


def test_create_features_missing_columns_raises():
    df = pd.DataFrame([{"amount": 10.0, "timestamp": "2026-01-01T10:00:00"}])
    try:
        create_features(df)
        assert False, "Expected ValueError for missing columns"
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
