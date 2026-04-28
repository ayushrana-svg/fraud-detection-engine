import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"amount", "user_id", "merchant_id", "timestamp"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns for feature creation: {missing}")

    features_df = df.copy()
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], errors="coerce")
    features_df["amount"] = pd.to_numeric(features_df["amount"], errors="coerce").fillna(0.0)
    features_df["log_amount"] = np.log1p(features_df["amount"])

    features_df["hour"] = features_df["timestamp"].dt.hour.fillna(0).astype(int)
    features_df["day_of_week"] = features_df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    features_df["is_night"] = features_df["hour"].between(0, 6, inclusive="both").astype(int)

    features_df = features_df.sort_values("timestamp", kind="stable").copy()

    features_df["user_tx_count"] = (
        features_df.groupby("user_id", dropna=False).cumcount() + 1
    ).clip(upper=10)

    features_df["user_avg_amount"] = (
        features_df.groupby("user_id", dropna=False)["amount"]
        .transform(lambda s: s.rolling(window=10, min_periods=1).mean())
        .fillna(0.0)
    )

    features_df["amount_deviation"] = (
        (features_df["amount"] - features_df["user_avg_amount"])
        / (features_df["user_avg_amount"] + 1.0)
    )

    if "is_fraud" in features_df.columns:
        global_mean = float(features_df["is_fraud"].mean())
        merchant_fraud_rate = (
            features_df.groupby("merchant_id", dropna=False)["is_fraud"]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        features_df["merchant_fraud_rate"] = merchant_fraud_rate.fillna(global_mean)
    else:
        features_df["merchant_fraud_rate"] = 0.0

    features_df["amount_to_avg_ratio"] = features_df["amount"] / (
        features_df["user_avg_amount"] + 1.0
    )

    return features_df.fillna(0)
