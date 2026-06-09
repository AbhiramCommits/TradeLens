import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(MODEL_DIR), "data")

ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "trade_model.pkl")

TIME_OF_DAY_MAP = {"Open": 0, "Mid": 1, "Close": 2}

FEATURE_COLS = [
    "notional_usd",
    "market_volatility",
    "spread_bps",
    "time_of_day",
    "venue_liquidity_score",
]


def main():
    csv_path = os.path.join(DATA_DIR, "trades.csv")
    df = pd.read_csv(csv_path)

    df["time_of_day"] = df["time_of_day"].map(TIME_OF_DAY_MAP)

    X = df[FEATURE_COLS].values
    y_raw = df["algo_label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    joblib.dump(le, ENCODER_PATH)
    print(f"Label encoder saved to {ENCODER_PATH}")
    print(f"Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    print("Model trained.")

    y_pred = pipeline.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(
        "\nClassification report:\n",
        classification_report(y_test, y_pred, target_names=le.classes_),
    )

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
