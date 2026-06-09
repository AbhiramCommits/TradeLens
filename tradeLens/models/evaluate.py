import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(MODEL_DIR), "data")

ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "trade_model.pkl")
IMPORTANCES_PATH = os.path.join(MODEL_DIR, "feature_importances.csv")

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

    le: LabelEncoder = joblib.load(ENCODER_PATH)
    y = le.transform(y_raw)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_
    print("Confusion matrix:\n")
    header = " " * 12 + "".join(f"{lbl:>10}" for lbl in labels)
    print(header)
    for i, lbl in enumerate(labels):
        row = f"{lbl:>10}: " + "".join(f"{cm[i, j]:10d}" for j in range(len(labels)))
        print(row)

    importances = pipeline.named_steps["clf"].feature_importances_
    fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(IMPORTANCES_PATH, index=False)

    print(f"\nFeature importances saved to {IMPORTANCES_PATH}")
    print(fi_df.to_string(index=False))


if __name__ == "__main__":
    main()
