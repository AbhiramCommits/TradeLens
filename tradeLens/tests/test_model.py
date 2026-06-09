import os

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
PIPELINE_PATH = os.path.join(MODEL_DIR, "trade_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

ALGO_CLASSES = {"Aggressive", "IS", "Passive", "TWAP", "VWAP"}


def test_model_file_exists():
    assert os.path.isfile(PIPELINE_PATH), f"Model not found at {PIPELINE_PATH}"


def test_encoder_classes():
    encoder: LabelEncoder = pytest.importorskip("joblib").load(ENCODER_PATH)
    classes = set(encoder.classes_)
    assert len(classes) == 5
    assert classes == ALGO_CLASSES


def test_predict_shape():
    import pandas as pd
    from sklearn.pipeline import Pipeline

    joblib = pytest.importorskip("joblib")
    pipeline: Pipeline = joblib.load(PIPELINE_PATH)

    df = pd.DataFrame(
        {
            "notional_usd": [1_000_000.0, 500_000.0],
            "market_volatility": [0.3, 0.7],
            "spread_bps": [5.0, 15.0],
            "time_of_day": [1.0, 2.0],
            "venue_liquidity_score": [0.8, 0.3],
        }
    )

    preds = pipeline.predict(df)
    assert preds.shape == (2,)
