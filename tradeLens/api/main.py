import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
)
from api.schemas import (
    HealthResponse,
    LoginRequest,
    PredictResponse,
    TokenResponse,
    TradeFeatures,
)

API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(API_DIR)
MODEL_PATH = os.path.join(PROJ_DIR, "models", "trade_model.pkl")
ENCODER_PATH = os.path.join(PROJ_DIR, "models", "label_encoder.pkl")

pipeline: Pipeline | None = None
encoder: LabelEncoder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global pipeline, encoder
    pipeline = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("Model and encoder loaded.")
    yield


app = FastAPI(title="TradeLens API", version="0.1.0", lifespan=lifespan)


@app.post("/auth/token", response_model=TokenResponse)
def login(body: LoginRequest):
    if not authenticate_user(body.username, body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return create_access_token(body.username)


@app.post("/predict", response_model=PredictResponse)
def predict(
    features: TradeFeatures, username: str = Depends(get_current_user)
):
    if pipeline is None or encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    X = np.array(
        [
            [
                features.notional_usd,
                features.market_volatility,
                features.spread_bps,
                features.time_of_day,
                features.venue_liquidity_score,
            ]
        ]
    )

    probs = pipeline.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    predicted_algo = str(encoder.inverse_transform([pred_idx])[0])

    confidence_scores = {
        str(encoder.inverse_transform([i])[0]): float(probs[i])
        for i in range(len(probs))
    }

    return PredictResponse(
        predicted_algo=predicted_algo, confidence_scores=confidence_scores
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="loaded" if pipeline is not None else "not loaded",
    )
