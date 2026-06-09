from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TradeFeatures(BaseModel):
    notional_usd: float = Field(..., gt=0)
    market_volatility: float = Field(..., ge=0, le=1)
    spread_bps: float = Field(..., ge=0)
    time_of_day: float = Field(..., ge=0, le=2)
    venue_liquidity_score: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_algo: str
    confidence_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model: str
