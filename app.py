import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from coingecko_client import CoinGeckoClient
from features import build_daily_frame, add_features
from forecast import run_forecast, ForecastResult

app = FastAPI(title="BTC Price Prediction API", version="1.0.0")

COIN_ID = "bitcoin"
DEFAULT_VS = os.getenv("VS_CURRENCY", "usd")

# Compute knobs (accuracy vs latency)
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "60"))  # increase for more accuracy (slower)
HISTORY_DAYS = os.getenv("HISTORY_DAYS", "max")        # "max" or number

cg = CoinGeckoClient()


class PredictResponse(BaseModel):
    coin_id: str
    vs_currency: str
    horizon_days: int
    last_date: str
    last_price: float
    predictions: list[dict]
    diagnostics: Dict[str, float]
    disclaimer: str


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/price/live")
async def live_price(vs_currency: str = Query(DEFAULT_VS)):
    try:
        data = await cg.simple_price(ids=COIN_ID, vs_currencies=vs_currency)
        return {
            "coin_id": COIN_ID,
            "vs_currency": vs_currency,
            "price": data.get(COIN_ID, {}).get(vs_currency),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/predict", response_model=PredictResponse)
async def predict(
    days: int = Query(7, ge=1, le=365, description="How many days forward to predict"),
    vs_currency: str = Query(DEFAULT_VS),
    optuna_trials: Optional[int] = Query(
        None,
        ge=10,
        le=500,
        description="Override Optuna trials (more = slower, potentially better)",
    ),
):
    """
    Predict BTC price path days forward.
    Very compute-heavy by design (Optuna tuning + CV).
    """
    trials = optuna_trials or OPTUNA_TRIALS

    try:
        btc_json = await cg.market_chart(
            COIN_ID,
            vs_currency=vs_currency,
            days=HISTORY_DAYS,
            interval="daily",
            precision="full",
        )
        btc_df = build_daily_frame(btc_json)

        # BTC-only features (no exogenous BTC/ETH joins)
        df_feat = add_features(btc_df)

        # Safety: need enough history after rolling windows + target shift
        if len(df_feat) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough usable history after feature engineering: {len(df_feat)} rows",
            )

        result: ForecastResult = run_forecast(
            df_feat,
            coin_id=COIN_ID,
            vs_currency=vs_currency,
            horizon_days=days,
            optuna_trials=trials,
        )

        return PredictResponse(
            **result.__dict__,
            disclaimer=(
                "Forecasts are statistical estimates from historical data, not financial advice, "
                "and can be very wrong—especially during regime changes/news-driven moves."
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
