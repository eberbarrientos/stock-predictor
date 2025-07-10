from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging
import os

from src.data_loader import fetch_stock_data
from src.features import add_features
from src.model_stock import load_model as load_stock_model
from src.model_etf import load_etf_model

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictionResponse(BaseModel):
    stock_confidence: Optional[float] = None
    etf_confidence: Optional[float] = None
    signal: str

def determine_signal(stock_prob, etf_prob, stock_thresh=0.65, etf_thresh=0.55):
    if stock_prob > stock_thresh and etf_prob > etf_thresh:
        return "Strong Buy"
    elif stock_prob > stock_thresh:
        return "Short-Term Buy"
    elif etf_prob > etf_thresh:
        return "Long-Term Buy"
    elif stock_prob < (1 - stock_thresh) and etf_prob < (1 - etf_thresh):
        return "Strong Sell"
    elif stock_prob < (1 - stock_thresh):
        return "Short-Term Sell"
    elif etf_prob < (1 - etf_thresh):
        return "Long-Term Sell"
    else:
        return "Hold"

@app.get("/predict_stock", response_model=PredictionResponse)
def predict_stock(
    ticker: str = Query(...),
    train_start: Optional[str] = Query(None),
    train_end: Optional[str] = Query(None)
):
    try:
        start = train_start or "2020-01-01"
        end = train_end or datetime.today().strftime("%Y-%m-%d")

        df = fetch_stock_data(ticker, start, end)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this ticker.")

        df = add_features(df)
        features = ['Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']
        df = df.dropna(subset=features)
        latest = df.iloc[[-1]]

        model = load_stock_model()
        prob = model.predict_proba(latest[features])[0][1]
        signal = "Buy" if prob > 0.65 else "Sell" if prob < 0.35 else "Hold"

        logger.info(f"/predict_stock | {ticker} | confidence: {prob:.3f} | signal: {signal}")
        return PredictionResponse(stock_confidence=round(prob, 3), signal=signal)

    except Exception as e:
        logger.exception(f"Error in /predict_stock: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/predict_etf", response_model=PredictionResponse)
def predict_etf(
    ticker: str = Query(...),
    horizon: int = Query(30, description="Prediction horizon (30, 60, 90, 120, 180)"),
    train_start: Optional[str] = Query(None),
    train_end: Optional[str] = Query(None)
):
    try:
        if horizon not in [30, 60, 90, 120, 180]:
            raise HTTPException(status_code=400, detail="Unsupported horizon. Choose from 30, 60, 90, 120, 180.")

        start = train_start or "2010-01-01"
        end = train_end or datetime.today().strftime("%Y-%m-%d")

        df = fetch_stock_data(ticker, start, end)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this ticker and date range.")

        df = add_features(df)
        features = ['Return', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'Volatility', 'RSI']
        df = df.dropna(subset=features)
        latest = df.iloc[[-1]]

        model_path = f"models/etf_model_{horizon}d.pkl"
        model = load_etf_model(model_path=model_path)

        prob = model.predict_proba(latest[features])[0][1]
        signal = "Buy" if prob > 0.55 else "Sell" if prob < 0.45 else "Hold"

        logger.info(f"/predict_etf | {ticker} | horizon: {horizon}d | confidence: {prob:.3f} | signal: {signal}")
        return PredictionResponse(etf_confidence=round(prob, 3), signal=signal)

    except FileNotFoundError:
        logger.error(f"Model file not found for horizon {horizon}d")
        raise HTTPException(status_code=500, detail="ETF model not trained for this horizon.")

    except Exception as e:
        logger.exception(f"Unhandled error in /predict_etf: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
