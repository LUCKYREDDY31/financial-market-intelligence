from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, MODELS_DIR, SEQUENCE_LENGTH, STOCK_SYMBOLS
from models.lstm_forecaster import StockForecaster
from models.sentiment_analyzer import FinBERTSentimentAnalyzer
from data_processing.feature_engineering import StockFeatureEngineer

app = FastAPI(title="Stock Market API", version="1.0")

sentiment_analyzer = None


@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer
    print("Loading sentiment model...")
    sentiment_analyzer = FinBERTSentimentAnalyzer()
    print("Ready!")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    positive: float
    negative: float
    neutral: float
    confidence: float


@app.get("/")
async def root():
    return {
        "message": "Stock Market API",
        "version": "1.0",
        "endpoints": ["/stocks", "/predict", "/sentiment", "/stock/{symbol}"]
    }


@app.get("/stocks")
async def get_stocks():
    return {"symbols": STOCK_SYMBOLS}


@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    if symbol not in STOCK_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")
    
    filepath = RAW_DATA_DIR / f"{symbol}_historical.csv"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    
    df = pd.read_csv(filepath)
    latest = df.iloc[-1]
    
    return {
        "symbol": symbol,
        "date": str(latest['Date']),
        "close": float(latest['Close']),
        "volume": int(latest['Volume']),
        "high": float(latest['High']),
        "low": float(latest['Low']),
        "open": float(latest['Open'])
    }


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    result = sentiment_analyzer.analyze_sentiment(request.text)
    return SentimentResponse(**result)


@app.post("/predict")
async def predict_stock(symbol: str, days: int = 5):
    symbol = symbol.upper()
    
    if symbol not in STOCK_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")
    
    model_path = MODELS_DIR / f"{symbol}_lstm.pth"
    scaler_path = MODELS_DIR / f"{symbol}_scalers.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not trained")
    
    # load scalers
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    feature_cols = scalers['feature_cols']
    
    # load data
    filepath = RAW_DATA_DIR / f"{symbol}_historical.csv"
    df = pd.read_csv(filepath)
    
    # prep features
    engineer = StockFeatureEngineer()
    df = engineer.add_technical_indicators(df)
    features, _ = engineer.prepare_for_lstm(df, feature_cols=feature_cols)
    features_normalized = feature_scaler.transform(features.values)
    
    # predict
    forecaster = StockForecaster(input_size=len(feature_cols), sequence_length=SEQUENCE_LENGTH)
    forecaster.load(model_path)
    
    last_sequence = features_normalized[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
    prediction = forecaster.predict(last_sequence)
    predicted_price = target_scaler.inverse_transform(prediction)[0][0]
    
    current_price = float(df.iloc[-1]['Close'])
    
    # generate future dates
    import datetime
    last_date = pd.to_datetime(df.iloc[-1]['Date'])
    future_dates = [(last_date + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d') 
                    for i in range(days)]
    
    # TODO: implement proper multi-step forecasting
    predicted_prices = [float(predicted_price)] * days
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_prices": predicted_prices,
        "dates": future_dates
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)