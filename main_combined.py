import pandas as pd
from datetime import datetime
from src.data_loader import fetch_stock_data
from src.features import add_features
from src.model_stock import load_model as load_stock_model
from src.model_etf import load_etf_model
from src.model_stock import train_model

# Check if model exists
import os
if not os.path.exists("models/random_forest.pkl"):
    df = fetch_stock_data("AAPL", "2015-01-01", "2023-12-31")
    df = add_features(df)
    train_model(df)


def combine_signals(
    ticker="AAPL", 
    start="2020-01-01", 
    end=datetime.today().strftime("%Y-%m-%d"), 
    stock_thresh=0.65, 
    etf_thresh=0.55
):
    df = fetch_stock_data(ticker, start, end)
    df = add_features(df)

    print(f"\n📅 Latest data used: {df['Date'].iloc[-1]}")

    # Load models
    stock_model = load_stock_model()
    etf_model = load_etf_model()

    # Define feature sets
    features_stock = ['Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']
    features_etf   = ['Return', 'SMA_10', 'SMA_20', 'Volatility', 'RSI']

    # Drop rows with missing values across all features used
    df = df.dropna(subset=list(set(features_stock + features_etf)))

    # Extract latest data for each model
    latest_stock = df[features_stock].iloc[[-1]]
    latest_etf   = df[features_etf].iloc[[-1]]

    # Predict probabilities
    stock_prob = stock_model.predict_proba(latest_stock)[0][1]  # Prob of 'Up'
    etf_prob   = etf_model.predict_proba(latest_etf)[0][1]

    print(f"\n🔍 Stock model confidence: {round(stock_prob, 3)}")
    print(f"📈 ETF model confidence:   {round(etf_prob, 3)}")

    # Combined signal logic
    if stock_prob > stock_thresh:
        if etf_prob > etf_thresh:
            action = "Strong Buy (both models agree)"
        else:
            action = "Short-Term Buy (stock high confidence)"
    elif stock_prob < (1 - stock_thresh):
        if etf_prob < (1 - etf_thresh):
            action = "Strong Sell (both models agree)"
        else:
            action = "Short-Term Sell (stock high confidence)"
    else:
        action = "Hold or Wait (no strong signal)"

    print(f"\n📊 Combined Signal for {ticker}: {action}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=False)

