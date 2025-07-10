import os
import pandas as pd
from datetime import datetime
from src.data_loader import fetch_stock_data
from src.features import add_features
from src.model_stock import load_model as load_stock_model, train_model
from src.model_etf import load_etf_model, train_etf_model
import uvicorn

# === Train stock model if not already saved ===
if not os.path.exists("models/random_forest.pkl"):
    df = fetch_stock_data("AAPL", "2015-01-01", "2023-12-31")
    df = add_features(df)
    train_model(df)

# === Train and save all ETF models ===
df_etf = fetch_stock_data("VOO", "2010-01-01", "2023-12-31")
df_etf = add_features(df_etf)
for horizon in [30, 60, 90, 120, 180]:
    model_path = f"models/etf_model_{horizon}d.pkl"
    if not os.path.exists(model_path):
        train_etf_model(df_etf, model_path=model_path, horizon_days=horizon, verbose=False)

# === Combined signal logic ===
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
    etf_model = load_etf_model()  # Default is 30-day

    # Define feature sets
    features_stock = ['Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']
    features_etf = ['Return', 'SMA_10', 'SMA_20', 'Volatility', 'RSI']

    df = df.dropna(subset=list(set(features_stock + features_etf)))
    latest_stock = df[features_stock].iloc[[-1]]
    latest_etf = df[features_etf].iloc[[-1]]

    stock_prob = stock_model.predict_proba(latest_stock)[0][1]
    etf_prob = etf_model.predict_proba(latest_etf)[0][1]

    print(f"\n🔍 Stock model confidence: {round(stock_prob, 3)}")
    print(f"📈 ETF model confidence:   {round(etf_prob, 3)}")

    # Combined decision logic
    if stock_prob > stock_thresh:
        action = "Strong Buy" if etf_prob > etf_thresh else "Short-Term Buy"
    elif stock_prob < (1 - stock_thresh):
        action = "Strong Sell" if etf_prob < (1 - etf_thresh) else "Short-Term Sell"
    else:
        action = "Hold or Wait"

    print(f"\n📊 Combined Signal for {ticker}: {action}")


if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=False)



