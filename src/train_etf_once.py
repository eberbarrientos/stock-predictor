from src.data_loader import fetch_stock_data
from src.features import add_features
from src.model_etf import train_etf_model

for horizon in [30, 60, 90, 120, 180]:
    df = fetch_stock_data("VOO", "2010-01-01", "2023-12-31")
    df = add_features(df)
    train_etf_model(df, model_path=f"models/etf_model_{horizon}d.pkl", horizon_days=horizon)
