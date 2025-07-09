from src.data_loader import fetch_stock_data
from src.features import add_features
from src.model import train_model

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2020-01-01", "2023-12-31")
    df =  add_features(df)
    model=train_model(df)
    print(df[['Date', 'Close', 'Return', 'SMA_5', 'SMA_10', 'Volatility', 'RSI']].head())