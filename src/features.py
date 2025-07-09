import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock data for ML modeling.
    
    Features:
    - Daily returns
    - SMA (5-day and 10-day)
    - Volatility (10-day rolling std)
    - RSI (14-day)
    
    Returns:
        pd.DataFrame with added features
    """

    df = df.copy()

    # Daily return
    df['Return'] = df['Close'].pct_change()

    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()



    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero since undefined
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # Drop initial rows due to rolling calculations
    df.dropna(inplace=True)

    return df
