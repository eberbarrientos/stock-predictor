import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save: bool = True) -> pd.DataFrame:
    """
    Fetch and format historical stock data using yfinance.
    Cleans MultiIndex and flattens DataFrame.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
        save (bool): Save to /data folder

    Returns:
        pd.DataFrame: Cleaned historical data
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # 🧹 CLEANUP MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = 'Date'
    data.reset_index(inplace=True)

    if save:
        os.makedirs("data", exist_ok=True)
        data.to_csv(f"data/{ticker}.csv", index=False)

    return data
