import yfinance as yf
import pandas as pd

def fetch_data_yf(symbol: str, period: str, interval: str = "1d"):
    df = yf.download(
        symbol, period=period, 
        interval=interval, 
        progress=False, 
        auto_adjust=True)
    df = pd.DataFrame(df)
    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

