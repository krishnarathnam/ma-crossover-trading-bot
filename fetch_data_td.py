from twelvedata import TDClient
import os
import pandas as pd
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "secrets.env"))
api_key = os.getenv("TD_API_KEY")

def fetch_data_td(symbol: str, interval: str, outputsize: int):
    td = TDClient(apikey=api_key)
    ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
    df = ts.as_pandas()
    df = df.reset_index()
    df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.dropna()
    
    # Remove flat candles (market closed - when OHLC are all the same)
    df = df[~((df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close']))]
    
    # Sort by date in ascending order (oldest to newest)
    df = df.sort_index()
    
    # Detect and remove large time gaps, then join the data blocks
    df['time_diff'] = df.index.to_series().diff()
    
    # Define what constitutes a "large gap" based on interval
    if interval == "5min":
        max_gap = pd.Timedelta(hours=1)
    elif interval in ["1min", "15min", "30min"]:
        max_gap = pd.Timedelta(hours=1)
    elif interval in ["1h", "1hour"]:
        max_gap = pd.Timedelta(hours=6)
    elif interval in ["1day", "1d"]:
        max_gap = pd.Timedelta(days=7)
    else:
        max_gap = pd.Timedelta(hours=2)
    
    # Find rows with large gaps
    large_gaps = df[df['time_diff'] > max_gap]
    
    if len(large_gaps) > 0:
        # Minimum block size needed for meaningful calculations (50 for RSI + 20 for BB)
        min_block_size = 70
        
        # Split data into contiguous blocks
        gap_indices = list(large_gaps.index)
        blocks = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            # Get data up to (but not including) the gap
            gap_loc = df.index.get_loc(gap_idx)
            block = df.iloc[start_idx:gap_loc]
            if len(block) >= min_block_size:
                blocks.append(block)
            start_idx = gap_loc + 1  # Skip the gap row itself and start from next row
        
        # Add the final block after the last gap
        if start_idx < len(df):
            final_block = df.iloc[start_idx:]
            if len(final_block) >= min_block_size:
                blocks.append(final_block)
        
        if len(blocks) > 0:
            # Concatenate all blocks together (this removes the gap rows and small blocks)
            df = pd.concat(blocks, axis=0)
    
    # Remove the temporary time_diff column
    df = df.drop(columns=['time_diff'])
    
    return df


