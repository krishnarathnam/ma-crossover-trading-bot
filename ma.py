import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fetch_data_td import fetch_data_td


# --------------------------------------------
# STEP 1 — Calculate Indicators
# --------------------------------------------
def calculate_ma_signals(df, ma_fast=9, ma_slow=21):
    """
    Calculate moving average crossover signals.
    
    Args:
        df: DataFrame with OHLC data
        ma_fast: Fast MA period (default 9)
        ma_slow: Slow MA period (default 21)
    
    Returns:
        DataFrame with signals
    """
    df = df.copy()
    
    # Calculate MAs
    df["MA_fast"] = df["close"].rolling(ma_fast).mean()
    df["MA_slow"] = df["close"].rolling(ma_slow).mean()
    
    # Generate signals (1 = long, -1 = short, 0 = no position)
    df["signal"] = 0
    df.loc[df["MA_fast"] > df["MA_slow"], "signal"] = 1   # Long when fast > slow
    df.loc[df["MA_fast"] < df["MA_slow"], "signal"] = -1  # Short when fast < slow
    
    # Detect crossovers (position changes)
    df["position_change"] = df["signal"].diff()
    
    # Drop NaN rows from rolling calculations
    df = df.dropna()
    
    return df


# --------------------------------------------
# STEP 2 — Backtest MA Strategy
# --------------------------------------------
def backtest_ma(df, capital=10000, leverage=20, commission=0.0002):
    """
    Backtest the MA crossover strategy.
    
    Args:
        df: DataFrame with signals
        capital: Starting capital
        leverage: Position leverage
        commission: Commission per trade (0.0002 = 0.02%)
    
    Returns:
        DataFrame with PnL and equity
    """
    df = df.copy()
    
    df["position"] = 0.0
    df["pnl"] = 0.0
    df["commission"] = 0.0
    
    for i in range(1, len(df)):
        # Get current signal
        current_signal = df["signal"].iloc[i]
        prev_signal = df["signal"].iloc[i-1]
        
        # Update position
        df.loc[df.index[i], "position"] = current_signal
        
        # Calculate price change
        price_change = df["close"].iloc[i] - df["close"].iloc[i-1]
        
        # PnL from previous position
        pnl = prev_signal * price_change * leverage
        df.loc[df.index[i], "pnl"] = pnl
        
        # Commission on position changes
        if current_signal != prev_signal and current_signal != 0:
            trade_commission = abs(df["close"].iloc[i]) * commission * leverage
            df.loc[df.index[i], "commission"] = trade_commission
    
    # Calculate cumulative metrics
    df["net_pnl"] = df["pnl"] - df["commission"]
    df["cum_pnl"] = df["net_pnl"].cumsum()
    df["equity"] = capital + df["cum_pnl"]
    
    return df


# --------------------------------------------
# STEP 3 — Print Statistics
# --------------------------------------------
def print_statistics(df, capital):
    """Print backtest performance statistics."""
    total_return = (df["equity"].iloc[-1] - capital) / capital * 100
    max_equity = df["equity"].max()
    drawdown = (df["equity"] - df["equity"].cummax()).min()
    drawdown_pct = (drawdown / max_equity) * 100
    
    # Count trades
    trades = (df["position_change"] != 0).sum()
    winning_trades = (df["net_pnl"] > 0).sum()
    losing_trades = (df["net_pnl"] < 0).sum()
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0
    
    # Total commissions
    total_commission = df["commission"].sum()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Starting Capital:     ${capital:,.2f}")
    print(f"Final Equity:         ${df['equity'].iloc[-1]:,.2f}")
    print(f"Total Return:         {total_return:.2f}%")
    print(f"Max Drawdown:         {drawdown_pct:.2f}%")
    print(f"Total Trades:         {trades}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print(f"Total Commission:     ${total_commission:.2f}")
    print(f"Net PnL:              ${df['cum_pnl'].iloc[-1]:.2f}")
    print("="*50)


# --------------------------------------------
# STEP 4 — Plot Results
# --------------------------------------------
def plot_results(df, symbol, interval, ma_fast, ma_slow):
    """Plot price, MAs, signals, and equity curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Price and MAs with signals
    ax1.plot(df.index, df["close"], label=f"{symbol} Close", linewidth=1, color='black')
    ax1.plot(df.index, df["MA_fast"], label=f"MA {ma_fast}", linewidth=1, color='blue')
    ax1.plot(df.index, df["MA_slow"], label=f"MA {ma_slow}", linewidth=1, color='red')
    
    # Buy signals (crossover up)
    buy_signals = df[df["position_change"] == 2]
    ax1.scatter(buy_signals.index, buy_signals["close"], 
                marker="^", color="green", s=100, label="Buy", zorder=5)
    
    # Sell signals (crossover down)
    sell_signals = df[df["position_change"] == -2]
    ax1.scatter(sell_signals.index, sell_signals["close"], 
                marker="v", color="red", s=100, label="Sell", zorder=5)
    
    ax1.legend(loc='best')
    ax1.set_title(f"{symbol} - MA {ma_fast}/{ma_slow} Crossover ({interval})")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity curve
    ax2.plot(df.index, df["equity"], label="Equity", linewidth=2, color='green')
    ax2.axhline(y=df["equity"].iloc[0], color='gray', linestyle='--', label='Starting Capital')
    ax2.fill_between(df.index, df["equity"].iloc[0], df["equity"], 
                      where=(df["equity"] >= df["equity"].iloc[0]), 
                      alpha=0.3, color='green')
    ax2.fill_between(df.index, df["equity"].iloc[0], df["equity"], 
                      where=(df["equity"] < df["equity"].iloc[0]), 
                      alpha=0.3, color='red')
    ax2.legend(loc='best')
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Equity ($)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# RUN EVERYTHING
# --------------------------------------------
def run_ma_backtest(symbol="XAU/USD", interval="5min", bars=5000, 
                     ma_fast=9, ma_slow=21, capital=10000, 
                     leverage=20, plot=True):
    """
    Run complete MA crossover backtest.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        bars: Number of bars to fetch
        ma_fast: Fast MA period
        ma_slow: Slow MA period
        capital: Starting capital
        leverage: Position leverage
        plot: Whether to show plots
    
    Returns:
        DataFrame with backtest results
    """
    print(f"\nFetching {symbol} data ({interval})...")
    df = fetch_data_td(symbol, interval, bars)
    
    print(f"Calculating MA {ma_fast}/{ma_slow} signals...")
    df = calculate_ma_signals(df, ma_fast, ma_slow)
    
    print("Running backtest...")
    df = backtest_ma(df, capital, leverage)
    
    print_statistics(df, capital)
    
    if plot:
        plot_results(df, symbol, interval, ma_fast, ma_slow)
    
    return df


# Run it
if __name__ == "__main__":
    df = run_ma_backtest(
        symbol="XAU/USD",
        interval="1h",
        bars=5000,
        ma_fast=9,
        ma_slow=21,
        capital=1000,
        leverage=20,
        plot=True
    )