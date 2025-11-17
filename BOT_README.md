# MT5 Trading Bot - User Guide

## Overview
Automated trading bot for MetaTrader 5 that runs every hour with:
- EMA 9/21 crossover strategy
- RSI filter (35-65)
- ADX trend filter (>20)
- Supertrend confirmation
- **1:2 Risk-Reward ratio** (ATR-based)

## Features

### Entry Conditions
**LONG (Buy)**:
- ✅ Fast EMA crosses above Slow EMA
- ✅ Supertrend direction = Bullish (1)
- ✅ RSI < 65 (not overbought)
- ✅ ADX > 20 (trending market)

**SHORT (Sell)**:
- ✅ Fast EMA crosses below Slow EMA
- ✅ Supertrend direction = Bearish (-1)
- ✅ RSI > 35 (not oversold)
- ✅ ADX > 20 (trending market)

### Exit Conditions
- ❌ Supertrend flips opposite
- ❌ ADX drops below threshold
- ❌ Opposite EMA crossover
- ✅ Stop Loss or Take Profit hit

### Risk Management
- **Stop Loss**: 1.5 × ATR from entry
- **Take Profit**: 3.0 × ATR from entry (2× SL distance)
- **Risk-Reward**: 1:2 ratio

## Installation

### 1. Requirements
```bash
pip install MetaTrader5 pandas numpy
```

### 2. Configuration
Edit `mt5_trading_bot.py` - **Config class** (lines 24-41):

```python
class Config:
    # Trading Parameters
    SYMBOL = "XAUUSD"              # Change to your symbol
    TIMEFRAME = mt5.TIMEFRAME_H1   # 1-hour chart
    LOT_SIZE = 0.01                # Position size
    RISK_REWARD_RATIO = 2.0        # 1:2 RR
    ATR_MULTIPLIER = 1.5           # SL distance
    
    # Strategy Parameters
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_THRESHOLD = 20
    RSI_UPPER = 65
    RSI_LOWER = 35
    SUPERTREND_ATR_PERIOD = 14
    SUPERTREND_MULTIPLIER = 3
    
    # Bot Settings
    MAGIC_NUMBER = 123456          # Unique ID for bot trades
```

## Usage

### Start the Bot
```bash
python3 mt5_trading_bot.py
```

### Stop the Bot
Press `Ctrl+C` to stop gracefully

### Logs
- Console output: Real-time
- File: `mt5_trading_bot.log` (saved in same directory)

## How It Works

### Execution Flow
1. **Every Hour** (at minute :00)
   - Fetches last 200 candles from MT5
   - Calculates all indicators
   - Checks exit conditions for open positions
   - Checks entry conditions for new signals
   - Executes trades if conditions met

2. **Position Management**
   - Max 1 position open at a time
   - Auto-closes on exit signals
   - SL/TP set automatically on entry

3. **Logging**
   - All decisions logged with reasons
   - Position entry/exit details
   - Indicator values for debugging

## Example Log Output

```
============================================================
Trading cycle started: 2025-11-16 10:00:00
============================================================
Analysis: Bullish EMA cross, ST=UP, RSI=58.2, ADX=24.5
Opening BUY position
Order opened: BUY at 2635.50000, SL=2620.75000, TP=2665.00000
Risk: 14.75000, Reward: 29.50000, RR=1:2.0
Cycle complete. Next check in 1 hour.
```

## Safety Features

### Built-in Protections
- ✅ Only trades during strong trends (ADX filter)
- ✅ Avoids overbought/oversold (RSI filter)
- ✅ Confirms with Supertrend
- ✅ Automatic stop loss on every trade
- ✅ Magic number prevents interference
- ✅ Max 1 position at a time

### Monitoring
- Check logs regularly
- Monitor MT5 terminal
- Verify positions match expected behavior

## Customization

### Change Symbol
```python
SYMBOL = "EURUSD"  # Forex
SYMBOL = "XAUUSD"  # Gold
SYMBOL = "BTCUSD"  # Crypto (if broker supports)
```

### Change Timeframe
```python
TIMEFRAME = mt5.TIMEFRAME_M15  # 15 minutes
TIMEFRAME = mt5.TIMEFRAME_H1   # 1 hour
TIMEFRAME = mt5.TIMEFRAME_H4   # 4 hours
TIMEFRAME = mt5.TIMEFRAME_D1   # Daily
```

### Adjust Risk-Reward
```python
RISK_REWARD_RATIO = 1.5  # 1:1.5 (more conservative)
RISK_REWARD_RATIO = 3.0  # 1:3 (more aggressive)
```

### Modify Filters
```python
ADX_THRESHOLD = 25     # More selective (stronger trends only)
RSI_UPPER = 70         # Less restrictive
RSI_LOWER = 30         # Less restrictive
ATR_MULTIPLIER = 2.0   # Wider stop loss
```

## Troubleshooting

### Bot won't start
- ✅ Check MT5 is running
- ✅ Verify symbol is available
- ✅ Check account has trading permissions

### No trades executed
- Check ADX is above threshold
- Verify Supertrend aligns with EMA cross
- Check RSI is within bounds
- Review logs for rejection reasons

### Positions close immediately
- Check Supertrend isn't flipping
- Verify ADX stays above threshold
- Review exit conditions in logs

## Important Notes

1. **Backtesting**: Test with small lot sizes first
2. **Broker**: Ensure your broker allows automated trading
3. **Spread**: High spread can affect profitability
4. **Slippage**: Set appropriate MAX_SLIPPAGE value
5. **VPS**: Consider running on VPS for 24/7 operation

## Risk Warning

⚠️ **Trading involves substantial risk of loss**
- Start with demo account
- Use small lot sizes initially
- Never risk more than you can afford to lose
- Monitor the bot regularly
- Understand the strategy before using real money

## Support

Check logs for detailed information:
```bash
tail -f mt5_trading_bot.log
```

## License
For personal use only. No warranty provided.

