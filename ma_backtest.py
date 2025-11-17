from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from fetch_data_td import fetch_data_td


# --------------------------------------------
# Indicator Calculations
# --------------------------------------------
def calculate_ema(data, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(data).ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """Calculate RSI (Relative Strength Index)."""
    data_series = pd.Series(data)
    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, period=14):
    """Calculate ATR (Average True Range)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) - measures trend strength."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_series - high_series.shift()
    down_move = low_series.shift() - low_series
    
    plus_dm = pd.Series(0.0, index=high_series.index)
    minus_dm = pd.Series(0.0, index=high_series.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Calculate ADX
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    return dx.rolling(period).mean()


def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
    """Calculate Supertrend indicator."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    atr = calculate_atr(high, low, close, atr_period)
    hl_avg = (high_series + low_series) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=close_series.index, dtype=float)
    direction = pd.Series(index=close_series.index, dtype=float)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(close_series)):
        if close_series.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close_series.iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            
        if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = supertrend.iloc[i-1]
        if direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = supertrend.iloc[i-1]
    
    return direction


# --------------------------------------------
# EMA Crossover Strategy
# --------------------------------------------
class EMACrossoverStrategy(Strategy):
    """
    EMA Crossover with ADX, RSI, Supertrend filters, and Pullback Entry.
    Entries: Crossover + ADX > threshold + RSI filter + Supertrend direction + Pullback
    Exits: Opposite crossover, ADX drops, or Supertrend reversal
    """
    
    # Strategy Parameters
    ema_fast = 9
    ema_slow = 21
    adx_period = 14
    adx_threshold = 20
    use_rsi_filter = True
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    use_pullback = True
    pullback_to_fast = True
    supertrend_atr_period = 10
    supertrend_multiplier = 3
    
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        self.ema_fast_line = self.I(calculate_ema, close, self.ema_fast)
        self.ema_slow_line = self.I(calculate_ema, close, self.ema_slow)
        self.adx = self.I(calculate_adx, high, low, close, self.adx_period)
        self.rsi = self.I(calculate_rsi, close, self.rsi_period)
        self.supertrend_dir = self.I(calculate_supertrend, high, low, close, 
                                     self.supertrend_atr_period, self.supertrend_multiplier)
        
        self.waiting_long = False
        self.waiting_short = False
    
    def next(self):
        # Exit if ADX below threshold
        if self.adx[-1] < self.adx_threshold:
            if self.position:
                self.position.close()
            self.waiting_long = False
            self.waiting_short = False
            return
        
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        supertrend_direction = self.supertrend_dir[-1]
        pullback_target = self.ema_fast_line[-1] if self.pullback_to_fast else self.ema_slow_line[-1]
        
        # Exit Logic - Supertrend reversal
        if self.position:
            if self.position.is_long and supertrend_direction == -1:
                self.position.close()
                self.waiting_long = False
                return
            elif self.position.is_short and supertrend_direction == 1:
                self.position.close()
                self.waiting_short = False
                return
        
        # Entry Logic
        if not self.position:
            # Bullish crossover
            if crossover(self.ema_fast_line, self.ema_slow_line):
                if supertrend_direction == 1:
                    if not self.use_rsi_filter or current_rsi < self.rsi_upper:
                        if self.use_pullback:
                            self.waiting_long = True
                            self.waiting_short = False
                        else:
                            self.buy()
            
            # Bearish crossover
            elif crossover(self.ema_slow_line, self.ema_fast_line):
                if supertrend_direction == -1:
                    if not self.use_rsi_filter or current_rsi > self.rsi_lower:
                        if self.use_pullback:
                            self.waiting_short = True
                            self.waiting_long = False
                        else:
                            self.sell()
            
            # Check pullback conditions
            if self.waiting_long:
                if self.ema_fast_line[-1] < self.ema_slow_line[-1] or supertrend_direction == -1:
                    self.waiting_long = False
                elif current_price <= pullback_target:
                    self.buy()
                    self.waiting_long = False
            
            if self.waiting_short:
                if self.ema_fast_line[-1] > self.ema_slow_line[-1] or supertrend_direction == 1:
                    self.waiting_short = False
                elif current_price >= pullback_target:
                    self.sell()
                    self.waiting_short = False
        
        # Exit Logic - Opposite crossover
        else:
            if self.position.is_long and crossover(self.ema_slow_line, self.ema_fast_line):
                self.position.close()
                self.sell()
            elif self.position.is_short and crossover(self.ema_fast_line, self.ema_slow_line):
                self.position.close()
                self.buy()


# --------------------------------------------
# Run Backtest
# --------------------------------------------
def run_ema_backtest(symbol="XAU/USD", interval="1h", bars=5000,
                     ema_fast=9, ema_slow=21, adx_threshold=20,
                     use_rsi_filter=True, rsi_upper=70, rsi_lower=30,
                     use_pullback=True, pullback_to_fast=True,
                     supertrend_atr_period=10, supertrend_multiplier=3,
                     capital=10000, commission=0.0002, plot=True):
    """Run EMA crossover backtest with Supertrend and other filters."""
    
    # Fetch and prepare data
    df = fetch_data_td(symbol, interval, bars)
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                            'close': 'Close', 'volume': 'Volume'})
    df = df.dropna()
    
    # Run backtest
    bt = Backtest(df, EMACrossoverStrategy, cash=capital, commission=commission,
                  exclusive_orders=True, trade_on_close=False)
    
    stats = bt.run(ema_fast=ema_fast, ema_slow=ema_slow, adx_threshold=adx_threshold,
                   use_rsi_filter=use_rsi_filter, rsi_upper=rsi_upper, rsi_lower=rsi_lower,
                   use_pullback=use_pullback, pullback_to_fast=pullback_to_fast,
                   supertrend_atr_period=supertrend_atr_period, 
                   supertrend_multiplier=supertrend_multiplier)
    
    print(f"\n{'='*60}")
    print(f"BACKTEST: {symbol} | {interval} | EMA {ema_fast}/{ema_slow}")
    print(f"{'='*60}")
    print(stats)
    
    if plot:
        bt.plot()
    
    return stats


# --------------------------------------------
# Grid Search
# --------------------------------------------
def simple_grid_search(symbol="XAU/USD", interval="1h", bars=5000, 
                       capital=10000, commission=0.0002):
    """Simple grid search for parameter optimization."""
    
    df = fetch_data_td(symbol, interval, bars)
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume'})
    df = df.dropna()
    
    results = []
    fast_range = range(5, 16, 2)
    slow_range = range(15, 45, 5)
    adx_thresholds = [15, 20, 25, 30]
    
    total = sum(1 for f in fast_range for s in slow_range for a in adx_thresholds if f < s)
    count = 0
    
    for ema_fast in fast_range:
        for ema_slow in slow_range:
            if ema_fast >= ema_slow:
                continue
            
            for adx_threshold in adx_thresholds:
                count += 1
                print(f"Testing {count}/{total}: EMA {ema_fast}/{ema_slow}, ADX>{adx_threshold}", end='\r')
                
                bt = Backtest(df, EMACrossoverStrategy, cash=capital, commission=commission,
                              exclusive_orders=True, trade_on_close=False)
                
                stats = bt.run(ema_fast=ema_fast, ema_slow=ema_slow, adx_threshold=adx_threshold)
                
                results.append({
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'adx_threshold': adx_threshold,
                    'return_pct': stats['Return [%]'],
                    'sharpe': stats['Sharpe Ratio'],
                    'max_dd': stats['Max. Drawdown [%]'],
                    'num_trades': stats['# Trades'],
                    'win_rate': stats['Win Rate [%]']
                })
    
    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    
    print(f"\n\n{'='*80}")
    print("TOP 10 PARAMETER COMBINATIONS (by Sharpe Ratio)")
    print(f"{'='*80}")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


# --------------------------------------------
# Main Execution
# --------------------------------------------
if __name__ == "__main__":
    stats = run_ema_backtest(
        symbol="XAU/USD",
        interval="1h",
        bars=5000,
        ema_fast=9,
        ema_slow=21,
        adx_threshold=20,
        use_rsi_filter=True,
        rsi_upper=65,
        rsi_lower=35,
        use_pullback=False,
        pullback_to_fast=False,
        supertrend_atr_period=14,
        supertrend_multiplier=3,
        capital=10000,
        commission=0.0002,
        plot=True
    )
    
    # Uncomment to run parameter search
    # results = simple_grid_search("XAU/USD", "1h", 5000)
