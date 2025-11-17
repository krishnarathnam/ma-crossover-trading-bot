import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mt5_trading_bot.log')
    ]
)

# Minimal console output
def print_status(action: str, signal: str, price: float = None, timestamp: str = None):
    """Print minimal CLI status."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%H:%M:%S')
    
    print(f"{signal.upper()} | [{timestamp}]")

# ============================================
# Configuration
# ============================================
class Config:
    # Trading Parameters
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_H1
    LOT_SIZE = 0.01
    RISK_REWARD_RATIO = 2.0
    ATR_MULTIPLIER = 1.5  # Stop loss distance
    
    # Strategy Parameters
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ADX_THRESHOLD = 20
    RSI_PERIOD = 14
    RSI_UPPER = 65
    RSI_LOWER = 35
    SUPERTREND_ATR_PERIOD = 14
    SUPERTREND_MULTIPLIER = 3
    
    # Bot Settings
    MAGIC_NUMBER = 123456
    MAX_SLIPPAGE = 10
    BARS_TO_FETCH = 200


# ============================================
# Indicator Calculations
# ============================================
def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate EMA."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    return dx.rolling(period).mean()


def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                        atr_period: int = 10, multiplier: float = 3) -> pd.Series:
    """Calculate Supertrend direction."""
    atr = calculate_atr(high, low, close, atr_period)
    hl_avg = (high + low) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < supertrend.iloc[i-1]:
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


# ============================================
# MT5 Connection & Data
# ============================================
class MT5Connection:
    @staticmethod
    def initialize() -> bool:
        """Initialize MT5 connection."""
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        logging.info("MT5 initialized successfully")
        return True
    
    @staticmethod
    def shutdown():
        """Shutdown MT5 connection."""
        mt5.shutdown()
        logging.info("MT5 connection closed")
    
    @staticmethod
    def get_data(symbol: str, timeframe: int, bars: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLC data from MT5."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'tick_volume']]
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None


# ============================================
# Trading Logic
# ============================================
class TradingSignals:
    def __init__(self, config: Config):
        self.config = config
        self.last_ema_fast = None
        self.last_ema_slow = None
        self.current_ema_fast = None
        self.current_ema_slow = None
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze data and generate trading signals."""
        # Calculate indicators
        df['ema_fast'] = calculate_ema(df['close'], self.config.EMA_FAST)
        df['ema_slow'] = calculate_ema(df['close'], self.config.EMA_SLOW)
        df['rsi'] = calculate_rsi(df['close'], self.config.RSI_PERIOD)
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'], self.config.ADX_PERIOD)
        df['supertrend_dir'] = calculate_supertrend(
            df['high'], df['low'], df['close'],
            self.config.SUPERTREND_ATR_PERIOD,
            self.config.SUPERTREND_MULTIPLIER
        )
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
        
        df.dropna(inplace=True)
        
        if len(df) < 2:
            return {'signal': 'NONE', 'reason': 'Insufficient data'}
        
        # Get current values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Store EMA values for crossover detection
        self.last_ema_fast = previous['ema_fast']
        self.last_ema_slow = previous['ema_slow']
        self.current_ema_fast = current['ema_fast']
        self.current_ema_slow = current['ema_slow']
        
        # Check ADX filter
        if current['adx'] < self.config.ADX_THRESHOLD:
            return {'signal': 'NONE', 'reason': f"ADX too low: {current['adx']:.2f}"}
        
        # Detect EMA crossover
        bullish_cross = (self.last_ema_fast <= self.last_ema_slow and 
                        self.current_ema_fast > self.current_ema_slow)
        bearish_cross = (self.last_ema_fast >= self.last_ema_slow and 
                        self.current_ema_fast < self.current_ema_slow)
        
        # Generate signals
        if bullish_cross:
            if current['supertrend_dir'] == 1:
                if current['rsi'] < self.config.RSI_UPPER:
                    return {
                        'signal': 'BUY',
                        'price': current['close'],
                        'atr': current['atr'],
                        'reason': f"Bullish EMA cross, ST=UP, RSI={current['rsi']:.1f}, ADX={current['adx']:.1f}"
                    }
                return {'signal': 'NONE', 'reason': f"RSI too high: {current['rsi']:.1f}"}
            return {'signal': 'NONE', 'reason': 'Supertrend bearish'}
        
        elif bearish_cross:
            if current['supertrend_dir'] == -1:
                if current['rsi'] > self.config.RSI_LOWER:
                    return {
                        'signal': 'SELL',
                        'price': current['close'],
                        'atr': current['atr'],
                        'reason': f"Bearish EMA cross, ST=DOWN, RSI={current['rsi']:.1f}, ADX={current['adx']:.1f}"
                    }
                return {'signal': 'NONE', 'reason': f"RSI too low: {current['rsi']:.1f}"}
            return {'signal': 'NONE', 'reason': 'Supertrend bullish'}
        
        # Check exit conditions for existing positions
        exit_signal = None
        if current['supertrend_dir'] != previous['supertrend_dir']:
            exit_signal = 'SUPERTREND_FLIP'
        elif current['adx'] < self.config.ADX_THRESHOLD:
            exit_signal = 'ADX_LOW'
        elif bearish_cross:
            exit_signal = 'BEARISH_CROSS'
        elif bullish_cross:
            exit_signal = 'BULLISH_CROSS'
        
        return {
            'signal': 'NONE',
            'exit_signal': exit_signal,
            'supertrend_dir': current['supertrend_dir'],
            'rsi': current['rsi'],
            'adx': current['adx'],
            'reason': 'No crossover detected'
        }


# ============================================
# Position Management
# ============================================
class PositionManager:
    def __init__(self, config: Config):
        self.config = config
    
    def get_positions(self) -> list:
        """Get all open positions for symbol."""
        positions = mt5.positions_get(symbol=self.config.SYMBOL)
        if positions is None:
            return []
        return [p._asdict() for p in positions if p.magic == self.config.MAGIC_NUMBER]
    
    def close_position(self, position: dict) -> bool:
        """Close a position."""
        try:
            symbol = position['symbol']
            ticket = position['ticket']
            volume = position['volume']
            position_type = position['type']
            
            # Determine close order type
            if position_type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": self.config.MAX_SLIPPAGE,
                "magic": self.config.MAGIC_NUMBER,
                "comment": "Bot close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to close position {ticket}: {result.comment}")
                return False
            
            logging.info(f"Position {ticket} closed successfully")
            position_type_str = "BUY" if position_type == mt5.POSITION_TYPE_BUY else "SELL"
            print_status("CLOSE", position_type_str, price)
            return True
            
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            return False
    
    def open_position(self, signal: dict) -> bool:
        """Open a new position with SL and TP."""
        try:
            symbol = self.config.SYMBOL
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                logging.error(f"Symbol {symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logging.error(f"Failed to select {symbol}")
                    return False
            
            point = symbol_info.point
            price = signal['price']
            atr = signal['atr']
            
            # Calculate SL and TP based on ATR and risk-reward ratio
            sl_distance = self.config.ATR_MULTIPLIER * atr
            tp_distance = sl_distance * self.config.RISK_REWARD_RATIO
            
            if signal['signal'] == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
                sl = price - sl_distance
                tp = price + tp_distance
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
                sl = price + sl_distance
                tp = price - tp_distance
            
            # Round to proper digits
            digits = symbol_info.digits
            sl = round(sl, digits)
            tp = round(tp, digits)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": self.config.LOT_SIZE,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.config.MAX_SLIPPAGE,
                "magic": self.config.MAGIC_NUMBER,
                "comment": f"Bot {signal['signal']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.comment}")
                return False
            
            logging.info(f"Order opened: {signal['signal']} at {price:.5f}, SL={sl:.5f}, TP={tp:.5f}")
            logging.info(f"Risk: {sl_distance:.5f}, Reward: {tp_distance:.5f}, RR=1:{self.config.RISK_REWARD_RATIO}")
            print_status("OPEN", signal['signal'], price)
            return True
            
        except Exception as e:
            logging.error(f"Error opening position: {e}")
            return False


# ============================================
# Main Trading Bot
# ============================================
class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.signals = TradingSignals(config)
        self.position_mgr = PositionManager(config)
        self.running = False
    
    def check_and_close_positions(self, analysis: dict):
        """Check if any positions need to be closed."""
        positions = self.position_mgr.get_positions()
        
        if not positions:
            return
        
        for position in positions:
            should_close = False
            reason = ""
            
            # Check Supertrend flip
            if 'supertrend_dir' in analysis:
                if position['type'] == mt5.POSITION_TYPE_BUY and analysis['supertrend_dir'] == -1:
                    should_close = True
                    reason = "Supertrend flipped bearish"
                elif position['type'] == mt5.POSITION_TYPE_SELL and analysis['supertrend_dir'] == 1:
                    should_close = True
                    reason = "Supertrend flipped bullish"
            
            # Check exit signal
            if 'exit_signal' in analysis and analysis['exit_signal'] is not None:
                should_close = True
                reason = f"Exit signal: {analysis['exit_signal']}"
            
            if should_close:
                logging.info(f"Closing position {position['ticket']}: {reason}")
                self.position_mgr.close_position(position)
    
    def run_once(self):
        """Execute one trading cycle."""
        logging.info(f"\n{'='*60}")
        logging.info(f"Trading cycle started: {datetime.now()}")
        logging.info(f"{'='*60}")
        
        # Fetch data
        df = MT5Connection.get_data(self.config.SYMBOL, self.config.TIMEFRAME, self.config.BARS_TO_FETCH)
        if df is None:
            logging.error("Failed to fetch data")
            return
        
        # Analyze
        analysis = self.signals.analyze(df)
        logging.info(f"Analysis: {analysis.get('reason', 'No reason')}")
        
        # Check and close positions if needed
        self.check_and_close_positions(analysis)
        
        # Check for new entry signals
        positions = self.position_mgr.get_positions()
        
        if analysis['signal'] in ['BUY', 'SELL']:
            if len(positions) > 0:
                logging.info(f"Signal {analysis['signal']} ignored: Position already open")
                print_status("HOLD", "HOLD")
            else:
                logging.info(f"Opening {analysis['signal']} position")
                self.position_mgr.open_position(analysis)
        else:
            print_status("HOLD", "HOLD")
        
        logging.info(f"Cycle complete. Next check in 1 hour.\n")
    
    def wait_for_next_hour(self):
        """Wait until the next hour starts (e.g., 10:00, 11:00, etc.)."""
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_seconds = (next_hour - now).total_seconds()
        
        logging.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Next run scheduled: {next_hour.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Waiting {wait_seconds:.0f} seconds ({wait_seconds/60:.1f} minutes)...")
        print(f"Waiting until {next_hour.strftime('%H:%M')}...")
        
        time.sleep(wait_seconds)
    
    def start(self):
        """Start the trading bot - runs at the start of each hour."""
        if not MT5Connection.initialize():
            return
        
        self.running = True
        logging.info("="*60)
        logging.info("MT5 TRADING BOT STARTED")
        logging.info("="*60)
        logging.info(f"Symbol: {self.config.SYMBOL}")
        logging.info(f"Timeframe: H1 (Hourly)")
        logging.info(f"Lot Size: {self.config.LOT_SIZE}")
        logging.info(f"Risk-Reward: 1:{self.config.RISK_REWARD_RATIO}")
        logging.info(f"Stop Loss: {self.config.ATR_MULTIPLIER}x ATR")
        logging.info(f"EMA: {self.config.EMA_FAST}/{self.config.EMA_SLOW}")
        logging.info(f"ADX: >{self.config.ADX_THRESHOLD}, RSI: {self.config.RSI_LOWER}-{self.config.RSI_UPPER}")
        logging.info(f"Supertrend: ATR={self.config.SUPERTREND_ATR_PERIOD}, Mult={self.config.SUPERTREND_MULTIPLIER}")
        logging.info("="*60)
        
        print(f"Bot Started: {self.config.SYMBOL} H1 | RR 1:{self.config.RISK_REWARD_RATIO}")
        
        try:
            # Run immediately on start
            self.run_once()
            
            while self.running:
                # Wait until next hour starts (synchronized with MT5 H1 candles)
                self.wait_for_next_hour()
                self.run_once()
                
        except KeyboardInterrupt:
            logging.info("\n" + "="*60)
            logging.info("Bot stopped by user")
            logging.info("="*60)
            print("\nBot stopped")
        except Exception as e:
            logging.error(f"Bot error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            print(f"\nError: {e}")
        finally:
            MT5Connection.shutdown()
    
    def stop(self):
        """Stop the trading bot."""
        self.running = False
        logging.info("Stopping bot...")


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    config = Config()
    bot = TradingBot(config)
    bot.start()

