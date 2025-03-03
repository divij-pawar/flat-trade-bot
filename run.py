import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)
logger = logging.getLogger('TradingBot')

class TradingBot:
    def __init__(self, initial_capital=10000, api_key=None, api_secret=None):
        """
        Initialize the trading bot with parameters
        
        Args:
            initial_capital (float): Starting capital for the bot
            api_key (str): API key for the trading platform
            api_secret (str): API secret for the trading platform
        """
        self.capital = initial_capital
        self.positions = {}  # Dictionary to track current positions
        self.trade_history = []  # List to track all trades
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_running = False
        logger.info(f"Trading bot initialized with ${initial_capital} capital")
    
    def connect_to_exchange(self, exchange_name):
        """
        Connect to the specified trading exchange
        
        Args:
            exchange_name (str): Name of the exchange to connect to
        """
        logger.info(f"Connecting to exchange: {exchange_name}")
        # Placeholder for exchange connection logic
        # In a real implementation, you would:
        # 1. Import the appropriate exchange API library
        # 2. Authenticate with your API credentials
        # 3. Establish a connection
        logger.info("Exchange connection established")
        return True
    
    def fetch_market_data(self, symbol, timeframe='1h', limit=100):
        """
        Fetch market data for a given symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USD')
            timeframe (str): Candlestick timeframe
            limit (int): Number of candlesticks to retrieve
        
        Returns:
            pd.DataFrame: Market data as a DataFrame
        """
        logger.info(f"Fetching market data for {symbol}, timeframe {timeframe}, limit {limit}")
        # Placeholder for market data retrieval
        # In a real implementation, you would call the exchange API
        
        # Simulated data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 5, limit),
            'high': np.random.normal(105, 5, limit),
            'low': np.random.normal(95, 5, limit),
            'close': np.random.normal(100, 5, limit),
            'volume': np.random.normal(1000, 200, limit)
        })
        return data
    
    def calculate_indicators(self, data):
        """
        Calculate technical indicators on market data
        
        Args:
            data (pd.DataFrame): Market data
        
        Returns:
            pd.DataFrame: Market data with added indicators
        """
        logger.info("Calculating technical indicators")
        df = data.copy()
        
        # Simple Moving Averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on indicators
        
        Args:
            data (pd.DataFrame): Market data with indicators
        
        Returns:
            pd.DataFrame: Market data with signals
        """
        logger.info("Generating trading signals")
        df = data.copy()
        
        # Initialize signal column
        df['signal'] = 0
        
        # SMA crossover strategy
        df['signal'] = np.where((df['sma20'] > df['sma50']) & 
                               (df['sma20'].shift(1) <= df['sma50'].shift(1)), 1, df['signal'])  # Buy signal
        df['signal'] = np.where((df['sma20'] < df['sma50']) & 
                               (df['sma20'].shift(1) >= df['sma50'].shift(1)), -1, df['signal'])  # Sell signal
        
        # RSI overbought/oversold strategy
        df['signal'] = np.where(df['rsi'] < 30, 1, df['signal'])  # Oversold -> Buy
        df['signal'] = np.where(df['rsi'] > 70, -1, df['signal'])  # Overbought -> Sell
        
        return df
    
    def execute_trade(self, symbol, side, quantity, price=None):
        """
        Execute a trade on the exchange
        
        Args:
            symbol (str): Trading pair symbol
            side (str): 'buy' or 'sell'
            quantity (float): Amount to buy/sell
            price (float, optional): Limit price, None for market orders
        
        Returns:
            dict: Trade result
        """
        order_type = "market" if price is None else "limit"
        logger.info(f"Executing {side} {order_type} order: {quantity} {symbol} @ {price if price else 'market price'}")
        
        # Placeholder for actual trade execution
        # In a real implementation, you would call the exchange API
        
        # Simulated trade execution
        execution_price = price if price else self.fetch_market_data(symbol, limit=1)['close'].iloc[-1]
        trade_value = quantity * execution_price
        
        # Record the trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'value': trade_value,
            'type': order_type
        }
        self.trade_history.append(trade)
        
        # Update positions and capital
        if side == 'buy':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.capital -= trade_value
        else:  # sell
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            self.capital += trade_value
        
        logger.info(f"Trade executed: {trade}")
        logger.info(f"Updated capital: ${self.capital:.2f}")
        logger.info(f"Updated positions: {self.positions}")
        
        return trade
    
    def risk_management(self, symbol, data):
        """
        Apply risk management rules
        
        Args:
            symbol (str): Trading pair symbol
            data (pd.DataFrame): Market data
        
        Returns:
            bool: True if trade passes risk checks
        """
        # Check if we have enough capital
        if self.capital < 1000:
            logger.warning("Risk check failed: Insufficient capital")
            return False
        
        # Check if volatility is acceptable
        recent_data = data.tail(20)
        volatility = (recent_data['high'] - recent_data['low']).mean() / recent_data['close'].mean()
        if volatility > 0.05:  # 5% volatility threshold
            logger.warning(f"Risk check failed: Volatility too high ({volatility:.2%})")
            return False
        
        # Position sizing (don't use more than 5% of capital per trade)
        max_position_size = self.capital * 0.05
        
        # Check current exposure to this symbol
        current_exposure = self.positions.get(symbol, 0) * data['close'].iloc[-1]
        if current_exposure > max_position_size:
            logger.warning(f"Risk check failed: Position size too large (${current_exposure:.2f})")
            return False
        
        logger.info("Risk management checks passed")
        return True
    
    def run_strategy(self, symbol, timeframe='1h'):
        """
        Execute the trading strategy in a loop
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candlestick timeframe
        """
        self.is_running = True
        logger.info(f"Starting trading strategy for {symbol} on {timeframe} timeframe")
        
        try:
            while self.is_running:
                # Fetch latest market data
                market_data = self.fetch_market_data(symbol, timeframe)
                
                # Calculate indicators
                data_with_indicators = self.calculate_indicators(market_data)
                
                # Generate signals
                data_with_signals = self.generate_signals(data_with_indicators)
                
                # Check the latest signal
                latest_signal = data_with_signals['signal'].iloc[-1]
                
                # Apply risk management
                if self.risk_management(symbol, data_with_signals) and latest_signal != 0:
                    # Determine trade size (simplified)
                    current_price = data_with_signals['close'].iloc[-1]
                    trade_size = min(self.capital * 0.05 / current_price, 1.0)
                    
                    if latest_signal > 0:  # Buy signal
                        self.execute_trade(symbol, 'buy', trade_size)
                    elif latest_signal < 0:  # Sell signal
                        # Check if we have a position to sell
                        position = self.positions.get(symbol, 0)
                        if position > 0:
                            self.execute_trade(symbol, 'sell', min(position, trade_size))
                
                # Wait before the next iteration
                logger.info(f"Waiting for next iteration... Current capital: ${self.capital:.2f}")
                time.sleep(60)  # Wait 60 seconds before checking again
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading strategy: {e}")
        finally:
            self.is_running = False
            logger.info("Trading strategy stopped")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Trading bot stopping...")
    
    def get_performance_metrics(self):
        """
        Calculate and return performance metrics
        
        Returns:
            dict: Performance metrics
        """
        if not self.trade_history:
            return {"message": "No trades executed yet"}
        
        # Calculate basic metrics
        initial_capital = 10000  # Assuming default initial capital
        current_value = self.capital
        for symbol, quantity in self.positions.items():
            # Get latest price for each symbol (simplified)
            latest_price = self.fetch_market_data(symbol, limit=1)['close'].iloc[-1]
            current_value += quantity * latest_price
        
        profit_loss = current_value - initial_capital
        profit_loss_percent = (profit_loss / initial_capital) * 100
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trade_history if 
                            (trade['side'] == 'buy' and trade['price'] < self.fetch_market_data(trade['symbol'], limit=1)['close'].iloc[-1]) or
                            (trade['side'] == 'sell' and trade['price'] > self.fetch_market_data(trade['symbol'], limit=1)['close'].iloc[-1]))
        
        win_rate = (winning_trades / len(self.trade_history)) * 100 if self.trade_history else 0
        
        metrics = {
            "initial_capital": initial_capital,
            "current_value": current_value,
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "current_positions": self.positions
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Create a trading bot instance
    bot = TradingBot(initial_capital=10000)
    
    # Connect to an exchange (simulated)
    bot.connect_to_exchange("ExampleExchange")
    
    # Run the trading strategy for BTC/USD
    try:
        bot.run_strategy("BTC/USD", timeframe="1h")
    except KeyboardInterrupt:
        # Gracefully stop the bot
        bot.stop()
        
        # Print performance metrics
        metrics = bot.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")