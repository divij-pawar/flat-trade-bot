import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import trading_core  # Import our C extension module

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
        self.max_position_percent = 0.05  # Max 5% of capital per position
        self.max_volatility = 0.05  # Max 5% volatility
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
        Calculate technical indicators on market data using C extension
        
        Args:
            data (pd.DataFrame): Market data
        
        Returns:
            pd.DataFrame: Market data with added indicators
        """
        logger.info("Calculating technical indicators using C extension")
        df = data.copy()
        
        # Convert to numpy arrays for C functions
        close_prices = df['close'].values.astype(np.float64)
        
        # Calculate indicators using C extension
        df['sma20'] = trading_core.calculate_sma(close_prices, 20)
        df['sma50'] = trading_core.calculate_sma(close_prices, 50)
        df['ema12'] = trading_core.calculate_ema(close_prices, 12)
        df['ema26'] = trading_core.calculate_ema(close_prices, 26)
        df['rsi'] = trading_core.calculate_rsi(close_prices, 14)
        
        # Calculate MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = trading_core.calculate_ema(df['macd'].values.astype(np.float64), 9)
        
        return df
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on indicators using C extension
        
        Args:
            data (pd.DataFrame): Market data with indicators
        
        Returns:
            pd.DataFrame: Market data with signals
        """
        logger.info("Generating trading signals")
        df = data.copy()
        
        # Generate signals using C extension
        crossover_signals = trading_core.generate_crossover_signals(
            df['sma20'].values.astype(np.float64), 
            df['sma50'].values.astype(np.float64)
        )
        
        # Initialize signal column with crossover signals
        df['signal'] = crossover_signals
        
        # RSI overbought/oversold logic (could also be moved to C)
        df.loc[df['rsi'] < 30, 'signal'] = 1  # Oversold -> Buy
        df.loc[df['rsi'] > 70, 'signal'] = -1  # Overbought -> Sell
        
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
        Apply risk management rules using C extension
        
        Args:
            symbol (str): Trading pair symbol
            data (pd.DataFrame): Market data
        
        Returns:
            bool: True if trade passes risk checks
        """
        # Calculate volatility from recent data
        recent_data = data.tail(20)
        volatility = (recent_data['high'] - recent_data['low']).mean() / recent_data['close'].mean()
        
        # Get current position value
        current_price = data['close'].iloc[-1]
        current_position = self.positions.get(symbol, 0)
        position_value = current_position * current_price
        
        # Perform risk check using C extension
        result, message = trading_core.check_risk(
            self.capital,
            position_value,
            volatility,
            self.max_position_percent,
            self.max_volatility
        )
        
        if result == 0:
            logger.warning(f"Risk check failed: {message}")
            return False
        
        logger.info(f"Risk management checks passed: {message}")
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
                
                # Calculate indicators (using C extension)
                data_with_indicators = self.calculate_indicators(market_data)
                
                # Generate signals (using C extension)
                data_with_signals = self.generate_signals(data_with_indicators)
                
                # Check the latest signal
                latest_signal = data_with_signals['signal'].iloc[-1]
                
                # Apply risk management (using C extension)
                if self.risk_management(symbol, data_with_signals) and latest_signal != 0:
                    # Determine trade size
                    current_price = data_with_signals['close'].iloc[-1]
                    trade_size = min(self.capital * self.max_position_percent / current_price, 1.0)
                    
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
            # Get latest price for each symbol
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