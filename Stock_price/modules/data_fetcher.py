import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Optional

class StockDataFetcher:
    """
    Handles fetching and preprocessing of stock market data using yfinance.
    """
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    def get_stock_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol and period.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
            
            # Clean and validate data
            data = self._clean_data(data)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stocks(self, symbols: list, period: str = '1y') -> dict:
        """
        Fetch data for multiple stock symbols.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Time period
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if data is not None:
                stock_data[symbol] = data
        
        return stock_data
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return stock_info
            
        except Exception as e:
            st.warning(f"Could not fetch info for {symbol}: {str(e)}")
            return {'name': symbol}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data.
        
        Args:
            data (pd.DataFrame): Raw stock data
        
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure positive values for prices and volume
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        data = data[data['Volume'] >= 0]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get the current/latest price for a stock symbol.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            float: Current price
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch real-time price for {symbol}: {str(e)}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data.
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d')
            return not data.empty
        except:
            return False
