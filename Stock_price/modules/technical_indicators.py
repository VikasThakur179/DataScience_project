import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TechnicalIndicators:
    """
    Calculate various technical indicators for stock analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def calculate_sma(self, window: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            window (int): Period for SMA calculation
        
        Returns:
            pd.Series: SMA values
        """
        return self.data['Close'].rolling(window=window).mean()
    
    def calculate_ema(self, window: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            window (int): Period for EMA calculation
        
        Returns:
            pd.Series: EMA values
        """
        return self.data['Close'].ewm(span=window).mean()
    
    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            window (int): Period for RSI calculation
        
        Returns:
            pd.Series: RSI values
        """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
        
        Returns:
            pd.DataFrame: MACD, Signal, and Histogram
        """
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            window (int): Period for moving average
            num_std (float): Number of standard deviations
        
        Returns:
            pd.DataFrame: Upper, Middle, and Lower bands
        """
        sma = self.calculate_sma(window)
        std = self.data['Close'].rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band
        })
    
    def calculate_stochastic(self, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_window (int): %K period
            d_window (int): %D period
        
        Returns:
            pd.DataFrame: %K and %D values
        """
        lowest_low = self.data['Low'].rolling(window=k_window).min()
        highest_high = self.data['High'].rolling(window=k_window).max()
        
        k_percent = 100 * (self.data['Close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        })
    
    def calculate_atr(self, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            window (int): Period for ATR calculation
        
        Returns:
            pd.Series: ATR values
        """
        high_low = self.data['High'] - self.data['Low']
        high_close_prev = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close_prev = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def calculate_williams_r(self, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            window (int): Period for calculation
        
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = self.data['High'].rolling(window=window).max()
        lowest_low = self.data['Low'].rolling(window=window).min()
        
        williams_r = -100 * (highest_high - self.data['Close']) / (highest_high - lowest_low)
        
        return williams_r
    
    def calculate_cci(self, window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index.
        
        Args:
            window (int): Period for calculation
        
        Returns:
            pd.Series: CCI values
        """
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def calculate_momentum(self, window: int = 10) -> pd.Series:
        """
        Calculate Price Momentum.
        
        Args:
            window (int): Period for calculation
        
        Returns:
            pd.Series: Momentum values
        """
        return self.data['Close'] / self.data['Close'].shift(window) - 1
    
    def calculate_roc(self, window: int = 12) -> pd.Series:
        """
        Calculate Rate of Change.
        
        Args:
            window (int): Period for calculation
        
        Returns:
            pd.Series: ROC values
        """
        return ((self.data['Close'] - self.data['Close'].shift(window)) / 
                self.data['Close'].shift(window)) * 100
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and return as DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with all indicators
        """
        result = self.data.copy()
        
        # Moving Averages
        result['SMA_10'] = self.calculate_sma(10)
        result['SMA_20'] = self.calculate_sma(20)
        result['SMA_50'] = self.calculate_sma(50)
        result['EMA_12'] = self.calculate_ema(12)
        result['EMA_26'] = self.calculate_ema(26)
        
        # Oscillators
        result['RSI'] = self.calculate_rsi()
        
        # MACD
        macd_data = self.calculate_macd()
        result = pd.concat([result, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands()
        result = pd.concat([result, bb_data], axis=1)
        
        # Stochastic
        stoch_data = self.calculate_stochastic()
        result = pd.concat([result, stoch_data], axis=1)
        
        # Other indicators
        result['ATR'] = self.calculate_atr()
        result['Williams_R'] = self.calculate_williams_r()
        result['CCI'] = self.calculate_cci()
        result['Momentum'] = self.calculate_momentum()
        result['ROC'] = self.calculate_roc()
        
        return result
    
    def get_trading_signals(self) -> pd.DataFrame:
        """
        Generate basic trading signals based on technical indicators.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # RSI signals
        rsi = self.calculate_rsi()
        signals['RSI_Oversold'] = rsi < 30
        signals['RSI_Overbought'] = rsi > 70
        
        # MACD signals
        macd_data = self.calculate_macd()
        signals['MACD_Bullish'] = (macd_data['MACD'] > macd_data['Signal']) & \
                                  (macd_data['MACD'].shift(1) <= macd_data['Signal'].shift(1))
        signals['MACD_Bearish'] = (macd_data['MACD'] < macd_data['Signal']) & \
                                  (macd_data['MACD'].shift(1) >= macd_data['Signal'].shift(1))
        
        # Bollinger Bands signals
        bb_data = self.calculate_bollinger_bands()
        signals['BB_Oversold'] = self.data['Close'] < bb_data['BB_Lower']
        signals['BB_Overbought'] = self.data['Close'] > bb_data['BB_Upper']
        
        # Moving Average signals
        sma_20 = self.calculate_sma(20)
        sma_50 = self.calculate_sma(50)
        signals['Golden_Cross'] = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
        signals['Death_Cross'] = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
        
        return signals
