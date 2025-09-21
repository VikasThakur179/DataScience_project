import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Tuple, Optional, Union
import streamlit as st

class DataPreprocessor:
    """
    Data preprocessing utilities for LSTM model training.
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize data preprocessor.
        
        Args:
            scaler_type (str): Type of scaler ('minmax', 'standard', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler()
        self.feature_columns = None
    
    def _get_scaler(self):
        """Get the appropriate scaler based on type."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def prepare_lstm_data(self, data: pd.DataFrame, sequence_length: int = 60,
                         target_column: str = 'Close', feature_columns: Optional[list] = None,
                         test_size: float = 0.2) -> Optional[Tuple]:
        """
        Prepare data for LSTM training.
        
        Args:
            data (pd.DataFrame): Input stock data
            sequence_length (int): Number of time steps for sequences
            target_column (str): Target column name
            feature_columns (list): List of feature columns to use
            test_size (float): Test set size ratio
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        try:
            # Select features
            if feature_columns is None:
                # Use all numeric columns except target
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_columns if col != target_column]
                
                # Prioritize important features if too many
                if len(feature_columns) > 10:
                    priority_features = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'SMA_20', 'EMA_12']
                    feature_columns = [col for col in priority_features if col in feature_columns]
                    feature_columns.append(target_column)  # Always include target
                else:
                    feature_columns.append(target_column)
            
            self.feature_columns = feature_columns
            
            # Select and clean data
            selected_data = data[feature_columns].dropna()
            
            if len(selected_data) < sequence_length + 10:
                st.error("Insufficient data for training. Need more historical data.")
                return None
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(selected_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, sequence_length, target_column, feature_columns)
            
            if len(X) == 0:
                st.error("Failed to create sequences. Check data quality.")
                return None
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            return X_train, X_test, y_train, y_test, self.scaler
            
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return None
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int,
                         target_column: str, feature_columns: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.ndarray): Scaled data
            sequence_length (int): Sequence length
            target_column (str): Target column name
            feature_columns (list): Feature column names
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X, y = [], []
        target_idx = feature_columns.index(target_column)
        
        for i in range(sequence_length, len(data)):
            # Features: all columns for the sequence
            X.append(data[i - sequence_length:i])
            # Target: only the target column for the next time step
            y.append(data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def inverse_transform_predictions(self, predictions: np.ndarray,
                                    target_column: str) -> np.ndarray:
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions (np.ndarray): Scaled predictions
            target_column (str): Target column name
        
        Returns:
            np.ndarray: Original scale predictions
        """
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Run prepare_lstm_data first.")
        
        target_idx = self.feature_columns.index(target_column)
        
        # Create a dummy array with the same shape as training data
        dummy_data = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_data[:, target_idx] = predictions.flatten()
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_data)
        
        return inverse_transformed[:, target_idx]
    
    def add_lag_features(self, data: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
        """
        Add lag features to the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create lags for
            lags (list): List of lag periods
        
        Returns:
            pd.DataFrame: Data with lag features
        """
        result = data.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    result[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return result.dropna()
    
    def add_rolling_features(self, data: pd.DataFrame, columns: list,
                           windows: list, functions: list = ['mean', 'std']) -> pd.DataFrame:
        """
        Add rolling window features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create rolling features for
            windows (list): List of window sizes
            functions (list): List of functions to apply
        
        Returns:
            pd.DataFrame: Data with rolling features
        """
        result = data.copy()
        
        for col in columns:
            if col in data.columns:
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            result[f'{col}_rolling_{window}_mean'] = data[col].rolling(window).mean()
                        elif func == 'std':
                            result[f'{col}_rolling_{window}_std'] = data[col].rolling(window).std()
                        elif func == 'min':
                            result[f'{col}_rolling_{window}_min'] = data[col].rolling(window).min()
                        elif func == 'max':
                            result[f'{col}_rolling_{window}_max'] = data[col].rolling(window).max()
        
        return result.dropna()
    
    def create_returns(self, data: pd.DataFrame, columns: list,
                      periods: list = [1, 5, 10]) -> pd.DataFrame:
        """
        Create return features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): Columns to create returns for
            periods (list): List of periods for returns
        
        Returns:
            pd.DataFrame: Data with return features
        """
        result = data.copy()
        
        for col in columns:
            if col in data.columns:
                for period in periods:
                    result[f'{col}_return_{period}'] = data[col].pct_change(periods=period)
        
        return result.dropna()
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Outlier detection method ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Data with outliers marked
        """
        result = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result[f'{col}_outlier'] = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                result[f'{col}_outlier'] = z_scores > threshold
        
        return result
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to handle missing values
        
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        result = data.copy()
        
        if method == 'forward_fill':
            result = result.fillna(method='ffill')
        elif method == 'backward_fill':
            result = result.fillna(method='bfill')
        elif method == 'interpolate':
            result = result.interpolate()
        elif method == 'drop':
            result = result.dropna()
        elif method == 'mean':
            for col in result.select_dtypes(include=[np.number]).columns:
                result[col] = result[col].fillna(result[col].mean())
        
        return result
    
    def create_features_for_lstm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for LSTM training.
        
        Args:
            data (pd.DataFrame): Input stock data
        
        Returns:
            pd.DataFrame: Enhanced data with features
        """
        result = data.copy()
        
        # Price-based features
        if 'Close' in result.columns:
            # Returns
            result = self.create_returns(result, ['Close'], [1, 3, 5, 10])
            
            # Rolling features
            result = self.add_rolling_features(
                result, ['Close'], [5, 10, 20], ['mean', 'std']
            )
            
            # Price ratios
            if 'High' in result.columns and 'Low' in result.columns:
                result['Price_Range'] = (result['High'] - result['Low']) / result['Close']
                result['Price_Position'] = (result['Close'] - result['Low']) / (result['High'] - result['Low'])
        
        # Volume features
        if 'Volume' in result.columns:
            result = self.add_rolling_features(
                result, ['Volume'], [5, 20], ['mean', 'std']
            )
            result['Volume_Ratio'] = result['Volume'] / result['Volume_rolling_20_mean']
        
        # Technical indicators ratios
        if 'RSI' in result.columns:
            result['RSI_normalized'] = (result['RSI'] - 50) / 50
        
        if 'MACD' in result.columns and 'Signal' in result.columns:
            result['MACD_Signal_Ratio'] = result['MACD'] / result['Signal']
        
        # Clean the data
        result = self.handle_missing_values(result, method='forward_fill')
        
        return result
    
    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """
        Validate data quality for LSTM training.
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            dict: Data quality report
        """
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'date_range': None,
            'data_gaps': 0
        }
        
        # Date range analysis
        if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
            report['date_range'] = (data.index.min(), data.index.max())
            
            # Check for gaps in daily data
            if len(data) > 1:
                expected_days = (data.index.max() - data.index.min()).days + 1
                actual_days = len(data)
                report['data_gaps'] = expected_days - actual_days
        
        # Data quality score
        quality_score = 100
        quality_score -= min(report['missing_percentage'], 50)  # Penalize missing values
        quality_score -= min((report['duplicate_rows'] / len(data)) * 100, 20)  # Penalize duplicates
        
        report['quality_score'] = max(quality_score, 0)
        
        return report
