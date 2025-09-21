import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import streamlit as st

class LSTMPredictor:
    """
    LSTM Neural Network for stock price prediction.
    """
    
    def __init__(self, sequence_length: int = 60, n_features: int = 1, lstm_units: int = 128):
        """
        Initialize LSTM predictor.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of features per time step
            lstm_units (int): Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def build_model(self, dropout_rate: float = 0.2) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            dropout_rate (float): Dropout rate for regularization
        
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(
                units=self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ),
            Dropout(dropout_rate),
            BatchNormalization(),
            
            # Second LSTM layer with return sequences
            LSTM(
                units=self.lstm_units // 2,
                return_sequences=True
            ),
            Dropout(dropout_rate),
            BatchNormalization(),
            
            # Third LSTM layer without return sequences
            LSTM(
                units=self.lstm_units // 4,
                return_sequences=False
            ),
            Dropout(dropout_rate),
            BatchNormalization(),
            
            # Dense layers
            Dense(units=50, activation='relu'),
            Dropout(dropout_rate / 2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data (np.ndarray): Input data
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, 0])  # Predict the first feature (Close price)
        
        return np.array(X), np.array(y)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   validation_split: float = 0.2, epochs: int = 50,
                   batch_size: int = 32) -> Tuple[Sequential, tf.keras.callbacks.History]:
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            validation_split (float): Validation data split ratio
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        
        Returns:
            Tuple[Sequential, History]: Trained model and training history
        """
        # Build model
        self.model = self.build_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model, history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input sequences
        
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        return self.model.predict(X)
    
    def predict_future(self, last_sequence: np.ndarray, scaler: MinMaxScaler, 
                      days: int = 7) -> np.ndarray:
        """
        Predict future prices for specified number of days.
        
        Args:
            last_sequence (np.ndarray): Last sequence of data
            scaler (MinMaxScaler): Fitted scaler object
            days (int): Number of days to predict
        
        Returns:
            np.ndarray: Future predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Prepare the last sequence
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        current_sequence = last_sequence_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # Calculate directional accuracy
        y_test_direction = np.diff(y_test) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not built yet."
        
        return self.model.summary()
    
    def calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Calculate approximate feature importance using permutation importance.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Get baseline score
        baseline_score = self.model.evaluate(X, y, verbose=0)[0]  # loss
        
        feature_importance = {}
        
        for feature_idx in range(X.shape[2]):
            # Create permuted version
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])
            
            # Calculate score with permuted feature
            permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[0]
            
            # Importance is the increase in loss
            importance = permuted_score - baseline_score
            feature_importance[f'Feature_{feature_idx}'] = importance
        
        return feature_importance
