import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Dict, Tuple
import tensorflow as tf

class ModelEvaluator:
    """
    Comprehensive model evaluation and performance metrics.
    """
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # Ensure arrays are flattened
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        y_true_direction = np.diff(y_true) > 0
        y_pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
        
        # Theil's U statistic
        theil_u = self._calculate_theil_u(y_true, y_pred)
        
        # Max error
        max_error = np.max(np.abs(y_true - y_pred))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Directional_Accuracy': directional_accuracy,
            'Theil_U': theil_u,
            'Max_Error': max_error
        }
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Theil's U statistic.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            float: Theil's U statistic
        """
        numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
        denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def plot_training_history(self, history: tf.keras.callbacks.History) -> go.Figure:
        """
        Plot training history.
        
        Args:
            history: Keras training history
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Loss', 'Model Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history['loss'],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color=self.colors['danger'])
                ),
                row=1, col=1
            )
        
        # Metrics plot
        if 'mean_absolute_error' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history['mean_absolute_error'],
                    mode='lines+markers',
                    name='Training MAE',
                    line=dict(color=self.colors['success'])
                ),
                row=1, col=2
            )
        
        if 'val_mean_absolute_error' in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history['val_mean_absolute_error'],
                    mode='lines+markers',
                    name='Validation MAE',
                    line=dict(color=self.colors['warning'])
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        
        fig.update_layout(
            title='Training History',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Create residual analysis plots.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            go.Figure: Plotly figure
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted',
                'Residuals vs Predicted',
                'Residual Distribution',
                'Q-Q Plot'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=y_true,
                mode='markers',
                name='Actual vs Predicted',
                marker=dict(color=self.colors['primary'], opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color=self.colors['warning'], opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Residual distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=30,
                marker_color=self.colors['success'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.colors['info'], opacity=0.6)
            ),
            row=2, col=2
        )
        
        # Q-Q line
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                name='Q-Q Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Actual", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        fig.update_layout(
            title='Residual Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                confidence_level: float = 0.95) -> go.Figure:
        """
        Plot prediction intervals.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            confidence_level (float): Confidence level for intervals
        
        Returns:
            go.Figure: Plotly figure
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        residuals = y_true - y_pred
        
        # Calculate prediction intervals
        std_residuals = np.std(residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_score * std_residuals
        
        upper_bound = y_pred + margin_error
        lower_bound = y_pred - margin_error
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_true))),
            y=y_true,
            mode='lines+markers',
            name='Actual',
            line=dict(color=self.colors['primary'])
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines+markers',
            name='Predicted',
            line=dict(color=self.colors['danger'])
        ))
        
        # Prediction intervals
        fig.add_trace(go.Scatter(
            x=list(range(len(upper_bound))),
            y=upper_bound,
            mode='lines',
            name=f'{confidence_level*100}% Upper Bound',
            line=dict(color='gray', dash='dash'),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(lower_bound))),
            y=lower_bound,
            mode='lines',
            name=f'{confidence_level*100}% Lower Bound',
            line=dict(color='gray', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.2)'
        ))
        
        fig.update_layout(
            title=f'Prediction Intervals ({confidence_level*100}% Confidence)',
            xaxis_title='Time Steps',
            yaxis_title='Value',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def calculate_portfolio_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns (np.ndarray): Portfolio returns
        
        Returns:
            Dict[str, float]: Portfolio metrics
        """
        # Annualized return
        annual_return = np.mean(returns) * 252
        
        # Annualized volatility
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean() * 100
        
        return {
            'Annual_Return': annual_return * 100,
            'Annual_Volatility': annual_volatility * 100,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown * 100,
            'Calmar_Ratio': calmar_ratio,
            'Win_Rate': win_rate
        }
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            str: Evaluation report
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        report = "📊 MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "🎯 ACCURACY METRICS:\n"
        report += f"  • Mean Squared Error (MSE): {metrics['MSE']:.4f}\n"
        report += f"  • Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}\n"
        report += f"  • Mean Absolute Error (MAE): {metrics['MAE']:.4f}\n"
        report += f"  • Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%\n"
        report += f"  • R-squared (R²): {metrics['R²']:.4f}\n\n"
        
        report += "📈 TRADING METRICS:\n"
        report += f"  • Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%\n"
        report += f"  • Theil's U Statistic: {metrics['Theil_U']:.4f}\n"
        report += f"  • Maximum Error: {metrics['Max_Error']:.4f}\n\n"
        
        # Performance interpretation
        report += "🔍 INTERPRETATION:\n"
        
        if metrics['MAPE'] < 5:
            report += "  • Excellent prediction accuracy (MAPE < 5%)\n"
        elif metrics['MAPE'] < 10:
            report += "  • Good prediction accuracy (MAPE < 10%)\n"
        elif metrics['MAPE'] < 20:
            report += "  • Moderate prediction accuracy (MAPE < 20%)\n"
        else:
            report += "  • Poor prediction accuracy (MAPE > 20%)\n"
        
        if metrics['Directional_Accuracy'] > 60:
            report += "  • Strong directional prediction capability\n"
        elif metrics['Directional_Accuracy'] > 50:
            report += "  • Moderate directional prediction capability\n"
        else:
            report += "  • Weak directional prediction capability\n"
        
        if metrics['R²'] > 0.8:
            report += "  • Excellent model fit (R² > 0.8)\n"
        elif metrics['R²'] > 0.6:
            report += "  • Good model fit (R² > 0.6)\n"
        elif metrics['R²'] > 0.4:
            report += "  • Moderate model fit (R² > 0.4)\n"
        else:
            report += "  • Poor model fit (R² < 0.4)\n"
        
        return report
