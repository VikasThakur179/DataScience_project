import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

class StockVisualizer:
    """
    Create interactive visualizations for stock market analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_price_chart(self, symbol: str) -> go.Figure:
        """
        Create an interactive price chart with volume.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Volume chart
        colors = ['red' if self.data['Close'].iloc[i] < self.data['Open'].iloc[i] 
                 else 'green' for i in range(len(self.data))]
        
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_candlestick_chart(self, symbol: str) -> go.Figure:
        """
        Create a candlestick chart.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure(data=go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name=symbol
        ))
        
        fig.update_layout(
            title=f'{symbol} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_prediction_chart(self, data: pd.DataFrame, train_pred: np.ndarray,
                              test_pred: np.ndarray, future_pred: np.ndarray,
                              sequence_length: int, symbol: str) -> go.Figure:
        """
        Create prediction visualization chart.
        
        Args:
            data (pd.DataFrame): Original stock data
            train_pred (np.ndarray): Training predictions
            test_pred (np.ndarray): Testing predictions
            future_pred (np.ndarray): Future predictions
            sequence_length (int): Sequence length used for training
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Original prices
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Actual Price',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Training predictions
        train_dates = data.index[sequence_length:sequence_length + len(train_pred)]
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_pred.flatten(),
            mode='lines',
            name='Training Predictions',
            line=dict(color=self.colors['success'], width=1, dash='dot')
        ))
        
        # Testing predictions
        test_start_idx = sequence_length + len(train_pred)
        test_dates = data.index[test_start_idx:test_start_idx + len(test_pred)]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_pred.flatten(),
            mode='lines',
            name='Testing Predictions',
            line=dict(color=self.colors['warning'], width=2)
        ))
        
        # Future predictions
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(future_pred),
            freq='D'
        )
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred.flatten(),
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color=self.colors['danger'], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'{symbol} LSTM Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified',
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def create_rsi_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create RSI indicator chart.
        
        Args:
            data (pd.DataFrame): Stock data with RSI
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # RSI chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color=self.colors['secondary'])
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} RSI Analysis',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        return fig
    
    def create_macd_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create MACD indicator chart.
        
        Args:
            data (pd.DataFrame): Stock data with MACD
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'MACD'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color=self.colors['secondary'])
            ),
            row=2, col=1
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color=self.colors['danger'])
            ),
            row=2, col=1
        )
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} MACD Analysis',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        return fig
    
    def create_bollinger_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create Bollinger Bands chart.
        
        Args:
            data (pd.DataFrame): Stock data with Bollinger Bands
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Upper band
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='red', dash='dash'),
            fill=None
        ))
        
        # Lower band
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='red', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Middle band (SMA)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            mode='lines',
            name='Middle Band (SMA)',
            line=dict(color='blue')
        ))
        
        # Close price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_moving_averages_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create moving averages chart.
        
        Args:
            data (pd.DataFrame): Stock data with moving averages
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Add available moving averages
        ma_columns = [col for col in data.columns if 'SMA' in col or 'EMA' in col]
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, col in enumerate(ma_columns):
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)])
                ))
        
        fig.update_layout(
            title=f'{symbol} Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            go.Figure: Plotly figure
        """
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            width=800
        )
        
        return fig
    
    def create_price_distribution(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create price distribution histogram.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Price distribution
        fig.add_trace(go.Histogram(
            x=data['Close'],
            name='Price Distribution',
            nbinsx=50,
            opacity=0.7,
            marker_color=self.colors['primary']
        ))
        
        # Add mean line
        mean_price = data['Close'].mean()
        fig.add_vline(
            x=mean_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{symbol} Price Distribution',
            xaxis_title='Price ($)',
            yaxis_title='Frequency',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_volume_analysis(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create volume analysis chart.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume Analysis'),
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Volume with moving average
        volume_ma = data['Volume'].rolling(window=20).mean()
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_ma,
                mode='lines',
                name='Volume MA(20)',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Volume Analysis',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
