import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.data_fetcher import StockDataFetcher
from modules.technical_indicators import TechnicalIndicators
from modules.lstm_model import LSTMPredictor
from modules.visualizations import StockVisualizer
from modules.model_evaluation import ModelEvaluator
from utils.preprocessing import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🚀 Stock Market Prediction Dashboard")
st.markdown("### AI-Powered Stock Analysis with LSTM Neural Networks")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Sidebar configuration
st.sidebar.header("📊 Configuration Panel")

# Stock symbol selection
popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
stock_symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    options=popular_stocks + ['Custom'],
    index=0
)

if stock_symbol == 'Custom':
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()

# Time period selection
period_options = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=list(period_options.keys()),
    index=3
)

period = period_options[selected_period]

# Model parameters
st.sidebar.subheader("🧠 LSTM Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length (days)", 10, 100, 60)
prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 128)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)

# Technical indicators selection
st.sidebar.subheader("📈 Technical Indicators")
use_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)", True)
use_macd = st.sidebar.checkbox("MACD", True)
use_bollinger = st.sidebar.checkbox("Bollinger Bands", True)
use_sma = st.sidebar.checkbox("Simple Moving Average", True)
use_ema = st.sidebar.checkbox("Exponential Moving Average", True)

# Initialize components
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period):
    fetcher = StockDataFetcher()
    return fetcher.get_stock_data(symbol, period)

# Main application logic
try:
    # Fetch stock data
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        stock_data = fetch_stock_data(stock_symbol, period)
        st.session_state.stock_data = stock_data
    
    if stock_data is not None and not stock_data.empty:
        st.success(f"Successfully fetched {len(stock_data)} days of data for {stock_symbol}")
        
        # Display basic stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
            st.metric("Daily Change", f"${price_change:.2f}", f"{change_pct:.2f}%")
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Volatility (Annual)", f"{volatility:.2f}%")
        
        # Calculate technical indicators
        indicators = TechnicalIndicators(stock_data)
        
        if use_rsi:
            stock_data['RSI'] = indicators.calculate_rsi()
        if use_macd:
            macd_data = indicators.calculate_macd()
            stock_data = pd.concat([stock_data, macd_data], axis=1)
        if use_bollinger:
            bollinger_data = indicators.calculate_bollinger_bands()
            stock_data = pd.concat([stock_data, bollinger_data], axis=1)
        if use_sma:
            stock_data['SMA_20'] = indicators.calculate_sma(20)
            stock_data['SMA_50'] = indicators.calculate_sma(50)
        if use_ema:
            stock_data['EMA_12'] = indicators.calculate_ema(12)
            stock_data['EMA_26'] = indicators.calculate_ema(26)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Price Analysis", 
            "🤖 LSTM Prediction", 
            "📈 Technical Indicators", 
            "📉 Model Performance", 
            "🔍 Feature Analysis"
        ])
        
        with tab1:
            st.subheader("Historical Price Analysis")
            visualizer = StockVisualizer(stock_data)
            
            # Price chart with volume
            price_fig = visualizer.create_price_chart(stock_symbol)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Candlestick chart
            candle_fig = visualizer.create_candlestick_chart(stock_symbol)
            st.plotly_chart(candle_fig, use_container_width=True)
        
        with tab2:
            st.subheader("LSTM Neural Network Prediction")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🚀 Train Model & Predict", type="primary"):
                    with st.spinner("Training LSTM model... This may take a few minutes."):
                        # Prepare data for LSTM
                        preprocessor = DataPreprocessor()
                        processed_data = preprocessor.prepare_lstm_data(
                            stock_data, 
                            sequence_length=sequence_length,
                            target_column='Close'
                        )
                        
                        if processed_data is not None:
                            X_train, X_test, y_train, y_test, scaler = processed_data
                            
                            # Train LSTM model
                            lstm_model = LSTMPredictor(
                                sequence_length=sequence_length,
                                n_features=X_train.shape[2],
                                lstm_units=lstm_units
                            )
                            
                            model, history = lstm_model.train_model(
                                X_train, y_train, 
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=32
                            )
                            
                            # Make predictions
                            train_predictions = lstm_model.predict(X_train)
                            test_predictions = lstm_model.predict(X_test)
                            
                            # Future predictions
                            future_predictions = lstm_model.predict_future(
                                np.array(stock_data['Close'].values), 
                                scaler, 
                                days=prediction_days
                            )
                            
                            st.session_state.model_trained = True
                            st.session_state.predictions = {
                                'train': train_predictions,
                                'test': test_predictions,
                                'future': future_predictions,
                                'scaler': scaler,
                                'history': history,
                                'y_train': y_train,
                                'y_test': y_test
                            }
                            
                            st.success("Model trained successfully!")
                        else:
                            st.error("Failed to prepare data for LSTM training.")
            
            with col1:
                if st.session_state.model_trained and st.session_state.predictions:
                    pred_data = st.session_state.predictions
                    
                    # Create prediction visualization
                    pred_fig = visualizer.create_prediction_chart(
                        stock_data,
                        pred_data['train'],
                        pred_data['test'],
                        pred_data['future'],
                        sequence_length,
                        stock_symbol
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Display future predictions
                    st.subheader("📅 Future Price Predictions")
                    future_dates = pd.date_range(
                        start=stock_data.index[-1] + pd.Timedelta(days=1),
                        periods=prediction_days,
                        freq='D'
                    )
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': pred_data['future'].flatten()
                    })
                    
                    st.dataframe(future_df, use_container_width=True)
                else:
                    st.info("Click 'Train Model & Predict' to generate LSTM predictions.")
        
        with tab3:
            st.subheader("Technical Indicators Dashboard")
            
            # RSI Chart
            if use_rsi and 'RSI' in stock_data.columns:
                rsi_fig = visualizer.create_rsi_chart(stock_data, stock_symbol)
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            # MACD Chart
            if use_macd and 'MACD' in stock_data.columns:
                macd_fig = visualizer.create_macd_chart(stock_data, stock_symbol)
                st.plotly_chart(macd_fig, use_container_width=True)
            
            # Bollinger Bands
            if use_bollinger and 'BB_Upper' in stock_data.columns:
                bb_fig = visualizer.create_bollinger_chart(stock_data, stock_symbol)
                st.plotly_chart(bb_fig, use_container_width=True)
            
            # Moving Averages
            if (use_sma or use_ema) and any(col in stock_data.columns for col in ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']):
                ma_fig = visualizer.create_moving_averages_chart(stock_data, stock_symbol)
                st.plotly_chart(ma_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Model Performance Metrics")
            
            if st.session_state.model_trained and st.session_state.predictions:
                evaluator = ModelEvaluator()
                pred_data = st.session_state.predictions
                
                # Calculate metrics
                train_metrics = evaluator.calculate_metrics(
                    pred_data['y_train'], 
                    pred_data['train']
                )
                test_metrics = evaluator.calculate_metrics(
                    pred_data['y_test'], 
                    pred_data['test']
                )
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Training Metrics")
                    for metric, value in train_metrics.items():
                        st.metric(metric, f"{value:.4f}")
                
                with col2:
                    st.subheader("Testing Metrics")
                    for metric, value in test_metrics.items():
                        st.metric(metric, f"{value:.4f}")
                
                # Training history
                if 'history' in pred_data:
                    history_fig = evaluator.plot_training_history(pred_data['history'])
                    st.plotly_chart(history_fig, use_container_width=True)
                
                # Residual analysis
                residual_fig = evaluator.plot_residual_analysis(
                    pred_data['y_test'], 
                    pred_data['test']
                )
                st.plotly_chart(residual_fig, use_container_width=True)
            else:
                st.info("Train the LSTM model to see performance metrics.")
        
        with tab5:
            st.subheader("Feature Importance & Analysis")
            
            if st.session_state.model_trained:
                # Correlation matrix
                correlation_fig = visualizer.create_correlation_matrix(stock_data)
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # Feature statistics
                st.subheader("📊 Feature Statistics")
                feature_stats = stock_data.describe()
                st.dataframe(feature_stats, use_container_width=True)
                
                # Price distribution
                price_dist_fig = visualizer.create_price_distribution(stock_data, stock_symbol)
                st.plotly_chart(price_dist_fig, use_container_width=True)
            else:
                st.info("Train the model to see detailed feature analysis.")
    
    else:
        st.error(f"Failed to fetch data for {stock_symbol}. Please check the symbol and try again.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try refreshing the page or selecting a different stock symbol.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, TensorFlow, and yfinance</p>
    </div>
    """, 
    unsafe_allow_html=True
)
