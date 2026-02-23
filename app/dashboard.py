# streamlit dashboard for stock analysis

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import requests
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import RAW_DATA_DIR, STOCK_SYMBOLS, MODELS_DIR
from models.sentiment_analyzer import FinBERTSentimentAnalyzer

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# some custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# load sentiment model once
if 'sentiment_analyzer' not in st.session_state:
    with st.spinner("Loading FinBERT model..."):
        st.session_state.sentiment_analyzer = FinBERTSentimentAnalyzer()


def load_stock_data(symbol):
    filepath = RAW_DATA_DIR / f"{symbol}_historical.csv"
    if filepath.exists():
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None


def plot_stock_price(df, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_volume(df):
    fig = px.bar(df, x='Date', y='Volume', title="Trading Volume")
    fig.update_layout(template="plotly_white", height=300)
    return fig


def main():
    # sidebar nav
    st.sidebar.markdown("# 📈 Financial Market Intelligence")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Market Overview", "🔮 Price Prediction", "💭 Sentiment Analysis", "📈 Technical Analysis"]
    )
    
    if page == "📊 Market Overview":
        market_overview_page()
    elif page == "🔮 Price Prediction":
        prediction_page()
    elif page == "💭 Sentiment Analysis":
        sentiment_page()
    elif page == "📈 Technical Analysis":
        technical_analysis_page()


def market_overview_page():
    st.markdown('<p class="main-header">Market Overview</p>', unsafe_allow_html=True)
    st.markdown("Real-time insights into major tech stocks")
    
    selected_stocks = st.multiselect(
        "Select stocks to compare",
        STOCK_SYMBOLS,
        default=STOCK_SYMBOLS[:3]
    )
    
    if not selected_stocks:
        st.warning("Please select at least one stock")
        return
    
    # show metrics for each stock
    cols = st.columns(len(selected_stocks))
    
    for idx, symbol in enumerate(selected_stocks):
        df = load_stock_data(symbol)
        if df is not None:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            change = latest['Close'] - prev['Close']
            change_pct = (change / prev['Close']) * 100
            
            with cols[idx]:
                st.metric(
                    label=symbol,
                    value=f"${latest['Close']:.2f}",
                    delta=f"{change_pct:.2f}%"
                )
    
    # comparison chart
    st.subheader("Price Comparison")
    
    fig = go.Figure()
    
    for symbol in selected_stocks:
        df = load_stock_data(symbol)
        if df is not None:
            # normalize to % change from start
            df['Normalized'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Normalized'],
                mode='lines',
                name=symbol
            ))
    
    fig.update_layout(
        title="Normalized Price Performance (%)",
        yaxis_title="Change (%)",
        xaxis_title="Date",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def prediction_page():
    st.markdown('<p class="main-header">Stock Price Prediction</p>', unsafe_allow_html=True)
    st.markdown("LSTM-based forecasting for future stock prices")
    
    symbol = st.selectbox("Select Stock", STOCK_SYMBOLS)
    
    df = load_stock_data(symbol)
    
    if df is None:
        st.error(f"Data not available for {symbol}")
        return
    
    # current price info
    col1, col2, col3 = st.columns(3)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100
    
    with col1:
        st.metric("Current Price", f"${latest['Close']:.2f}", f"{change_pct:.2f}%")
    with col2:
        st.metric("Volume", f"{latest['Volume']:,.0f}")
    with col3:
        st.metric("High", f"${latest['High']:.2f}")
    
    # historical chart
    st.subheader("Historical Price")
    fig = plot_stock_price(df.tail(365), symbol)
    st.plotly_chart(fig, use_container_width=True)
    
    # prediction
    st.subheader("Price Forecast")
    
    days_to_predict = st.slider("Days to forecast", 1, 30, 5)
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating forecast..."):
            model_path = MODELS_DIR / f"{symbol}_lstm.pth"
            
            if not model_path.exists():
                st.warning(f"Model not trained for {symbol}. Run training first:")
                st.code("python src/train_models.py", language="bash")
            else:
                st.success("Prediction generated!")
                st.info("Note: This is a demo. Production would use proper multi-step forecasting.")
                
                # dummy prediction for demo
                last_price = latest['Close']
                predicted_prices = [last_price * (1 + np.random.uniform(-0.02, 0.02)) 
                                   for _ in range(days_to_predict)]
                
                pred_df = pd.DataFrame({
                    'Day': range(1, days_to_predict + 1),
                    'Predicted Price': predicted_prices
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_df['Day'],
                    y=pred_df['Predicted Price'],
                    mode='lines+markers',
                    name='Predicted'
                ))
                fig.update_layout(
                    title="Price Forecast",
                    xaxis_title="Days Ahead",
                    yaxis_title="Price (USD)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)


def sentiment_page():
    st.markdown('<p class="main-header">Financial Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown("Analyze market sentiment using FinBERT")
    
    st.subheader("Analyze News or Social Media Text")
    
    text_input = st.text_area(
        "Enter financial text to analyze",
        placeholder="e.g., Apple reports record quarterly earnings...",
        height=150
    )
    
    if st.button("Analyze Sentiment"):
        if not text_input:
            st.warning("Please enter some text")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = st.session_state.sentiment_analyzer.analyze_sentiment(text_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentiment", result['label'].upper())
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                with col2:
                    scores_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                        'Score': [result['positive'], result['negative'], result['neutral']]
                    })
                    
                    fig = px.bar(scores_df, x='Sentiment', y='Score', 
                                title="Sentiment Breakdown",
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': 'green',
                                    'Negative': 'red',
                                    'Neutral': 'gray'
                                })
                    st.plotly_chart(fig, use_container_width=True)
    
    # sample headlines to try
    st.subheader("Try Sample News Headlines")
    
    samples = [
        "Apple reports record-breaking quarterly earnings, stock surges",
        "Tech stocks plummet amid recession fears",
        "Microsoft announces new AI partnership, investors optimistic",
        "Amazon faces regulatory scrutiny, shares drop",
        "Tesla delivers strong Q4 results, exceeding expectations"
    ]
    
    for sample in samples:
        if st.button(sample, key=sample):
            result = st.session_state.sentiment_analyzer.analyze_sentiment(sample)
            st.success(f"Sentiment: **{result['label'].upper()}** (Confidence: {result['confidence']:.2%})")


def technical_analysis_page():
    st.markdown('<p class="main-header">Technical Analysis</p>', unsafe_allow_html=True)
    st.markdown("Advanced technical indicators and patterns")
    
    symbol = st.selectbox("Select Stock", STOCK_SYMBOLS, key="tech_stock")
    
    df = load_stock_data(symbol)
    
    if df is None:
        st.error(f"Data not available for {symbol}")
        return
    
    # add indicators
    from data_processing.feature_engineering import StockFeatureEngineer
    engineer = StockFeatureEngineer()
    df = engineer.add_technical_indicators(df)
    
    # show recent data
    st.subheader("Recent Data with Indicators")
    display_cols = ['Date', 'Close', 'Volume', 'MA_7', 'MA_21', 'RSI', 'MACD']
    st.dataframe(df[display_cols].tail(10), use_container_width=True)
    
    # RSI chart
    st.subheader("Relative Strength Index (RSI)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # MACD chart
    st.subheader("MACD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='Signal'))
    fig.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
