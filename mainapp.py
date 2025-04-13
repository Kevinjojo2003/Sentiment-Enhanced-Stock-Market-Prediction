import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import finnhub
from datetime import datetime, timedelta

# API Keys
ALPHA_VANTAGE_API_KEY = "QV6ZUTOC3GE2JXWM"
MEDIASTACK_API_KEY = "e56268ec21362ed46e6bddae3798ec27"
GEMINI_API_KEY = "AIzaSyDInX6FBZ0n_Fno0nijRH0cd0baLKTSxR0"
FINNHUB_API_KEY = "c1h2h3h4h5h6h7h8"

# Configure APIs
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Fetch Stock Data
def get_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Predict Next 7 Days
def predict_next_7_days(data):
    try:
        last_price = data['Close'].iloc[-1]
        date_range = [datetime.today() + timedelta(days=i) for i in range(1, 8)]
        predictions = pd.DataFrame({
            'Date': date_range,
            'Predicted Open': np.round(last_price * (1 + np.random.uniform(-0.01, 0.01, 7)), 2),
            'Predicted Close': np.round(last_price * (1 + np.random.uniform(-0.01, 0.01, 7)), 2)
        })
        return predictions
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return pd.DataFrame()

# Fetch Technical Indicators
def get_technical_indicators(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Bollinger High'] = df['SMA'] + (df['Close'].rolling(20).std() * 2)
        df['Bollinger Low'] = df['SMA'] - (df['Close'].rolling(20).std() * 2)
        return df
    except Exception as e:
        st.error(f"Error fetching technical indicators: {e}")
        return pd.DataFrame()

# Fetch Stock News
def get_stock_news(symbol):
    try:
        url = f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_API_KEY}&keywords={symbol}&countries=us,gb,de"
        response = requests.get(url).json()
        return response.get("data", [])
    except Exception as e:
        st.warning("No news available.")
        return []

# Streamlit UI Setup
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

tabs = st.tabs(["📈 Stock Overview", "📊 Technical Indicators", "📰 News & Market Updates", "📉 Predictions", "💬 Chatbot"])

with tabs[0]:
    st.title("📈 Stock Market Dashboard")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "5y"])
    if st.button("Fetch Data"):
        data = get_stock_data(ticker, period)
        if not data.empty:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            fig = go.Figure(data=[go.Candlestick(
                x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]
            )])
            fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)
            
            # Download Button for Historical Data
            csv = data.to_csv().encode('utf-8')
            st.download_button(label="Download Historical Data", data=csv, file_name=f"{ticker}_historical_data.csv", mime='text/csv')

with tabs[1]:
    st.title("📊 Technical Indicators")
    if ticker:
        indicators = get_technical_indicators(ticker)
        if not indicators.empty:
            for col in ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger High', 'Bollinger Low']:
                st.subheader(col)
                st.line_chart(indicators[['Close', col]])
                st.write(f"Latest {col} Value: {indicators[col].iloc[-1]:.2f}")

with tabs[2]:
    st.title("📰 News & Market Updates")
    if ticker:
        news = get_stock_news(ticker)
        if news:
            for article in news:
                st.subheader(article.get("title", "No Title"))
                st.write(article.get("description", "No Description"))
                st.write(f"[Read more]({article.get('url', '#')})")

with tabs[3]:
    st.title("📉 7-Day Stock Price Predictions")
    if st.button("Predict Next 7 Days"):
        data = get_stock_data(ticker, "3mo")
        if not data.empty:
            predictions = predict_next_7_days(data)
            st.line_chart(predictions.set_index("Date"))
            st.write(predictions)  # Display predictions
            actual = data.tail(7).reset_index()[['Date', 'Open', 'Close']]
            actual.rename(columns={'Open': 'Actual Open', 'Close': 'Actual Close'}, inplace=True)
            merged = pd.merge(predictions, actual, on='Date', how='left')
            st.line_chart(merged.set_index("Date"))

with tabs[4]:
    st.title("💬 Gemini AI Chatbot")
    user_input = st.text_input("Ask anything about the stock market:")
    if user_input:
        response = "This is a mock response from Gemini chatbot."
        st.write(response)
