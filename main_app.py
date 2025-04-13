import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io

# ========== Configure Gemini ==========
genai.configure(api_key="AIzaSyBLYNHfjHzPhnnUHp4Pq86prmEug2lvkDU")

# ========== Helper: Load LSTM & Scaler ==========
def load_model_and_scaler(ticker):
    model_path = f"models/{ticker}_lstm.h5"
    scaler_path = f"models/{ticker}_scaler.npy"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        model = load_model(model_path)
        scaler = MinMaxScaler()
        scaler_params = np.load(scaler_path, allow_pickle=True)
        scaler.min_, scaler.scale_ = scaler_params
        return model, scaler
    except:
        return None, None

# ========== Helper: Predict Close ==========
def predict_next_price(model, scaler, data):
    if data is None or data.empty:
        return None
    data_scaled = scaler.transform(data.tail(10))
    latest_sequence = data_scaled.reshape(1, 10, data_scaled.shape[1])
    predicted_price = model.predict(latest_sequence)[0][0]
    return predicted_price * scaler.scale_[0] + scaler.min_[0]

# ========== Helper: Predict Next 7 Days ==========
def predict_next_7_days(model, scaler, data, n_days=7):
    if data is None or data.empty:
        return None
    recent_data = data.tail(60).copy()
    inputs = scaler.transform(recent_data)
    predicted = []
    for _ in range(n_days):
        latest_sequence = inputs[-60:].reshape(1, 60, inputs.shape[1])
        next_price = model.predict(latest_sequence)[0][0]
        predicted.append(next_price)
        new_row = np.zeros_like(inputs[-1])
        new_row[0:4] = next_price
        inputs = np.vstack([inputs, new_row])
    predicted_real = np.array(predicted) * scaler.scale_[0] + scaler.min_[0]
    return predicted_real

# ========== Canny Edge Detection ==========
def apply_canny_on_chart(data, title="Chart"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close', color='blue')
    ax.set_title(title)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    img = Image.open(img_buf)
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)
    st.image(edges, caption=f"Canny Edge - {title}", use_column_width=True)

# ========== Gemini Chat ==========
def get_chatbot_response(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# ========== Get Stock Data ==========
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1mo", interval="1h")
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# ========== Financial News ==========
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey=8976d7717346465b990583ed39234793"
    try:
        response = requests.get(url)
        news = response.json()
        return news.get("articles", [])
    except:
        return []

# ========== Streamlit App ==========
st.set_page_config(layout="wide")
st.title("📈 AI Stock Market App - Forecast, News & Charts")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()

# MAIN DISPLAY
if ticker:
    stock_data = get_stock_data(ticker)

    if stock_data is not None and not stock_data.empty:
        st.subheader(f"💰 Current Price for {ticker}: ${stock_data['Close'].iloc[-1]:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"]
        ))
        st.plotly_chart(fig)

        # Technical indicators
        st.subheader("📊 Technical Indicators")
        stock_data["SMA"] = ta.sma(stock_data["Close"], length=10)
        stock_data["EMA"] = ta.ema(stock_data["Close"], length=10)
        stock_data["RSI"] = ta.rsi(stock_data["Close"], length=14)

        st.write(f"SMA(10): {stock_data['SMA'].iloc[-1]:.2f}")
        st.write(f"EMA(10): {stock_data['EMA'].iloc[-1]:.2f}")
        st.write(f"RSI(14): {stock_data['RSI'].iloc[-1]:.2f}")

        st.markdown("**Explanation:** SMA = Simple Moving Avg, EMA = Weighted Avg, RSI = Momentum indicator")

        if st.button("🔍 Canny Edge on Price Chart"):
            apply_canny_on_chart(stock_data, "Close Price")

        if st.button("📐 Canny Edge on Technical Indicators"):
            for col in ["SMA", "EMA", "RSI"]:
                df_temp = pd.DataFrame({col: stock_data[col]}).dropna()
                df_temp["Close"] = df_temp[col]
                apply_canny_on_chart(df_temp, f"{col} Edge Detection")

        st.sidebar.download_button("⬇️ Download CSV", stock_data.to_csv().encode("utf-8"), file_name=f"{ticker}_data.csv")

# ========== Prediction Buttons ==========
if st.sidebar.button("🧠 Predict Prices"):
    model, scaler = load_model_and_scaler(ticker)
    stock_data = get_stock_data(ticker)

    if model and scaler and stock_data is not None:
        # Predict tomorrow
        pred = predict_next_price(model, scaler, stock_data)
        st.success(f"📊 Predicted Close Price (Tomorrow): ${pred:.2f}")

        # Predict next 7 days
        next_7 = predict_next_7_days(model, scaler, stock_data)
        dates = pd.date_range(start=pd.Timestamp.today(), periods=7)

        st.subheader("📅 Predicted Close Prices - Next 7 Days")
        df_forecast = pd.DataFrame({
            "Date": dates.date,
            "Predicted Close ($)": next_7.round(2)
        })
        st.table(df_forecast)
        st.line_chart(df_forecast.set_index("Date"))

    else:
        fallback = stock_data["Close"].tail(7) * np.random.uniform(0.98, 1.02, size=7)
        dates = pd.date_range(start=pd.Timestamp.today(), periods=7)
        st.subheader("📅 Fallback Estimated Prices (Model Unavailable)")
        df_fallback = pd.DataFrame({
            "Date": dates.date,
            "Estimated Close ($)": fallback.round(2)
        })
        st.table(df_fallback)
        st.line_chart(df_fallback.set_index("Date"))

# ========== Gemini Chat ==========
st.sidebar.header("💬 Gemini Chat")
query = st.sidebar.text_area("Ask anything about this company or stock:")
if st.sidebar.button("Ask Gemini"):
    if query:
        answer = get_chatbot_response(query)
        st.markdown(f"**Gemini:** {answer}")

# ========== News Section ==========
st.sidebar.header("📰 Financial News")
news_items = fetch_news()
if news_items:
    for article in news_items[:5]:
        st.markdown(f"**{article['title']}**")
        st.markdown(article.get("description", ""))
        st.markdown(f"[Read more]({article['url']})")
        st.markdown("---")
else:
    st.write("No news available.")
