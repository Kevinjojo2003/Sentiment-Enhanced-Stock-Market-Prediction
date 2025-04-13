import numpy as np
import pandas as pd
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(ticker):
    indicators_path = f"data/{ticker}_indicators.csv"
    sentiment_path = f"data/{ticker}_sentiment.csv"
    
    if not os.path.exists(indicators_path) or not os.path.exists(sentiment_path):
        print("Error: Data files not found!")
        return None
    
    stock_data = pd.read_csv(indicators_path, index_col=0, parse_dates=True)
    sentiment_data = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    
    sentiment_data["Sentiment_Count"] = 1
    sentiment_counts = sentiment_data.resample("T").sum()["Sentiment_Count"]
    stock_data = stock_data.join(sentiment_counts, how="left").fillna(0)
    
    return stock_data

def plot_heatmap(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Stock Data Correlation Heatmap")
    plt.show()

def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 3])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

past_actuals, past_predictions = [], []

def compute_errors(actual, predicted):
    global past_actuals, past_predictions
    past_actuals.append(actual)
    past_predictions.append(predicted)
    
    if len(past_actuals) > 10:
        past_actuals.pop(0)
        past_predictions.pop(0)
    
    mae = mean_absolute_error([actual], [predicted])
    rmse = np.sqrt(mean_squared_error([actual], [predicted]))
    r2 = r2_score(past_actuals, past_predictions) if len(past_actuals) > 1 else np.nan
    
    return mae, rmse, r2

def real_time_prediction(ticker):
    print(f"📊 Starting real-time LSTM prediction for {ticker}...")
    
    stock_data = load_data(ticker)
    if stock_data is None:
        return
    
    plot_heatmap(stock_data)
    
    scaled_data, scaler = preprocess_data(stock_data)
    X, y = create_sequences(scaled_data)
    
    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    
    while True:
        stock_data = load_data(ticker)
        scaled_data, _ = preprocess_data(stock_data)
        latest_sequence = scaled_data[-10:].reshape(1, 10, scaled_data.shape[1])
        
        predicted_price = model.predict(latest_sequence)[0][0]
        predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0][3]
        
        actual_price = stock_data["Close"].iloc[-1]
        
        mae, rmse, r2 = compute_errors(actual_price, predicted_price)
        
        print(f"📈 Real-time Prediction: {ticker} -> Predicted Close: {predicted_price:.2f}, Actual Close: {actual_price:.2f}")
        print(f"📊 Error Metrics: MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")
        
        time.sleep(60)

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    real_time_prediction(ticker)
