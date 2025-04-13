import numpy as np
import pandas as pd
import time
import os
import joblib  # For loading the saved scaler
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load CNN Model
def load_cnn_model(ticker):
    model_file = f"models/{ticker}_cnn.h5"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    return tf.keras.models.load_model(model_file, compile=False)  # Don't compile again

# Load Scaler
def load_scaler(ticker):
    scaler_file = f"models/{ticker}_scaler.pkl"
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler not found: {scaler_file}")
    
    return joblib.load(scaler_file)

# Load real-time stock data
def load_data(ticker):
    indicators_path = f"data/{ticker}_indicators.csv"
    sentiment_path = f"data/{ticker}_sentiment.csv"

    if not os.path.exists(indicators_path) or not os.path.exists(sentiment_path):
        print("Error: Data files not found!")
        return None

    stock_data = pd.read_csv(indicators_path, index_col=0, parse_dates=True)
    sentiment_data = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)

    # Aggregate sentiment count per minute
    sentiment_data["Sentiment_Count"] = 1
    sentiment_counts = sentiment_data.resample("T").sum()["Sentiment_Count"]

    # Merge sentiment data with stock indicators
    stock_data = stock_data.join(sentiment_counts, how="left").fillna(0)

    return stock_data

# Compute error metrics
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

# Real-time CNN prediction loop
def real_time_cnn_prediction(ticker):
    print(f"📊 Starting real-time CNN prediction for {ticker}...")

    model = load_cnn_model(ticker)
    scaler = load_scaler(ticker)

    while True:
        stock_data = load_data(ticker)
        if stock_data is None:
            time.sleep(60)
            continue  # Retry after 1 minute

        # Scale the latest data
        features = stock_data.drop(columns=["Close"])
        scaled_features = scaler.transform(features)

        # Reshape for CNN input
        latest_sequence = scaled_features[-1].reshape(1, scaled_features.shape[1], 1)

        # Make prediction
        predicted_price = model.predict(latest_sequence)[0][0]

        # Fix scaling issue: Correctly inverse-transform the prediction
        dummy_input = np.zeros((1, scaled_features.shape[1]))  # Same shape as training features
        dummy_input[0, -1] = predicted_price  # Place predicted value in last column

        predicted_price = scaler.inverse_transform(dummy_input)[0, -1]  # Extract only 'Close' price

        # Get actual latest close price
        actual_price = stock_data["Close"].iloc[-1]

        # Compute error metrics
        mae, rmse, r2 = compute_errors(actual_price, predicted_price)

        print(f"📈 Real-time Prediction: {ticker} -> Predicted Close: {predicted_price:.2f}, Actual Close: {actual_price:.2f}")
        print(f"📊 Error Metrics: MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")

        time.sleep(60)  # Update every 1 minute

# Run real-time prediction
if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    real_time_cnn_prediction(ticker)
