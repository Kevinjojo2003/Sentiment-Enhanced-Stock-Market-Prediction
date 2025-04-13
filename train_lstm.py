import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(ticker, time_steps=60, forecast_horizon=1):
    """
    Loads processed stock data, scales it, and prepares training sequences.
    Uses all features but predicts 'Close' price.
    """
    data_path = f"data/{ticker}_processed.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None, None, None, None, None, None

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Select all features
    features = df.columns.tolist()
    target_col = "Close"  # Predicting Close price

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_steps - forecast_horizon):
        X.append(scaled_data[i:i + time_steps])  # Use all features
        y.append(scaled_data[i + time_steps + forecast_horizon, features.index(target_col)])  # Predict Close price

    X, y = np.array(X), np.array(y)

    # Split into train & test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, features

def build_lstm_model(input_shape):
    """
    Builds an LSTM model using all stock market features for prediction.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")  # Predict Close price
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def train_lstm(ticker, epochs=100, batch_size=32):
    """
    Trains the LSTM model using all available stock data features.
    """
    X_train, X_test, y_train, y_test, scaler, features = load_data(ticker)

    if X_train is None:
        print("No data to train.")
        return

    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    # Save model & scaler
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{ticker}_lstm.h5")
    np.save(f"models/{ticker}_scaler.npy", scaler)

    print(f"Model saved: models/{ticker}_lstm.h5")
    print(f"Scaler saved: models/{ticker}_scaler.npy")

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    train_lstm(ticker)
