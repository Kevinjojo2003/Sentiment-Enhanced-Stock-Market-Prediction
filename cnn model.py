import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib  # For saving scaler properly

# Function to load and preprocess data
def load_data(ticker):
    data_path = f"data/{ticker}_processed.csv"
    
    # Load CSV
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)  # Drop missing values
    
    # Select all features except 'Close' as input
    features = df.drop(columns=["Close"])
    target = df["Close"]

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Reshape for CNN input
    X = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    y = target.values

    return X, y, scaler

# Get user input for ticker
ticker = input("Enter stock ticker: ").upper()
X, y, scaler = load_data(ticker)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build CNN Model
model = Sequential([
    Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(filters=64, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)  # Output layer for regression
])

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save Model & Scaler
if not os.path.exists("models"):
    os.makedirs("models")

model.save(f"models/{ticker}_cnn.h5")

# Save the entire scaler object properly
joblib.dump(scaler, f"models/{ticker}_scaler.pkl")

print(f"✅ CNN Model for {ticker} Trained & Saved Successfully!")
