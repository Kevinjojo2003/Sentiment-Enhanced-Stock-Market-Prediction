import pandas as pd
import pandas_ta as ta
import os
import time

def compute_indicators(ticker):
    """
    Computes technical indicators for a given ticker and updates the CSV.
    Reads from both historical and real-time stock data.
    """
    hist_csv = f"data/{ticker}_historical.csv"
    real_time_csv = f"data/{ticker}_realtime.csv"
    indicators_csv = f"data/{ticker}_indicators.csv"

    while True:
        try:
            # Ensure data files exist
            if not os.path.exists(hist_csv) or not os.path.exists(real_time_csv):
                print(f"Waiting for stock data files for {ticker}...")
                time.sleep(30)
                continue

            # Load historical and real-time data
            hist_data = pd.read_csv(hist_csv, index_col=0, parse_dates=True)
            real_time_data = pd.read_csv(real_time_csv, index_col=0, parse_dates=True)

            # Merge historical and real-time data, remove duplicates
            df = pd.concat([hist_data, real_time_data]).drop_duplicates().sort_index()

            # Compute Technical Indicators
            df["SMA_50"] = ta.sma(df["Close"], length=50)  # 50-day SMA
            df["SMA_200"] = ta.sma(df["Close"], length=200)  # 200-day SMA
            df["EMA_50"] = ta.ema(df["Close"], length=50)  # 50-day EMA
            df["RSI_14"] = ta.rsi(df["Close"], length=14)  # RSI (14)
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = ta.macd(df["Close"], fast=12, slow=26, signal=9).T.values
            bbands = ta.bbands(df["Close"], length=20)  # Bollinger Bands
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bbands["BBU_20_2.0"], bbands["BBM_20_2.0"], bbands["BBL_20_2.0"]

            # Save updated data
            df.to_csv(indicators_csv)
            print(f"Indicators updated and saved: {indicators_csv}")

        except Exception as e:
            print(f"Error computing indicators for {ticker}: {e}")

        # Update every minute
        time.sleep(60)

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    compute_indicators(ticker)
