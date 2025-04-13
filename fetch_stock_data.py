import yfinance as yf
import pandas as pd
import os
import time

def fetch_stock_data(ticker, period="5y", interval="1d"):
    """
    Fetches historical stock data for a given ticker and saves it to a CSV file.
    """
    try:
        # Create 'data' directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=interval)
        
        if hist_data.empty:
            print(f"No data found for {ticker}.")
            return
        
        # Save historical data
        hist_csv = f"data/{ticker}_historical.csv"
        hist_data.to_csv(hist_csv)
        print(f"Historical data saved: {hist_csv}")

    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")

def fetch_realtime_data(ticker):
    """
    Fetches real-time stock data (1-minute updates) and appends it to a CSV file.
    Ensures no duplicate entries.
    """
    real_time_csv = f"data/{ticker}_realtime.csv"

    while True:
        try:
            # Fetch latest 1-minute data
            stock = yf.Ticker(ticker)
            real_time_data = stock.history(period="1d", interval="1m")

            if real_time_data.empty:
                print(f"No real-time data found for {ticker}. Retrying...")
                time.sleep(60)
                continue

            # Ensure the CSV exists before appending
            if os.path.exists(real_time_csv):
                existing_data = pd.read_csv(real_time_csv, index_col=0, parse_dates=True)

                # Remove duplicates by comparing timestamps
                new_data = real_time_data.loc[~real_time_data.index.isin(existing_data.index)]
                if new_data.empty:
                    print("No new data. Waiting for the next update...")
                else:
                    new_data.to_csv(real_time_csv, mode="a", header=False)
                    print(f"Appended {len(new_data)} new rows to {real_time_csv}")
            else:
                # Save first-time real-time data
                real_time_data.to_csv(real_time_csv)
                print(f"Real-time data saved: {real_time_csv}")

        except Exception as e:
            print(f"Error fetching real-time data for {ticker}: {e}")

        # Wait 1 minute before fetching again
        time.sleep(60)

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    fetch_stock_data(ticker)  # Fetch historical data once
    fetch_realtime_data(ticker)  # Continuously fetch real-time data
