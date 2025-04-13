import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(ticker):
    """
    Preprocesses sentiment and technical indicator data.
    Merges both datasets and normalizes numeric features.
    Updates 'data/{ticker}_processed.csv' every minute.
    """
    # File paths
    sentiment_csv = f"data/{ticker}_sentiment.csv"
    indicators_csv = f"data/{ticker}_indicators.csv"
    output_csv = f"data/{ticker}_processed.csv"

    while True:
        try:
            # Ensure both files exist
            if not os.path.exists(sentiment_csv) or not os.path.exists(indicators_csv):
                print(f"Waiting for {sentiment_csv} and {indicators_csv} to be available...")
                time.sleep(30)
                continue

            # Load sentiment data
            sentiment_data = pd.read_csv(sentiment_csv)
            if "PublishedAt" in sentiment_data.columns:
                sentiment_data["PublishedAt"] = pd.to_datetime(sentiment_data["PublishedAt"], errors="coerce")
                sentiment_data.dropna(subset=["PublishedAt"], inplace=True)
                sentiment_data.set_index("PublishedAt", inplace=True)
            else:
                print(f"Error: 'PublishedAt' column missing in {sentiment_csv}")
                time.sleep(60)
                continue

            # Resample sentiment data to count occurrences per minute
            sentiment_scores = sentiment_data.resample("min").count()[["Sentiment"]]
            sentiment_scores.rename(columns={"Sentiment": "Sentiment_Count"}, inplace=True)

            # Load indicators data
            indicators_data = pd.read_csv(indicators_csv, index_col=0, parse_dates=True)

            # Merge sentiment with indicators
            processed_data = indicators_data.merge(sentiment_scores, how="left", left_index=True, right_index=True)
            processed_data["Sentiment_Count"] = processed_data["Sentiment_Count"].fillna(0)

            # Fill missing values
            processed_data = processed_data.ffill().bfill()

            # Normalize numeric features
            numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
                            "SMA_50", "SMA_200", "EMA_50", "RSI_14", "MACD", "MACD_Signal",
                            "MACD_Hist", "BB_Upper", "BB_Middle", "BB_Lower", "Sentiment_Count"]
            numeric_cols = [col for col in numeric_cols if col in processed_data.columns]
            scaler = MinMaxScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])

            # Save to CSV
            processed_data.to_csv(output_csv)
            print(f"Processed data updated: {output_csv}")

        except Exception as e:
            print(f"Error in preprocessing for {ticker}: {e}")

        time.sleep(60)  # Update every minute

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    preprocess_data(ticker)
