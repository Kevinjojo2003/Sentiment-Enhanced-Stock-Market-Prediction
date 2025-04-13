import requests
import pandas as pd
import os
import time
from transformers import pipeline

# Replace with your NewsAPI key
NEWS_API_KEY = "463f65548cd9492ea77cbc061fc501e3"

# Load Hugging Face sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_news_sentiment(ticker):
    """
    Fetches latest news for the stock ticker and performs sentiment analysis.
    Saves results in a CSV file and updates every 5 minutes.
    """
    sentiment_csv = f"data/{ticker}_sentiment.csv"

    while True:
        try:
            # Create 'data' folder if not exists
            if not os.path.exists("data"):
                os.makedirs("data")

            # Fetch news related to the stock ticker
            url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            news_data = response.json()

            if "articles" not in news_data:
                print(f"Error fetching news for {ticker}: {news_data.get('message', 'Unknown error')}")
                time.sleep(300)  # Wait 5 minutes before retrying
                continue

            articles = news_data["articles"]
            if not articles:
                print(f"No news found for {ticker}. Retrying in 5 minutes...")
                time.sleep(300)
                continue

            # Extract relevant fields
            news_list = []
            for article in articles[:10]:  # Limit to 10 latest articles
                title = article["title"]
                description = article["description"]
                url = article["url"]
                published_at = article["publishedAt"]  # Extract date

                # Convert publishedAt to datetime format
                published_at = pd.to_datetime(published_at)

                # Perform sentiment analysis
                sentiment_result = sentiment_analyzer(title)[0]
                sentiment = sentiment_result["label"]
                confidence = sentiment_result["score"]

                news_list.append([published_at, title, description, sentiment, confidence, url])

            # Convert to DataFrame
            df = pd.DataFrame(news_list, columns=["PublishedAt", "Title", "Description", "Sentiment", "Confidence", "URL"])

            # Save or append to CSV
            if os.path.exists(sentiment_csv):
                existing_df = pd.read_csv(sentiment_csv, parse_dates=["PublishedAt"])
                df = pd.concat([existing_df, df]).drop_duplicates(subset=["Title"]).reset_index(drop=True)

            df.to_csv(sentiment_csv, index=False)
            print(f"Sentiment analysis updated: {sentiment_csv}")

        except Exception as e:
            print(f"Error in sentiment analysis for {ticker}: {e}")

        time.sleep(300)  # Wait 5 minutes before fetching again

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    fetch_news_sentiment(ticker)
