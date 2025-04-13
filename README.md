# 📈 AI Stock Market Prediction & Analysis App

This Streamlit-based AI application allows users to predict stock prices, analyze technical indicators, detect chart patterns, and access real-time financial news — all powered by LSTM deep learning models, Gemini chatbot integration, and edge detection via computer vision.

---

## 🚀 Features

### ✅ Real-Time Stock Analysis
- Fetches 1-month, 1-hour interval data using **Yahoo Finance (yfinance)**
- Displays live candlestick charts with **Plotly**

### 📊 Technical Indicators
- **SMA (Simple Moving Average)**
- **EMA (Exponential Moving Average)**
- **RSI (Relative Strength Index)**

### 🧠 LSTM-Based Predictions
- Predicts:
  - **Tomorrow’s closing price**
  - **Next 7 days' closing prices**
- Uses a trained **LSTM deep learning model**

### 🧠 Gemini AI Chatbot
- Ask natural language queries about stocks or companies
- Powered by **Google Generative AI (Gemini)**

### 📰 Financial News
- Live news from **NewsAPI** (top business headlines)

### 🧠 Computer Vision
- Detect chart patterns using **Canny Edge Detection**
- Applies edge detection on price & technical indicator charts

---

## 📂 Project Structure

```bash
.
├── main_app.py                # Streamlit app file
├── models/
│   ├── AAPL_lstm.h5           # Trained LSTM model
│   └── AAPL_scaler.npy        # Scaler for preprocessing input
├── requirements.txt           # Python dependencies
├── README.md                  # This file

⚙️ Setup Instructions
🔧 1. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📦 2. Folder Setup
Create a models/ folder and add:

Trained LSTM model (e.g. AAPL_lstm.h5)

Corresponding MinMaxScaler parameters (e.g. AAPL_scaler.npy)

🔑 3. Add Your API Keys
Gemini (Google Generative AI): Replace YOUR_GEMINI_API_KEY in the code

NewsAPI: Already integrated (you can swap your key if needed)

▶️ Run the App
bash
Copy
Edit
streamlit run main_app.py
🧠 Model Training
To train your own LSTM:

Use lstm_hyperparameter_tuning.py (not included here)

Save the .h5 and .npy files inside the models/ folder

Model expects 60 timesteps with features like OHLC, volume, RSI, etc.

📈 Example Output
📊 Line chart showing 7-day forecast

📋 Table of predicted prices

🧠 Gemini chatbot response

🔍 Canny edge detected chart on price trends



🛡️ Disclaimer
This project is for educational/demo purposes only and should not be used for real financial decisions. Predictions are based on historical patterns and do not guarantee future performance.

