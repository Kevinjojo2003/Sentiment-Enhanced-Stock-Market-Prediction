import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# 📦 Function to process and save a single chart as Canny edge
def process_and_save_canny(data, column, title, filename, save_dir="cannyedge", display=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data[column], label=column, color='blue')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    # Save chart to buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)

    # Convert to OpenCV image
    img = Image.open(img_buf).convert("RGB")
    img_cv = np.array(img)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Save to folder
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, edges)

    if display:
        st.subheader(f"{title} - Canny Edge Detection")
        st.image(edges, caption=filename, use_column_width=True)

    return path

# 🧠 Function to process real-time chart & indicators
def apply_and_save_canny_on_all(stock_data, ticker, display=False):
    if stock_data is None or stock_data.empty:
        st.warning("No stock data available for Canny edge processing.")
        return
    
    save_paths = []

    # Apply to Closing Price Chart
    path1 = process_and_save_canny(stock_data, 'Close', f"{ticker} Price Chart", f"{ticker}_price_edges.png", display=display)
    save_paths.append(path1)

    # Apply to SMA
    if 'SMA' in stock_data.columns:
        path2 = process_and_save_canny(stock_data, 'SMA', f"{ticker} SMA (Simple Moving Avg)", f"{ticker}_sma_edges.png", display=display)
        save_paths.append(path2)

    # Apply to EMA
    if 'EMA' in stock_data.columns:
        path3 = process_and_save_canny(stock_data, 'EMA', f"{ticker} EMA (Exp Moving Avg)", f"{ticker}_ema_edges.png", display=display)
        save_paths.append(path3)

    # Apply to RSI
    if 'RSI' in stock_data.columns:
        path4 = process_and_save_canny(stock_data, 'RSI', f"{ticker} RSI (Relative Strength Index)", f"{ticker}_rsi_edges.png", display=display)
        save_paths.append(path4)

    return save_paths
