import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
sns.set_style("whitegrid")

st.title("📈 LSTM-Based Stock Price Prediction App")

# --------------------- Sidebar UI ---------------------
st.sidebar.header("Select a Stock")
stocks = {
    "Tata Steel": "TATASTEEL.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS"
}

selected_stock = st.sidebar.selectbox("Choose a company:", list(stocks.keys()))

# Model Parameters
future_days = st.sidebar.slider("Forecast Days", 5, 30, 10)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

# --------------------- Data Fetching ---------------------
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

ticker = stocks[selected_stock]
st.write(f"### Analyzing Stock: {selected_stock} ({ticker})")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

df = load_data(ticker, start, end)
st.dataframe(df.tail())

# --------------------- Data Visualization ---------------------
st.subheader("Closing Price History")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["Close"])
ax.set_title(f"{selected_stock} Closing Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
st.pyplot(fig)

# --------------------- Moving Averages ---------------------
ma_days = [10, 20, 50]
for ma in ma_days:
    df[f"MA_{ma}"] = df["Close"].rolling(ma).mean()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["Close"], label="Close", linewidth=2)
for ma in ma_days:
    ax.plot(df[f"MA_{ma}"], label=f"{ma}-Day MA")
ax.legend()
ax.set_title(f"{selected_stock} Moving Averages")
st.pyplot(fig)

# --------------------- Daily Returns ---------------------
df["Daily Return"] = df["Close"].pct_change()
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df["Daily Return"].dropna(), bins=50, kde=True, color='orange', ax=ax)
ax.set_title(f"{selected_stock} Daily Returns Distribution")
st.pyplot(fig)

# --------------------- LSTM Model ---------------------
data = df[["Close"]].values
training_data_len = int(np.ceil(len(data) * 0.85))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_data = scaled_data[:training_data_len]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

with st.spinner("⏳ Training LSTM Model..."):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)

# --------------------- Predictions ---------------------
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = data[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
mae = mean_absolute_error(y_test, predictions)

train = df[:training_data_len]
valid = df[training_data_len:].copy()
valid["Predictions"] = predictions

st.subheader("Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")

# Plot Predictions
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(train["Close"], label="Train")
ax.plot(valid["Close"], label="Actual")
ax.plot(valid["Predictions"], label="Predicted")
ax.legend()
ax.set_title(f"{selected_stock} Actual vs Predicted Prices")
st.pyplot(fig)

# --------------------- Future Forecast ---------------------
last_60_days = scaled_data[-60:]
future_predictions = []
current_batch = last_60_days.reshape(1, 60, 1)

for i in range(future_days):
    next_pred = model.predict(current_batch)[0]
    future_predictions.append(next_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)

future_predictions = scaler.inverse_transform(future_predictions)
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=["Predicted Close"])

st.subheader(f"{future_days}-Day Forecast for {selected_stock}")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["Close"], label="Historical Close")
ax.plot(future_df["Predicted Close"], label="Future Predicted", linestyle="--", marker="o")
ax.legend()
st.pyplot(fig)

st.dataframe(future_df)
