import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.arima_model import arima_forecast
from models.sarima_model import sarima_forecast
from models.prophet_model import prophet_forecast
from models.lstm_model import lstm_forecast

st.set_page_config(page_title="📈 Stock Price Forecasting", layout="wide")
st.title("📊 Stock Price Forecasting Dashboard")

# Load dataset options
DATA_DIR = "cleaned_data"
available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
stock_options = [f.replace('_cleaned.csv', '') for f in available_files]

# Sidebar
selected_stock = st.selectbox("Select Stock", stock_options)
forecast_days = st.slider("Days to Forecast", min_value=10, max_value=60, value=30, step=5)

# Load selected stock data
data_path = os.path.join(DATA_DIR, selected_stock.lower() + "_cleaned.csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Show data
st.write(f"### 📁 Loaded Data for {selected_stock}")
st.dataframe(df.tail())

# Trend Visualization
st.write("### 📈 Stock Price Trend")
st.line_chart(df.set_index('date')['close'])

# Optional indicators
show_ma = st.checkbox("Show Moving Averages")
show_bollinger = st.checkbox("Show Bollinger Bands")

if show_ma or show_bollinger:
    df_plot = df.copy()
    df_plot['MA20'] = df_plot['close'].rolling(window=20).mean()
    df_plot['Upper'] = df_plot['MA20'] + 2 * df_plot['close'].rolling(window=20).std()
    df_plot['Lower'] = df_plot['MA20'] - 2 * df_plot['close'].rolling(window=20).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], name='Close'))
    if show_ma:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['MA20'], name='MA20'))
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Upper'], name='Upper Band', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Lower'], name='Lower Band', line=dict(dash='dot')))
    fig.update_layout(title="Price with MA/Bollinger Bands", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Forecast tabs
tabs = st.tabs(["ARIMA", "SARIMA", "Prophet", "LSTM"])

with tabs[0]:
    st.subheader("ARIMA Forecast")
    forecast, actual, fig = arima_forecast(df, selected_stock, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, forecast)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(actual, forecast):.2f}")

with tabs[1]:
    st.subheader("SARIMA Forecast")
    forecast, actual, fig = sarima_forecast(df, selected_stock, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, forecast)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(actual, forecast):.2f}")

with tabs[2]:
    st.subheader("Prophet Forecast")
    forecast, actual, fig = prophet_forecast(df, selected_stock, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, forecast)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(actual, forecast):.2f}")

with tabs[3]:
    st.subheader("LSTM Forecast")
    forecast, actual, fig = lstm_forecast(df, selected_stock, epochs=20)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, forecast)):.2f}")
    st.metric("MAE", f"{mean_absolute_error(actual, forecast):.2f}")

# Forecast CSV download (uses ARIMA by default)
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=forecast_days),
    'Forecast': forecast
})
st.download_button(
    label="📅 Download Forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f"{selected_stock}_forecast.csv",
    mime='text/csv'
)
