import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go


def lstm_forecast(data: pd.DataFrame, stock_name: str, epochs: int = 20, look_back: int = 60):
    """
    Forecasts stock prices using LSTM and returns forecast, actual, and Plotly figure.


    Parameters:
    - data: DataFrame with 'date' and 'close' columns.
    - stock_name: Name of the stock.
    - epochs: Number of training epochs.
    - look_back: Number of past days to use for prediction.


    Returns:
    - forecast_values: Forecasted prices (array)
    - actual_values: Last real prices from dataset (array)
    - fig: Plotly figure with forecast and historical data
    """


    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[['date', 'close']]


    # Normalize the closing prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))


    # Create training data
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i, 0])


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)


    # Predict next 30 days
    test_inputs = scaled_data[-look_back:].reshape(1, look_back, 1)
    future_preds = []
    for _ in range(30):
        pred = model.predict(test_inputs, verbose=0)[0][0]
        future_preds.append(pred)
        test_inputs = np.append(test_inputs[:, 1:, :], [[[pred]]], axis=1)


    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    forecast_values = future_preds.flatten()


    # Prepare actual values for comparison
    if len(df['close']) >= 30:
        actual_values = df['close'].values[-30:]
    else:
        actual_values = df['close'].values  # fallback


    # Prepare future dates
    future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.Timedelta(days=1), periods=30)


    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode='lines', name='LSTM Forecast'))


    fig.update_layout(
        title=f"LSTM Forecast for {stock_name}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(x=0, y=1)
    )


    return forecast_values, actual_values, fig