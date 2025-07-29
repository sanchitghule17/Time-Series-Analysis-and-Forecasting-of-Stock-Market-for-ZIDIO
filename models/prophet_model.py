import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

def prophet_forecast(data: pd.DataFrame, stock_name: str, periods: int = 30):
    """
    Forecasts stock prices using Prophet and returns forecast, actual, and Plotly figure.

    Parameters:
    - data: DataFrame with 'date' and 'close' columns.
    - stock_name: Name of the stock.
    - periods: Number of days to forecast.

    Returns:
    - forecast_values: Forecasted prices (array)
    - actual_values: Last real prices from dataset (array)
    - fig: Plotly figure
    """

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Prepare Prophet format
    prophet_df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Extract forecasted values
    forecast_values = forecast[['ds', 'yhat']].tail(periods)['yhat'].values

    # Use last N actual values for comparison (if available)
    if len(df['close']) >= periods:
        actual_values = df['close'].values[-periods:]
    else:
        actual_values = df['close'].values  # fallback

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'].tail(periods), y=forecast['yhat'].tail(periods),
                             mode='lines', name='Forecast'))

    fig.update_layout(
        title=f'Prophet Forecast for {stock_name}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    return forecast_values, actual_values, fig
