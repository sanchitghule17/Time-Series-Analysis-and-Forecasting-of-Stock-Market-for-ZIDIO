import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

def sarima_forecast(data: pd.DataFrame, stock_name: str, periods: int = 30):
    """
    Fits SARIMA model and returns forecast values, actual values, and a Plotly forecast figure.

    Parameters:
    - data: DataFrame with 'date' and 'close' columns.
    - stock_name: Name of the stock.
    - periods: Number of future days to forecast.

    Returns:
    - forecast_values: Forecasted prices (array)
    - actual_values: Actual last known prices for comparison (array)
    - fig: Plotly forecast figure
    """

    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    ts = df['close']

    # Fit SARIMA model (adjust seasonal order if needed)
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecast future values
    forecast = model_fit.forecast(steps=periods)
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=periods)

    # Use last N actual values if available
    if len(ts) >= periods:
        actual_values = ts.values[-periods:]
    else:
        actual_values = ts.values

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast.values, mode='lines', name='Forecast'))

    fig.update_layout(
        title=f"SARIMA Forecast for {stock_name}",
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    return forecast.values, actual_values, fig
