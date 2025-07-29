import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

def arima_forecast(data: pd.DataFrame, stock_name: str, periods: int = 30):
    """
    Fits ARIMA model and returns forecast, actual values, and Plotly plot.

    Parameters:
    - data: DataFrame with 'date' and 'close' columns.
    - stock_name: Name of the stock for labeling.
    - periods: Days into the future to forecast.

    Returns:
    - forecast.values: forecasted prices (array)
    - actual.values: actual recent prices (array)
    - fig: Plotly figure object
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    ts = df['close']

    # Use all data for training (or split if needed)
    stepwise_model = auto_arima(ts, start_p=1, start_q=1,
                                 max_p=5, max_q=5, m=1,
                                 seasonal=False, trace=False,
                                 error_action='ignore', suppress_warnings=True)

    model = ARIMA(ts, order=stepwise_model.order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=periods)
    forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')

    # For evaluation: take last N days from original data to compare
    if len(ts) >= periods:
        actual = ts[-periods:]
    else:
        actual = ts.copy()  # fallback

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast'))
    fig.update_layout(
        title=f"ARIMA Forecast for {stock_name}",
        xaxis_title="Date", yaxis_title="Price", template="plotly_white"
    )

    return forecast.values, actual.values, fig
