
import numpy as np
import pandas as pd

from modules.analysis import fit_model


def rolling_forecast(
    returns,
    model_name="EGARCH",
    distribution="t",
    window=1000
):

    returns = pd.Series(returns).dropna()

    if len(returns) <= window + 10:
        raise ValueError("Insufficient observations.")

    forecasts = []

    for i in range(window, len(returns)):

        training = returns.iloc[i-window:i]

        result = fit_model(
            training,
            model_name=model_name,
            distribution=distribution
        )

        forecast = result.forecast(
            horizon=1,
            reindex=False
        )

        variance = float(
            forecast.variance.iloc[-1, 0]
        )

        mean = float(
            forecast.mean.iloc[-1, 0]
        )

        forecasts.append({
            "Date": returns.index[i],
            "Actual Return": float(returns.iloc[i]),
            "Forecast Mean": mean,
            "Forecast Variance": variance,
            "Forecast Volatility": np.sqrt(variance)
        })

    return pd.DataFrame(forecasts).set_index("Date")
