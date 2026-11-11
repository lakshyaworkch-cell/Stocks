import numpy as np
import pandas as pd

from modules.analysis import fit_model


def rolling_forecast(
    returns,
    model_name="EGARCH",
    distribution="t",
    window=750
):
    """
    Generate rolling one-step-ahead variance forecasts.

    At each date:
    1. Fit the model using only the preceding window of returns.
    2. Forecast the next observation's variance.
    3. Move forward by one observation.

    The first forecast is made after `window` observations.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if window < 250:
        raise ValueError(
            "The rolling estimation window must be at least 250."
        )

    if len(returns) <= window:
        raise ValueError(
            "The return series must contain more observations "
            "than the rolling window."
        )

    forecasts = []

    for i in range(window, len(returns)):

        train = returns.iloc[i - window:i]
        actual_return = float(returns.iloc[i])
        forecast_date = returns.index[i]

        try:
            result = fit_model(
                train,
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

            if not np.isfinite(variance) or variance <= 0:
                continue

            forecasts.append({
                "Date": forecast_date,
                "Actual Return": actual_return,
                "Forecast Variance": variance,
                "Forecast Volatility": np.sqrt(variance)
            })

        except Exception:
            # Skip an origin if estimation or forecasting fails.
            continue

    if not forecasts:
        raise RuntimeError(
            "No rolling forecasts were produced. "
            "Try a different model or a longer sample."
        )

    return pd.DataFrame(forecasts).set_index("Date")


def evaluate_forecasts(forecasts):
    """
    Evaluate variance forecasts using MSE and QLIKE.

    Squared returns are used as a noisy proxy for realized variance.
    """

    if forecasts.empty:
        raise ValueError("The forecast DataFrame is empty.")

    required = [
        "Actual Return",
        "Forecast Variance"
    ]

    for column in required:
        if column not in forecasts.columns:
            raise ValueError(
                f"Missing required forecast column: {column}"
            )

    actual_variance = (
        forecasts["Actual Return"].astype(float) ** 2
    )

    forecast_variance = (
        forecasts["Forecast Variance"].astype(float)
    )

    valid = (
        np.isfinite(actual_variance)
        & np.isfinite(forecast_variance)
        & (forecast_variance > 0)
    )

    actual_variance = actual_variance[valid]
    forecast_variance = forecast_variance[valid]

    if len(actual_variance) == 0:
        raise ValueError("No valid forecasts are available for evaluation.")

    mse = np.mean(
        (actual_variance - forecast_variance) ** 2
    )

    qlike = np.mean(
        np.log(forecast_variance)
        + actual_variance / forecast_variance
    )

    return {
        "MSE": float(mse),
        "QLIKE": float(qlike),
        "Forecast Observations": int(len(actual_variance))
    }
