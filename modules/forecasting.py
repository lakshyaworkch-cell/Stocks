
import numpy as np
import pandas as pd

from modules.analysis import fit_model


def rolling_forecast(
    returns,
    model_name="EGARCH",
    distribution="t",
    window=1000,
    refit_every=1
):

    returns = pd.Series(returns).dropna()

    if len(returns) <= window + 10:
        raise ValueError(
            "Not enough observations for rolling estimation."
        )

    if refit_every < 1:
        raise ValueError("refit_every must be positive.")

    output = []
    current_result = None

    for i in range(window, len(returns)):

        if current_result is None or (
            i - window
        ) % refit_every == 0:

            training = returns.iloc[i-window:i]

            current_result = fit_model(
                training,
                model_name=model_name,
                distribution=distribution
            )

        forecast = current_result.forecast(
            horizon=1,
            reindex=False
        )

        variance = float(
            forecast.variance.iloc[-1, 0]
        )

        mean = float(
            forecast.mean.iloc[-1, 0]
        )

        if variance <= 0 or not np.isfinite(variance):
            raise ValueError("Invalid forecast variance.")

        output.append({
            "Date": returns.index[i],
            "Actual Return": float(returns.iloc[i]),
            "Forecast Mean": mean,
            "Forecast Variance": variance,
            "Forecast Volatility": np.sqrt(variance)
        })

    return pd.DataFrame(output).set_index("Date")


def evaluate_forecasts(forecasts):

    actual = forecasts["Actual Return"].values
    variance = forecasts["Forecast Variance"].values

    if np.any(variance <= 0):
        raise ValueError("Forecast variances must be positive.")

    realized_proxy = actual ** 2

    mse = np.mean(
        (realized_proxy - variance) ** 2
    )

    qlike = np.mean(
        np.log(variance) +
        realized_proxy / variance
    )

    return {
        "MSE": float(mse),
        "QLIKE": float(qlike),
        "Observations": len(actual)
    }

