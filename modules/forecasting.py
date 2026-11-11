
import numpy as np
import pandas as pd

from modules.analysis import fit_model


def rolling_forecast(
    returns,
    model_name="EGARCH",
    distribution="t",
    window=750
):
    returns = pd.Series(returns).dropna().astype(float)

    if window < 250:
        raise ValueError("Window must be at least 250 observations.")

    if len(returns) <= window:
        raise ValueError(
            "You need more observations than the rolling window."
        )

    forecasts = []

    for i in range(window, len(returns)):
        train = returns.iloc[i - window:i]

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

            variance = float(forecast.variance.iloc[-1, 0])

            if not np.isfinite(variance) or variance <= 0:
                continue

            forecasts.append({
                "Date": returns.index[i],
                "Actual Return": float(returns.iloc[i]),
                "Forecast Variance": variance,
                "Forecast Volatility": np.sqrt(variance)
            })

        except Exception:
            continue

    if not forecasts:
        raise RuntimeError("No valid rolling forecasts were generated.")

    return pd.DataFrame(forecasts).set_index("Date")


def evaluate_forecasts(forecasts):
    if forecasts.empty:
        raise ValueError("No forecasts available.")

    actual = forecasts["Actual Return"].astype(float) ** 2
    predicted = forecasts["Forecast Variance"].astype(float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (predicted > 0)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid forecasts to evaluate.")

    mse = np.mean((actual - predicted) ** 2)

    qlike = np.mean(
        np.log(predicted) + actual / predicted
    )

    return {
        "MSE": float(mse),
        "QLIKE": float(qlike),
        "Forecast Observations": int(len(actual))
    }
