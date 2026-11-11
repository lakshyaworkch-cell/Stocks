
import numpy as np
import pandas as pd

from scipy.stats import t


def student_t_risk(
    result,
    confidence=0.95,
    forecast_variance=None
):
    """
    Student-t parametric VaR and Expected Shortfall.

    If forecast_variance is supplied, it is used for next-period
    risk forecasting. Otherwise, the last fitted conditional
    volatility is used.
    
    VaR and ES are returned as positive loss percentages.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    params = result.params

    if "nu" not in params.index or "mu" not in params.index:
        raise ValueError(
            "A constant-mean Student-t model is required."
        )

    nu = float(params["nu"])
    mean = float(params["mu"])

    if nu <= 2:
        raise ValueError(
            "Student-t degrees of freedom must exceed 2."
        )

    if forecast_variance is None:
        sigma = float(
            pd.Series(
                result.conditional_volatility
            ).dropna().iloc[-1]
        )
    else:
        variance = float(forecast_variance)

        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Forecast variance must be positive.")

        sigma = np.sqrt(variance)

    alpha = 1.0 - confidence

    raw_quantile = t.ppf(alpha, df=nu)
    scale = np.sqrt((nu - 2.0) / nu)

    standardized_quantile = raw_quantile * scale

    raw_es = (
        -((nu + raw_quantile ** 2) / (nu - 1.0))
        * t.pdf(raw_quantile, df=nu)
        / alpha
    )

    standardized_es = raw_es * scale

    var = -(mean + sigma * standardized_quantile)
    es = -(mean + sigma * standardized_es)

    return {
        "Confidence": confidence,
        "VaR": float(var),
        "ES": float(es),
        "Degrees of Freedom": nu,
        "Forecast Volatility": float(sigma)
    }


def historical_risk(returns, confidence=0.95):
    """Historical simulation VaR and Expected Shortfall."""

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) < 30:
        raise ValueError("At least 30 observations are required.")

    alpha = 1.0 - confidence

    quantile = float(returns.quantile(alpha))
    tail = returns[returns <= quantile]

    if tail.empty:
        raise ValueError("Insufficient observations in the tail.")

    return {
        "Confidence": confidence,
        "VaR": float(-quantile),
        "ES": float(-tail.mean()),
        "Tail Observations": int(len(tail))
    }
