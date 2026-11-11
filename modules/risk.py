
import numpy as np
import pandas as pd

from scipy.stats import t


def student_t_risk(result, confidence=0.95):

    if not 0.5 < confidence < 1:
        raise ValueError("Confidence must be between 0.5 and 1.")

    params = result.params

    nu = float(params["nu"])

    if nu <= 2:
        raise ValueError(
            "Student-t degrees of freedom must exceed 2."
        )

    mean = float(result.params["mu"])

    sigma = float(result.conditional_volatility.iloc[-1])

    alpha = 1 - confidence

    raw_quantile = t.ppf(alpha, df=nu)

    scale = np.sqrt((nu - 2) / nu)

    standardized_quantile = raw_quantile * scale

    raw_es = -(
        (nu + raw_quantile ** 2) /
        (nu - 1)
    ) * t.pdf(raw_quantile, df=nu) / alpha

    standardized_es = raw_es * scale

    return {
        "Confidence": confidence,
        "VaR (%)": -(mean + sigma * standardized_quantile),
        "Expected Shortfall (%)": -(mean + sigma * standardized_es)
    }


def historical_risk(returns, confidence=0.95):

    returns = pd.Series(returns).dropna()

    alpha = 1 - confidence

    quantile = returns.quantile(alpha)

    tail = returns[returns <= quantile]

    return {
        "Confidence": confidence,
        "Historical VaR (%)": -float(quantile),
        "Historical ES (%)": -float(tail.mean())
    }
