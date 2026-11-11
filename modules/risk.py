import numpy as np
import pandas as pd

from scipy.stats import t


def student_t_risk(result, confidence=0.95):
    """
    Calculate one-period parametric VaR and Expected Shortfall
    using a fitted Student-t EGARCH/GARCH model.

    Returns VaR and ES as positive loss percentages.

    Important:
    - This function requires a Student-t fitted model.
    - The arch package uses a standardized Student-t distribution.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    params = result.params

    if "nu" not in params.index:
        raise ValueError(
            "The fitted model does not contain a Student-t degrees-of-freedom "
            "parameter. Fit the model with distribution='t'."
        )

    if "mu" not in params.index:
        raise ValueError(
            "The fitted model does not contain a constant mean parameter."
        )

    nu = float(params["nu"])
    mean = float(params["mu"])

    if nu <= 2:
        raise ValueError(
            "Student-t degrees of freedom must exceed 2 "
            "for finite variance."
        )

    sigma = float(
        pd.Series(result.conditional_volatility).dropna().iloc[-1]
    )

    alpha = 1 - confidence

    # Standardize the Student-t distribution to unit variance.
    raw_quantile = t.ppf(alpha, df=nu)
    standardization = np.sqrt((nu - 2) / nu)

    standardized_quantile = raw_quantile * standardization

    # Expected shortfall of the standardized Student-t lower tail.
    raw_es = (
        -((nu + raw_quantile ** 2) / (nu - 1))
        * t.pdf(raw_quantile, df=nu)
        / alpha
    )

    standardized_es = raw_es * standardization

    var = -(mean + sigma * standardized_quantile)
    expected_shortfall = -(mean + sigma * standardized_es)

    return {
        "Confidence": confidence,
        "VaR": float(var),
        "ES": float(expected_shortfall),
        "Degrees of Freedom": nu,
        "Conditional Volatility": sigma
    }


def historical_risk(returns, confidence=0.95):
    """
    Calculate historical VaR and Expected Shortfall.

    Returns positive loss percentages.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) < 30:
        raise ValueError(
            "At least 30 observations are required for historical risk."
        )

    alpha = 1 - confidence

    quantile = float(
        returns.quantile(alpha)
    )

    tail_returns = returns[returns <= quantile]

    if tail_returns.empty:
        raise ValueError("Could not calculate historical tail risk.")

    var = -quantile
    expected_shortfall = -float(tail_returns.mean())

    return {
        "Confidence": confidence,
        "VaR": float(var),
        "ES": float(expected_shortfall),
        "Tail Observations": int(len(tail_returns))
    }
