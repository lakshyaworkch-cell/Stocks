import numpy as np
import pandas as pd


def calculate_returns(prices):
    """
    Calculate daily logarithmic returns in percentage units.

    Formula:
        r_t = 100 * ln(P_t / P_(t-1))
    """

    if isinstance(prices, pd.DataFrame):
        if "Close" not in prices.columns:
            raise ValueError("The input DataFrame must contain a Close column.")

        prices = prices["Close"]

    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices.dropna()
    prices = prices[prices > 0]

    if len(prices) < 251:
        raise ValueError(
            "At least 251 valid price observations are required "
            "to calculate a useful return series."
        )

    returns = 100 * np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    returns.name = "Returns"

    if len(returns) < 250:
        raise ValueError(
            "At least 250 valid return observations are required."
        )

    return returns


def validate_returns(returns, minimum_observations=250):
    """
    Validate a return series before model estimation.
    """

    returns = pd.Series(returns).dropna()

    if len(returns) < minimum_observations:
        raise ValueError(
            f"At least {minimum_observations} observations are required."
        )

    if not np.isfinite(returns).all():
        raise ValueError("Returns contain infinite or invalid values.")

    if returns.std() == 0:
        raise ValueError("Returns have zero variance.")

    return returns
