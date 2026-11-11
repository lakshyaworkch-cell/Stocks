
import numpy as np
import pandas as pd


def validate_prices(prices):
    prices = pd.Series(prices).copy()

    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna()

    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found in price data.")

    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive.")

    if len(prices) < 250:
        raise ValueError(
            "At least 250 price observations are recommended."
        )

    return prices


def calculate_returns(prices):
    prices = validate_prices(prices)

    returns = np.log(prices).diff() * 100

    returns = returns.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if returns.std() < 1e-10:
        raise ValueError("Returns have almost zero variance.")

    return returns


def validate_returns(returns, minimum=250):
    returns = pd.Series(returns).copy()

    returns = pd.to_numeric(
        returns, errors="coerce"
    )

    returns = returns.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if not np.isfinite(returns).all():
        raise ValueError("Non-finite returns detected.")

    if len(returns) < minimum:
        raise ValueError(
            f"At least {minimum} return observations required."
        )

    return returns

