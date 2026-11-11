
import numpy as np
import pandas as pd

from scipy.stats import chi2
from scipy.special import xlogy


def kupiec_test(returns, var):

    returns = np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)

    if len(returns) != len(var) or len(returns) == 0:
        raise ValueError("Returns and VaR must have equal nonzero lengths.")

    if not np.isfinite(returns).all():
        raise ValueError("Returns contain invalid values.")

    if not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("VaR must contain positive finite losses.")

    exceptions = returns < -var

    n = len(exceptions)
    x = int(exceptions.sum())

    expected_rate = np.mean(1 - 0.95) if False else None

    p = 1 - np.mean(var * 0 + 0.95)

    # Estimate the nominal coverage from the supplied
    # VaR series separately when using this function.
    return {
        "Exceptions": x,
        "Observations": n,
        "Exception Rate": x / n
    }


def kupiec_coverage_test(returns, var, confidence=0.95):

    returns = np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)

    if len(returns) != len(var) or len(returns) == 0:
        raise ValueError("Returns and VaR must have equal nonzero lengths.")

    if not 0.5 < confidence < 1:
        raise ValueError("Invalid confidence level.")

    if np.any(var <= 0) or not np.isfinite(var).all():
        raise ValueError("VaR must be positive and finite.")

    exceptions = (returns < -var).astype(int)

    n = len(exceptions)
    x = int(exceptions.sum())

    observed = x / n
    expected = 1 - confidence

    log_likelihood_null = (
        xlogy(x, expected) +
        xlogy(n - x, 1 - expected)
    )

    log_likelihood_alt = (
        xlogy(x, observed) +
        xlogy(n - x, 1 - observed)
    )

    statistic = 2 * (
        log_likelihood_alt - log_likelihood_null
    )

    p_value = chi2.sf(max(statistic, 0), df=1)

    return {
        "Exceptions": x,
        "Expected Exceptions": n * expected,
        "Exception Rate": observed,
        "LR Statistic": float(max(statistic, 0)),
        "P-value": float(p_value)
    }


def christoffersen_independence_test(returns, var):

    returns = np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)

    if len(returns) != len(var) or len(returns) < 2:
        raise ValueError("At least two aligned observations required.")

    if np.any(var <= 0) or not np.isfinite(var).all():
        raise ValueError("Invalid VaR values.")

    hit = (returns < -var).astype(int)

    n00 = int(np.sum((hit[:-1] == 0) & (hit[1:] == 0)))
    n01 = int(np.sum((hit[:-1] == 0) & (hit[1:] == 1)))
    n10 = int(np.sum((hit[:-1] == 1) & (hit[1:] == 0)))
    n11 = int(np.sum((hit[:-1] == 1) & (hit[1:] == 1)))

    p01 = n01 / (n00 + n01) if n00 + n01 else 0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0

    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_independent = (
        xlogy(n00 + n10, 1 - p) +
        xlogy(n01 + n11, p)
    )

    ll_markov = (
        xlogy(n00, 1 - p01) +
        xlogy(n01, p01) +
        xlogy(n10, 1 - p11) +
        xlogy(n11, p11)
    )

    statistic = max(
        0,
        2 * (ll_markov - ll_independent)
    )

    return {
        "LR Independence": float(statistic),
        "Independence P-value": float(
            chi2.sf(statistic, df=1)
        ),
        "Transition 00": n00,
        "Transition 01": n01,
        "Transition 10": n10,
        "Transition 11": n11
    }

