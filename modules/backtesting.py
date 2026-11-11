
import numpy as np
import pandas as pd

from scipy.stats import chi2
from scipy.special import xlogy


def _lr_statistic(log_likelihood_null, log_likelihood_alt):
    """Calculate a non-negative likelihood-ratio statistic."""
    return max(
        0.0,
        2.0 * (log_likelihood_alt - log_likelihood_null)
    )


def var_backtests(returns, var, confidence=0.95):
    """
    Kupiec unconditional coverage test,
    Christoffersen independence test,
    and conditional coverage test.

    returns: actual returns in percentage units.
    var: positive VaR loss forecasts in percentage units.
    Both inputs must be chronologically aligned.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    returns = pd.Series(returns, dtype=float)
    var = pd.Series(var, dtype=float)

    if returns.index.equals(var.index):
        aligned = pd.DataFrame({
            "returns": returns,
            "var": var
        }).dropna()
    else:
        aligned = pd.concat(
            [returns.rename("returns"), var.rename("var")],
            axis=1,
            join="inner"
        ).dropna()

    aligned = aligned.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(aligned) < 30:
        raise ValueError("At least 30 aligned observations are required.")

    if (aligned["var"] <= 0).any():
        raise ValueError("VaR forecasts must be positive loss amounts.")

    actual = aligned["returns"].to_numpy()
    predicted_var = aligned["var"].to_numpy()

    # A violation occurs when actual loss exceeds predicted VaR.
    exceptions = (
        actual < -predicted_var
    ).astype(int)

    n = len(exceptions)
    x = int(exceptions.sum())

    p0 = 1.0 - confidence
    phat = x / n

    # 1. Kupiec unconditional coverage
    ll_null = (
        xlogy(x, p0)
        + xlogy(n - x, 1.0 - p0)
    )

    ll_alt = (
        xlogy(x, phat)
        + xlogy(n - x, 1.0 - phat)
    )

    lr_uc = _lr_statistic(ll_null, ll_alt)
    p_uc = float(chi2.sf(lr_uc, df=1))

    # 2. Christoffersen independence
    previous = exceptions[:-1]
    current = exceptions[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    transitions = len(current)

    total_0 = n00 + n01
    total_1 = n10 + n11

    total_exceptions = n01 + n11
    p_all = total_exceptions / transitions

    p01 = n01 / total_0 if total_0 else 0.0
    p11 = n11 / total_1 if total_1 else 0.0

    ll_ind_null = (
        xlogy(total_exceptions, p_all)
        + xlogy(transitions - total_exceptions, 1.0 - p_all)
    )

    ll_ind_alt = (
        xlogy(n01, p01)
        + xlogy(n00, 1.0 - p01)
        + xlogy(n11, p11)
        + xlogy(n10, 1.0 - p11)
    )

    lr_ind = _lr_statistic(ll_ind_null, ll_ind_alt)
    p_ind = float(chi2.sf(lr_ind, df=1))

    # 3. Conditional coverage
    lr_cc = lr_uc + lr_ind
    p_cc = float(chi2.sf(lr_cc, df=2))

    return {
        "Observations": n,
        "Exceptions": x,
        "Expected Exception Rate": p0,
        "Observed Exception Rate": phat,
        "Kupiec LR Statistic": lr_uc,
        "Kupiec P-value": p_uc,
        "Christoffersen Independence LR": lr_ind,
        "Christoffersen Independence P-value": p_ind,
        "Conditional Coverage LR": lr_cc,
        "Conditional Coverage P-value": p_cc,
        "Kupiec Result": (
            "Do not reject coverage"
            if p_uc >= 0.05 else "Reject coverage"
        ),
        "Independence Result": (
            "Do not reject independence"
            if p_ind >= 0.05 else "Reject independence"
        ),
        "Conditional Coverage Result": (
            "Do not reject conditional coverage"
            if p_cc >= 0.05 else "Reject conditional coverage"
        )
    }
