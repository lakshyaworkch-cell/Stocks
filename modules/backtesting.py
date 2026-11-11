import numpy as np
import pandas as pd

from scipy.stats import chi2


def _safe_log(value):
    return np.log(max(float(value), 1e-12))


def var_backtests(returns, var, confidence=0.95):
    """
    Backtest VaR forecasts.

    Parameters
    ----------
    returns : pandas.Series or array
        Actual returns in percentage units.
    var : pandas.Series or array
        Positive VaR loss forecasts in percentage units.
        Must be aligned with returns.
    confidence : float
        VaR confidence level, e.g. 0.95.

    An exception occurs when:
        return < -VaR

    Returns
    -------
    dict
        Backtesting statistics and p-values.
    """

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    returns = pd.Series(returns).astype(float)
    var = pd.Series(var).astype(float)

    if isinstance(returns.index, pd.RangeIndex):
        returns = returns.reset_index(drop=True)

    if isinstance(var.index, pd.RangeIndex):
        var = var.reset_index(drop=True)

    aligned = pd.concat(
        [
            returns.rename("returns"),
            var.rename("var")
        ],
        axis=1
    ).dropna()

    aligned = aligned.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(aligned) < 30:
        raise ValueError(
            "At least 30 aligned observations are required."
        )

    if (aligned["var"] < 0).any():
        raise ValueError(
            "VaR values must be positive loss amounts."
        )

    actual = aligned["returns"].to_numpy()
    predicted_var = aligned["var"].to_numpy()

    violations = (
        actual < -predicted_var
    ).astype(int)

    n = len(violations)
    exceptions = int(violations.sum())

    expected_rate = 1 - confidence
    observed_rate = exceptions / n

    # Kupiec unconditional coverage test
    if exceptions == 0 or exceptions == n:
        lr_uc = 0.0 if exceptions == 0 else np.inf
        p_uc = 1.0 if exceptions == 0 else 0.0
    else:
        likelihood_null = (
            (1 - expected_rate) ** (n - exceptions)
            * expected_rate ** exceptions
        )

        likelihood_alt = (
            (1 - observed_rate) ** (n - exceptions)
            * observed_rate ** exceptions
        )

        lr_uc = -2 * (
            _safe_log(likelihood_null)
            - _safe_log(likelihood_alt)
        )

        p_uc = float(chi2.sf(lr_uc, df=1))

    # Christoffersen independence test
    n00 = n01 = n10 = n11 = 0

    for previous, current in zip(
        violations[:-1],
        violations[1:]
    ):
        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        else:
            n11 += 1

    total_0 = n00 + n01
    total_1 = n10 + n11

    p01 = n01 / total_0 if total_0 > 0 else 0.0
    p11 = n11 / total_1 if total_1 > 0 else 0.0

    total_transitions = n00 + n01 + n10 + n11

    p_all = (
        (n01 + n11) / total_transitions
        if total_transitions > 0
        else 0.0
    )

    log_likelihood_null = (
        (n00 + n10) * _safe_log(1 - p_all)
        + (n01 + n11) * _safe_log(p_all)
    )

    log_likelihood_alt = (
        n00 * _safe_log(1 - p01)
        + n01 * _safe_log(p01)
        + n10 * _safe_log(1 - p11)
        + n11 * _safe_log(p11)
    )

    lr_ind = max(
        0.0,
        -2 * (log_likelihood_null - log_likelihood_alt)
    )

    p_ind = float(chi2.sf(lr_ind, df=1))

    # Conditional coverage test
    lr_cc = lr_uc + lr_ind
    p_cc = float(chi2.sf(lr_cc, df=2))

    return {
        "Observations": int(n),
        "Exceptions": exceptions,
        "Expected Exception Rate": float(expected_rate),
        "Observed Exception Rate": float(observed_rate),
        "Kupiec LR Statistic": float(lr_uc),
        "Kupiec P-value": float(p_uc),
        "Christoffersen Independence LR": float(lr_ind),
        "Christoffersen Independence P-value": float(p_ind),
        "Conditional Coverage LR": float(lr_cc),
        "Conditional Coverage P-value": float(p_cc)
    }
