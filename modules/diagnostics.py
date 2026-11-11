import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from scipy.stats import skew, kurtosis


def get_diagnostics(result):
    """
    Calculate residual diagnostic statistics.

    Includes:
    - Ljung-Box tests on standardized residuals
    - Ljung-Box tests on squared standardized residuals
    - ARCH LM test
    - Residual skewness and excess kurtosis
    """

    residuals = pd.Series(result.std_resid).dropna()

    if len(residuals) < 30:
        raise ValueError(
            "Not enough standardized residuals for diagnostics."
        )

    residuals = residuals.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    squared_residuals = residuals ** 2

    diagnostics = {}

    # Ljung-Box tests on standardized residuals
    for lag in [10, 20]:
        safe_lag = min(lag, max(1, len(residuals) // 5))

        lb = acorr_ljungbox(
            residuals,
            lags=[safe_lag],
            return_df=True
        )

        diagnostics[f"Ljung-Box residual ({lag})"] = float(
            lb["lb_stat"].iloc[0]
        )

        diagnostics[f"Ljung-Box residual ({lag}) p-value"] = float(
            lb["lb_pvalue"].iloc[0]
        )

    # Ljung-Box tests on squared standardized residuals
    for lag in [10, 20]:
        safe_lag = min(lag, max(1, len(squared_residuals) // 5))

        lb = acorr_ljungbox(
            squared_residuals,
            lags=[safe_lag],
            return_df=True
        )

        diagnostics[f"Ljung-Box squared ({lag})"] = float(
            lb["lb_stat"].iloc[0]
        )

        diagnostics[f"Ljung-Box squared ({lag}) p-value"] = float(
            lb["lb_pvalue"].iloc[0]
        )

    # ARCH LM test
    arch_lags = min(10, max(1, len(residuals) // 5))

    arch_stat, arch_pvalue, _, _ = het_arch(
        residuals,
        nlags=arch_lags
    )

    diagnostics["ARCH LM statistic"] = float(arch_stat)
    diagnostics["ARCH LM p-value"] = float(arch_pvalue)

    diagnostics["Residual skewness"] = float(skew(residuals))
    diagnostics["Residual excess kurtosis"] = float(
        kurtosis(residuals, fisher=True)
    )

    return diagnostics
