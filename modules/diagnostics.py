
import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)


def get_diagnostics(result):

    residuals = pd.Series(
        result.std_resid
    ).dropna()

    squared = residuals ** 2

    lb_returns = acorr_ljungbox(
        residuals,
        lags=[10, 20],
        return_df=True
    )

    lb_squared = acorr_ljungbox(
        squared,
        lags=[10, 20],
        return_df=True
    )

    arch_test = het_arch(
        residuals,
        nlags=10
    )

    return {
        "Ljung-Box residual (10)": float(
            lb_returns["lb_pvalue"].iloc[0]
        ),
        "Ljung-Box residual (20)": float(
            lb_returns["lb_pvalue"].iloc[1]
        ),
        "Ljung-Box squared (10)": float(
            lb_squared["lb_pvalue"].iloc[0]
        ),
        "Ljung-Box squared (20)": float(
            lb_squared["lb_pvalue"].iloc[1]
        ),
        "ARCH LM statistic": float(arch_test[0]),
        "ARCH LM p-value": float(arch_test[1]),
        "Residual skewness": float(residuals.skew()),
        "Residual excess kurtosis": float(residuals.kurtosis())
    }

