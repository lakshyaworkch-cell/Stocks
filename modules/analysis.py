
import numpy as np
import pandas as pd

from arch import arch_model
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from scipy.stats import jarque_bera


def fit_egarch(returns):
    model = arch_model(
        returns,
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="StudentsT"
    )

    result = model.fit(
        disp="off",
        update_freq=0
    )

    return result


def get_forecast(result, horizon=10):
    forecast = result.forecast(
        horizon=horizon,
        method="simulation",
        simulations=1000,
        reindex=False
    )

    variance = forecast.variance.iloc[-1]

    volatility = np.sqrt(variance)

    forecast_df = pd.DataFrame({
        "Day": range(1, horizon + 1),
        "Variance": variance.values,
        "Daily Volatility (%)": volatility.values,
        "Annualized Volatility (%)":
            volatility.values * np.sqrt(252)
    })

    return forecast_df


def get_diagnostics(result):
    residuals = result.std_resid.dropna()

    lb = acorr_ljungbox(
        residuals,
        lags=[10, 20],
        return_df=True
    )

    arch_test = het_arch(residuals, nlags=10)

    jb = jarque_bera(residuals)

    diagnostics = {
        "Ljung-Box p-value (10)": lb["lb_pvalue"].iloc[0],
        "Ljung-Box p-value (20)": lb["lb_pvalue"].iloc[1],
        "ARCH LM p-value": arch_test[1],
        "Jarque-Bera statistic": jb.statistic,
        "Jarque-Bera p-value": jb.pvalue
    }

    return diagnostics


def get_parameters(result):
    return pd.DataFrame({
        "Parameter": result.params.index,
        "Estimate": result.params.values,
        "P-value": result.pvalues.values
    })

