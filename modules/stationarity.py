import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tests(returns):
    """
    Run ADF and KPSS tests.

    ADF:
        Null hypothesis: the series has a unit root.

    KPSS:
        Null hypothesis: the series is stationary.
    """

    series = pd.Series(returns).dropna()

    if len(series) < 30:
        raise ValueError(
            "Not enough observations for stationarity testing."
        )

    results = []

    # Augmented Dickey-Fuller test
    try:
        adf_result = adfuller(series, autolag="AIC")

        results.append({
            "Test": "Augmented Dickey-Fuller",
            "Test Statistic": adf_result[0],
            "P-value": adf_result[1],
            "Lags Used": adf_result[2],
            "Observations": adf_result[3],
            "Conclusion": (
                "Stationary"
                if adf_result[1] < 0.05
                else "Non-stationary"
            )
        })

    except Exception as error:
        results.append({
            "Test": "Augmented Dickey-Fuller",
            "Test Statistic": None,
            "P-value": None,
            "Lags Used": None,
            "Observations": len(series),
            "Conclusion": f"Test failed: {error}"
        })

    # KPSS test
    try:
        kpss_result = kpss(
            series,
            regression="c",
            nlags="auto"
        )

        results.append({
            "Test": "KPSS",
            "Test Statistic": kpss_result[0],
            "P-value": kpss_result[1],
            "Lags Used": kpss_result[2],
            "Observations": len(series),
            "Conclusion": (
                "Stationary"
                if kpss_result[1] >= 0.05
                else "Non-stationary"
            )
        })

    except Exception as error:
        results.append({
            "Test": "KPSS",
            "Test Statistic": None,
            "P-value": None,
            "Lags Used": None,
            "Observations": len(series),
            "Conclusion": f"Test failed: {error}"
        })

    return pd.DataFrame(results)
