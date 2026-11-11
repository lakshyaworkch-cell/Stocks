
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tests(series):

    series = series.dropna()

    adf = adfuller(
        series,
        regression="c",
        autolag="AIC"
    )

    kpss_result = kpss(
        series,
        regression="c",
        nlags="auto"
    )

    return {
        "ADF Statistic": float(adf[0]),
        "ADF p-value": float(adf[1]),
        "ADF Lags": int(adf[2]),
        "KPSS Statistic": float(kpss_result[0]),
        "KPSS p-value": float(kpss_result[1]),
        "KPSS Lags": int(kpss_result[2])
    }

