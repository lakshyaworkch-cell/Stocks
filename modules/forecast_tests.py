
import numpy as np

from scipy.stats import t as student_t


def diebold_mariano_test(
    actual,
    forecast_a,
    forecast_b,
    horizon=1,
    loss="qlike",
    bandwidth=None
):
    """
    Compare two variance forecasts using a Diebold-Mariano test.

    actual:
        Realized variance proxy, normally squared returns.

    forecast_a and forecast_b:
        Positive variance forecasts aligned with actual.

    loss:
        'qlike' or 'mse'.

    Uses a Bartlett-weighted HAC estimate of the variance
    of the loss differential.
    """

    actual = np.asarray(actual, dtype=float)
    forecast_a = np.asarray(forecast_a, dtype=float)
    forecast_b = np.asarray(forecast_b, dtype=float)

    if not (
        len(actual) == len(forecast_a) == len(forecast_b)
    ):
        raise ValueError("All forecast arrays must have equal length.")

    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")

    valid = (
        np.isfinite(actual)
        & np.isfinite(forecast_a)
        & np.isfinite(forecast_b)
        & (actual >= 0)
        & (forecast_a > 0)
        & (forecast_b > 0)
    )

    actual = actual[valid]
    forecast_a = forecast_a[valid]
    forecast_b = forecast_b[valid]

    n = len(actual)

    if n < 30:
        raise ValueError("At least 30 valid forecasts are required.")

    if loss == "qlike":

        loss_a = (
            np.log(forecast_a)
            + actual / forecast_a
        )

        loss_b = (
            np.log(forecast_b)
            + actual / forecast_b
        )

    elif loss == "mse":

        loss_a = (actual - forecast_a) ** 2
        loss_b = (actual - forecast_b) ** 2

    else:
        raise ValueError("Loss must be 'qlike' or 'mse'.")

    differential = loss_a - loss_b
    mean_difference = float(differential.mean())

    if bandwidth is None:
        bandwidth = horizon - 1

    bandwidth = min(int(bandwidth), n - 1)

    centered = differential - mean_difference

    long_run_variance = float(
        np.dot(centered, centered) / n
    )

    for lag in range(1, bandwidth + 1):

        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )

        weight = 1.0 - lag / (bandwidth + 1.0)

        long_run_variance += (
            2.0 * weight * autocovariance
        )

    if long_run_variance <= 0:
        raise ValueError(
            "Non-positive HAC variance; DM statistic cannot be calculated."
        )

    statistic = (
        mean_difference
        / np.sqrt(long_run_variance / n)
    )

    # Harvey-Leybourne-Newbold finite-sample adjustment.
    correction = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n)
        / n
    )

    statistic *= correction

    p_value = float(
        2 * student_t.sf(abs(statistic), df=n - 1)
    )

    if mean_difference < 0:
        better = "Forecast A"
    elif mean_difference > 0:
        better = "Forecast B"
    else:
        better = "No difference in average loss"

    return {
        "DM Statistic": float(statistic),
        "P-value": p_value,
        "Mean Loss Difference (A-B)": mean_difference,
        "Observations": n,
        "Lower Average Loss": better
    }
