
import numpy as np
import pandas as pd

from scipy.stats import t, kstest


def student_t_pit(result):
    """
    Calculate probability integral transform values
    for standardized residuals from an arch Student-t model.

    The arch Student-t distribution is standardized to unit variance.
    """

    params = result.params

    if "nu" not in params.index:
        raise ValueError(
            "PIT calibration requires a Student-t fitted model."
        )

    nu = float(params["nu"])

    if nu <= 2:
        raise ValueError("Degrees of freedom must exceed 2.")

    residuals = pd.Series(result.std_resid).dropna()

    residuals = residuals.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    scale = np.sqrt((nu - 2.0) / nu)

    raw_t_values = residuals / scale

    pit = pd.Series(
        t.cdf(raw_t_values, df=nu),
        index=residuals.index,
        name="PIT"
    )

    pit = pit.clip(1e-10, 1 - 1e-10)

    ks_stat, ks_pvalue = kstest(
        pit.to_numpy(),
        "uniform"
    )

    summary = {
        "Observations": len(pit),
        "KS Statistic": float(ks_stat),
        "KS P-value": float(ks_pvalue),
        "PIT Mean": float(pit.mean()),
        "PIT Variance": float(pit.var()),
        "PIT Below 5%": float((pit < 0.05).mean()),
        "PIT Above 95%": float((pit > 0.95).mean())
    }

    return pit, summary

