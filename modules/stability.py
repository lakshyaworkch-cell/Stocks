
import numpy as np
import pandas as pd

from modules.analysis import fit_model


def rolling_parameter_stability(
    returns,
    model_name="EGARCH",
    distribution="t",
    window=750,
    step=50
):
    """
    Estimate model parameters over successive rolling windows.

    This helps identify changes in volatility dynamics and
    parameter instability.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if window < 250:
        raise ValueError("Window must be at least 250.")

    if step < 1:
        raise ValueError("Step must be positive.")

    if len(returns) < window:
        raise ValueError("Insufficient observations for stability analysis.")

    records = []

    for end in range(window, len(returns) + 1, step):

        sample = returns.iloc[end - window:end]

        try:
            result = fit_model(
                sample,
                model_name=model_name,
                distribution=distribution
            )

            row = {
                "Date": sample.index[-1],
                "Log Likelihood": result.loglikelihood,
                "AIC": result.aic,
                "BIC": result.bic
            }

            for name, value in result.params.items():
                row[name] = float(value)

            records.append(row)

        except Exception:
            continue

    if len(records) < 3:
        raise RuntimeError(
            "Too few successful rolling estimations for stability analysis."
        )

    estimates = pd.DataFrame(records).set_index("Date")

    parameter_columns = [
        col for col in estimates.columns
        if col not in ["Log Likelihood", "AIC", "BIC"]
    ]

    summary = estimates[parameter_columns].agg(
        ["mean", "std", "min", "max"]
    ).T

    summary["Coefficient of Variation"] = (
        summary["std"] / summary["mean"].abs().replace(0, np.nan)
    )

    return estimates, summary
