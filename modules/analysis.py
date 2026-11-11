
import warnings
import numpy as np
import pandas as pd

from arch import arch_model

from modules.preprocessing import validate_returns


VOLATILITY_MODELS = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
}

DISTRIBUTIONS = [
    "normal",
    "t",
    "skewt"
]


def fit_model(returns, model_name="EGARCH",
              distribution="t"):

    returns = validate_returns(returns)

    if model_name not in VOLATILITY_MODELS:
        raise ValueError("Unsupported volatility model.")

    if distribution not in DISTRIBUTIONS:
        raise ValueError("Unsupported distribution.")

    config = VOLATILITY_MODELS[model_name]

    model = arch_model(
        returns,
        mean="Constant",
        vol=config["vol"],
        p=config["p"],
        o=config["o"],
        q=config["q"],
        dist=distribution,
        rescale=False
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        result = model.fit(
            disp="off",
            show_warning=False,
            cov_type="robust",
            options={"maxiter": 2000}
        )

    if result.convergence_flag != 0:
        raise RuntimeError(
            f"{model_name} estimation did not converge. "
            "Check data and model specification."
        )

    if not np.isfinite(result.params).all():
        raise RuntimeError("Non-finite model parameters.")

    return result


def fit_egarch(returns):
    return fit_model(
        returns,
        model_name="EGARCH",
        distribution="t"
    )


def compare_models(returns):

    results = []

    for model_name in VOLATILITY_MODELS:

        for distribution in DISTRIBUTIONS:

            try:
                fitted = fit_model(
                    returns,
                    model_name,
                    distribution
                )

                results.append({
                    "Model": model_name,
                    "Distribution": distribution,
                    "Log Likelihood": fitted.loglikelihood,
                    "AIC": fitted.aic,
                    "BIC": fitted.bic,
                    "Observations": fitted.nobs,
                    "Converged": True
                })

            except Exception as error:

                results.append({
                    "Model": model_name,
                    "Distribution": distribution,
                    "Log Likelihood": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Observations": len(returns),
                    "Converged": False
                })

    return pd.DataFrame(results).sort_values(
        "BIC",
        na_position="last"
    )


def get_parameters(result):

    return pd.DataFrame({
        "Parameter": result.params.index,
        "Estimate": result.params.values,
        "Std Error": result.std_err.values,
        "T Statistic": result.tvalues.values,
        "P-value": result.pvalues.values
    })
