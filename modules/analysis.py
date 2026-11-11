import warnings
import numpy as np
import pandas as pd

from arch import arch_model


VOLATILITY_MODELS = {
    "GARCH": {
        "vol": "GARCH",
        "p": 1,
        "o": 0,
        "q": 1
    },
    "GJR-GARCH": {
        "vol": "GARCH",
        "p": 1,
        "o": 1,
        "q": 1
    },
    "EGARCH": {
        "vol": "EGARCH",
        "p": 1,
        "o": 1,
        "q": 1
    }
}

DISTRIBUTIONS = ["normal", "t", "skewt"]


def fit_model(
    returns,
    model_name="EGARCH",
    distribution="t"
):
    """
    Fit a volatility model.

    Returns should be expressed in percentage units.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if model_name not in VOLATILITY_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(VOLATILITY_MODELS.keys())}."
        )

    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Choose from {DISTRIBUTIONS}."
        )

    if len(returns) < 250:
        raise ValueError(
            "At least 250 return observations are required."
        )

    if not np.isfinite(returns).all():
        raise ValueError("Returns contain invalid values.")

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = model.fit(
            disp="off",
            show_warning=False,
            cov_type="robust",
            options={"maxiter": 2000}
        )

    if result.convergence_flag != 0:
        raise RuntimeError(
            f"{model_name} model did not converge. "
            "Try a longer sample or another distribution."
        )

    if not np.isfinite(result.params).all():
        raise RuntimeError("The model produced invalid parameter estimates.")

    return result


def fit_egarch(returns):
    """
    Convenience function for EGARCH(1,1) with Student-t innovations.
    """

    return fit_model(
        returns,
        model_name="EGARCH",
        distribution="t"
    )


def get_parameters(result):
    """
    Return estimated model parameters and robust inference statistics.
    """

    params = result.params
    standard_errors = result.std_err
    t_statistics = result.tvalues
    p_values = result.pvalues

    table = pd.DataFrame({
        "Parameter": params.index,
        "Estimate": params.values,
        "Std Error": standard_errors.values,
        "T Statistic": t_statistics.values,
        "P-value": p_values.values
    })

    return table


def compare_models(returns):
    """
    Compare all supported volatility-model and distribution combinations.

    Lower AIC and BIC values indicate a better information-criterion fit.
    """

    returns = pd.Series(returns).dropna().astype(float)

    records = []

    for model_name in VOLATILITY_MODELS:
        for distribution in DISTRIBUTIONS:

            try:
                result = fit_model(
                    returns,
                    model_name=model_name,
                    distribution=distribution
                )

                records.append({
                    "Model": model_name,
                    "Distribution": distribution,
                    "Log Likelihood": result.loglikelihood,
                    "AIC": result.aic,
                    "BIC": result.bic,
                    "Observations": result.nobs,
                    "Converged": result.convergence_flag == 0,
                    "Status": "Success"
                })

            except Exception as error:
                records.append({
                    "Model": model_name,
                    "Distribution": distribution,
                    "Log Likelihood": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Observations": len(returns),
                    "Converged": False,
                    "Status": str(error)
                })

    comparison = pd.DataFrame(records)

    return comparison.sort_values(
        "AIC",
        ascending=True,
        na_position="last"
    ).reset_index(drop=True)

def generate_forecast(result, horizon=5, simulations=10000):
    """
    Generate volatility forecasts.

    EGARCH uses simulation for multi-step forecasts.
    Other models can use analytic forecasts.
    """

    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    model_name = getattr(result.model.volatility, "name", "")

    if model_name.upper() == "EGARCH" and horizon > 1:
        forecast = result.forecast(
            horizon=horizon,
            method="simulation",
            simulations=simulations,
            reindex=False
        )
    else:
        forecast = result.forecast(
            horizon=horizon,
            method="analytic",
            reindex=False
        )

    return forecast
