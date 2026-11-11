
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from datetime import date, timedelta
from scipy.stats import t as student_t

from modules.data import DUTCH_STOCKS, get_stock_data
from modules.preprocessing import calculate_returns
from modules.stationarity import stationarity_tests
from modules.analysis import (
    fit_model,
    fit_egarch,
    compare_models,
    get_parameters
)
from modules.diagnostics import get_diagnostics
from modules.forecasting import (
    rolling_forecast,
    evaluate_forecasts
)
from modules.risk import (
    student_t_risk,
    historical_risk
)
from modules.backtesting import var_backtests


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dutch Quant | Institutional Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5EAF3;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
    font-weight: 650 !important;
}

section[data-testid="stSidebar"] {
    background-color: #101A2B;
    border-right: 1px solid #293951;
}

div[data-testid="stMetric"] {
    background: #151F32;
    border: 1px solid #293951;
    padding: 17px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #00D4AA;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293951;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    background-color: #151F32;
    border: 1px solid #293951;
    border-radius: 10px;
}

.stButton > button {
    background-color: #00A88F;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #00C6A5;
    color: white;
}

.stDownloadButton > button {
    background-color: #17263C;
    color: #00D4AA;
    border: 1px solid #00A88F;
    border-radius: 8px;
}

hr {
    border-color: #293951;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE
# =========================================================

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "model_result" not in st.session_state:
    st.session_state.model_result = None

if "model_parameters" not in st.session_state:
    st.session_state.model_parameters = None

if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None

if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

if "analysis_metadata" not in st.session_state:
    st.session_state.analysis_metadata = None


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def dark_chart(fig, title=None):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#151F32",
        font=dict(
            family="Arial",
            color="#E5EAF3",
            size=12
        ),
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=30
        ),
        title=title
    )

    return fig


def display_table(data, height=350):

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        height=height
    )


def display_diagnostic_card(title, pvalue):

    st.metric(
        title,
        f"{pvalue:.4f}",
        help=(
            "P-value below 0.05 indicates evidence "
            "against the null hypothesis."
        )
    )


def build_rolling_var(
    returns,
    window=750,
    confidence=0.95
):
    """
    Strict one-step-ahead rolling VaR estimation.

    Each forecast uses only observations available
    before the forecast date.

    This implementation refits the model at every
    forecast origin, so it can be computationally expensive.
    """

    from modules.analysis import fit_model

    returns = pd.Series(returns).dropna()

    if len(returns) <= window + 10:
        raise ValueError(
            "Insufficient observations for rolling VaR."
        )

    if not 0.5 < confidence < 1:
        raise ValueError("Invalid confidence level.")

    alpha = 1 - confidence

    output = []

    for i in range(window, len(returns)):

        training = returns.iloc[i-window:i]

        fitted = fit_model(
            training,
            model_name="EGARCH",
            distribution="t"
        )

        forecast = fitted.forecast(
            horizon=1,
            reindex=False
        )

        mu = float(forecast.mean.iloc[-1, 0])

        variance = float(
            forecast.variance.iloc[-1, 0]
        )

        nu = float(fitted.params["nu"])

        if variance <= 0 or nu <= 2:
            raise ValueError(
                "Invalid variance or Student-t degrees of freedom."
            )

        sigma = np.sqrt(variance)

        scale = np.sqrt((nu - 2) / nu)

        quantile = (
            student_t.ppf(alpha, df=nu) * scale
        )

        var_loss = -(mu + sigma * quantile)

        output.append({
            "Date": returns.index[i],
            "Actual Return": float(returns.iloc[i]),
            "Forecast Mean": mu,
            "Forecast Volatility": sigma,
            "VaR (%)": max(0.0, var_loss)
        })

    return pd.DataFrame(output).set_index("Date")


# =========================================================
# 5. HEADER
# =========================================================

st.title("📈 DUTCH QUANT")

st.subheader(
    "Institutional Quantitative Research Terminal"
)

st.caption(
    "Euronext Amsterdam | Econometric Modelling | "
    "Volatility Forecasting | Risk Management"
)

st.divider()


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("DUTCH QUANT")

st.sidebar.caption("RESEARCH CONTROL PANEL")

page = st.sidebar.radio(
    "Research Sections",
    [
        "Market Data",
        "Statistical Analysis",
        "Volatility Models",
        "Model Diagnostics",
        "Forecasting",
        "Risk Management",
        "Portfolio Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Market Configuration")

company = st.sidebar.selectbox(
    "Select Dutch Company",
    list(DUTCH_STOCKS.keys())
)

ticker = DUTCH_STOCKS[company]

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2018, 1, 1),
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

st.sidebar.divider()

st.sidebar.subheader("Model Configuration")

model_choice = st.sidebar.selectbox(
    "Volatility Model",
    ["EGARCH", "GARCH", "GJR-GARCH"]
)

distribution_choice = st.sidebar.selectbox(
    "Innovation Distribution",
    ["t", "normal", "skewt"],
    index=0
)

st.sidebar.divider()

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    1,
    30,
    10
)

st.sidebar.caption(
    "Default estimation: maximum likelihood "
    "with robust covariance estimates."
)

run_analysis = st.sidebar.button(
    "RUN RESEARCH ANALYSIS",
    type="primary",
    use_container_width=True
)


# =========================================================
# 7. RUN MAIN ANALYSIS
# =========================================================

if run_analysis:

    if start_date >= end_date:

        st.error(
            "Start date must be earlier than end date."
        )

        st.stop()

    with st.spinner("Downloading historical market data..."):

        try:

            df = get_stock_data(
                ticker,
                start_date,
                end_date + timedelta(days=1)
            )

            if len(df) < 250:

                st.error(
                    "At least 250 return observations "
                    "are required for this research setup."
                )

                st.stop()

            returns = calculate_returns(df["Close"])

            # Align prices and returns
            df = df.loc[returns.index].copy()
            df["Returns"] = returns

        except Exception as error:

            st.error(
                f"Market data error: {error}"
            )

            st.stop()

    with st.spinner("Estimating volatility model..."):

        try:

            result = fit_model(
                returns,
                model_name=model_choice,
                distribution=distribution_choice
            )

            parameters = get_parameters(result)

            diagnostics = get_diagnostics(result)

            from modules.analysis import generate_forecast

forecast = generate_forecast(
    result,
    horizon=forecast_horizon,
    simulations=10000
)

            forecast_variance = (
                forecast.variance.iloc[-1].values
            )

            forecast_mean = (
                forecast.mean.iloc[-1].values
            )

            forecast_df = pd.DataFrame({
                "Day": range(1, forecast_horizon + 1),
                "Expected Return (%)": forecast_mean,
                "Variance": forecast_variance,
                "Daily Volatility (%)":
                    np.sqrt(forecast_variance),
                "Annualized Volatility (%)":
                    np.sqrt(forecast_variance) * np.sqrt(252)
            })

            st.session_state.market_data = df
            st.session_state.model_result = result
            st.session_state.model_parameters = parameters
            st.session_state.diagnostics = diagnostics
            st.session_state.forecast_data = forecast_df

            st.session_state.analysis_metadata = {
                "company": company,
                "ticker": ticker,
                "model": model_choice,
                "distribution": distribution_choice,
                "start": start_date,
                "end": end_date
            }

            st.success(
                f"Research analysis completed for {company}."
            )

        except Exception as error:

            st.error(
                f"Model estimation failed: {error}"
            )

            st.stop()


# =========================================================
# 8. LOAD STORED ANALYSIS
# =========================================================

df = st.session_state.market_data
result = st.session_state.model_result
parameters = st.session_state.model_parameters
diagnostics = st.session_state.diagnostics
forecast_df = st.session_state.forecast_data
metadata = st.session_state.analysis_metadata


if df is None or result is None:

    st.info(
        "Select a company, configure your model and click "
        "'RUN RESEARCH ANALYSIS' to begin."
    )

    st.markdown("### Research Capabilities")

    c1, c2, c3 = st.columns(3)

    c1.info(
        "**Market Research**\n\n"
        "Historical prices, returns and statistical analysis."
    )

    c2.info(
        "**Econometric Modelling**\n\n"
        "GARCH, GJR-GARCH and EGARCH specifications."
    )

    c3.info(
        "**Risk Management**\n\n"
        "Conditional volatility, VaR and Expected Shortfall."
    )

    st.stop()


# Use the actual model metadata from the last successful run.

company = metadata["company"]
ticker = metadata["ticker"]
model_choice = metadata["model"]
distribution_choice = metadata["distribution"]

returns = df["Returns"]

st.caption(
    f"Active research: {company} ({ticker}) | "
    f"{model_choice} | {distribution_choice} | "
    f"{metadata['start']} – {metadata['end']}"
)


# =========================================================
# 9. MARKET DATA PAGE
# =========================================================

if page == "Market Data":

    st.header("01 | Market Data")

    total_return = (
        df["Close"].iloc[-1] /
        df["Close"].iloc[0] - 1
    ) * 100

    annual_volatility = (
        returns.std() * np.sqrt(252)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Price",
        f"€{df['Close'].iloc[-1]:.2f}"
    )

    c2.metric(
        "Total Return",
        f"{total_return:.2f}%"
    )

    c3.metric(
        "Annualized Volatility",
        f"{annual_volatility:.2f}%"
    )

    c4.metric(
        "Observations",
        f"{len(df):,}"
    )

    st.subheader("Historical Price")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Adjusted Close"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "Historical Adjusted Price"),
        use_container_width=True
    )

    st.subheader("Daily Log Returns")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="Log Returns"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "Daily Log Returns (%)"),
        use_container_width=True
    )

    st.subheader("Historical Data")

    display_table(df.reset_index())

    st.download_button(
        "Download Market Data",
        df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_market_data.csv",
        mime="text/csv"
    )


# =========================================================
# 10. STATISTICAL ANALYSIS PAGE
# =========================================================

elif page == "Statistical Analysis":

    st.header("02 | Statistical Analysis")

    st.subheader("Descriptive Statistics")

    descriptive = returns.describe().to_frame(
        "Daily Returns (%)"
    )

    descriptive.loc["Skewness"] = returns.skew()
    descriptive.loc["Excess Kurtosis"] = returns.kurtosis()

    display_table(
        descriptive.reset_index().rename(
            columns={"index": "Statistic"}
        )
    )

    st.divider()

    st.subheader("Stationarity Tests")

    with st.spinner("Running stationarity tests..."):

        try:

            stationarity = stationarity_tests(returns)

            stationarity_df = pd.DataFrame(
                stationarity.items(),
                columns=["Test", "Value"]
            )

            display_table(stationarity_df)

        except Exception as error:

            st.error(
                f"Stationarity test error: {error}"
            )

    st.markdown("""
    **ADF test**

    Null hypothesis: the series contains a unit root.

    **KPSS test**

    Null hypothesis: the series is stationary.

    Interpret the tests jointly, considering their
    respective deterministic specifications.
    """)

    st.divider()

    st.subheader("Return Distribution")

    fig = px.histogram(
        x=returns,
        nbins=60,
        labels={"x": "Daily Return (%)"},
        title="Distribution of Daily Returns"
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True
    )

    st.subheader("Rolling Volatility")

    rolling_vol = returns.rolling(30).std() * np.sqrt(252)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode="lines",
            name="30-day volatility"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "30-Day Annualized Rolling Volatility (%)"),
        use_container_width=True
    )


# =========================================================
# 11. VOLATILITY MODELS PAGE
# =========================================================

elif page == "Volatility Models":

    st.header("03 | Volatility Models")

    st.subheader("Current Model Specification")

    c1, c2, c3 = st.columns(3)

    c1.metric("Model", model_choice)
    c2.metric("Distribution", distribution_choice)
    c3.metric("Observations", len(returns))

    st.subheader("Parameter Estimates")

    display_table(parameters)

    with st.expander("Full Econometric Model Summary"):

        st.code(
            result.summary().as_text(),
            language="text"
        )

    st.divider()

    st.subheader("Compare Nine Model Specifications")

    st.warning(
        "This estimates nine separate models and may take "
        "some time. Failed or non-converged models are "
        "reported as unavailable."
    )

    if st.button("RUN MODEL COMPARISON"):

        with st.spinner("Estimating competing models..."):

            try:

                comparison = compare_models(returns)

                st.session_state.model_comparison = comparison

                st.success("Model comparison completed.")

            except Exception as error:

                st.error(
                    f"Comparison error: {error}"
                )

    comparison = st.session_state.get(
        "model_comparison"
    )

    if comparison is not None:

        display_table(comparison)

        st.caption(
            "Lower AIC and BIC indicate better relative "
            "in-sample fit among successfully estimated "
            "models with comparable likelihood samples. "
            "They do not establish superior forecasting."
        )

        st.download_button(
            "Download Model Comparison",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Conditional Volatility")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result.conditional_volatility.index,
            y=result.conditional_volatility,
            mode="lines",
            name="Conditional Volatility"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "Estimated Conditional Volatility (%)"),
        use_container_width=True
    )


# =========================================================
# 12. MODEL DIAGNOSTICS PAGE
# =========================================================

elif page == "Model Diagnostics":

    st.header("04 | Model Diagnostics")

    st.subheader("Residual Diagnostic Tests")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Ljung-Box (10)",
        f"{diagnostics['Ljung-Box residual (10)']:.4f}"
    )

    c2.metric(
        "Ljung-Box Squared (10)",
        f"{diagnostics['Ljung-Box squared (10)']:.4f}"
    )

    c3.metric(
        "ARCH LM",
        f"{diagnostics['ARCH LM p-value']:.4f}"
    )

    c4.metric(
        "Ljung-Box Squared (20)",
        f"{diagnostics['Ljung-Box squared (20)']:.4f}"
    )

    diagnostic_df = pd.DataFrame({

        "Diagnostic": [
            "Ljung-Box residual (10)",
            "Ljung-Box residual (20)",
            "Ljung-Box squared (10)",
            "Ljung-Box squared (20)",
            "ARCH LM"
        ],

        "P-value": [
            diagnostics["Ljung-Box residual (10)"],
            diagnostics["Ljung-Box residual (20)"],
            diagnostics["Ljung-Box squared (10)"],
            diagnostics["Ljung-Box squared (20)"],
            diagnostics["ARCH LM p-value"]
        ],

        "Interpretation": [
            "Residual autocorrelation",
            "Residual autocorrelation",
            "Remaining volatility dependence",
            "Remaining volatility dependence",
            "Remaining ARCH effects"
        ]

    })

    diagnostic_df["Conclusion"] = diagnostic_df[
        "P-value"
    ].apply(
        lambda x: (
            "Reject H0" if x < 0.05 else "Do not reject H0"
        )
    )

    display_table(diagnostic_df)

    st.divider()

    st.subheader("Standardized Residuals")

    residuals = pd.Series(
        result.std_resid
    ).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals,
            mode="lines",
            name="Standardized Residuals"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "Standardized Residuals"),
        use_container_width=True
    )

    st.subheader("Squared Standardized Residuals")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals ** 2,
            mode="lines",
            name="Squared Residuals"
        )
    )

    st.plotly_chart(
        dark_chart(fig, "Squared Standardized Residuals"),
        use_container_width=True
    )

    st.subheader("Residual Distribution")

    fig = px.histogram(
        x=residuals,
        nbins=60,
        title="Standardized Residual Distribution"
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True
    )

    st.markdown("""
    **Interpretation**

    - Ljung-Box on residuals tests remaining linear dependence.
    - Ljung-Box on squared residuals tests remaining dependence
      in squared innovations.
    - ARCH LM tests remaining ARCH effects.

    A significant test suggests that the model may not have
    captured all relevant dependence.
    """)

    st.download_button(
        "Download Diagnostics",
        diagnostic_df.to_csv(index=False).encode("utf-8"),
        file_name="model_diagnostics.csv",
        mime="text/csv"
    )


# =========================================================
# 13. FORECASTING PAGE
# =========================================================

elif page == "Forecasting":

    st.header("05 | Out-of-Sample Forecasting")

    st.markdown("""
    Rolling-window forecasts use historical information
    available before each forecast date.

    Forecast performance is evaluated against subsequent
    observed returns.
    """)

    window = st.number_input(
        "Rolling Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 20),
        value=min(750, len(returns) - 20),
        step=50
    )

    st.warning(
        "Rolling estimation can be computationally expensive "
        "because the model is re-estimated at every forecast origin."
    )

    if st.button("RUN ROLLING FORECAST"):

        with st.spinner("Running rolling forecast..."):

            try:

                rolling_df = rolling_forecast(
                    returns,
                    model_name=model_choice,
                    distribution=distribution_choice,
                    window=int(window)
                )

                scores = evaluate_forecasts(rolling_df)

                st.session_state.rolling_forecast = rolling_df
                st.session_state.forecast_scores = scores

                st.success(
                    "Out-of-sample forecasting completed."
                )

            except Exception as error:

                st.error(
                    f"Forecasting error: {error}"
                )

    rolling_df = st.session_state.get(
        "rolling_forecast"
    )

    scores = st.session_state.get(
        "forecast_scores"
    )

    if rolling_df is not None:

        st.subheader("Forecast Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "QLIKE",
            f"{scores['QLIKE']:.6f}"
        )

        c2.metric(
            "MSE",
            f"{scores['MSE']:.6f}"
        )

        c3.metric(
            "Forecast Observations",
            scores["Observations"]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rolling_df.index,
                y=rolling_df["Forecast Volatility"],
                name="Forecast Volatility"
            )
        )

        st.plotly_chart(
            dark_chart(fig, "Rolling Forecast Volatility (%)"),
            use_container_width=True
        )

        display_table(rolling_df.reset_index())

        st.download_button(
            "Download Rolling Forecast",
            rolling_df.to_csv().encode("utf-8"),
            file_name="rolling_forecasts.csv",
            mime="text/csv"
        )


# =========================================================
# 14. RISK MANAGEMENT PAGE
# =========================================================

elif page == "Risk Management":

    st.header("06 | Risk Management")

    confidence = st.selectbox(
        "Confidence Level",
        [0.95, 0.99],
        format_func=lambda x: f"{x:.0%}"
    )

    st.subheader("Parametric Student-t Risk")

    try:

        parametric = student_t_risk(
            result,
            confidence=confidence
        )

        historical = historical_risk(
            returns,
            confidence=confidence
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Student-t VaR",
            f"{parametric['VaR (%)']:.4f}%"
        )

        c2.metric(
            "Student-t Expected Shortfall",
            f"{parametric['Expected Shortfall (%)']:.4f}%"
        )

        risk_table = pd.DataFrame([
            {
                "Method": "Student-t EGARCH",
                "VaR (%)": parametric["VaR (%)"],
                "Expected Shortfall (%)":
                    parametric["Expected Shortfall (%)"]
            },
            {
                "Method": "Historical",
                "VaR (%)": historical["Historical VaR (%)"],
                "Expected Shortfall (%)":
                    historical["Historical ES (%)"]
            }
        ])

        st.subheader("Risk Measure Comparison")

        display_table(risk_table)

    except Exception as error:

        st.error(
            f"Risk calculation error: {error}"
        )

    st.divider()

    st.subheader("Rolling VaR Backtesting")

    st.caption(
        "This procedure re-estimates EGARCH at every "
        "forecast origin and uses only past observations."
    )

    risk_window = st.number_input(
        "VaR Backtesting Window",
        min_value=250,
        max_value=max(250, len(returns) - 20),
        value=min(750, len(returns) - 20),
        step=50,
        key="risk_window"
    )

    if st.button("RUN VaR BACKTEST"):

        with st.spinner(
            "Running one-step-ahead EGARCH VaR backtesting..."
        ):

            try:

                risk_forecasts = build_rolling_var(
                    returns,
                    window=int(risk_window),
                    confidence=confidence
                )

                backtest = var_backtests(
                    risk_forecasts["Actual Return"],
                    risk_forecasts["VaR (%)"],
                    confidence=confidence
                )

                st.session_state.risk_forecasts = risk_forecasts
                st.session_state.risk_backtest = backtest

                st.success("VaR backtesting completed.")

            except Exception as error:

                st.error(
                    f"Backtesting error: {error}"
                )

    risk_forecasts = st.session_state.get(
        "risk_forecasts"
    )

    backtest = st.session_state.get(
        "risk_backtest"
    )

    if risk_forecasts is not None and backtest is not None:

        st.subheader("Backtesting Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Exceptions",
            backtest["Exceptions"]
        )

        c2.metric(
            "Expected Exceptions",
            f"{backtest['Expected Exceptions']:.2f}"
        )

        c3.metric(
            "Exception Rate",
            f"{backtest['Exception Rate']:.2%}"
        )

        display_table(
            pd.DataFrame(
                backtest.items(),
                columns=["Test", "Result"]
            )
        )

        st.markdown("""
        **Backtesting interpretation**

        - Kupiec tests unconditional coverage.
        - Christoffersen tests exception independence.
        - Conditional coverage combines both tests.

        A p-value below 0.05 indicates evidence against
        the relevant null hypothesis.
        """)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=risk_forecasts.index,
                y=risk_forecasts["Actual Return"],
                name="Actual Return",
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=risk_forecasts.index,
                y=-risk_forecasts["VaR (%)"],
                name="VaR Threshold",
                mode="lines"
            )
        )

        st.plotly_chart(
            dark_chart(fig, "Actual Returns versus VaR Threshold"),
            use_container_width=True
        )

        st.download_button(
            "Download VaR Backtest",
            risk_forecasts.to_csv().encode("utf-8"),
            file_name="var_backtest.csv",
            mime="text/csv"
        )


# =========================================================
# 15. PORTFOLIO ANALYSIS PAGE
# =========================================================

elif page == "Portfolio Analysis":

    st.header("07 | Portfolio Analysis")

    st.markdown(
        "Construct an equally weighted portfolio of Dutch equities."
    )

    selected_companies = st.multiselect(
        "Select Companies",
        list(DUTCH_STOCKS.keys()),
        default=[
            "ASML",
            "Shell",
            "ING Group"
        ]
    )

    if len(selected_companies) < 2:

        st.info(
            "Select at least two companies to construct a portfolio."
        )

    elif st.button("RUN PORTFOLIO ANALYSIS"):

        with st.spinner("Downloading portfolio data..."):

            try:

                portfolio_returns = {}

                for name in selected_companies:

                    stock_ticker = DUTCH_STOCKS[name]

                    stock_df = get_stock_data(
                        stock_ticker,
                        metadata["start"],
                        metadata["end"] + timedelta(days=1)
                    )

                    stock_returns = calculate_returns(
                        stock_df["Close"]
                    )

                    portfolio_returns[name] = stock_returns

                return_df = pd.DataFrame(
                    portfolio_returns
                ).dropna()

                if len(return_df) < 100:
                    raise ValueError(
                        "Insufficient common observations."
                    )

                # Equal-weighted portfolio

                weights = np.repeat(
                    1 / len(selected_companies),
                    len(selected_companies)
                )

                portfolio_return = (
                    return_df @ weights
                )

                annual_return = (
                    portfolio_return.mean() * 252
                )

                annual_volatility = (
                    portfolio_return.std() * np.sqrt(252)
                )

                sharpe = (
                    annual_return / annual_volatility
                    if annual_volatility > 0
                    else np.nan
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Annualized Return",
                    f"{annual_return:.2f}%"
                )

                c2.metric(
                    "Annualized Volatility",
                    f"{annual_volatility:.2f}%"
                )

                c3.metric(
                    "Annualized Sharpe",
                    f"{sharpe:.4f}"
                )

                st.subheader("Correlation Matrix")

                correlation = return_df.corr()

                fig = px.imshow(
                    correlation,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )

                st.plotly_chart(
                    dark_chart(fig, "Equity Return Correlations"),
                    use_container_width=True
                )

                st.subheader("Portfolio Performance")

                cumulative = (
                    portfolio_return / 100
                ).add(1).cumprod()

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=cumulative.index,
                        y=cumulative,
                        mode="lines",
                        name="Equal-weighted portfolio"
                    )
                )

                st.plotly_chart(
                    dark_chart(fig, "Portfolio Growth of €1"),
                    use_container_width=True
                )

                st.subheader("Portfolio Returns")

                display_table(
                    return_df.describe().reset_index()
                )

                st.session_state.portfolio_returns = return_df

            except Exception as error:

                st.error(
                    f"Portfolio analysis failed: {error}"
                )

    portfolio_data = st.session_state.get(
        "portfolio_returns"
    )

    if portfolio_data is not None:

        st.download_button(
            "Download Portfolio Returns",
            portfolio_data.to_csv().encode("utf-8"),
            file_name="portfolio_returns.csv",
            mime="text/csv"
        )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.caption(
    "DUTCH QUANT | Institutional Quantitative Research | "
    "Euronext Amsterdam | Educational and Research Use"
)
