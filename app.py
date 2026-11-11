
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta

from modules.data import DUTCH_STOCKS, get_stock_data
from modules.preprocessing import calculate_returns
from modules.stationarity import stationarity_tests

from modules.analysis import (
    fit_model,
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

from modules.stability import rolling_parameter_stability
from modules.pit_diagnostics import student_t_pit
from modules.forecast_tests import diebold_mariano_test


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dutch Quant | Quantitative Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DARK THEME
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #263449;
        padding: 15px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }

    .main-header {
        font-size: 36px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 4px;
    }

    hr {
        border-color: #263449;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-header">DUTCH QUANT</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-header">'
    'Quantitative Financial Research | Volatility Modelling | Risk Analytics'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "stock_data": None,
    "returns": None,
    "model_result": None,
    "model_parameters": None,
    "diagnostics": None,
    "forecast_result": None,
    "selected_ticker": None,
    "selected_model": None,
    "selected_distribution": None,
    "model_comparison": None,
    "rolling_forecasts": None,
    "forecast_evaluation": None,
    "forecast_comparison": None,
    "stability_estimates": None,
    "stability_summary": None,
    "pit_values": None,
    "pit_summary": None,
    "var_forecasts": None,
    "var_backtest_results": None,
    "portfolio_returns": None,
    "portfolio_series": None,
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_volatility_forecast(
    result,
    horizon,
    model_name,
    simulations=10000
):
    """
    Generate conditional variance forecasts.

    EGARCH does not support analytic forecasts for horizons
    greater than one in the arch package.

    EGARCH with horizon > 1 uses simulation.
    Other cases use analytic forecasts.
    """

    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

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


def build_rolling_var(
    returns,
    confidence=0.95,
    window=500
):
    """
    Chronological one-step-ahead EGARCH Student-t VaR forecasts.

    Each model is estimated using only observations available
    before the forecast date.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) <= window:
        raise ValueError(
            "There are not enough observations for rolling VaR."
        )

    records = []
    failed = 0

    for i in range(window, len(returns)):

        training_data = returns.iloc[i-window:i]

        try:

            fitted = fit_model(
                training_data,
                model_name="EGARCH",
                distribution="t"
            )

            one_day_forecast = fitted.forecast(
                horizon=1,
                method="analytic",
                reindex=False
            )

            next_variance = float(
                one_day_forecast.variance.iloc[-1, 0]
            )

            risk = student_t_risk(
                fitted,
                confidence=confidence,
                forecast_variance=next_variance
            )

            records.append({
                "Date": returns.index[i],
                "Actual Return": float(returns.iloc[i]),
                "VaR": risk["VaR"],
                "ES": risk["ES"],
                "Forecast Volatility": risk["Forecast Volatility"]
            })

        except Exception:
            failed += 1
            continue

    if not records:
        raise RuntimeError(
            "Rolling VaR estimation failed for every forecast date."
        )

    output = pd.DataFrame(records).set_index("Date")

    output.attrs["attempted"] = len(returns) - window
    output.attrs["failed"] = failed

    return output


def make_ewma_forecasts(
    returns,
    forecast_index,
    window=750,
    decay=0.94
):
    """
    Produce one-step-ahead RiskMetrics-style EWMA variance forecasts.

    Variance recursion:
        sigma2_t = lambda * sigma2_(t-1)
                   + (1-lambda) * r_(t-1)^2

    Only past returns are used for each forecast.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if not 0 < decay < 1:
        raise ValueError("EWMA decay must be between 0 and 1.")

    records = []

    for forecast_date in forecast_index:

        position = returns.index.get_loc(forecast_date)

        if not isinstance(position, (int, np.integer)):
            continue

        if position < window:
            continue

        training = returns.iloc[position-window:position]

        initial_count = min(30, len(training))

        variance = float(
            training.iloc[:initial_count].var(ddof=1)
        )

        if not np.isfinite(variance) or variance <= 0:
            continue

        for observation in training.iloc[initial_count:]:
            variance = (
                decay * variance
                + (1 - decay) * float(observation) ** 2
            )

        if np.isfinite(variance) and variance > 0:
            records.append({
                "Date": forecast_date,
                "Actual Return": float(returns.loc[forecast_date]),
                "EWMA Variance": variance,
                "EWMA Volatility": np.sqrt(variance)
            })

    if not records:
        raise RuntimeError("No valid EWMA forecasts were generated.")

    return pd.DataFrame(records).set_index("Date")


def make_chart_layout(fig, title, y_title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font=dict(color="#e5e7eb"),
        xaxis_title="Date",
        yaxis_title=y_title,
        height=420,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    return fig


def show_error(error):
    st.error(str(error))


def clear_analysis_results():
    """
    Clear stored outputs when a new model is estimated.
    """

    keys = [
        "model_comparison",
        "rolling_forecasts",
        "forecast_evaluation",
        "forecast_comparison",
        "stability_estimates",
        "stability_summary",
        "pit_values",
        "pit_summary",
        "var_forecasts",
        "var_backtest_results",
        "portfolio_returns",
        "portfolio_series"
    ]

    for key in keys:
        st.session_state[key] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dutch Quant")
st.sidebar.caption("Quantitative Research Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Data",
        "Statistical Analysis",
        "Volatility Models",
        "Model Diagnostics",
        "Forecasting",
        "Risk Management",
        "Model Stability & Calibration",
        "Forecast Comparison",
        "Portfolio Analysis"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Analysis Settings")

company = st.sidebar.selectbox(
    "Select Dutch Stock",
    list(DUTCH_STOCKS.keys())
)

ticker = DUTCH_STOCKS[company]

today = date.today()
default_start = today - timedelta(days=365 * 5)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=date(2000, 1, 1),
    max_value=today
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today,
    min_value=date(2000, 1, 1),
    max_value=today
)

model_name = st.sidebar.selectbox(
    "Volatility Model",
    ["EGARCH", "GARCH", "GJR-GARCH"]
)

distribution = st.sidebar.selectbox(
    "Innovation Distribution",
    ["t", "normal", "skewt"]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Trading Days)",
    min_value=1,
    max_value=20,
    value=5
)

st.sidebar.caption(
    "Student-t innovations are a useful baseline for "
    "heavy-tailed financial returns."
)

run_analysis = st.sidebar.button(
    "Run Analysis",
    type="primary",
    use_container_width=True
)


# ============================================================
# DATA DOWNLOAD AND MODEL ESTIMATION
# ============================================================

if run_analysis:

    if start_date >= end_date:

        st.sidebar.error("End date must be after start date.")

    else:

        try:

            clear_analysis_results()

            with st.spinner("Downloading market data..."):

                data = get_stock_data(
                    ticker,
                    start_date,
                    end_date + timedelta(days=1)
                )

                returns = calculate_returns(data["Close"])

            with st.spinner("Estimating volatility model..."):

                fitted_model = fit_model(
                    returns,
                    model_name=model_name,
                    distribution=distribution
                )

                parameters = get_parameters(fitted_model)
                diagnostics = get_diagnostics(fitted_model)

            with st.spinner("Generating volatility forecasts..."):

                forecast = make_volatility_forecast(
                    fitted_model,
                    horizon=forecast_horizon,
                    model_name=model_name,
                    simulations=10000
                )

            st.session_state.stock_data = data
            st.session_state.returns = returns
            st.session_state.model_result = fitted_model
            st.session_state.model_parameters = parameters
            st.session_state.diagnostics = diagnostics
            st.session_state.forecast_result = forecast

            st.session_state.selected_ticker = ticker
            st.session_state.selected_model = model_name
            st.session_state.selected_distribution = distribution

            st.sidebar.success("Analysis completed successfully.")

        except Exception as error:

            st.error(f"Model estimation failed: {error}")


# ============================================================
# CHECK ANALYSIS AVAILABILITY
# ============================================================

data = st.session_state.stock_data
returns = st.session_state.returns
result = st.session_state.model_result

if data is None or returns is None or result is None:

    st.info(
        "Select a company and date range in the sidebar, "
        "then click Run Analysis."
    )

    st.markdown("### Quantitative Research Tools")

    col1, col2, col3 = st.columns(3)

    col1.metric("Volatility Models", "3")
    col2.metric("Innovation Distributions", "3")
    col3.metric("Risk Frameworks", "2")

    st.markdown("""
    - EGARCH, GARCH and GJR-GARCH modelling
    - Student-t, normal and skewed Student-t innovations
    - ADF and KPSS stationarity tests
    - Residual and ARCH diagnostics
    - Multi-step volatility forecasts
    - Rolling out-of-sample forecast evaluation
    - EWMA benchmark comparison
    - Diebold-Mariano forecast tests
    - Parameter stability analysis
    - Probability integral transform diagnostics
    - VaR and Expected Shortfall
    - Kupiec and Christoffersen backtesting
    - Equal-weighted portfolio analysis
    """)

    st.stop()


# ============================================================
# GLOBAL MODEL INFORMATION
# ============================================================

st.caption(
    f"Company: {company} | "
    f"Ticker: {st.session_state.selected_ticker} | "
    f"Model: {st.session_state.selected_model} | "
    f"Distribution: {st.session_state.selected_distribution}"
)


# ============================================================
# PAGE 1: MARKET DATA
# ============================================================

if page == "Market Data":

    st.header("Market Data")

    latest_price = float(data["Close"].iloc[-1])
    daily_return = float(returns.iloc[-1])

    annual_volatility = float(
        returns.std() * np.sqrt(252)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Adjusted Close",
        f"€{latest_price:,.2f}"
    )

    col2.metric(
        "Latest Daily Return",
        f"{daily_return:.2f}%"
    )

    col3.metric(
        "Annualized Historical Volatility",
        f"{annual_volatility:.2f}%"
    )

    st.subheader("Historical Stock Price")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Adjusted Close"
        )
    )

    st.plotly_chart(
        make_chart_layout(
            fig,
            "Historical Stock Price",
            "Price (€)"
        ),
        use_container_width=True
    )

    st.subheader("Daily Log Returns")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="Daily Returns"
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        make_chart_layout(
            fig,
            "Daily Percentage Log Returns",
            "Return (%)"
        ),
        use_container_width=True
    )

    st.subheader("Historical Data")

    st.dataframe(
        data.tail(20),
        use_container_width=True
    )


# ============================================================
# PAGE 2: STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.header("Statistical Analysis")

    st.subheader("Descriptive Statistics")

    stats = returns.describe().to_frame(
        name="Returns"
    )

    stats.loc["Skewness"] = returns.skew()
    stats.loc["Excess Kurtosis"] = returns.kurtosis()
    stats.loc["Annualized Volatility"] = (
        returns.std() * np.sqrt(252)
    )

    st.dataframe(
        stats,
        use_container_width=True
    )

    st.subheader("Stationarity Tests")

    try:

        stationarity = stationarity_tests(returns)

        st.dataframe(
            stationarity,
            use_container_width=True
        )

        st.caption(
            "ADF null: unit root. KPSS null: stationarity. "
            "A p-value below 0.05 rejects the relevant null hypothesis."
        )

    except Exception as error:
        show_error(error)

    st.subheader("Return Distribution")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=60,
            name="Returns"
        )
    )

    st.plotly_chart(
        make_chart_layout(
            fig,
            "Distribution of Daily Returns",
            "Frequency"
        ),
        use_container_width=True
    )


# ============================================================
# PAGE 3: VOLATILITY MODELS
# ============================================================

elif page == "Volatility Models":

    st.header("Volatility Models")

    st.subheader("Estimated Model Parameters")

    st.dataframe(
        st.session_state.model_parameters,
        use_container_width=True
    )

    st.subheader("Model Fit Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Log Likelihood",
        f"{result.loglikelihood:.2f}"
    )

    col2.metric(
        "AIC",
        f"{result.aic:.2f}"
    )

    col3.metric(
        "BIC",
        f"{result.bic:.2f}"
    )

    st.subheader("Conditional Volatility")

    volatility = pd.Series(
        result.conditional_volatility
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility,
            mode="lines",
            name="Conditional Volatility"
        )
    )

    st.plotly_chart(
        make_chart_layout(
            fig,
            "Estimated Conditional Volatility",
            "Volatility (%)"
        ),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    st.warning(
        "This comparison estimates nine specifications "
        "and may take some time."
    )

    if st.button("Run Model Comparison"):

        try:

            with st.spinner("Estimating nine models..."):

                comparison = compare_models(returns)

            st.session_state.model_comparison = comparison

            st.success("Model comparison completed.")

        except Exception as error:
            show_error(error)

    if st.session_state.model_comparison is not None:

        st.dataframe(
            st.session_state.model_comparison,
            use_container_width=True
        )

        st.caption(
            "Lower AIC and BIC indicate better relative in-sample fit. "
            "They do not establish superior out-of-sample performance."
        )


# ============================================================
# PAGE 4: MODEL DIAGNOSTICS
# ============================================================

elif page == "Model Diagnostics":

    st.header("Model Diagnostics")

    diagnostics = st.session_state.diagnostics

    diagnostics_df = pd.DataFrame(
        list(diagnostics.items()),
        columns=["Diagnostic", "Value"]
    )

    st.dataframe(
        diagnostics_df,
        use_container_width=True
    )

    st.subheader("Diagnostic Interpretation")

    for name, value in diagnostics.items():

        if "p-value" in name.lower():

            if value < 0.05:
                st.warning(
                    f"{name}: {value:.4f} — Reject the null hypothesis."
                )

            else:
                st.success(
                    f"{name}: {value:.4f} — Do not reject the null hypothesis."
                )

    st.caption(
        "Ljung-Box tests examine residual autocorrelation. "
        "ARCH LM examines remaining ARCH effects. "
        "A high p-value does not prove correct model specification."
    )

    st.subheader("Standardized Residuals")

    residuals = pd.Series(result.std_resid).dropna()

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
        make_chart_layout(
            fig,
            "Standardized Residuals",
            "Residual"
        ),
        use_container_width=True
    )


# ============================================================
# PAGE 5: FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("Volatility Forecasting")

    st.subheader(
        f"{forecast_horizon}-Day Conditional Variance Forecast"
    )

    try:

        forecast = st.session_state.forecast_result

        variance_values = (
            forecast.variance.iloc[-1].to_numpy(dtype=float)
        )

        volatility_values = np.sqrt(variance_values)

        forecast_table = pd.DataFrame({
            "Forecast Day": np.arange(
                1,
                len(variance_values) + 1
            ),
            "Forecast Variance": variance_values,
            "Forecast Volatility (%)": volatility_values
        })

        st.dataframe(
            forecast_table,
            use_container_width=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=forecast_table["Forecast Day"],
                y=forecast_table["Forecast Volatility (%)"],
                mode="lines+markers",
                name="Forecast Volatility"
            )
        )

        fig.update_layout(
            title="Future Conditional Volatility",
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            font=dict(color="#e5e7eb"),
            xaxis_title="Forecast Horizon (Trading Days)",
            yaxis_title="Volatility (%)",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "EGARCH multi-step forecasts use simulation. "
            "These are conditional model forecasts, not guaranteed outcomes."
        )

    except Exception as error:
        show_error(error)

    st.divider()

    st.subheader("Rolling Out-of-Sample Evaluation")

    rolling_window = st.number_input(
        "Rolling Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 20)),
        step=50,
        key="rolling_window"
    )

    if st.button("Run Rolling Forecast Evaluation"):

        try:

            with st.spinner(
                "Estimating rolling forecasts. This may take some time..."
            ):

                rolling = rolling_forecast(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(rolling_window)
                )

                evaluation = evaluate_forecasts(rolling)

            st.session_state.rolling_forecasts = rolling
            st.session_state.forecast_evaluation = evaluation

            st.success("Rolling forecast evaluation completed.")

        except Exception as error:
            show_error(error)

    if st.session_state.forecast_evaluation is not None:

        evaluation = st.session_state.forecast_evaluation

        col1, col2, col3 = st.columns(3)

        col1.metric("MSE", f"{evaluation['MSE']:.6f}")
        col2.metric("QLIKE", f"{evaluation['QLIKE']:.6f}")
        col3.metric(
            "Forecast Observations",
            evaluation["Forecast Observations"]
        )

        st.dataframe(
            st.session_state.rolling_forecasts.tail(20),
            use_container_width=True
        )

        st.caption(
            "Squared returns are a noisy proxy for realized variance. "
            "Use the same forecast dates when comparing models."
        )


# ============================================================
# PAGE 6: RISK MANAGEMENT
# ============================================================

elif page == "Risk Management":

    st.header("Risk Management")

    confidence = st.selectbox(
        "Confidence Level",
        [0.90, 0.95, 0.975, 0.99],
        index=1,
        format_func=lambda x: f"{x * 100:.1f}%"
    )

    st.subheader("Historical VaR and Expected Shortfall")

    try:

        historical = historical_risk(
            returns,
            confidence=confidence
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Historical VaR",
            f"{historical['VaR']:.4f}%"
        )

        col2.metric(
            "Historical Expected Shortfall",
            f"{historical['ES']:.4f}%"
        )

    except Exception as error:
        show_error(error)

    st.divider()

    st.subheader("Parametric Student-t Risk")

    if st.session_state.selected_distribution == "t":

        try:

            one_day_forecast = result.forecast(
                horizon=1,
                method="analytic",
                reindex=False
            )

            next_variance = float(
                one_day_forecast.variance.iloc[-1, 0]
            )

            parametric = student_t_risk(
                result,
                confidence=confidence,
                forecast_variance=next_variance
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Forecast Student-t VaR",
                f"{parametric['VaR']:.4f}%"
            )

            col2.metric(
                "Forecast Student-t ES",
                f"{parametric['ES']:.4f}%"
            )

            st.caption(
                f"Degrees of freedom: "
                f"{parametric['Degrees of Freedom']:.3f}"
            )

        except Exception as error:
            show_error(error)

    else:

        st.info(
            "Student-t VaR requires the Student-t distribution. "
            "Select 't' in the sidebar and rerun the analysis."
        )

    st.divider()

    st.subheader("Rolling VaR Backtesting")

    st.warning(
        "This procedure repeatedly estimates EGARCH Student-t models "
        "and can take several minutes."
    )

    backtest_window = st.number_input(
        "Backtesting Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(500, len(returns) - 20)),
        step=50,
        key="backtest_window"
    )

    if st.button("Run VaR Backtest"):

        try:

            with st.spinner(
                "Estimating chronological VaR forecasts..."
            ):

                var_forecasts = build_rolling_var(
                    returns,
                    confidence=confidence,
                    window=int(backtest_window)
                )

                backtest_results = var_backtests(
                    var_forecasts["Actual Return"],
                    var_forecasts["VaR"],
                    confidence=confidence
                )

            st.session_state.var_forecasts = var_forecasts
            st.session_state.var_backtest_results = backtest_results

            st.success("VaR backtesting completed.")

        except Exception as error:
            show_error(error)

    if st.session_state.var_backtest_results is not None:

        results = st.session_state.var_backtest_results

        st.subheader("Backtesting Results")

        result_df = pd.DataFrame(
            list(results.items()),
            columns=["Statistic", "Value"]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        var_data = st.session_state.var_forecasts

        st.caption(
            f"Attempted forecasts: {var_data.attrs.get('attempted', 'N/A')} | "
            f"Failed fits: {var_data.attrs.get('failed', 'N/A')}"
        )

        st.caption(
            "A p-value below 0.05 rejects the relevant test null. "
            "Non-rejection does not prove that VaR is accurate."
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=-var_data["VaR"],
                mode="lines",
                name="Negative VaR Threshold"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data["Actual Return"],
                mode="lines",
                name="Actual Return"
            )
        )

        st.plotly_chart(
            make_chart_layout(
                fig,
                "Actual Returns and VaR Threshold",
                "Return (%)"
            ),
            use_container_width=True
        )


# ============================================================
# PAGE 7: MODEL STABILITY AND CALIBRATION
# ============================================================

elif page == "Model Stability & Calibration":

    st.header("Model Stability & Calibration")

    st.markdown(
        "Evaluate whether model parameters and innovation "
        "distribution assumptions remain consistent over time."
    )

    # --------------------------------------------------------
    # ROLLING PARAMETER STABILITY
    # --------------------------------------------------------

    st.subheader("Rolling Parameter Stability")

    stability_window = st.number_input(
        "Stability Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 100)),
        step=50,
        key="stability_window"
    )

    stability_step = st.number_input(
        "Estimation Step",
        min_value=10,
        max_value=250,
        value=50,
        step=10
    )

    if st.button("Run Stability Analysis"):

        try:

            with st.spinner("Estimating rolling model parameters..."):

                estimates, summary = rolling_parameter_stability(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(stability_window),
                    step=int(stability_step)
                )

            st.session_state.stability_estimates = estimates
            st.session_state.stability_summary = summary

            st.success("Stability analysis completed.")

        except Exception as error:
            show_error(error)

    if st.session_state.stability_estimates is not None:

        estimates = st.session_state.stability_estimates
        summary = st.session_state.stability_summary

        st.subheader("Parameter Summary")

        st.dataframe(
            summary,
            use_container_width=True
        )

        parameter_names = [
            name for name in estimates.columns
            if name not in ["Log Likelihood", "AIC", "BIC"]
        ]

        selected_parameter = st.selectbox(
            "Select Parameter",
            parameter_names
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=estimates.index,
                y=estimates[selected_parameter],
                mode="lines+markers",
                name=selected_parameter
            )
        )

        st.plotly_chart(
            make_chart_layout(
                fig,
                f"Rolling Estimate: {selected_parameter}",
                "Parameter Estimate"
            ),
            use_container_width=True
        )

        st.caption(
            "Rolling estimates are descriptive evidence of parameter "
            "variation. They are not, by themselves, a formal structural "
            "break test."
        )

    st.divider()

    # --------------------------------------------------------
    # PIT DIAGNOSTICS
    # --------------------------------------------------------

    st.subheader("Student-t Distribution Calibration")

    if st.session_state.selected_distribution == "t":

        if st.button("Run PIT Diagnostics"):

            try:

                pit, pit_summary = student_t_pit(result)

                st.session_state.pit_values = pit
                st.session_state.pit_summary = pit_summary

                st.success("PIT diagnostics completed.")

            except Exception as error:
                show_error(error)

        if st.session_state.pit_summary is not None:

            pit_summary = st.session_state.pit_summary

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "KS Statistic",
                f"{pit_summary['KS Statistic']:.4f}"
            )

            col2.metric(
                "KS P-value",
                f"{pit_summary['KS P-value']:.4f}"
            )

            col3.metric(
                "PIT Mean",
                f"{pit_summary['PIT Mean']:.4f}"
            )

            st.write("PIT Summary")

            st.dataframe(
                pd.DataFrame(
                    list(pit_summary.items()),
                    columns=["Statistic", "Value"]
                ),
                use_container_width=True
            )

            pit = st.session_state.pit_values

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=pit,
                    nbinsx=20,
                    name="PIT Values"
                )
            )

            st.plotly_chart(
                make_chart_layout(
                    fig,
                    "Probability Integral Transform",
                    "Frequency"
                ),
                use_container_width=True
            )

            st.caption(
                "A well-calibrated conditional distribution should produce "
                "approximately uniform PIT values. The ordinary KS test "
                "does not adjust for estimated model parameters."
            )

    else:

        st.info(
            "PIT calibration in this module is implemented for "
            "Student-t innovations. Select distribution 't' and "
            "rerun the model to use it."
        )


# ============================================================
# PAGE 8: FORECAST COMPARISON
# ============================================================

elif page == "Forecast Comparison":

    st.header("Forecast Comparison")

    st.markdown(
        "Compare your selected volatility model against a "
        "RiskMetrics-style EWMA benchmark."
    )

    st.info(
        "Both forecasts use the same chronological forecast dates. "
        "Squared returns are used as the realized variance proxy."
    )

    comparison_window = st.number_input(
        "Common Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 20)),
        step=50,
        key="comparison_window"
    )

    ewma_decay = st.slider(
        "EWMA Decay Factor",
        min_value=0.80,
        max_value=0.99,
        value=0.94,
        step=0.01
    )

    if st.button("Run EGARCH vs EWMA Comparison"):

        try:

            with st.spinner(
                "Estimating model forecasts and EWMA benchmark..."
            ):

                model_forecasts = rolling_forecast(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(comparison_window)
                )

                ewma_forecasts = make_ewma_forecasts(
                    returns,
                    forecast_index=model_forecasts.index,
                    window=int(comparison_window),
                    decay=ewma_decay
                )

                common = model_forecasts.join(
                    ewma_forecasts[
                        ["EWMA Variance", "EWMA Volatility"]
                    ],
                    how="inner"
                ).dropna()

                if len(common) < 30:
                    raise ValueError(
                        "Fewer than 30 common forecasts are available."
                    )

                actual_variance = (
                    common["Actual Return"] ** 2
                )

                model_variance = common["Forecast Variance"]
                ewma_variance = common["EWMA Variance"]

                model_qlike = float(
                    np.mean(
                        np.log(model_variance)
                        + actual_variance / model_variance
                    )
                )

                ewma_qlike = float(
                    np.mean(
                        np.log(ewma_variance)
                        + actual_variance / ewma_variance
                    )
                )

                model_mse = float(
                    np.mean(
                        (actual_variance - model_variance) ** 2
                    )
                )

                ewma_mse = float(
                    np.mean(
                        (actual_variance - ewma_variance) ** 2
                    )
                )

                dm_qlike = diebold_mariano_test(
                    actual_variance,
                    model_variance,
                    ewma_variance,
                    horizon=1,
                    loss="qlike",
                    bandwidth=0
                )

                dm_mse = diebold_mariano_test(
                    actual_variance,
                    model_variance,
                    ewma_variance,
                    horizon=1,
                    loss="mse",
                    bandwidth=0
                )

                comparison_results = {
                    "Model QLIKE": model_qlike,
                    "EWMA QLIKE": ewma_qlike,
                    "Model MSE": model_mse,
                    "EWMA MSE": ewma_mse,
                    "DM QLIKE": dm_qlike,
                    "DM MSE": dm_mse,
                    "Forecast Count": len(common)
                }

                st.session_state.forecast_comparison = comparison_results
                st.session_state.comparison_forecasts = common

            st.success("Forecast comparison completed.")

        except Exception as error:
            show_error(error)

    if st.session_state.forecast_comparison is not None:

        comparison = st.session_state.forecast_comparison

        st.subheader("Forecast Performance")

        col1, col2 = st.columns(2)

        col1.metric(
            f"{st.session_state.selected_model} QLIKE",
            f"{comparison['Model QLIKE']:.6f}"
        )

        col2.metric(
            "EWMA QLIKE",
            f"{comparison['EWMA QLIKE']:.6f}"
        )

        col3, col4 = st.columns(2)

        col3.metric(
            f"{st.session_state.selected_model} MSE",
            f"{comparison['Model MSE']:.6f}"
        )

        col4.metric(
            "EWMA MSE",
            f"{comparison['EWMA MSE']:.6f}"
        )

        st.write(
            f"Common forecast observations: "
            f"{comparison['Forecast Count']}"
        )

        st.divider()

        st.subheader("Diebold-Mariano Test: QLIKE")

        dm_qlike = comparison["DM QLIKE"]

        st.write(
            f"DM statistic: {dm_qlike['DM Statistic']:.4f}"
        )

        st.write(
            f"P-value: {dm_qlike['P-value']:.4f}"
        )

        st.write(
            f"Lower average loss: {dm_qlike['Lower Average Loss']}"
        )

        st.subheader("Diebold-Mariano Test: MSE")

        dm_mse = comparison["DM MSE"]

        st.write(
            f"DM statistic: {dm_mse['DM Statistic']:.4f}"
        )

        st.write(
            f"P-value: {dm_mse['P-value']:.4f}"
        )

        st.write(
            f"Lower average loss: {dm_mse['Lower Average Loss']}"
        )

        common = st.session_state.comparison_forecasts

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=common.index,
                y=common["Forecast Volatility"],
                mode="lines",
                name="Selected Model"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=common.index,
                y=common["EWMA Volatility"],
                mode="lines",
                name="EWMA"
            )
        )

        st.plotly_chart(
            make_chart_layout(
                fig,
                "Out-of-Sample Volatility Forecasts",
                "Forecast Volatility (%)"
            ),
            use_container_width=True
        )

        st.caption(
            "Negative DM mean loss difference means the selected model "
            "has lower average loss than EWMA. A p-value below 0.05 "
            "indicates evidence of a difference in predictive accuracy. "
            "The one-step test uses a zero-lag HAC variance estimate."
        )


# ============================================================
# PAGE 9: PORTFOLIO ANALYSIS
# ============================================================

elif page == "Portfolio Analysis":

    st.header("Portfolio Analysis")

    st.write(
        "Construct an equal-weighted portfolio from selected Dutch equities."
    )

    portfolio_companies = st.multiselect(
        "Select Portfolio Companies",
        list(DUTCH_STOCKS.keys()),
        default=[
            name for name in ["ASML", "Shell", "ING Group"]
            if name in DUTCH_STOCKS
        ]
    )

    if len(portfolio_companies) < 2:

        st.info("Select at least two companies.")

    elif st.button("Calculate Portfolio"):

        try:

            portfolio_returns = {}

            with st.spinner("Downloading portfolio market data..."):

                for name in portfolio_companies:

                    symbol = DUTCH_STOCKS[name]

                    stock = get_stock_data(
                        symbol,
                        start_date,
                        end_date + timedelta(days=1)
                    )

                    stock_returns = calculate_returns(
                        stock["Close"]
                    )

                    portfolio_returns[name] = stock_returns

            returns_df = pd.DataFrame(
                portfolio_returns
            ).dropna()

            if len(returns_df) < 30:
                raise ValueError(
                    "Not enough common observations for portfolio analysis."
                )

            weights = np.repeat(
                1 / len(portfolio_companies),
                len(portfolio_companies)
            )

            portfolio = returns_df.dot(weights)

            annual_return = float(
                portfolio.mean() * 252
            )

            annual_volatility = float(
                portfolio.std() * np.sqrt(252)
            )

            sharpe_ratio = (
                annual_return / annual_volatility
                if annual_volatility > 0
                else np.nan
            )

            portfolio_growth = np.exp(
                portfolio.cumsum() / 100
            )

            portfolio_var = historical_risk(
                portfolio,
                confidence=0.95
            )

            st.session_state.portfolio_returns = returns_df
            st.session_state.portfolio_series = portfolio

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Annualized Mean Log Return",
                f"{annual_return:.2f}%"
            )

            col2.metric(
                "Annualized Volatility",
                f"{annual_volatility:.2f}%"
            )

            col3.metric(
                "Sharpe Ratio (0% risk-free rate)",
                f"{sharpe_ratio:.3f}"
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "Portfolio 95% Historical VaR",
                f"{portfolio_var['VaR']:.3f}%"
            )

            col5.metric(
                "Portfolio 95% Historical ES",
                f"{portfolio_var['ES']:.3f}%"
            )

            st.subheader("Portfolio Growth")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=portfolio_growth.index,
                    y=portfolio_growth,
                    mode="lines",
                    name="Portfolio"
                )
            )

            st.plotly_chart(
                make_chart_layout(
                    fig,
                    "Growth of €1 Invested",
                    "Portfolio Value (€)"
                ),
                use_container_width=True
            )

            st.subheader("Correlation Matrix")

            correlation = returns_df.corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.index,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1
                )
            )

            fig.update_layout(
                title="Asset Return Correlations",
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#111827",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Portfolio Return Statistics")

            st.dataframe(
                returns_df.describe(),
                use_container_width=True
            )

            st.caption(
                "Equal-weighted baseline. The Sharpe ratio assumes a "
                "zero risk-free rate. Transaction costs, taxes, rebalancing "
                "and dynamic covariance are not included."
            )

        except Exception as error:
            show_error(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Dutch Quant | Quantitative Financial Research Platform | "
    "Educational and research use only. Not investment advice."
)
