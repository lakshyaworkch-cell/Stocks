
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dutch Quant | Financial Research",
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

    .info-card {
        background-color: #172033;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #263449;
        margin-bottom: 12px;
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

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "returns" not in st.session_state:
    st.session_state.returns = None

if "model_result" not in st.session_state:
    st.session_state.model_result = None

if "model_parameters" not in st.session_state:
    st.session_state.model_parameters = None

if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "selected_distribution" not in st.session_state:
    st.session_state.selected_distribution = None


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
    Generate future conditional variance forecasts.

    EGARCH does not support analytic forecasts for horizons
    greater than one in the arch package.

    Therefore:
        EGARCH + horizon > 1 -> simulation
        Other cases -> analytic

    Returns the arch forecast object.
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
    Produce chronological one-step-ahead EGARCH Student-t VaR forecasts.

    Each VaR forecast is estimated using only past observations.
    """

    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) <= window:
        raise ValueError(
            "There are not enough observations for rolling VaR."
        )

    records = []

    for i in range(window, len(returns)):

        training_data = returns.iloc[i-window:i]

        try:
            result = fit_model(
                training_data,
                model_name="EGARCH",
                distribution="t"
            )

            one_day_forecast = result.forecast(
    horizon=1,
    method="analytic",
    reindex=False
)

next_variance = float(
    one_day_forecast.variance.iloc[-1, 0]
)

risk = student_t_risk(
    result,
    confidence=confidence,
    forecast_variance=next_variance
)

            records.append({
                "Date": returns.index[i],
                "Actual Return": float(returns.iloc[i]),
                "VaR": risk["VaR"]
            })

        except Exception:
            continue

    if not records:
        raise RuntimeError(
            "Rolling VaR estimation failed for all forecast dates."
        )

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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dutch Quant")

st.sidebar.caption("Research & Analytics Platform")

page = st.sidebar.radio(
    "Navigation",
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
    "Student-t innovations are recommended for heavy-tailed "
    "financial return distributions."
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

            with st.spinner(
                "Generating volatility forecasts..."
            ):

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
            st.error(
                f"Model estimation failed: {error}"
            )


# ============================================================
# CHECK WHETHER ANALYSIS HAS BEEN RUN
# ============================================================

data = st.session_state.stock_data
returns = st.session_state.returns
result = st.session_state.model_result

if data is None or returns is None or result is None:

    st.info(
        "Select a company and date range in the sidebar, "
        "then click Run Analysis."
    )

    st.markdown("### Platform Features")

    col1, col2, col3 = st.columns(3)

    col1.metric("Volatility Models", "3")
    col2.metric("Innovation Distributions", "3")
    col3.metric("Risk Frameworks", "2")

    st.markdown("""
    **Available quantitative tools**

    - EGARCH(1,1), GARCH(1,1), and GJR-GARCH
    - Normal, Student-t, and skewed Student-t innovations
    - ADF and KPSS stationarity testing
    - Residual and ARCH diagnostics
    - Rolling volatility forecasting
    - Parametric and historical VaR and Expected Shortfall
    - Kupiec and Christoffersen VaR backtesting
    - Equal-weighted portfolio analysis
    """)

    st.stop()


# ============================================================
# GLOBAL MODEL INFORMATION
# ============================================================

st.caption(
    f"Company: {company} | "
    f"Ticker: {ticker} | "
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
        line_dash="dash",
        line_color="white"
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
            "ADF null hypothesis: unit root. "
            "KPSS null hypothesis: stationarity. "
            "A p-value below 0.05 rejects the relevant null."
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

    st.subheader("Selected Model Parameters")

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
        result.conditional_volatility,
        index=returns.index[-len(result.conditional_volatility):]
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

    st.subheader("Compare Alternative Models")

    st.warning(
        "Model comparison estimates nine models and may take some time."
    )

    if st.button("Run Model Comparison"):

        try:

            with st.spinner("Estimating nine volatility models..."):

                comparison = compare_models(returns)

            st.session_state.model_comparison = comparison

            st.success("Model comparison completed.")

        except Exception as error:
            show_error(error)

    if "model_comparison" in st.session_state:

        st.dataframe(
            st.session_state.model_comparison,
            use_container_width=True
        )

        st.caption(
            "Lower AIC and BIC indicate a better relative fit, "
            "but do not establish superior out-of-sample forecasting."
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
        "For Ljung-Box tests, the null is no serial correlation. "
        "For the ARCH LM test, the null is no remaining ARCH effects. "
        "A high p-value does not prove that a model is correctly specified."
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
            "For EGARCH forecasts beyond one day, the application "
            "uses Monte Carlo simulation with 10,000 simulations. "
            "These are model-based forecasts, not guaranteed future returns."
        )

    except Exception as error:
        show_error(error)

    st.divider()

    st.subheader("Rolling Out-of-Sample Forecast Evaluation")

    rolling_window = st.number_input(
        "Rolling Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 20)),
        step=50
    )

    if st.button("Run Rolling Forecast Evaluation"):

        try:

            with st.spinner(
                "Running chronological rolling forecasts..."
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

    if "forecast_evaluation" in st.session_state:

        evaluation = st.session_state.forecast_evaluation

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MSE",
            f"{evaluation['MSE']:.6f}"
        )

        col2.metric(
            "QLIKE",
            f"{evaluation['QLIKE']:.6f}"
        )

        col3.metric(
            "Forecast Observations",
            evaluation["Forecast Observations"]
        )

        st.dataframe(
            st.session_state.rolling_forecasts.tail(20),
            use_container_width=True
        )

        st.caption(
            "Squared returns are used as a noisy proxy for realized variance. "
            "MSE and QLIKE should be compared across models on the same dates."
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

            parametric = student_t_risk(
                result,
                confidence=confidence
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Student-t VaR",
                f"{parametric['VaR']:.4f}%"
            )

            col2.metric(
                "Student-t Expected Shortfall",
                f"{parametric['ES']:.4f}%"
            )

            st.caption(
                f"Estimated Student-t degrees of freedom: "
                f"{parametric['Degrees of Freedom']:.3f}"
            )

        except Exception as error:
            show_error(error)

    else:

        st.info(
            "Parametric Student-t VaR requires the model to be fitted "
            "with the Student-t distribution. Select distribution 't' "
            "in the sidebar and run the analysis again."
        )

    st.divider()

    st.subheader("Rolling VaR Backtesting")

    st.warning(
        "This procedure repeatedly estimates EGARCH Student-t models "
        "and may take several minutes."
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
                "Estimating rolling VaR forecasts. Please wait..."
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

    if "var_backtest_results" in st.session_state:

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

        st.caption(
            "For Kupiec, Christoffersen independence and conditional "
            "coverage tests, a p-value below 0.05 rejects the relevant "
            "null hypothesis. A non-rejection does not prove model accuracy."
        )

        var_data = st.session_state.var_forecasts

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
# PAGE 7: PORTFOLIO ANALYSIS
# ============================================================

elif page == "Portfolio Analysis":

    st.header("Portfolio Analysis")

    st.write(
        "Build an equal-weighted portfolio from selected Dutch equities."
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

            st.session_state.portfolio_returns = returns_df
            st.session_state.portfolio_series = portfolio

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Annualized Return",
                f"{annual_return:.2f}%"
            )

            col2.metric(
                "Annualized Volatility",
                f"{annual_volatility:.2f}%"
            )

            col3.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.3f}"
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
                "This is a simple equal-weighted portfolio baseline. "
                "It excludes transaction costs, taxes, rebalancing costs "
                "and currency effects."
            )

        except Exception as error:
            show_error(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Dutch Quant | Quantitative Research Platform | "
    "For educational and research purposes only. "
    "Not investment advice."
)
