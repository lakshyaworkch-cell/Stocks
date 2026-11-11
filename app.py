
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta

from modules.data import DUTCH_STOCKS, get_stock_data
from modules.preprocessing import calculate_returns
from modules.stationarity import stationarity_tests
from modules.analysis import fit_model, compare_models, get_parameters
from modules.diagnostics import get_diagnostics
from modules.forecasting import rolling_forecast, evaluate_forecasts
from modules.risk import student_t_risk, historical_risk
from modules.backtesting import var_backtests
from modules.stability import rolling_parameter_stability
from modules.pit_diagnostics import student_t_pit
from modules.forecast_tests import diebold_mariano_test


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HIDE STREAMLIT AUTOMATIC MULTIPAGE NAVIGATION
# ============================================================

st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# VISUAL DESIGN
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111b2e;
        border-right: 1px solid #26364d;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #cbd5e1;
    }

    div[data-testid="stMetric"] {
        background: #17243a;
        border: 1px solid #2b3b54;
        padding: 16px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    .research-card {
        background: #17243a;
        border: 1px solid #2b3b54;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    hr {
        border-color: #26364d;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "stock_data": None,
    "returns": None,
    "model_result": None,
    "model_parameters": None,
    "diagnostics": None,
    "forecast_result": None,

    "selected_company": None,
    "selected_ticker": None,
    "selected_model": None,
    "selected_distribution": None,

    "analysis_start": None,
    "analysis_end": None,
    "analysis_horizon": 5,

    "model_comparison": None,
    "rolling_forecasts": None,
    "forecast_evaluation": None,

    "stability_estimates": None,
    "stability_summary": None,

    "pit_values": None,
    "pit_summary": None,

    "forecast_comparison": None,
    "comparison_forecasts": None,

    "var_forecasts": None,
    "var_backtest_results": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clear_old_results():

    keys = [
        "model_comparison",
        "rolling_forecasts",
        "forecast_evaluation",
        "stability_estimates",
        "stability_summary",
        "pit_values",
        "pit_summary",
        "forecast_comparison",
        "comparison_forecasts",
        "var_forecasts",
        "var_backtest_results",
    ]

    for key in keys:
        st.session_state[key] = None


def make_chart(fig, title, y_title, height=420):

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111b2e",
        font=dict(color="#e5e7eb"),
        xaxis_title="Date",
        yaxis_title=y_title,
        height=height,
        margin=dict(l=25, r=25, t=60, b=25)
    )

    return fig


def show_error(error):
    st.error(f"Error: {error}")


def make_volatility_forecast(result, horizon, model_name):

    if model_name.upper() == "EGARCH" and horizon > 1:

        return result.forecast(
            horizon=horizon,
            method="simulation",
            simulations=10000,
            reindex=False
        )

    return result.forecast(
        horizon=horizon,
        method="analytic",
        reindex=False
    )


def build_rolling_var(returns, confidence=0.95, window=500):

    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) <= window:
        raise ValueError("Insufficient observations for rolling VaR.")

    records = []
    failed = 0

    for i in range(window, len(returns)):

        train = returns.iloc[i-window:i]

        try:

            fitted = fit_model(
                train,
                model_name="EGARCH",
                distribution="t"
            )

            forecast = fitted.forecast(
                horizon=1,
                method="analytic",
                reindex=False
            )

            variance = float(forecast.variance.iloc[-1, 0])

            risk = student_t_risk(
                fitted,
                confidence=confidence,
                forecast_variance=variance
            )

            records.append({
                "Date": returns.index[i],
                "Actual Return": float(returns.iloc[i]),
                "VaR": risk["VaR"],
                "ES": risk["ES"]
            })

        except Exception:
            failed += 1

    if not records:
        raise RuntimeError("No valid rolling VaR forecasts.")

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

    returns = pd.Series(returns).dropna().astype(float)

    records = []

    for forecast_date in forecast_index:

        position = returns.index.get_loc(forecast_date)

        if not isinstance(position, (int, np.integer)):
            continue

        if position < window:
            continue

        train = returns.iloc[position-window:position]

        initial_count = min(30, len(train))

        variance = float(
            train.iloc[:initial_count].var(ddof=1)
        )

        if not np.isfinite(variance) or variance <= 0:
            continue

        for observation in train.iloc[initial_count:]:

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
        raise RuntimeError("No valid EWMA forecasts.")

    return pd.DataFrame(records).set_index("Date")


# ============================================================
# SINGLE SIDEBAR
# ============================================================

st.sidebar.title("Stock Research")


# 1. MARKET SELECTION

st.sidebar.header("1. Market Selection")

company = st.sidebar.selectbox(
    "Select Dutch Stock",
    options=list(DUTCH_STOCKS.keys()),
    format_func=lambda x: f"{x} — {DUTCH_STOCKS[x]}"
)

ticker = DUTCH_STOCKS[company]


# 2. DATE SELECTION

st.sidebar.header("2. Date Selection")

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)


# 3. MODEL CONFIGURATION

st.sidebar.header("3. Model Configuration")

model_name = st.sidebar.selectbox(
    "Volatility Model",
    ["EGARCH", "GARCH", "GJR-GARCH"]
)

distribution = st.sidebar.selectbox(
    "Error Distribution",
    ["t", "normal", "skewt"]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=10,
    value=1
)


# 4. RUN ANALYSIS

st.sidebar.header("4. Run Analysis")

run_analysis = st.sidebar.button(
    "Load Data & Run Analysis",
    use_container_width=True
)


# 5. RESEARCH NAVIGATION

st.sidebar.header("5. Research Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Market Data",
        "Statistical Analysis",
        "Volatility Models",
        "Model Diagnostics",
        "Forecasting",
        "Risk Management",
        "Model Stability & Calibration",
        "Forecast Comparison"
    ]
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:

    if start_date >= end_date:

        st.sidebar.error(
            "End date must be after start date."
        )

    else:

        try:

            with st.spinner("Downloading stock market data..."):

                data = get_stock_data(
                    ticker,
                    start_date,
                    end_date + timedelta(days=1)
                )

                returns = calculate_returns(data["Close"])

            with st.spinner("Estimating volatility model..."):

                fitted = fit_model(
                    returns,
                    model_name=model_name,
                    distribution=distribution
                )

                parameters = get_parameters(fitted)
                diagnostics = get_diagnostics(fitted)

            with st.spinner("Generating volatility forecasts..."):

                forecast = make_volatility_forecast(
                    fitted,
                    forecast_horizon,
                    model_name
                )

            clear_old_results()

            st.session_state.stock_data = data
            st.session_state.returns = returns
            st.session_state.model_result = fitted
            st.session_state.model_parameters = parameters
            st.session_state.diagnostics = diagnostics
            st.session_state.forecast_result = forecast

            st.session_state.selected_company = company
            st.session_state.selected_ticker = ticker
            st.session_state.selected_model = model_name
            st.session_state.selected_distribution = distribution

            st.session_state.analysis_start = start_date
            st.session_state.analysis_end = end_date
            st.session_state.analysis_horizon = forecast_horizon

            st.sidebar.success(
                "Analysis completed successfully!"
            )

        except Exception as error:

            st.error(f"Analysis failed: {error}")


# ============================================================
# LOAD STORED DATA
# ============================================================

data = st.session_state.stock_data
returns = st.session_state.returns
result = st.session_state.model_result

has_data = (
    data is not None
    and returns is not None
    and result is not None
)


# ============================================================
# CURRENT SESSION - OVERVIEW ONLY
# ============================================================

if has_data and page == "Overview":

    st.markdown("### Current Research Session")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Stock",
        st.session_state.selected_company
    )

    c2.metric(
        "Ticker",
        st.session_state.selected_ticker
    )

    c3.metric(
        "Volatility Model",
        st.session_state.selected_model
    )

    c4.metric(
        "Observations",
        f"{len(returns):,}"
    )

    st.caption(
        f"Period: {st.session_state.analysis_start} to "
        f"{st.session_state.analysis_end} | "
        f"Distribution: {st.session_state.selected_distribution}"
    )

    st.divider()


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Quantitative Research Dashboard")

    st.write(
        "Select a Dutch stock from the left sidebar, configure "
        "your model and load the data to begin your analysis."
    )

    if has_data:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Latest Price",
            f"€{data['Close'].iloc[-1]:,.2f}"
        )

        col2.metric(
            "Latest Return",
            f"{returns.iloc[-1]:.2f}%"
        )

        col3.metric(
            "Historical Annual Volatility",
            f"{returns.std() * np.sqrt(252):.2f}%"
        )

        col4.metric(
            "Model AIC",
            f"{result.aic:.2f}"
        )

        st.subheader("Stock Price Overview")

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
            make_chart(fig, "Historical Stock Price", "Price (€)"),
            use_container_width=True
        )

        cols = st.columns(3)

        sections = [
            ("Market Data", "Price and return analysis"),
            ("Statistical Analysis", "Distribution and stationarity"),
            ("Volatility Models", "EGARCH and GARCH estimation"),
            ("Model Diagnostics", "Residual and ARCH testing"),
            ("Forecasting", "Future conditional volatility"),
            ("Risk Management", "VaR and Expected Shortfall"),
            ("Model Stability & Calibration", "Parameter and PIT testing"),
            ("Forecast Comparison", "EWMA and DM comparison"),
        ]

        for i, (name, description) in enumerate(sections):

            with cols[i % 3]:

                st.markdown(
                    f"""
                    <div class="research-card">
                        <b>{name}</b><br>
                        <span style="color:#94a3b8">{description}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.info(
            "Your dashboard is ready. Select a stock and click "
            "'Load Data & Run Analysis' in the left sidebar."
        )

        st.subheader("Available Research Tools")

        cols = st.columns(3)

        cols[0].metric("Volatility Models", "3")
        cols[1].metric("Return Distributions", "3")
        cols[2].metric("Research Pages", "8")

        st.markdown("""
        **Available analysis**

        - EGARCH(1,1), GARCH(1,1), GJR-GARCH
        - Student-t, normal and skewed Student-t innovations
        - ADF and KPSS stationarity tests
        - Residual and ARCH diagnostics
        - Multi-step volatility forecasts
        - Rolling out-of-sample evaluation
        - EWMA benchmark and Diebold-Mariano tests
        - Parameter stability and PIT calibration
        - Historical and parametric VaR and ES
        - Kupiec and Christoffersen backtesting
        """)


# ============================================================
# PAGES REQUIRING DATA
# ============================================================

elif not has_data:

    st.header(page)

    st.warning(
        "No stock analysis has been loaded yet."
    )

    st.markdown("""
    To display this research section:

    1. Select a company in the left sidebar.
    2. Select the start and end dates.
    3. Select the volatility model and distribution.
    4. Click **Load Data & Run Analysis**.
    """)

    st.info(
        "Load market data to populate this section's charts and statistics."
    )


# ============================================================
# PAGE 2: MARKET DATA
# ============================================================

elif page == "Market Data":

    st.header("Market Data")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest Price",
        f"€{data['Close'].iloc[-1]:,.2f}"
    )

    c2.metric(
        "Latest Daily Return",
        f"{returns.iloc[-1]:.2f}%"
    )

    c3.metric(
        "Annualized Volatility",
        f"{returns.std() * np.sqrt(252):.2f}%"
    )

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
        make_chart(fig, "Historical Stock Price", "Price (€)"),
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="Daily Returns"
        )
    )

    fig.add_hline(y=0, line_dash="dash")

    st.plotly_chart(
        make_chart(fig, "Daily Log Returns", "Return (%)"),
        use_container_width=True
    )

    st.subheader("Recent Historical Data")

    concise_columns = [
        column for column in ["Close", "Volume"]
        if column in data.columns
    ]

    st.dataframe(
        data[concise_columns].tail(10).round(2),
        use_container_width=True,
        height=320
    )


# ============================================================
# PAGE 3: STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.header("Statistical Analysis")

    st.subheader("Descriptive Statistics")

    stat_cols = st.columns(4)

    stat_cols[0].metric(
        "Observations",
        f"{returns.count():,}"
    )

    stat_cols[1].metric(
        "Mean Return",
        f"{returns.mean():.3f}%"
    )

    stat_cols[2].metric(
        "Daily Std. Dev.",
        f"{returns.std():.3f}%"
    )

    stat_cols[3].metric(
        "Annualized Volatility",
        f"{returns.std() * np.sqrt(252):.2f}%"
    )

    concise_stats = pd.DataFrame({
        "Statistic": [
            "Minimum",
            "Median",
            "Maximum",
            "Skewness",
            "Excess Kurtosis"
        ],
        "Value": [
            returns.min(),
            returns.median(),
            returns.max(),
            returns.skew(),
            returns.kurtosis()
        ]
    })

    st.dataframe(
        concise_stats.round(3),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Stationarity Tests")

    try:

        stationarity = stationarity_tests(returns)

        st.dataframe(
            stationarity,
            use_container_width=True
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
        make_chart(fig, "Distribution of Returns", "Frequency"),
        use_container_width=True
    )


# ============================================================
# PAGE 4: VOLATILITY MODELS
# ============================================================

elif page == "Volatility Models":

    st.header("Volatility Models")

    st.subheader("Parameter Estimates")

    st.dataframe(
        st.session_state.model_parameters,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Log Likelihood", f"{result.loglikelihood:.2f}")
    c2.metric("AIC", f"{result.aic:.2f}")
    c3.metric("BIC", f"{result.bic:.2f}")

    volatility = pd.Series(result.conditional_volatility)

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
        make_chart(fig, "Conditional Volatility", "Volatility (%)"),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    if st.button("Run Nine-Model Comparison"):

        try:

            with st.spinner("Estimating models..."):

                comparison = compare_models(returns)

            st.session_state.model_comparison = comparison

            st.success("Comparison completed.")

        except Exception as error:

            show_error(error)

    if st.session_state.model_comparison is not None:

        st.dataframe(
            st.session_state.model_comparison,
            use_container_width=True
        )


# ============================================================
# PAGE 5: MODEL DIAGNOSTICS
# ============================================================

elif page == "Model Diagnostics":

    st.header("Model Diagnostics")

    diagnostics = st.session_state.diagnostics

    diagnostic_table = pd.DataFrame(
        [
            {
                "Diagnostic": name,
                "Result": (
                    f"{value:.4f}"
                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating)
                    )
                    else str(value)
                )
            }
            for name, value in diagnostics.items()
        ]
    )

    st.dataframe(
        diagnostic_table,
        hide_index=True,
        use_container_width=True,
        height=min(260, 45 + 38 * len(diagnostic_table))
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
        make_chart(fig, "Standardized Residuals", "Residual"),
        use_container_width=True
    )

    st.caption(
        "For p-values below 0.05, reject the corresponding null hypothesis. "
        "A larger p-value does not prove the model is correctly specified."
    )


# ============================================================
# PAGE 6: FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("Volatility Forecasting")

    forecast = st.session_state.forecast_result

    variance_values = (
        forecast.variance.iloc[-1].to_numpy(dtype=float)
    )

    forecast_table = pd.DataFrame({
        "Forecast Day": np.arange(1, len(variance_values) + 1),
        "Forecast Variance": variance_values,
        "Forecast Volatility (%)": np.sqrt(variance_values)
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
        template="plotly_dark",
        title="Future Conditional Volatility",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111b2e",
        font=dict(color="#e5e7eb"),
        xaxis_title="Forecast Day",
        yaxis_title="Volatility (%)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Rolling Forecast Evaluation")

    window = st.number_input(
        "Rolling Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 20)),
        step=50
    )

    if st.button("Run Rolling Forecast"):

        try:

            with st.spinner("Running rolling estimation..."):

                rolling = rolling_forecast(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(window)
                )

                evaluation = evaluate_forecasts(rolling)

            st.session_state.rolling_forecasts = rolling
            st.session_state.forecast_evaluation = evaluation

            st.success("Forecast evaluation completed.")

        except Exception as error:

            show_error(error)

    if st.session_state.forecast_evaluation is not None:

        evaluation = st.session_state.forecast_evaluation

        c1, c2, c3 = st.columns(3)

        c1.metric("MSE", f"{evaluation['MSE']:.6f}")
        c2.metric("QLIKE", f"{evaluation['QLIKE']:.6f}")

        c3.metric(
            "Forecast Count",
            evaluation["Forecast Observations"]
        )

        st.dataframe(
            st.session_state.rolling_forecasts.tail(30),
            use_container_width=True
        )


# ============================================================
# PAGE 7: RISK MANAGEMENT
# ============================================================

elif page == "Risk Management":

    st.header("Risk Management")

    confidence = st.selectbox(
        "Confidence Level",
        [0.90, 0.95, 0.975, 0.99],
        index=1
    )

    st.subheader("Historical Risk")

    historical = historical_risk(
        returns,
        confidence=confidence
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Historical VaR",
        f"{historical['VaR']:.4f}%"
    )

    c2.metric(
        "Historical ES",
        f"{historical['ES']:.4f}%"
    )

    st.divider()

    st.subheader("Parametric Student-t Risk")

    if st.session_state.selected_distribution == "t":

        try:

            forecast = result.forecast(
                horizon=1,
                method="analytic",
                reindex=False
            )

            variance = float(forecast.variance.iloc[-1, 0])

            risk = student_t_risk(
                result,
                confidence=confidence,
                forecast_variance=variance
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Forecast VaR",
                f"{risk['VaR']:.4f}%"
            )

            c2.metric(
                "Forecast ES",
                f"{risk['ES']:.4f}%"
            )

            st.caption(
                f"Student-t degrees of freedom: "
                f"{risk['Degrees of Freedom']:.3f}"
            )

        except Exception as error:

            show_error(error)

    else:

        st.info(
            "Select Student-t distribution in the sidebar for "
            "parametric Student-t risk calculations."
        )

    st.divider()

    st.subheader("Rolling VaR Backtesting")

    window = st.number_input(
        "Backtest Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(500, len(returns) - 20)),
        step=50,
        key="var_window"
    )

    if st.button("Run VaR Backtest"):

        try:

            with st.spinner("Running rolling VaR backtest..."):

                var_data = build_rolling_var(
                    returns,
                    confidence=confidence,
                    window=int(window)
                )

                backtest = var_backtests(
                    var_data["Actual Return"],
                    var_data["VaR"],
                    confidence=confidence
                )

            st.session_state.var_forecasts = var_data
            st.session_state.var_backtest_results = backtest

            st.success("Backtesting completed.")

        except Exception as error:

            show_error(error)

    if st.session_state.var_backtest_results is not None:

        st.dataframe(
            pd.DataFrame(
                list(st.session_state.var_backtest_results.items()),
                columns=["Statistic", "Value"]
            ),
            use_container_width=True
        )

        var_data = st.session_state.var_forecasts

        st.caption(
            f"Failed estimations: "
            f"{var_data.attrs.get('failed', 'N/A')} "
            f"of {var_data.attrs.get('attempted', 'N/A')} attempts."
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data["Actual Return"],
                mode="lines",
                name="Actual Return"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=-var_data["VaR"],
                mode="lines",
                name="Negative VaR Threshold"
            )
        )

        st.plotly_chart(
            make_chart(fig, "Returns vs VaR", "Return (%)"),
            use_container_width=True
        )


# ============================================================
# PAGE 8: MODEL STABILITY & CALIBRATION
# ============================================================

elif page == "Model Stability & Calibration":

    st.header("Model Stability & Calibration")

    st.subheader("Rolling Parameter Stability")

    window = st.number_input(
        "Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 100)),
        step=50,
        key="stability_window"
    )

    step = st.number_input(
        "Estimation Step",
        min_value=10,
        max_value=250,
        value=50,
        step=10
    )

    if st.button("Run Stability Analysis"):

        try:

            with st.spinner("Estimating rolling parameters..."):

                estimates, summary = rolling_parameter_stability(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(window),
                    step=int(step)
                )

            st.session_state.stability_estimates = estimates
            st.session_state.stability_summary = summary

            st.success("Stability analysis completed.")

        except Exception as error:

            show_error(error)

    if st.session_state.stability_estimates is not None:

        estimates = st.session_state.stability_estimates

        st.subheader("Parameter Summary")

        st.dataframe(
            st.session_state.stability_summary,
            use_container_width=True
        )

        parameters = [
            col for col in estimates.columns
            if col not in ["Log Likelihood", "AIC", "BIC"]
        ]

        if parameters:

            selected = st.selectbox(
                "Select Parameter",
                parameters
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=estimates.index,
                    y=estimates[selected],
                    mode="lines+markers",
                    name=selected
                )
            )

            st.plotly_chart(
                make_chart(fig, f"Rolling {selected}", "Estimate"),
                use_container_width=True
            )

    st.divider()

    st.subheader("Student-t PIT Calibration")

    if st.session_state.selected_distribution == "t":

        if st.button("Run PIT Diagnostics"):

            try:

                pit, summary = student_t_pit(result)

                st.session_state.pit_values = pit
                st.session_state.pit_summary = summary

                st.success("PIT diagnostics completed.")

            except Exception as error:

                show_error(error)

        if st.session_state.pit_summary is not None:

            summary = st.session_state.pit_summary

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "KS Statistic",
                f"{summary['KS Statistic']:.4f}"
            )

            c2.metric(
                "KS P-value",
                f"{summary['KS P-value']:.4f}"
            )

            c3.metric(
                "PIT Mean",
                f"{summary['PIT Mean']:.4f}"
            )

            st.dataframe(
                pd.DataFrame(
                    list(summary.items()),
                    columns=["Statistic", "Value"]
                ),
                use_container_width=True
            )

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=st.session_state.pit_values,
                    nbinsx=20,
                    name="PIT"
                )
            )

            st.plotly_chart(
                make_chart(fig, "PIT Distribution", "Frequency"),
                use_container_width=True
            )

            st.caption(
                "The ordinary KS test does not correct for parameter "
                "estimation. Treat it as a diagnostic."
            )

    else:

        st.info("PIT diagnostic requires Student-t innovations.")


# ============================================================
# PAGE 9: FORECAST COMPARISON
# ============================================================

elif page == "Forecast Comparison":

    st.header("Forecast Comparison")

    st.write(
        "Compare your selected volatility model against the "
        "RiskMetrics-style EWMA benchmark."
    )

    window = st.number_input(
        "Common Estimation Window",
        min_value=250,
        max_value=max(250, len(returns) - 1),
        value=max(250, min(750, len(returns) - 20)),
        step=50,
        key="forecast_compare_window"
    )

    decay = st.slider(
        "EWMA Decay Factor",
        min_value=0.80,
        max_value=0.99,
        value=0.94,
        step=0.01
    )

    if st.button("Run Forecast Comparison"):

        try:

            with st.spinner("Comparing volatility forecasts..."):

                model_forecast = rolling_forecast(
                    returns,
                    model_name=st.session_state.selected_model,
                    distribution=st.session_state.selected_distribution,
                    window=int(window)
                )

                ewma = make_ewma_forecasts(
                    returns,
                    model_forecast.index,
                    window=int(window),
                    decay=decay
                )

                common = model_forecast.join(
                    ewma[["EWMA Variance", "EWMA Volatility"]],
                    how="inner"
                ).dropna()

                if len(common) < 30:
                    raise ValueError(
                        "Not enough common forecast observations."
                    )

                actual = common["Actual Return"] ** 2
                model_var = common["Forecast Variance"]
                ewma_var = common["EWMA Variance"]

                model_qlike = np.mean(
                    np.log(model_var) + actual / model_var
                )

                ewma_qlike = np.mean(
                    np.log(ewma_var) + actual / ewma_var
                )

                model_mse = np.mean(
                    (actual - model_var) ** 2
                )

                ewma_mse = np.mean(
                    (actual - ewma_var) ** 2
                )

                dm_qlike = diebold_mariano_test(
                    actual,
                    model_var,
                    ewma_var,
                    horizon=1,
                    loss="qlike",
                    bandwidth=0
                )

                dm_mse = diebold_mariano_test(
                    actual,
                    model_var,
                    ewma_var,
                    horizon=1,
                    loss="mse",
                    bandwidth=0
                )

                st.session_state.forecast_comparison = {
                    "Model QLIKE": float(model_qlike),
                    "EWMA QLIKE": float(ewma_qlike),
                    "Model MSE": float(model_mse),
                    "EWMA MSE": float(ewma_mse),
                    "DM QLIKE": dm_qlike,
                    "DM MSE": dm_mse,
                    "Forecast Count": len(common)
                }

                st.session_state.comparison_forecasts = common

            st.success("Forecast comparison completed.")

        except Exception as error:

            show_error(error)

    if st.session_state.forecast_comparison is not None:

        comparison = st.session_state.forecast_comparison

        st.subheader("Forecast Performance")

        c1, c2 = st.columns(2)

        c1.metric(
            f"{st.session_state.selected_model} QLIKE",
            f"{comparison['Model QLIKE']:.6f}"
        )

        c2.metric(
            "EWMA QLIKE",
            f"{comparison['EWMA QLIKE']:.6f}"
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Model MSE",
            f"{comparison['Model MSE']:.6f}"
        )

        c4.metric(
            "EWMA MSE",
            f"{comparison['EWMA MSE']:.6f}"
        )

        st.subheader("Diebold-Mariano QLIKE Test")

        dm = comparison["DM QLIKE"]

        st.write(f"Statistic: {dm['DM Statistic']:.4f}")
        st.write(f"P-value: {dm['P-value']:.4f}")
        st.write(f"Lower average loss: {dm['Lower Average Loss']}")

        st.subheader("Diebold-Mariano MSE Test")

        dm = comparison["DM MSE"]

        st.write(f"Statistic: {dm['DM Statistic']:.4f}")
        st.write(f"P-value: {dm['P-value']:.4f}")
        st.write(f"Lower average loss: {dm['Lower Average Loss']}")

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
            make_chart(fig, "Out-of-Sample Forecasts", "Volatility (%)"),
            use_container_width=True
        )

        st.caption(
            "Squared returns are used as a noisy realized variance proxy. "
            "A negative DM mean loss difference favors the selected model."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Quantitative Financial Research | "
    "Educational and research purposes only. Not investment advice."
)
