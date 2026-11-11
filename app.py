
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
# TEAL UI THEME
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B1923;
    color: #E1F1F0;
}

/* Main content spacing - FIXED */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

/* Keep Streamlit header above page content */
[data-testid="stHeader"] {
    z-index: 999;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #102630;
    border-right: 1px solid #21434B;
}

[data-testid="stSidebar"] * {
    color: #D9EEEE;
}

/* Headings */
h1, h2, h3 {
    color: #E1F1F0 !important;
    letter-spacing: -0.3px;
}

h1 {
    font-size: 1.8rem !important;
}

h2 {
    font-size: 1.35rem !important;
}

h3 {
    font-size: 1.05rem !important;
}

/* Paragraphs */
p, label {
    color: #B5CFD0;
}

/* Compact metric cards */
div[data-testid="stMetric"] {
    background: #12313A;
    border: 1px solid #24515A;
    border-radius: 9px;
    padding: 9px 12px;
    min-height: 0;
}

/* Metric labels */
div[data-testid="stMetricLabel"] {
    color: #9ABDC0 !important;
    font-size: 0.76rem !important;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #5DE0CF !important;
    font-size: 1.35rem !important;
    font-weight: 700;
}

/* Metric delta */
div[data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
}

/* Reduce vertical gaps between elements */
div[data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}

/* Buttons */
div.stButton > button {
    background: #167C78;
    color: #FFFFFF;
    border: 1px solid #2AA69A;
    border-radius: 7px;
    font-weight: 600;
    min-height: 36px;
    padding: 5px 12px;
}

div.stButton > button:hover {
    background: #209A8D;
    color: white;
    border-color: #5DE0CF;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #16323B;
    border-color: #31545C;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #28505A;
    border-radius: 8px;
    overflow: hidden;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #A9C7C9;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #5DE0CF;
    border-bottom-color: #36B9A9;
}

/* Dividers */
hr {
    border-color: #24444D;
}

/* Research cards */
.research-card {
    background: #12313A;
    border: 1px solid #28505A;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 6px;
    min-height: 65px;
}

.research-card b {
    color: #5DE0CF;
    font-size: 0.9rem;
}

.research-card span {
    color: #A9C7C9;
    font-size: 0.76rem;
}

/* Compact info panels */
.compact-panel {
    background: #12313A;
    border: 1px solid #28505A;
    border-radius: 8px;
    padding: 10px 12px;
}

/* Hide automatic Streamlit navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
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
    "analysis_horizon": 1,

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


def make_chart(fig, title, y_title, height=360):

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        paper_bgcolor="#0B1923",
        plot_bgcolor="#102630",
        font=dict(color="#D9EEEE"),
        xaxis_title="Date",
        yaxis_title=y_title,
        height=height,
        margin=dict(l=25, r=20, t=48, b=25),
        colorway=["#42CFC0", "#5B9BD5", "#D9A441", "#A78BFA"]
    )

    fig.update_xaxes(
        gridcolor="#203C45",
        zerolinecolor="#31535B"
    )

    fig.update_yaxes(
        gridcolor="#203C45",
        zerolinecolor="#31535B"
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

st.sidebar.header("1. Market Selection")

company = st.sidebar.selectbox(
    "Select Dutch Stock",
    options=list(DUTCH_STOCKS.keys()),
    format_func=lambda x: f"{x} — {DUTCH_STOCKS[x]}"
)

ticker = DUTCH_STOCKS[company]


st.sidebar.header("2. Date Selection")

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)


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


st.sidebar.header("4. Run Analysis")

run_analysis = st.sidebar.button(
    "Load Data & Run Analysis",
    use_container_width=True
)


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
        st.sidebar.error("End date must be after start date.")

    else:
        try:

            with st.spinner("Downloading stock market data..."):

                new_data = get_stock_data(
                    ticker,
                    start_date,
                    end_date + timedelta(days=1)
                )

                new_returns = calculate_returns(new_data["Close"])

            with st.spinner("Estimating volatility model..."):

                fitted = fit_model(
                    new_returns,
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

            st.session_state.stock_data = new_data
            st.session_state.returns = new_returns
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

            st.sidebar.success("Analysis completed!")

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

    c1.metric("Stock", st.session_state.selected_company)
    c2.metric("Ticker", st.session_state.selected_ticker)
    c3.metric("Model", st.session_state.selected_model)
    c4.metric("Observations", f"{len(returns):,}")

    st.caption(
        f"{st.session_state.analysis_start} — "
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
        "Select a Dutch stock, configure the model and load data "
        "to begin your analysis."
    )

    if has_data:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Latest Price",
            f"€{data['Close'].iloc[-1]:,.2f}"
        )

        col2.metric(
            "Daily Return",
            f"{returns.iloc[-1]:.2f}%"
        )

        col3.metric(
            "Annual Volatility",
            f"{returns.std() * np.sqrt(252):.2f}%"
        )

        col4.metric(
            "Model AIC",
            f"{result.aic:.2f}"
        )

        st.subheader("Stock Price")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Adjusted Close",
                line=dict(color="#42CFC0", width=2)
            )
        )

        st.plotly_chart(
            make_chart(fig, "Historical Stock Price", "Price (€)"),
            use_container_width=True
        )

        cols = st.columns(4)

        sections = [
            ("Market Data", "Price and returns"),
            ("Statistical Analysis", "Distribution and stationarity"),
            ("Volatility Models", "EGARCH and GARCH"),
            ("Model Diagnostics", "Residual tests"),
            ("Forecasting", "Volatility forecasts"),
            ("Risk Management", "VaR and ES"),
            ("Model Stability", "Stability and PIT"),
            ("Forecast Comparison", "EWMA and DM test"),
        ]

        for i, (name, description) in enumerate(sections):
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div class="research-card">
                        <b>{name}</b><br>
                        <span>{description}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.info(
            "Choose a stock and click Load Data & Run Analysis "
            "in the sidebar."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Volatility Models", "3")
        c2.metric("Distributions", "3")
        c3.metric("Research Pages", "8")

        st.markdown("""
        - EGARCH, GARCH and GJR-GARCH
        - Student-t, normal and skewed Student-t
        - ADF and KPSS stationarity tests
        - Residual and ARCH diagnostics
        - Rolling volatility forecasts
        - EWMA benchmark and DM tests
        - Parameter stability and PIT calibration
        - Historical and parametric VaR and ES
        - Kupiec and Christoffersen backtesting
        """)


# ============================================================
# DATA REQUIRED MESSAGE
# ============================================================

elif not has_data:

    st.header(page)

    st.warning("No stock analysis has been loaded yet.")

    st.write(
        "Select a company, set the date range and model, "
        "then click Load Data & Run Analysis."
    )


# ============================================================
# PAGE 2: MARKET DATA
# ============================================================

elif page == "Market Data":

    st.header("Market Data")

    c1, c2, c3 = st.columns(3)

    c1.metric("Latest Price", f"€{data['Close'].iloc[-1]:,.2f}")
    c2.metric("Daily Return", f"{returns.iloc[-1]:.2f}%")
    c3.metric(
        "Annual Volatility",
        f"{returns.std() * np.sqrt(252):.2f}%"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="#42CFC0", width=2)
        )
    )

    st.plotly_chart(
        make_chart(fig, "Stock Price", "Price (€)"),
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="Returns",
            line=dict(color="#5B9BD5", width=1)
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#6B8A91")

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
        data[concise_columns].tail(8).round(2),
        use_container_width=True,
        height=260
    )


# ============================================================
# PAGE 3: STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.header("Statistical Analysis")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", f"{returns.count():,}")
    c2.metric("Mean", f"{returns.mean():.3f}%")
    c3.metric("Std. Dev.", f"{returns.std():.3f}%")
    c4.metric(
        "Annual Volatility",
        f"{returns.std() * np.sqrt(252):.2f}%"
    )

    st.subheader("Distribution Summary")

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
        use_container_width=True,
        height=240
    )

    st.subheader("Stationarity")

    try:
        stationarity = stationarity_tests(returns)
        st.dataframe(
            stationarity,
            use_container_width=True,
            height=220
        )
    except Exception as error:
        show_error(error)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns",
            marker_color="#42CFC0"
        )
    )

    st.plotly_chart(
        make_chart(fig, "Return Distribution", "Frequency"),
        use_container_width=True
    )


# ============================================================
# PAGE 4: VOLATILITY MODELS
# ============================================================

elif page == "Volatility Models":

    st.header("Volatility Models")

    st.subheader("Estimated Parameters")

    st.dataframe(
        st.session_state.model_parameters,
        use_container_width=True,
        height=250
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
            name="Conditional Volatility",
            line=dict(color="#42CFC0", width=1.7)
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

    diagnostic_table = pd.DataFrame([
        {
            "Test": name,
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
    ])

    st.dataframe(
        diagnostic_table,
        hide_index=True,
        use_container_width=True,
        height=250
    )

    residuals = pd.Series(result.std_resid).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals,
            mode="lines",
            name="Standardized Residuals",
            line=dict(color="#42CFC0", width=1)
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#6B8A91")

    st.plotly_chart(
        make_chart(fig, "Standardized Residuals", "Residual"),
        use_container_width=True
    )

    st.caption(
        "For p-values below 0.05, reject the corresponding null hypothesis. "
        "Non-rejection does not prove correct model specification."
    )


# ============================================================
# PAGE 6: FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("Volatility Forecasting")

    forecast = st.session_state.forecast_result

    variance_values = forecast.variance.iloc[-1].to_numpy(dtype=float)

    forecast_table = pd.DataFrame({
        "Day": np.arange(1, len(variance_values) + 1),
        "Variance": variance_values,
        "Volatility (%)": np.sqrt(variance_values)
    })

    st.dataframe(
        forecast_table.round(4),
        hide_index=True,
        use_container_width=True,
        height=230
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_table["Day"],
            y=forecast_table["Volatility (%)"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#42CFC0", width=2)
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Future Conditional Volatility",
        paper_bgcolor="#0B1923",
        plot_bgcolor="#102630",
        font=dict(color="#D9EEEE"),
        xaxis_title="Forecast Day",
        yaxis_title="Volatility (%)",
        height=340
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

        c1.metric("MSE", f"{evaluation['MSE']:.5f}")
        c2.metric("QLIKE", f"{evaluation['QLIKE']:.5f}")
        c3.metric(
            "Forecasts",
            evaluation["Forecast Observations"]
        )

        st.dataframe(
            st.session_state.rolling_forecasts.tail(10),
            use_container_width=True,
            height=260
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

    c1.metric("Historical VaR", f"{historical['VaR']:.3f}%")
    c2.metric("Historical ES", f"{historical['ES']:.3f}%")

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

            c1.metric("Forecast VaR", f"{risk['VaR']:.3f}%")
            c2.metric("Forecast ES", f"{risk['ES']:.3f}%")

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
        "Estimation Window",
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

        st.subheader("Backtest Results")

        backtest = st.session_state.var_backtest_results

        if isinstance(backtest, dict):

            # Compact display instead of a large table
            result_items = list(backtest.items())

            result_cols = st.columns(min(3, len(result_items)))

            for i, (name, value) in enumerate(result_items):
                with result_cols[i % len(result_cols)]:

                    if isinstance(value, (int, float, np.integer, np.floating)):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = str(value)

                    st.metric(name, display_value)

        else:
            st.write(backtest)

        var_data = st.session_state.var_forecasts

        st.caption(
            f"Failed estimations: "
            f"{var_data.attrs.get('failed', 'N/A')} / "
            f"{var_data.attrs.get('attempted', 'N/A')}"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data["Actual Return"],
                mode="lines",
                name="Actual Return",
                line=dict(color="#42CFC0", width=1)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=-var_data["VaR"],
                mode="lines",
                name="VaR Threshold",
                line=dict(color="#E4A45B", width=1.5)
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
            use_container_width=True,
            height=220
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
                    name=selected,
                    line=dict(color="#42CFC0")
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

            c1.metric("KS Statistic", f"{summary['KS Statistic']:.4f}")
            c2.metric("KS P-value", f"{summary['KS P-value']:.4f}")
            c3.metric("PIT Mean", f"{summary['PIT Mean']:.4f}")

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=st.session_state.pit_values,
                    nbinsx=20,
                    name="PIT",
                    marker_color="#42CFC0"
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
        "Compare your selected volatility model against "
        "the RiskMetrics-style EWMA benchmark."
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

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model QLIKE",
            f"{comparison['Model QLIKE']:.5f}"
        )

        c2.metric(
            "EWMA QLIKE",
            f"{comparison['EWMA QLIKE']:.5f}"
        )

        c3.metric(
            "Forecasts",
            comparison["Forecast Count"]
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Model MSE",
            f"{comparison['Model MSE']:.5f}"
        )

        c2.metric(
            "EWMA MSE",
            f"{comparison['EWMA MSE']:.5f}"
        )

        st.subheader("Diebold-Mariano Tests")

        dm1 = comparison["DM QLIKE"]
        dm2 = comparison["DM MSE"]

        test_cols = st.columns(2)

        with test_cols[0]:
            st.markdown("**QLIKE**")
            st.write(f"Statistic: {dm1['DM Statistic']:.4f}")
            st.write(f"P-value: {dm1['P-value']:.4f}")
            st.write(f"Lower loss: {dm1['Lower Average Loss']}")

        with test_cols[1]:
            st.markdown("**MSE**")
            st.write(f"Statistic: {dm2['DM Statistic']:.4f}")
            st.write(f"P-value: {dm2['P-value']:.4f}")
            st.write(f"Lower loss: {dm2['Lower Average Loss']}")

        common = st.session_state.comparison_forecasts

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=common.index,
                y=common["Forecast Volatility"],
                mode="lines",
                name="Selected Model",
                line=dict(color="#42CFC0")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=common.index,
                y=common["EWMA Volatility"],
                mode="lines",
                name="EWMA",
                line=dict(color="#E4A45B")
            )
        )

        st.plotly_chart(
            make_chart(fig, "Out-of-Sample Forecasts", "Volatility (%)"),
            use_container_width=True
        )

        st.caption(
            "Squared returns are a noisy realized variance proxy. "
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
