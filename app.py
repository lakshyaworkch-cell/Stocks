
import streamlit as st
import pandas as pd
import numpy as np

from datetime import date, timedelta

from modules.data import DUTCH_STOCKS, get_stock_data

from modules.analysis import (
    fit_egarch,
    get_forecast,
    get_diagnostics,
    get_parameters
)

from modules.charts import (
    price_chart,
    returns_chart,
    volatility_chart
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Dutch Quant | Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. DARK THEME AND CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0B1120;
    color: #E5EAF3;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */

h1, h2, h3 {
    color: #F1F5F9 !important;
    font-weight: 650 !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #101A2B;
    border-right: 1px solid #293951;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: #151F32;
    border: 1px solid #293951;
    border-radius: 12px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #00D4AA;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

/* Buttons */

.stButton > button {
    background-color: #00A88F;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #00C6A5;
    color: white;
}

/* Download buttons */

.stDownloadButton > button {
    background-color: #17263C;
    color: #00D4AA;
    border: 1px solid #00A88F;
    border-radius: 8px;
}

/* Tables */

div[data-testid="stDataFrame"] {
    border: 1px solid #293951;
    border-radius: 10px;
}

/* Dividers */

hr {
    border-color: #293951;
}

/* Expanders */

div[data-testid="stExpander"] {
    background-color: #151F32;
    border: 1px solid #293951;
    border-radius: 10px;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Tabs */

button[data-baseweb="tab"] {
    color: #94A3B8;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #00D4AA;
}

/* Footer */

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# =====================================================
# 3. HEADER
# =====================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #101A2B, #17263C);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #293951;
        margin-bottom: 25px;
    ">

        <div style="
            color: #00D4AA;
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 3px;
        ">
            QUANTITATIVE RESEARCH TERMINAL
        </div>

        <h1 style="
            color: #F1F5F9;
            font-size: 34px;
            margin-top: 8px;
            margin-bottom: 5px;
        ">
            DUTCH QUANT
        </h1>

        <p style="
            color: #94A3B8;
            font-size: 15px;
            margin-bottom: 0;
        ">
            Dutch Equity Market | EGARCH Volatility Modelling |
            Student-t Distribution
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 4. SIDEBAR CONFIGURATION
# =====================================================

st.sidebar.markdown("""
<h2 style="color:#00D4AA;">
    QUANT CONTROL
</h2>
""", unsafe_allow_html=True)

st.sidebar.caption("MARKET CONFIGURATION")

company = st.sidebar.selectbox(
    "Select Company",
    list(DUTCH_STOCKS.keys())
)

ticker = DUTCH_STOCKS[company]

st.sidebar.markdown("---")

st.sidebar.caption("HISTORICAL DATA")

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

st.sidebar.markdown("---")

st.sidebar.caption("FORECAST SETTINGS")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=1,
    max_value=30,
    value=10
)

st.sidebar.markdown("---")

run_analysis = st.sidebar.button(
    "RUN QUANT ANALYSIS",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Data source: Yahoo Finance via yfinance"
)

st.sidebar.caption(
    "Model: EGARCH(1,1) Student-t"
)


# =====================================================
# 5. ANALYSIS EXECUTION
# =====================================================

if run_analysis:

    # ---------------------------------------------
    # DATE VALIDATION
    # ---------------------------------------------

    if start_date >= end_date:

        st.error(
            "Start date must be earlier than end date."
        )

        st.stop()

    # ---------------------------------------------
    # DOWNLOAD DATA
    # ---------------------------------------------

    with st.spinner("Downloading Dutch market data..."):

        try:

            df = get_stock_data(
                ticker,
                start_date,
                end_date + timedelta(days=1)
            )

            if df.empty or len(df) < 100:

                st.error(
                    "Insufficient observations. "
                    "Please select a longer historical period."
                )

                st.stop()

        except Exception as e:

            st.error(
                f"Market data error: {str(e)}"
            )

            st.stop()

    st.success(
        f"Market data successfully downloaded: {company} ({ticker})"
    )

    # ---------------------------------------------
    # ESTIMATE EGARCH MODEL
    # ---------------------------------------------

    with st.spinner(
        "Estimating EGARCH(1,1) Student-t model..."
    ):

        try:

            result = fit_egarch(
                df["Returns"]
            )

        except Exception as e:

            st.error(
                f"EGARCH estimation failed: {str(e)}"
            )

            st.stop()

    # ---------------------------------------------
    # MODEL FORECAST
    # ---------------------------------------------

    try:

        forecast_df = get_forecast(
            result,
            forecast_horizon
        )

    except Exception as e:

        st.error(
            f"Forecast calculation failed: {str(e)}"
        )

        st.stop()

    # ---------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------

    try:

        diagnostics = get_diagnostics(result)

    except Exception as e:

        st.error(
            f"Diagnostic calculation failed: {str(e)}"
        )

        st.stop()

    parameters = get_parameters(result)

    # =================================================
    # 6. MARKET OVERVIEW
    # =================================================

    st.header("Market Overview")

    total_return = (
        df["Close"].iloc[-1] /
        df["Close"].iloc[0] - 1
    ) * 100

    annual_vol = (
        df["Returns"].std() * np.sqrt(252)
    )

    latest_price = df["Close"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Price",
        f"€{latest_price:.2f}"
    )

    col2.metric(
        "Total Return",
        f"{total_return:.2f}%"
    )

    col3.metric(
        "Annualized Volatility",
        f"{annual_vol:.2f}%"
    )

    col4.metric(
        "Observations",
        f"{len(df):,}"
    )

    st.caption(
        f"Company: {company} | Ticker: {ticker} | "
        f"Period: {start_date} to {end_date}"
    )

    st.divider()

    # =================================================
    # 7. HISTORICAL STOCK PRICE
    # =================================================

    st.header("01 | Historical Market Performance")

    tab1, tab2 = st.tabs([
        "Stock Price",
        "Daily Returns"
    ])

    with tab1:

        st.plotly_chart(
            price_chart(df),
            use_container_width=True
        )

    with tab2:

        st.plotly_chart(
            returns_chart(df),
            use_container_width=True
        )

    # =================================================
    # 8. DESCRIPTIVE STATISTICS
    # =================================================

    st.header("02 | Descriptive Statistics")

    returns = df["Returns"]

    descriptive = pd.DataFrame({
        "Metric": [
            "Mean",
            "Standard Deviation",
            "Minimum",
            "Maximum",
            "Skewness",
            "Excess Kurtosis",
            "Median"
        ],

        "Value": [
            returns.mean(),
            returns.std(),
            returns.min(),
            returns.max(),
            returns.skew(),
            returns.kurtosis(),
            returns.median()
        ]
    })

    col1, col2 = st.columns([1.4, 1])

    with col1:

        st.dataframe(
            descriptive,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(
                    "Value (%)",
                    format="%.4f"
                )
            }
        )

    with col2:

        st.markdown("""
        <div style="
            background:#151F32;
            padding:20px;
            border-radius:12px;
            border:1px solid #293951;
        ">

        <h4 style="color:#00D4AA;">
        Return Distribution
        </h4>

        <p style="color:#94A3B8;">
        Skewness measures asymmetry in returns.
        </p>

        <p style="color:#94A3B8;">
        Excess kurtosis measures tail heaviness
        relative to a normal distribution.
        </p>

        <p style="color:#94A3B8;">
        High kurtosis may indicate extreme market
        movements occur more frequently.
        </p>

        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # 9. EGARCH ESTIMATION
    # =================================================

    st.header("03 | EGARCH(1,1) Student-t Estimation")

    st.markdown("""
    The conditional variance equation is:

    \[
    \\log(\\sigma_t^2) =
    \\omega +
    \\alpha (|z_{t-1}|-E|z_{t-1}|) +
    \\gamma z_{t-1} +
    \\beta\\log(\\sigma_{t-1}^2)
    \]

    Innovations follow a standardized Student-t distribution.
    """)

    with st.expander(
        "View Full Econometric Model Summary",
        expanded=False
    ):

        st.code(
            result.summary().as_text(),
            language="text"
        )

    st.subheader("Estimated Model Parameters")

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Estimate": st.column_config.NumberColumn(
                "Coefficient",
                format="%.6f"
            ),
            "P-value": st.column_config.NumberColumn(
                "P-value",
                format="%.6f"
            )
        }
    )

    st.caption(
        "The estimated asymmetry coefficient gamma "
        "measures the impact of positive versus negative "
        "standardized shocks."
    )

    # =================================================
    # 10. CONDITIONAL VOLATILITY
    # =================================================

    st.header("04 | Conditional Volatility")

    st.plotly_chart(
        volatility_chart(result),
        use_container_width=True
    )

    st.caption(
        "Conditional volatility estimated using the "
        "EGARCH(1,1) Student-t specification."
    )

    # =================================================
    # 11. VOLATILITY FORECASTING
    # =================================================

    st.header("05 | Volatility Forecasting")

    forecast_col1, forecast_col2 = st.columns(2)

    with forecast_col1:

        st.metric(
            "Forecast Horizon",
            f"{forecast_horizon} Trading Days"
        )

    with forecast_col2:

        st.metric(
            "Final Forecast Volatility",
            f"{forecast_df['Daily Volatility (%)'].iloc[-1]:.4f}%"
        )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Variance": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "Daily Volatility (%)": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Annualized Volatility (%)": st.column_config.NumberColumn(
                format="%.4f"
            )
        }
    )

    st.subheader("Forecasted Annualized Volatility")

    st.line_chart(
        forecast_df.set_index("Day")[
            "Annualized Volatility (%)"
        ]
    )

    st.download_button(
        label="Download Volatility Forecast",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_egarch_forecast.csv",
        mime="text/csv"
    )

    # =================================================
    # 12. MODEL DIAGNOSTICS
    # =================================================

    st.header("06 | Model Diagnostics")

    st.markdown(
        "Statistical tests for residual adequacy and "
        "remaining conditional heteroskedasticity."
    )

    # Extract diagnostic values

    lb10 = diagnostics["Ljung-Box p-value (10)"]

    lb20 = diagnostics["Ljung-Box p-value (20)"]

    arch_p = diagnostics["ARCH LM p-value"]

    jb_stat = diagnostics["Jarque-Bera statistic"]

    jb_p = diagnostics["Jarque-Bera p-value"]

    # Diagnostic KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Ljung-Box (10)",
        f"{lb10:.4f}"
    )

    c2.metric(
        "Ljung-Box (20)",
        f"{lb20:.4f}"
    )

    c3.metric(
        "ARCH LM",
        f"{arch_p:.4f}"
    )

    c4.metric(
        "Jarque-Bera",
        f"{jb_p:.4f}"
    )

    # Organized diagnostic table

    diagnostic_table = pd.DataFrame({

        "Diagnostic": [
            "Ljung-Box (10)",
            "Ljung-Box (20)",
            "ARCH LM",
            "Jarque-Bera"
        ],

        "Null Hypothesis": [
            "No residual autocorrelation",
            "No residual autocorrelation",
            "No remaining ARCH effects",
            "Normal residual distribution"
        ],

        "Test Statistic": [
            np.nan,
            np.nan,
            np.nan,
            jb_stat
        ],

        "P-value": [
            lb10,
            lb20,
            arch_p,
            jb_p
        ],

        "Conclusion": [
            "Reject H0" if lb10 < 0.05 else "Do not reject H0",
            "Reject H0" if lb20 < 0.05 else "Do not reject H0",
            "Reject H0" if arch_p < 0.05 else "Do not reject H0",
            "Reject H0" if jb_p < 0.05 else "Do not reject H0"
        ]

    })

    st.dataframe(
        diagnostic_table,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Test Statistic": st.column_config.NumberColumn(
                "Test Statistic",
                format="%.4f"
            ),

            "P-value": st.column_config.NumberColumn(
                "P-value",
                format="%.6f"
            )

        }
    )

    with st.expander("Understanding Model Diagnostics"):

        st.markdown("""
        **Ljung-Box test**

        Tests whether standardized residuals exhibit
        significant autocorrelation.

        **ARCH LM test**

        Tests whether conditional heteroskedasticity
        remains in the standardized residuals.

        **Jarque-Bera test**

        Tests whether standardized residuals follow
        a normal distribution.

        A p-value below 0.05 indicates evidence against
        the respective null hypothesis.

        Note: Student-t innovations do not imply that
        standardized residuals should be normally distributed.
        Therefore, rejection by Jarque-Bera is not necessarily
        evidence that the Student-t model is inadequate.
        """)

    # =================================================
    # 13. STANDARDIZED RESIDUALS
    # =================================================

    st.header("07 | Standardized Residual Analysis")

    residuals = result.std_resid.dropna()

    residual_col1, residual_col2 = st.columns(2)

    with residual_col1:

        st.markdown("### Residual Statistics")

        residual_stats = pd.DataFrame({
            "Metric": [
                "Mean",
                "Standard Deviation",
                "Skewness",
                "Excess Kurtosis"
            ],

            "Value": [
                residuals.mean(),
                residuals.std(),
                residuals.skew(),
                residuals.kurtosis()
            ]
        })

        st.dataframe(
            residual_stats,
            use_container_width=True,
            hide_index=True
        )

    with residual_col2:

        st.markdown("### Residual Distribution")

        st.line_chart(
            residuals,
            height=260
        )

    # =================================================
    # 14. DATA EXPORT
    # =================================================

    st.header("08 | Data Export")

    export_col1, export_col2 = st.columns(2)

    with export_col1:

        st.download_button(
            "Download Historical Data",
            data=df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with export_col2:

        st.download_button(
            "Download EGARCH Parameters",
            data=parameters.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_egarch_parameters.csv",
            mime="text/csv",
            use_container_width=True
        )

    # =================================================
    # 15. FOOTER
    # =================================================

    st.divider()

    st.markdown("""
    <div style="
        text-align:center;
        color:#64748B;
        padding:15px;
    ">

    DUTCH QUANTITATIVE FINANCE TERMINAL

    <br>

    EGARCH(1,1) | Student-t | Volatility Research

    <br><br>

    For educational and quantitative research purposes.
    Not investment advice.

    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 16. INITIAL SCREEN
# =====================================================

else:

    st.markdown("""
    <div style="
        background:#151F32;
        border:1px solid #293951;
        border-radius:15px;
        padding:35px;
        text-align:center;
        margin-top:20px;
    ">

        <h2 style="color:#00D4AA;">
            Welcome to Dutch Quant
        </h2>

        <p style="
            color:#94A3B8;
            font-size:16px;
        ">
            Quantitative financial research for
            Dutch equity markets.
        </p>

        <br>

        <p style="color:#CBD5E1;">
            Select a company and configure the
            historical period using the sidebar.
        </p>

        <p style="
            color:#00D4AA;
            font-weight:bold;
        ">
            Click RUN QUANT ANALYSIS to begin.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Available Research Modules")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**Market Analysis**\n\n"
            "Historical prices, returns and descriptive statistics."
        )

    with col2:

        st.info(
            "**EGARCH Modelling**\n\n"
            "EGARCH(1,1) with Student-t innovations."
        )

    with col3:

        st.info(
            "**Risk & Forecasting**\n\n"
            "Conditional volatility and model diagnostics."
        )

    st.divider()

    st.caption(
        "Dutch Quant | Quantitative Finance Research Platform"
    )
