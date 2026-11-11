
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
# 2. DARK THEME CSS
# =====================================================

st.markdown("""
<style>

/* Application background */

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
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #101A2B;
}

/* Metrics */

div[data-testid="stMetric"] {
    background-color: #151F32;
    border: 1px solid #293951;
    border-radius: 12px;
    padding: 18px;
}

div[data-testid="stMetricValue"] {
    color: #00D4AA;
}

/* Buttons */

.stButton > button {
    background-color: #00A88F;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

/* Tables */

div[data-testid="stDataFrame"] {
    border: 1px solid #293951;
    border-radius: 10px;
}

/* Expanders */

div[data-testid="stExpander"] {
    background-color: #151F32;
    border: 1px solid #293951;
    border-radius: 10px;
}

/* Dividers */

hr {
    border-color: #293951;
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
# 3. DASHBOARD HEADER
# =====================================================

st.title("📈 DUTCH QUANT")

st.subheader("Quantitative Research Terminal")

st.caption(
    "Dutch Equity Market | EGARCH(1,1) | Student-t Distribution"
)

st.divider()


# =====================================================
# 4. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Quant Control")

st.sidebar.caption("MARKET CONFIGURATION")

company = st.sidebar.selectbox(
    "Select Dutch Company",
    list(DUTCH_STOCKS.keys())
)

ticker = DUTCH_STOCKS[company]

st.sidebar.divider()

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

st.sidebar.divider()

st.sidebar.caption("FORECAST CONFIGURATION")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Trading Days)",
    1,
    30,
    10
)

run_analysis = st.sidebar.button(
    "RUN QUANT ANALYSIS",
    type="primary",
    use_container_width=True
)


# =====================================================
# 5. MAIN ANALYSIS
# =====================================================

if run_analysis:

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    # Download data

    with st.spinner("Downloading market data..."):

        try:
            df = get_stock_data(
                ticker,
                start_date,
                end_date + timedelta(days=1)
            )

            if df.empty or len(df) < 100:
                st.error(
                    "Insufficient observations. "
                    "Choose a longer historical period."
                )
                st.stop()

        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    st.success(f"Data downloaded: {company} ({ticker})")

    # Estimate EGARCH

    with st.spinner("Estimating EGARCH Student-t model..."):

        try:
            result = fit_egarch(df["Returns"])

        except Exception as e:
            st.error(f"Model estimation failed: {e}")
            st.stop()

    # Forecast

    try:
        forecast_df = get_forecast(
            result,
            forecast_horizon
        )

        diagnostics = get_diagnostics(result)

        parameters = get_parameters(result)

    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.stop()


    # =================================================
    # 6. MARKET OVERVIEW
    # =================================================

    st.header("01 | Market Overview")

    total_return = (
        df["Close"].iloc[-1] /
        df["Close"].iloc[0] - 1
    ) * 100

    annual_vol = df["Returns"].std() * np.sqrt(252)

    latest_price = df["Close"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest Price", f"€{latest_price:.2f}")

    c2.metric("Total Return", f"{total_return:.2f}%")

    c3.metric("Annual Volatility", f"{annual_vol:.2f}%")

    c4.metric("Observations", f"{len(df):,}")

    st.caption(
        f"Company: {company} | Ticker: {ticker} | "
        f"Period: {start_date} to {end_date}"
    )

    st.divider()


    # =================================================
    # 7. STOCK PRICE AND RETURNS
    # =================================================

    st.header("02 | Historical Market Performance")

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

    st.header("03 | Descriptive Statistics")

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


    # =================================================
    # 9. EGARCH ESTIMATION
    # =================================================

    st.header("04 | EGARCH(1,1) Student-t Estimation")

    st.latex(
        r"""
        \log(\sigma_t^2)=
        \omega+
        \alpha(|z_{t-1}|-E|z_{t-1}|)
        +\gamma z_{t-1}
        +\beta\log(\sigma_{t-1}^2)
        """
    )

    with st.expander(
        "View Full Econometric Model Summary",
        expanded=False
    ):
        st.code(
            result.summary().as_text(),
            language="text"
        )

    st.subheader("Estimated Parameters")

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Estimate": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "P-value": st.column_config.NumberColumn(
                format="%.6f"
            )
        }
    )


    # =================================================
    # 10. CONDITIONAL VOLATILITY
    # =================================================

    st.header("05 | Conditional Volatility")

    st.plotly_chart(
        volatility_chart(result),
        use_container_width=True
    )


    # =================================================
    # 11. VOLATILITY FORECAST
    # =================================================

    st.header("06 | Volatility Forecast")

    f1, f2 = st.columns(2)

    f1.metric(
        "Forecast Horizon",
        f"{forecast_horizon} Trading Days"
    )

    f2.metric(
        "Final Forecast Volatility",
        f"{forecast_df['Daily Volatility (%)'].iloc[-1]:.4f}%"
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    st.line_chart(
        forecast_df.set_index("Day")[
            "Annualized Volatility (%)"
        ]
    )

    st.download_button(
        "Download Volatility Forecast",
        forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )


    # =================================================
    # 12. MODEL DIAGNOSTICS
    # =================================================

    st.header("07 | Model Diagnostics")

    lb10 = diagnostics["Ljung-Box p-value (10)"]
    lb20 = diagnostics["Ljung-Box p-value (20)"]
    arch_p = diagnostics["ARCH LM p-value"]
    jb_stat = diagnostics["Jarque-Bera statistic"]
    jb_p = diagnostics["Jarque-Bera p-value"]

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Ljung-Box (10)", f"{lb10:.4f}")
    d2.metric("Ljung-Box (20)", f"{lb20:.4f}")
    d3.metric("ARCH LM", f"{arch_p:.4f}")
    d4.metric("Jarque-Bera", f"{jb_p:.4f}")

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

    st.subheader("Diagnostic Test Results")

    st.dataframe(
        diagnostic_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Test Statistic": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "P-value": st.column_config.NumberColumn(
                format="%.6f"
            )
        }
    )

    with st.expander("Diagnostic Interpretation"):

        st.markdown("""
        **Ljung-Box:** Tests residual autocorrelation.

        **ARCH LM:** Tests remaining ARCH effects.

        **Jarque-Bera:** Tests normality of standardized residuals.

        A p-value below 0.05 indicates evidence against
        the respective null hypothesis.

        Since the model uses Student-t innovations,
        rejection of normality by Jarque-Bera is not
        automatically evidence of model failure.
        """)


    # =================================================
    # 13. STANDARDIZED RESIDUALS
    # =================================================

    st.header("08 | Standardized Residual Analysis")

    residuals = result.std_resid.dropna()

    r1, r2 = st.columns(2)

    with r1:

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

    with r2:

        st.line_chart(
            residuals,
            height=280
        )


    # =================================================
    # 14. DOWNLOAD RESULTS
    # =================================================

    st.header("09 | Download Research Data")

    x1, x2, x3 = st.columns(3)

    with x1:

        st.download_button(
            "Historical Data",
            df.to_csv().encode("utf-8"),
            file_name=f"{ticker}_historical.csv",
            mime="text/csv",
            use_container_width=True
        )

    with x2:

        st.download_button(
            "EGARCH Parameters",
            parameters.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_parameters.csv",
            mime="text/csv",
            use_container_width=True
        )

    with x3:

        st.download_button(
            "Diagnostics",
            diagnostic_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_diagnostics.csv",
            mime="text/csv",
            use_container_width=True
        )


    # =================================================
    # 15. FOOTER
    # =================================================

    st.divider()

    st.caption(
        "DUTCH QUANT | Quantitative Financial Research | "
        "EGARCH Student-t | Educational Use Only"
    )


# =====================================================
# 16. INITIAL SCREEN
# =====================================================

else:

    st.info(
        "Welcome to Dutch Quant. Select a company "
        "and click RUN QUANT ANALYSIS in the sidebar."
    )

    st.subheader("Research Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 Market Analysis")
        st.write(
            "Historical prices, logarithmic returns "
            "and descriptive statistics."
        )

    with c2:
        st.markdown("### 📈 EGARCH Modelling")
        st.write(
            "EGARCH(1,1) with Student-t innovations "
            "and conditional volatility."
        )

    with c3:
        st.markdown("### 🔬 Model Diagnostics")
        st.write(
            "Ljung-Box, ARCH LM and Jarque-Bera tests."
        )

    st.divider()

    st.caption(
        "Dutch Quantitative Finance Research Platform"
    )
