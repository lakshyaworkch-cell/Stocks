
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


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Dutch Quant Finance",
    page_icon="📈",
    layout="wide"
)

st.title("🇳🇱 Dutch Quantitative Finance Platform")

st.markdown("""
### EGARCH(1,1) - Student-t Volatility Analysis

Quantitative financial modelling of Dutch equities
listed on Euronext Amsterdam.
""")

st.divider()

# ==========================================
# CUSTOM DARK QUANT TERMINAL DESIGN
# ==========================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0B1120;
    color: #E5EAF3;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Main headings */
h1, h2, h3 {
    color: #F1F5F9 !important;
    font-weight: 650 !important;
    letter-spacing: -0.5px;
}

/* Paragraphs */
p, label, span {
    color: #CBD5E1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #101A2B;
    border-right: 1px solid #25334A;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #151F32;
    border: 1px solid #293951;
    padding: 18px;
    border-radius: 12px;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #00D4AA;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #00A88F;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1rem;
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
    overflow: hidden;
}

/* Input fields */
.stSelectbox, .stDateInput, .stSlider {
    margin-bottom: 12px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #94A3B8;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #00D4AA;
}

/* Dividers */
hr {
    border-color: #293951;
}

/* Information messages */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("Model Configuration")

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

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=30,
    value=10
)

run_analysis = st.sidebar.button(
    "Run Quantitative Analysis",
    type="primary"
)


# -------------------------------
# DATA DOWNLOAD
# -------------------------------

if run_analysis:

    if start_date >= end_date:
        st.error("End date must be after start date.")
        st.stop()

    with st.spinner("Downloading market data..."):

        try:
            df = get_stock_data(
                ticker,
                start_date,
                end_date + timedelta(days=1)
            )

            if len(df) < 100:
                st.error(
                    "Insufficient observations. "
                    "Select a longer historical period."
                )
                st.stop()

        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    st.success(f"Data downloaded for {company}")

    # -------------------------------
    # MARKET OVERVIEW
    # -------------------------------

    st.header("1. Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_return = (
        df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    ) * 100

    annual_vol = df["Returns"].std() * np.sqrt(252)

    col1.metric(
        "Latest Price",
        f"€{df['Close'].iloc[-1]:.2f}"
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
        len(df)
    )

    st.plotly_chart(
        price_chart(df),
        use_container_width=True
    )

    st.plotly_chart(
        returns_chart(df),
        use_container_width=True
    )

    # -------------------------------
    # DESCRIPTIVE STATISTICS
    # -------------------------------

    st.header("2. Descriptive Statistics")

    descriptive = df["Returns"].describe().to_frame(
        name="Daily Log Returns (%)"
    )

    descriptive.loc["Skewness"] = df["Returns"].skew()
    descriptive.loc["Kurtosis"] = df["Returns"].kurtosis()

    st.dataframe(
        descriptive,
        use_container_width=True
    )

    # -------------------------------
    # EGARCH MODEL
    # -------------------------------

    st.header("3. EGARCH(1,1) Student-t Estimation")

    with st.spinner("Estimating EGARCH model..."):

        try:
            result = fit_egarch(df["Returns"])
        except Exception as e:
            st.error(f"Model estimation failed: {e}")
            st.stop()

    st.subheader("Model Summary")

    st.text(result.summary().as_text())

    st.subheader("Estimated Parameters")

    parameters = get_parameters(result)

    st.dataframe(
        parameters,
        use_container_width=True
    )

    st.markdown("""
    **Interpretation:**

    - Alpha measures the magnitude response.
    - Gamma measures the asymmetric volatility response.
    - Beta measures volatility persistence.
    - The Student-t degrees of freedom parameter captures
      the thickness of the innovation tails.
    """)

    # -------------------------------
    # CONDITIONAL VOLATILITY
    # -------------------------------

    st.header("4. Conditional Volatility")

    st.plotly_chart(
        volatility_chart(result),
        use_container_width=True
    )

    # -------------------------------
    # VOLATILITY FORECAST
    # -------------------------------

    st.header("5. EGARCH Volatility Forecast")

    try:
        forecast_df = get_forecast(
            result,
            forecast_horizon
        )

        st.dataframe(
            forecast_df.round(4),
            use_container_width=True
        )

        st.line_chart(
            forecast_df.set_index("Day")[
                "Annualized Volatility (%)"
            ]
        )

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "Download Volatility Forecast",
            data=csv,
            file_name=f"{ticker}_egarch_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Forecast error: {e}")

    # -------------------------------
    # MODEL DIAGNOSTICS
    # -------------------------------

    
# ==========================================
# 6. MODEL DIAGNOSTICS
# ==========================================

st.header("6. Model Diagnostics")

diagnostics = get_diagnostics(result)


def pvalue_status(p):
    if p < 0.05:
        return "Significant"
    return "Not significant"


# Four compact diagnostic cards
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Ljung-Box (10)",
    f"{diagnostics['Ljung-Box p-value (10)']:.4f}",
    help="Tests residual autocorrelation."
)

c2.metric(
    "Ljung-Box (20)",
    f"{diagnostics['Ljung-Box p-value (20)']:.4f}",
    help="Tests residual autocorrelation at 20 lags."
)

c3.metric(
    "ARCH LM",
    f"{diagnostics['ARCH LM p-value']:.4f}",
    help="Tests remaining ARCH effects."
)

c4.metric(
    "Jarque-Bera",
    f"{diagnostics['Jarque-Bera p-value']:.4f}",
    help="Tests normality of standardized residuals."
)

st.markdown("### Statistical Diagnostic Results")

diagnostic_table = pd.DataFrame([
    {
        "Test": "Ljung-Box (10)",
        "Null Hypothesis": "No residual autocorrelation",
        "Statistic": np.nan,
        "P-value": diagnostics["Ljung-Box p-value (10)"]
    },
    {
        "Test": "Ljung-Box (20)",
        "Null Hypothesis": "No residual autocorrelation",
        "Statistic": np.nan,
        "P-value": diagnostics["Ljung-Box p-value (20)"]
    },
    {
        "Test": "ARCH LM",
        "Null Hypothesis": "No remaining ARCH effects",
        "Statistic": np.nan,
        "P-value": diagnostics["ARCH LM p-value"]
    },
    {
        "Test": "Jarque-Bera",
        "Null Hypothesis": "Residuals are normally distributed",
        "Statistic": diagnostics["Jarque-Bera statistic"],
        "P-value": diagnostics["Jarque-Bera p-value"]
    }
])

diagnostic_table["Result"] = diagnostic_table["P-value"].apply(
    pvalue_status
)

st.dataframe(
    diagnostic_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Statistic": st.column_config.NumberColumn(
            "Test Statistic",
            format="%.4f"
        ),
        "P-value": st.column_config.NumberColumn(
            "P-value",
            format="%.4f"
        )
    }
)

st.caption(
    "Interpretation: A p-value below 0.05 indicates "
    "evidence against the respective null hypothesis. "
    "Statistical significance does not automatically imply "
    "that the model is unsuitable."
)


    diagnostics = get_diagnostics(result)

    diag_df = pd.DataFrame(
        diagnostics.items(),
        columns=["Diagnostic", "Value"]
    )

    st.dataframe(
        diag_df,
        use_container_width=True
    )

    st.markdown("""
    **Diagnostic interpretation:**

    - Ljung-Box: tests residual autocorrelation.
    - ARCH LM: tests remaining ARCH effects.
    - Jarque-Bera: tests normality of standardized residuals.

    A small p-value (typically below 0.05) indicates
    evidence against the relevant null hypothesis.
    """)

    # -------------------------------
    # DOWNLOAD DATA
    # -------------------------------

    st.header("7. Export Historical Data")

    st.download_button(
        "Download Historical Data",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_historical_data.csv",
        mime="text/csv"
    )

    st.success("Quantitative analysis completed!")

else:

    st.info(
        "Select a Dutch company and click "
        "'Run Quantitative Analysis' to begin."
    )

    st.markdown("""
    ### Available model

    - EGARCH(1,1)
    - Student-t innovations
    - Conditional volatility estimation
    - Volatility forecasting
    - Residual diagnostics
    - Historical return analysis
    """)

