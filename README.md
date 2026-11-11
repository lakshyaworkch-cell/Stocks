
# Dutch Quantitative Finance

A Streamlit-based quantitative financial analysis
platform for Dutch equities listed on Euronext Amsterdam.

## Features

- Historical stock price data using Yahoo Finance
- Daily logarithmic returns
- Descriptive statistics
- EGARCH(1,1) volatility modelling
- Student-t innovation distribution
- Conditional volatility estimation
- Volatility forecasting
- Model diagnostics
- CSV downloads

## Econometric Model

EGARCH(1,1) with Student-t innovations.

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Data Source

Yahoo Finance via yfinance.

## Disclaimer

This project is intended for educational and
quantitative research purposes. It does not provide
investment advice.
