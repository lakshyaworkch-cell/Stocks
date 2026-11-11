
import yfinance as yf
import numpy as np
import pandas as pd


DUTCH_STOCKS = {
    "ASML": "ASML.AS",
    "Shell": "SHELL.AS",
    "ING Group": "INGA.AS",
    "Philips": "PHIA.AS",
    "Heineken": "HEIA.AS",
    "Ahold Delhaize": "AD.AS",
    "Prosus": "PRX.AS",
    "NN Group": "NN.AS",
    "KPN": "KPN.AS",
    "AkzoNobel": "AKZA.AS",
}


def get_stock_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data available for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("Closing prices not available.")

    df = df[["Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)

    df["Returns"] = (
        np.log(df["Close"] / df["Close"].shift(1)) * 100
    )

    df.dropna(inplace=True)

    return df

