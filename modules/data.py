import yfinance as yf
import pandas as pd


DUTCH_STOCKS = {
    "ASML": "ASML.AS",
    "Shell": "SHEL.AS",
    "ING Group": "INGA.AS",
    "ABN AMRO": "ABN.AS",
    "Adyen": "ADYEN.AS",
    "Prosus": "PRX.AS",
    "Philips": "PHIA.AS",
    "Heineken": "HEIA.AS",
    "Ahold Delhaize": "AD.AS",
    "KPN": "KPN.AS",
    "NN Group": "NN.AS",
    "ASR Nederland": "ASRNL.AS",
    "DSM-Firmenich": "DSFIR.AS",
    "Wolters Kluwer": "WKL.AS",
    "AkzoNobel": "AKZA.AS",
    "Randstad": "RAND.AS",
    "IMCD": "IMCD.AS",
    "BE Semiconductor": "BESI.AS",
    "ASM International": "ASM.AS",
    "Universal Music Group": "UMG.AS",
}


def get_stock_data(ticker, start, end):
    """
    Download historical stock data.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker, e.g. ASML.AS.
    start : str or datetime
        Start date.
    end : str or datetime
        End date.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing Close and Volume columns.
    """

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty:
            raise ValueError(
                f"No stock data was returned for {ticker}. "
                "Check the ticker symbol and date range."
            )

        # Handle newer yfinance versions that may return MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = ["Close", "Volume"]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(
                    f"Required column '{column}' is missing for {ticker}."
                )

        data = data[required_columns].copy()

        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")

        data = data.dropna(subset=["Close"])
        data = data[data["Close"] > 0]

        if data.empty:
            raise ValueError(f"No valid closing prices found for {ticker}.")

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        return data

    except Exception as error:
        raise RuntimeError(
            f"Could not download data for {ticker}: {error}"
        ) from error
