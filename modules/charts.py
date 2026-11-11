import pandas as pd
import plotly.graph_objects as go


def plot_price(data):
    """Plot historical closing prices."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Closing Price"
        )
    )

    fig.update_layout(
        title="Historical Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=450
    )

    return fig


def plot_returns(returns):
    """Plot daily percentage log returns."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="Daily Returns"
        )
    )

    fig.update_layout(
        title="Daily Log Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_volatility(result):
    """Plot fitted conditional volatility."""

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

    fig.update_layout(
        title="Conditional Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_dark",
        height=400
    )

    return fig
