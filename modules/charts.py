
import plotly.graph_objects as go


def price_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Adjusted Close"
    ))

    fig.update_layout(
        title="Historical Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (EUR)",
        template="plotly_white"
    )

    return fig


def returns_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Returns"],
        mode="lines",
        name="Daily Log Returns"
    ))

    fig.update_layout(
        title="Daily Log Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_dark"
    )

    return fig


def volatility_chart(result):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.conditional_volatility.index,
        y=result.conditional_volatility,
        mode="lines",
        name="EGARCH Volatility"
    ))

    fig.update_layout(
        title="EGARCH Conditional Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_dark"
    )

    return fig

