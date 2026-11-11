
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

    c1.metric("Historical VaR", f"{historical['VaR']:.4f}%")
    c2.metric("Historical ES", f"{historical['ES']:.4f}%")

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

            c1.metric("Forecast VaR", f"{risk['VaR']:.4f}%")
            c2.metric("Forecast ES", f"{risk['ES']:.4f}%")

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
            f"Failed estimations: {var_data.attrs.get('failed', 'N/A')} "
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
