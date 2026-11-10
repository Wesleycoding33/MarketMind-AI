import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

from src.predictions import generate_forecasts, load_history
from src.paper_trading import (
    get_account,
    get_open_positions,
    get_closed_trades,
    open_position,
    close_position,
    calculate_portfolio,
)
from src.trade_evaluator import evaluate_trade_setup
from src.backtesting import run_backtest, format_metrics
from src.predictions import load_history

# ---------------------------------------------------------
# PAGE CONFIGURATION AND CACHED DATA HELPERS
# ---------------------------------------------------------

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 MarketMind AI")
st.caption("AI-powered market analysis and paper-trading research")

st.info(
    "Development mode: predictions and trading assessments are experimental. "
    "No real trades are placed."
)


@st.cache_data(ttl=1800)
def cached_forecasts(symbol):
    return generate_forecasts(symbol)


@st.cache_data(ttl=300)
def load_market_data(symbol, selected_period):
    data = yf.Ticker(symbol).history(
        period=selected_period,
        interval="1d",
        auto_adjust=True,
    )
    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()
    return data.dropna(subset=["Close"])


@st.cache_data(ttl=300)
def load_latest_price(symbol):
    data = yf.Ticker(symbol).history(
        period="5d",
        interval="1d",
        auto_adjust=True,
    )
    if data.empty or data["Close"].dropna().empty:
        return None
    return float(data["Close"].dropna().iloc[-1])


# ---------------------------------------------------------
# SIDEBAR / WATCHLIST
# ---------------------------------------------------------

WATCHLIST = {
    "NVDA": "Nvidia",
    "AMD": "Advanced Micro Devices",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "SPY": "S&P 500 ETF",
}

st.sidebar.header("Market Controls")

selected_symbol = st.sidebar.selectbox(
    "Select a stock or ETF",
    options=list(WATCHLIST.keys()),
    format_func=lambda symbol: f"{symbol} — {WATCHLIST[symbol]}",
)

period = st.sidebar.selectbox(
    "Historical period",
    options=["3mo", "6mo", "1y", "2y"],
    index=2,
)


# ---------------------------------------------------------
# MARKET OVERVIEW
# ---------------------------------------------------------

stock_data = pd.DataFrame()

with st.spinner("Retrieving historical market data..."):
    try:
        stock_data = load_market_data(selected_symbol, period)
    except Exception as error:
        st.error(f"Market data retrieval failed: {error}")

if stock_data.empty:
    st.warning(
        "No market data was returned. Check your internet connection "
        "and try again later."
    )
else:
    latest_close = float(stock_data["Close"].iloc[-1])
    previous_close = (
        float(stock_data["Close"].iloc[-2]) if len(stock_data) > 1 else latest_close
    )
    daily_change = (
        ((latest_close / previous_close) - 1) * 100 if previous_close else 0.0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Latest available close",
        f"${latest_close:,.2f}",
        f"{daily_change:+.2f}%",
    )
    col2.metric("Trading sessions loaded", f"{len(stock_data):,}")
    col3.metric("Selected asset", selected_symbol)

    st.subheader(f"{selected_symbol} — Historical Price")

    chart_data = stock_data.reset_index()
    date_column = "Date" if "Date" in chart_data.columns else chart_data.columns[0]

    fig = px.line(
        chart_data,
        x=date_column,
        y="Close",
        title="Adjusted Historical Closing Prices",
        labels={date_column: "Date", "Close": "Price (USD)"},
    )
    fig.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Market Data")
    recent_data = stock_data.tail(10).copy()
    recent_data.index = recent_data.index.strftime("%Y-%m-%d")
    visible_columns = [
        column
        for column in ["Open", "High", "Low", "Close", "Volume"]
        if column in recent_data.columns
    ]
    st.dataframe(
        recent_data[visible_columns].round(2),
        use_container_width=True,
    )


# ---------------------------------------------------------
# AI PREDICTION ENGINE
# ---------------------------------------------------------

st.divider()
st.header("🤖 AI Prediction Engine")
st.write(
    "Estimates the probability that the closing price will be higher "
    "after 1, 3, or 5 trading sessions."
)
st.caption(
    "Forecasts are experimental. They are not live trading signals "
    "or guarantees of future returns."
)

refresh_col, _ = st.columns([1, 4])
with refresh_col:
    if st.button("Refresh AI forecasts"):
        cached_forecasts.clear()
        st.session_state.pop("forecast_error", None)

forecasts = None

try:
    with st.spinner(f"Loading forecasts for {selected_symbol}..."):
        forecasts = cached_forecasts(selected_symbol)
    st.session_state["last_forecasts"] = forecasts
    st.session_state["last_forecast_symbol"] = selected_symbol
    st.session_state.pop("forecast_error", None)
except Exception as error:
    st.session_state["forecast_error"] = str(error)
    st.error(f"Could not generate forecasts for {selected_symbol}: {error}")
    st.caption(
        "Check your internet connection, historical data availability, "
        "and the error details before trying again."
    )

if forecasts:
    forecast_columns = st.columns(3)
    for column, forecast in zip(forecast_columns, forecasts):
        with column:
            st.metric(
                label=f"{forecast['horizon']}-day upward probability",
                value=f"{forecast['up_probability']:.1%}",
            )
            st.write(
                f"Downward / no-rise probability: "
                f"{forecast['down_probability']:.1%}"
            )
            st.caption(
                f"Forecast data through {forecast.get('forecast_date', 'unknown date')}"
            )

    st.subheader("Historical Model Evaluation")
    st.write(
        "These results come from a chronological holdout period, "
        "not from the latest forecast."
    )

    evaluation_rows = []
    for forecast in forecasts:
        evaluation_rows.append(
            {
                "Horizon": f"{forecast['horizon']} day(s)",
                "Holdout accuracy": forecast["test_accuracy"],
                "Baseline accuracy": forecast["baseline_accuracy"],
                "Brier score": forecast["brier_score"],
                "Test observations": forecast["test_observations"],
                "Training observations": forecast["training_observations"],
            }
        )

    evaluation = pd.DataFrame(evaluation_rows)
    st.dataframe(
        evaluation.style.format(
            {
                "Holdout accuracy": "{:.1%}",
                "Baseline accuracy": "{:.1%}",
                "Brier score": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Compare holdout accuracy with the baseline and examine the Brier "
        "score (lower is better). Neither metric alone proves profitability."
    )


# ---------------------------------------------------------
# AI TRADE EVALUATION ENGINE
# ---------------------------------------------------------

st.divider()
st.header("🧠 AI Trade Evaluation")
st.caption(
    "Combines forecast probabilities, technical indicators, and historical "
    "model diagnostics. This is a research assessment, not a buy/sell order."
)

if forecasts:
    try:
        with st.spinner("Evaluating forecast and technical evidence..."):
            evaluation_history = load_history(selected_symbol)
            assessment = evaluate_trade_setup(evaluation_history, forecasts)

        st.subheader(f"Assessment: {assessment['signal']}")
        st.write(assessment["explanation"])

        probability_col, downside_col, trend_col = st.columns(3)
        probability_col.metric(
            "Consensus probability of rise",
            f"{assessment['consensus_up_probability']:.1%}",
        )
        downside_col.metric(
            "Consensus probability of no rise",
            f"{assessment['consensus_down_probability']:.1%}",
        )
        trend_col.metric(
            "Technical trend",
            assessment["technical"]["trend"],
        )

        st.info(assessment["confirmation_note"])

        technical = assessment["technical"]
        st.subheader("Technical Indicators")
        t1, t2, t3 = st.columns(3)
        t1.metric("Latest close", f"${technical['latest_price']:,.2f}")
        t2.metric("20-day moving average", f"${technical['ma20']:,.2f}")
        t3.metric("50-day moving average", f"${technical['ma50']:,.2f}")

        t4, t5, t6 = st.columns(3)
        t4.metric("5-day momentum", f"{technical['momentum_5d']:.2%}")
        t5.metric("20-day momentum", f"{technical['momentum_20d']:.2%}")
        t6.metric("20-day daily volatility", f"{technical['volatility_20d']:.2%}")

        st.subheader("Model Quality Diagnostics")
        quality = assessment["model_quality"].copy()
        for column in [
            "Holdout accuracy",
            "Baseline accuracy",
            "Difference vs baseline",
        ]:
            quality[column] = quality[column].map(lambda value: f"{value:.2%}")
        quality["Brier score"] = quality["Brier score"].map(
            lambda value: f"{value:.4f}"
        )
        st.dataframe(quality, use_container_width=True, hide_index=True)

        underperforming = assessment["underperforming_horizons"]
        if underperforming:
            horizon_text = ", ".join(f"{h}-day" for h in underperforming)
            st.warning(
                "These models did not beat the majority-class baseline on "
                f"the holdout sample: {horizon_text}. Treat those forecasts "
                "with additional caution."
            )

        st.caption(
            f"Forecast data date: {assessment['forecast_date']}. "
            "Historical performance does not guarantee future results."
        )
    except Exception as error:
        st.error(f"Trade evaluation could not be completed: {error}")
else:
    st.info("Forecasts are unavailable, so the trade evaluator cannot run yet.")


# ---------------------------------------------------------
# PAPER TRADING PORTFOLIO
# ---------------------------------------------------------

st.divider()
st.header("💼 Paper Trading Portfolio")
st.caption(
    "Simulation only. Prices are latest available daily data, not guaranteed "
    "executable prices. Illustrative fees are applied to each transaction. "
    "Stop-loss and target levels are recorded references and are not "
    "automatically enforced by this version."
)

try:
    open_positions = get_open_positions()
    closed_trades = get_closed_trades()

    current_prices = {}
    if not open_positions.empty:
        for symbol in open_positions["symbol"].dropna().unique():
            try:
                price = load_latest_price(symbol)
                if price is not None:
                    current_prices[symbol] = price
            except Exception:
                # The missing-price warning below will identify unvalued positions.
                continue

    if not stock_data.empty:
        current_prices[selected_symbol] = float(stock_data["Close"].iloc[-1])

    portfolio = calculate_portfolio(current_prices)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Available virtual cash", f"${portfolio['cash']:,.2f}")
    metric2.metric("Known position market value", f"${portfolio['market_value']:,.2f}")
    metric3.metric("Unrealised P&L", f"${portfolio['unrealised_pnl']:+,.2f}")
    metric4.metric("Realised P&L", f"${portfolio['realised_pnl']:+,.2f}")

    st.metric("Known portfolio value", f"${portfolio['portfolio_value']:,.2f}")

    if portfolio["missing_prices"]:
        st.warning(
            "Some positions could not be valued because their latest prices "
            "were unavailable: "
            + ", ".join(portfolio["missing_prices"])
            + ". The displayed portfolio value is incomplete."
        )

    # Open simulated trade
    st.subheader("Open a Simulated Trade")
    if stock_data.empty:
        st.warning("Market data is unavailable, so a new position cannot be opened.")
    else:
        entry_price = float(stock_data["Close"].iloc[-1])
        daily_returns = stock_data["Close"].pct_change().dropna()
        daily_volatility = (
            float(daily_returns.tail(20).std()) if len(daily_returns) >= 10 else 0.02
        )
        if pd.isna(daily_volatility) or daily_volatility <= 0:
            daily_volatility = 0.02

        stop_distance = entry_price * 2 * daily_volatility
        target_distance = entry_price * 3 * daily_volatility
        suggested_stop = round(max(0.01, entry_price - stop_distance), 2)
        if suggested_stop >= entry_price:
            suggested_stop = round(max(0.01, entry_price - 0.01), 2)
        suggested_target = round(
            max(entry_price + 0.01, entry_price + target_distance), 2
        )

        with st.form("paper_trade_form"):
            st.write(f"Selected asset: **{selected_symbol}**")
            st.write(f"Illustrative entry price: **${entry_price:,.2f}**")

            quantity = st.number_input(
                "Number of shares",
                min_value=1,
                max_value=10000,
                value=1,
                step=1,
            )
            stop_price = st.number_input(
                "Stop-loss reference price ($)",
                min_value=0.01,
                value=float(suggested_stop),
                step=0.01,
                format="%.2f",
            )
            target_price = st.number_input(
                "Target reference price ($)",
                min_value=0.01,
                value=float(suggested_target),
                step=0.01,
                format="%.2f",
            )

            estimated_cost = int(quantity) * entry_price * 1.001
            st.write(
                "Estimated purchase cost including illustrative entry fee: "
                f"**${estimated_cost:,.2f}**"
            )
            confirm_trade = st.checkbox(
                "I confirm this is a simulated trade.",
                key="confirm_open_paper_trade",
            )
            submitted = st.form_submit_button("Open Paper Position")

            if submitted:
                if not confirm_trade:
                    st.error("Confirm that this is a simulated trade.")
                else:
                    try:
                        open_position(
                            selected_symbol,
                            int(quantity),
                            entry_price,
                            float(stop_price),
                            float(target_price),
                        )
                        st.success("Paper position opened and saved.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not open paper position: {error}")

    # Current open positions
    st.subheader("Current Open Positions")
    if open_positions.empty:
        st.info("There are no open paper positions yet.")
    else:
        display_positions = open_positions.copy()
        display_positions["Latest Price"] = display_positions["symbol"].map(
            current_prices
        )

        def unrealised_pnl(row):
            price = row["Latest Price"]
            if pd.isna(price):
                return None
            return int(row["quantity"]) * (
                float(price) - float(row["entry_price"])
            ) - float(row["entry_fee"])

        display_positions["Unrealised P&L"] = display_positions.apply(
            unrealised_pnl, axis=1
        )
        st.dataframe(
            display_positions[
                [
                    "id",
                    "symbol",
                    "quantity",
                    "entry_price",
                    "Latest Price",
                    "stop_price",
                    "target_price",
                    "Unrealised P&L",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Displayed unrealised P&L includes entry fees but excludes a "
            "hypothetical exit fee until the position is closed."
        )

        st.subheader("Close a Simulated Position")
        options = {
            int(row["id"]): (
                f"Trade #{int(row['id'])} — {row['symbol']} "
                f"({int(row['quantity'])} shares)"
            )
            for _, row in open_positions.iterrows()
        }

        with st.form("close_trade_form"):
            trade_id = st.selectbox(
                "Select position",
                options=list(options.keys()),
                format_func=lambda item: options[item],
            )
            selected_trade = open_positions[open_positions["id"] == trade_id].iloc[0]
            default_exit = current_prices.get(selected_trade["symbol"])
            if default_exit is None:
                default_exit = float(selected_trade["entry_price"])

            exit_price = st.number_input(
                "Illustrative exit price ($)",
                min_value=0.01,
                value=round(float(default_exit), 2),
                step=0.01,
                format="%.2f",
            )
            confirm_close = st.checkbox(
                "Confirm closing this simulated position.",
                key="confirm_close_paper_trade",
            )
            close_submitted = st.form_submit_button("Close Paper Position")

            if close_submitted:
                if not confirm_close:
                    st.error("Confirm the simulated position closure.")
                else:
                    try:
                        close_position(int(trade_id), float(exit_price))
                        st.success("Paper position closed and recorded.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not close paper position: {error}")

    # Closed trade history
    st.subheader("Closed Trade History")
    if closed_trades.empty:
        st.info("Closed trades will appear here.")
    else:
        closed_display = closed_trades.copy()
        closed_display["Net P&L"] = (
            closed_display["quantity"]
            * (closed_display["exit_price"] - closed_display["entry_price"])
            - closed_display["entry_fee"]
            - closed_display["exit_fee"]
        )
        st.dataframe(
            closed_display[
                [
                    "id",
                    "symbol",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "entry_time",
                    "exit_time",
                    "Net P&L",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

except Exception as error:
    st.error(f"Paper-trading portfolio could not be loaded: {error}")

st.caption(
    "MarketMind AI is a developing research and simulation tool. "
    "It does not place real orders or guarantee forecast accuracy."
)
# ==================================================
# MARKETMIND AI — HISTORICAL BACKTESTING
# ==================================================

st.divider()
st.header("📊 Historical Strategy Backtesting")

st.write(
    "Test a historical trading strategy against a buy-and-hold "
    "benchmark. Results are simulations, not predictions of future returns."
)

with st.expander("Backtesting settings", expanded=True):

    initial_capital = st.number_input(
        "Starting virtual capital ($)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=10_000.0,
        step=500.0,
        key="bt_initial_capital",
    )

    fast_window = st.number_input(
        "Fast moving average (days)",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
        key="bt_fast_window",
    )

    slow_window = st.number_input(
        "Slow moving average (days)",
        min_value=3,
        max_value=250,
        value=50,
        step=1,
        key="bt_slow_window",
    )

    fee_rate = st.number_input(
        "Transaction fee (%) per side",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.05,
        key="bt_fee_rate",
    )

    slippage_rate = st.number_input(
        "Estimated slippage (%) per side",
        min_value=0.0,
        max_value=2.0,
        value=0.05,
        step=0.05,
        key="bt_slippage_rate",
    )

    run_button = st.button(
        "Run Historical Backtest",
        type="primary",
        key="run_historical_backtest",
    )


if run_button:

    if fast_window >= slow_window:
        st.error(
            "The fast moving average must be smaller " "than the slow moving average."
        )

    else:

        try:
            with st.spinner(f"Testing the strategy on {selected_symbol}..."):

                history = load_history(selected_symbol)

                result = run_backtest(
                    history=history,
                    initial_capital=float(initial_capital),
                    fee_rate=float(fee_rate) / 100,
                    slippage_rate=float(slippage_rate) / 100,
                    fast_window=int(fast_window),
                    slow_window=int(slow_window),
                )

                st.session_state["backtest_result"] = result
                st.session_state["backtest_symbol"] = selected_symbol

        except Exception as error:
            st.error(f"Backtesting failed: {error}")


if (
    "backtest_result" in st.session_state
    and st.session_state.get("backtest_symbol") == selected_symbol
):

    result = st.session_state["backtest_result"]
    metrics = result["metrics"]
    daily_results = result["daily_results"]

    st.subheader(f"Results for {selected_symbol}")

    st.caption(
        f"Historical period: {metrics['start_date']} to " f"{metrics['end_date']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Strategy return",
        f"{metrics['total_return']:.2%}",
    )

    col2.metric(
        "Buy-and-hold return",
        f"{metrics['buy_hold_return']:.2%}",
    )

    col3.metric(
        "Maximum drawdown",
        f"{metrics['max_drawdown']:.2%}",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Final simulated equity",
        f"${metrics['final_equity']:,.2f}",
    )

    col2.metric(
        "Completed trades",
        metrics["completed_trades"],
    )

    col3.metric(
        "Market exposure",
        f"{metrics['market_exposure']:.1%}",
    )

    st.subheader("Portfolio Equity Over Time")

    equity_chart = daily_results[["Strategy_Equity", "Buy_Hold_Equity"]].rename(
        columns={
            "Strategy_Equity": "Strategy",
            "Buy_Hold_Equity": "Buy and Hold",
        }
    )

    st.line_chart(equity_chart)

    st.subheader("Detailed Performance Metrics")

    st.dataframe(
        format_metrics(metrics),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Recent Backtest Observations")

    st.dataframe(
        daily_results[
            [
                "Close",
                "Fast_MA",
                "Slow_MA",
                "Signal",
                "Position",
                "Market_Return",
                "Strategy_Return",
                "Strategy_Equity",
            ]
        ].tail(20),
        use_container_width=True,
    )

    st.download_button(
        "Download Backtest Results (CSV)",
        data=daily_results.to_csv().encode("utf-8"),
        file_name=f"{selected_symbol}_backtest.csv",
        mime="text/csv",
    )

    st.warning(
        "Historical simulations can be misleading. These results do not "
        "guarantee future performance. The strategy uses daily closing "
        "prices and simplified execution-cost assumptions."
    )
