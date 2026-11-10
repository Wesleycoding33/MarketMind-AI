"""
MarketMind AI — basic historical backtesting engine.

This module tests a simple long-only moving-average strategy on daily
closing prices. It is a research tool, not a guarantee of future results.

Important design choices:
- Signals are calculated using data available at the close of day t.
- Positions are shifted by one trading day, so day-t signals only affect
  day-(t+1) returns (avoids same-close look-ahead bias).
- Transaction costs and slippage are charged when exposure changes.
- No leverage, short selling, or intraday stop/target simulation is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Return the minimum percentage drawdown from a previous equity peak."""
    if equity_curve.empty:
        return 0.0
    peaks = equity_curve.cummax()
    drawdowns = equity_curve / peaks - 1.0
    return float(drawdowns.min())


def _trade_stats(position: pd.Series, strategy_returns: pd.Series) -> dict[str, Any]:
    """
    Estimate completed position episodes.

    A trade episode starts when exposure changes from 0 to 1 and ends when
    exposure changes from 1 to 0. Open positions at the end are marked open
    and are not counted as completed trades.
    """
    pos = position.fillna(0).astype(int)
    changes = pos.diff().fillna(pos.iloc[0] if len(pos) else 0)
    entries = list(pos.index[changes == 1])
    exits = list(pos.index[changes == -1])

    completed = []
    exit_index = 0
    for entry in entries:
        while exit_index < len(exits) and exits[exit_index] <= entry:
            exit_index += 1
        if exit_index >= len(exits):
            break
        exit_date = exits[exit_index]
        episode = strategy_returns.loc[entry:exit_date]
        completed.append(float((1.0 + episode).prod() - 1.0))
        exit_index += 1

    wins = sum(value > 0 for value in completed)
    return {
        "completed_trades": len(completed),
        "winning_trades": wins,
        "win_rate": wins / len(completed) if completed else None,
        "average_trade_return": float(np.mean(completed)) if completed else None,
        "open_position_at_end": bool(len(pos) and pos.iloc[-1] == 1),
    }


def run_backtest(
    history: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    fast_window: int = 20,
    slow_window: int = 50,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Backtest a simple long-only moving-average strategy.

    Parameters
    ----------
    history:
        DataFrame with a 'Close' column and a datetime index.
    initial_capital:
        Starting simulated portfolio value.
    fee_rate:
        One-way transaction fee fraction; 0.001 means 0.10%.
    slippage_rate:
        One-way slippage assumption; 0.0005 means 0.05%.
    fast_window / slow_window:
        Moving-average windows. The fast window must be smaller.
    periods_per_year:
        Trading periods used to annualise volatility and Sharpe ratio.

    Returns
    -------
    Dictionary containing metrics, a daily results DataFrame, and settings.
    """
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("Provide a non-empty historical price DataFrame.")
    if "Close" not in history.columns:
        raise ValueError("Historical data must contain a 'Close' column.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("Fees and slippage cannot be negative.")
    if fast_window < 1 or slow_window < 2 or fast_window >= slow_window:
        raise ValueError("Use positive windows where fast_window < slow_window.")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive.")

    data = history[["Close"]].copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["Close"] > 0]
    data = data[~data.index.duplicated(keep="last")].sort_index()

    if len(data) < slow_window + 2:
        raise ValueError(
            f"At least {slow_window + 2} valid closing prices are required."
        )

    data["Fast_MA"] = data["Close"].rolling(fast_window).mean()
    data["Slow_MA"] = data["Close"].rolling(slow_window).mean()

    # Desired exposure known after each close; execute from the next session.
    data["Signal"] = (
        (data["Close"] > data["Fast_MA"]) & (data["Fast_MA"] > data["Slow_MA"])
    ).astype(int)
    data["Position"] = data["Signal"].shift(1).fillna(0).astype(int)

    data["Market_Return"] = data["Close"].pct_change().fillna(0.0)
    data["Turnover"] = data["Position"].diff().abs()
    data.loc[data.index[0], "Turnover"] = abs(float(data["Position"].iloc[0]))

    one_way_cost = fee_rate + slippage_rate
    data["Trading_Cost"] = data["Turnover"].fillna(0.0) * one_way_cost

    # Costs are deducted from the daily strategy return on position changes.
    data["Strategy_Return"] = (
        data["Position"] * data["Market_Return"] - data["Trading_Cost"]
    )
    data["Strategy_Equity"] = (
        initial_capital * (1.0 + data["Strategy_Return"]).cumprod()
    )

    # Buy-and-hold benchmark: first return is zero because no prior close exists.
    data["Buy_Hold_Return"] = data["Market_Return"]
    data["Buy_Hold_Equity"] = (
        initial_capital * (1.0 + data["Buy_Hold_Return"]).cumprod()
    )

    strategy_returns = data["Strategy_Return"].dropna()
    daily_volatility = (
        float(strategy_returns.std(ddof=1)) if len(strategy_returns) > 1 else 0.0
    )
    annual_volatility = daily_volatility * np.sqrt(periods_per_year)
    mean_return = float(strategy_returns.mean()) if len(strategy_returns) else 0.0
    sharpe = (
        mean_return / daily_volatility * np.sqrt(periods_per_year)
        if daily_volatility > 0
        else None
    )

    total_return = float(data["Strategy_Equity"].iloc[-1] / initial_capital - 1.0)
    buy_hold_return = float(data["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1.0)
    years = max(len(data) - 1, 1) / periods_per_year
    cagr = (
        float((data["Strategy_Equity"].iloc[-1] / initial_capital) ** (1 / years) - 1)
        if years > 0 and data["Strategy_Equity"].iloc[-1] > 0
        else None
    )

    trade_stats = _trade_stats(data["Position"], data["Strategy_Return"])

    metrics = {
        "initial_capital": float(initial_capital),
        "final_equity": float(data["Strategy_Equity"].iloc[-1]),
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return_vs_buy_hold": total_return - buy_hold_return,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(data["Strategy_Equity"]),
        "annualised_volatility": float(annual_volatility),
        "sharpe_ratio_zero_risk_free": sharpe,
        "market_exposure": float(data["Position"].mean()),
        "total_trading_cost_fraction": float(data["Trading_Cost"].sum()),
        "start_date": str(data.index[0].date()),
        "end_date": str(data.index[-1].date()),
        "observations": int(len(data)),
        **trade_stats,
    }

    return {
        "metrics": metrics,
        "daily_results": data,
        "settings": {
            "strategy": "Long-only moving-average trend",
            "fast_window": fast_window,
            "slow_window": slow_window,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "initial_capital": initial_capital,
        },
    }


def format_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return a dashboard-friendly two-column table of backtest metrics."""
    rows = [
        ("Initial capital", f"${metrics['initial_capital']:,.2f}"),
        ("Final simulated equity", f"${metrics['final_equity']:,.2f}"),
        ("Strategy total return", f"{metrics['total_return']:.2%}"),
        ("Buy-and-hold return", f"{metrics['buy_hold_return']:.2%}"),
        ("Excess return vs benchmark", f"{metrics['excess_return_vs_buy_hold']:.2%}"),
        (
            "CAGR (approx.)",
            "N/A" if metrics["cagr"] is None else f"{metrics['cagr']:.2%}",
        ),
        ("Maximum drawdown", f"{metrics['max_drawdown']:.2%}"),
        ("Annualised volatility", f"{metrics['annualised_volatility']:.2%}"),
        (
            "Sharpe ratio (zero risk-free rate)",
            (
                "N/A"
                if metrics["sharpe_ratio_zero_risk_free"] is None
                else f"{metrics['sharpe_ratio_zero_risk_free']:.2f}"
            ),
        ),
        ("Market exposure", f"{metrics['market_exposure']:.2%}"),
        ("Completed trades", str(metrics["completed_trades"])),
        (
            "Win rate",
            "N/A" if metrics["win_rate"] is None else f"{metrics['win_rate']:.2%}",
        ),
        (
            "Average completed-trade return",
            (
                "N/A"
                if metrics["average_trade_return"] is None
                else f"{metrics['average_trade_return']:.2%}"
            ),
        ),
        ("Observations", str(metrics["observations"])),
        ("Period", f"{metrics['start_date']} to {metrics['end_date']}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
