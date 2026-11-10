import pandas as pd


# These are configurable research thresholds, not statistically validated
# trading rules. Reassess them with out-of-sample testing before use.
HORIZON_WEIGHTS = {1: 1, 3: 2, 5: 2}
BULLISH_THRESHOLD = 0.58
BEARISH_THRESHOLD = 0.42
MIN_SUPPORTING_HORIZONS = 2


def analyse_technical_trend(history):
    """Calculate simple moving-average, momentum and volatility indicators."""
    if history is None or history.empty:
        raise ValueError("Historical price data is unavailable.")
    if "Close" not in history.columns:
        raise ValueError("Historical data must contain a Close column.")

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 50:
        raise ValueError("At least 50 closing-price observations are required.")

    latest_price = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    momentum_5d = float(close.pct_change(5).iloc[-1])
    momentum_20d = float(close.pct_change(20).iloc[-1])
    volatility_20d = float(close.pct_change().dropna().tail(20).std())

    if latest_price > ma20 > ma50:
        trend = "Uptrend"
    elif latest_price < ma20 < ma50:
        trend = "Downtrend"
    else:
        trend = "Mixed"

    return {
        "latest_price": latest_price,
        "ma20": ma20,
        "ma50": ma50,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "volatility_20d": volatility_20d,
        "trend": trend,
    }


def evaluate_forecasts(forecasts):
    """Create a weighted consensus from available 1-, 3- and 5-day forecasts."""
    if not forecasts:
        raise ValueError("No forecasts were provided.")

    forecast_map = {}
    for forecast in forecasts:
        horizon = int(forecast["horizon"])
        probability = float(forecast["up_probability"])
        if horizon not in HORIZON_WEIGHTS:
            continue
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid upward probability for {horizon}-day forecast.")
        forecast_map[horizon] = forecast

    if not forecast_map:
        raise ValueError("No supported forecast horizons were found.")

    total_weight = sum(HORIZON_WEIGHTS[h] for h in forecast_map)
    consensus = sum(
        float(forecast_map[h]["up_probability"]) * HORIZON_WEIGHTS[h]
        for h in forecast_map
    ) / total_weight

    bullish_horizons = sum(
        float(item["up_probability"]) >= 0.55
        for item in forecast_map.values()
    )
    bearish_horizons = sum(
        float(item["up_probability"]) <= 0.45
        for item in forecast_map.values()
    )

    if consensus >= BULLISH_THRESHOLD and bullish_horizons >= MIN_SUPPORTING_HORIZONS:
        signal = "Bullish Watch"
        explanation = (
            "The weighted upward probability is elevated and multiple "
            "forecast horizons support the direction."
        )
    elif consensus <= BEARISH_THRESHOLD and bearish_horizons >= MIN_SUPPORTING_HORIZONS:
        signal = "Bearish Risk"
        explanation = (
            "The weighted upward probability is low and multiple forecast "
            "horizons indicate downside risk."
        )
    else:
        signal = "Neutral / Mixed"
        explanation = (
            "The forecasts do not meet the configured threshold for a "
            "stronger directional assessment."
        )

    return {
        "signal": signal,
        "explanation": explanation,
        "consensus_up_probability": float(consensus),
        "consensus_down_probability": float(1.0 - consensus),
        "bullish_horizons": int(bullish_horizons),
        "bearish_horizons": int(bearish_horizons),
        "horizons_available": sorted(forecast_map),
    }


def evaluate_model_quality(forecasts):
    """Present existing holdout diagnostics without asserting profitability."""
    rows = []
    for forecast in forecasts:
        accuracy = float(forecast["test_accuracy"])
        baseline = float(forecast["baseline_accuracy"])
        brier = float(forecast["brier_score"])

        if accuracy > baseline:
            comparison = "Beat baseline"
        elif accuracy < baseline:
            comparison = "Below baseline"
        else:
            comparison = "Matched baseline"

        rows.append(
            {
                "Horizon": f"{int(forecast['horizon'])} day(s)",
                "Holdout accuracy": accuracy,
                "Baseline accuracy": baseline,
                "Difference vs baseline": accuracy - baseline,
                "Brier score": brier,
                "Baseline comparison": comparison,
                "Test observations": int(forecast["test_observations"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_trade_setup(history, forecasts):
    """Combine forecasts, technical trend and historical diagnostics."""
    if not forecasts:
        raise ValueError("No forecasts were provided.")

    technical = analyse_technical_trend(history)
    forecast_assessment = evaluate_forecasts(forecasts)
    model_quality = evaluate_model_quality(forecasts)
    signal = forecast_assessment["signal"]

    if signal == "Bullish Watch":
        technical_confirmation = (
            technical["trend"] == "Uptrend" and technical["momentum_20d"] > 0
        )
    elif signal == "Bearish Risk":
        technical_confirmation = (
            technical["trend"] == "Downtrend" and technical["momentum_20d"] < 0
        )
    else:
        technical_confirmation = False

    if signal in ("Bullish Watch", "Bearish Risk"):
        confirmation_note = (
            "The technical trend supports the forecast assessment."
            if technical_confirmation
            else "The technical trend does not fully confirm the forecast assessment."
        )
    else:
        confirmation_note = "No strong directional assessment was produced."

    underperforming_horizons = [
        int(item["horizon"])
        for item in forecasts
        if float(item["test_accuracy"]) < float(item["baseline_accuracy"])
    ]

    return {
        "signal": signal,
        "explanation": forecast_assessment["explanation"],
        "consensus_up_probability": forecast_assessment["consensus_up_probability"],
        "consensus_down_probability": forecast_assessment["consensus_down_probability"],
        "technical": technical,
        "technical_confirmation": technical_confirmation,
        "confirmation_note": confirmation_note,
        "model_quality": model_quality,
        "underperforming_horizons": underperforming_horizons,
        "forecast_date": forecasts[0].get("forecast_date", "Unknown"),
    }
