import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 3, 5)


def load_history(symbol):
    """Load up to five years of daily historical price data."""
    data = yf.Ticker(symbol).history(period="5y", interval="1d", auto_adjust=True)

    if data.empty:
        raise ValueError(f"No historical data available for {symbol}.")

    required = ["Close", "High", "Low", "Volume"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    data = data[required].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if len(data) < 300:
        raise ValueError(f"Not enough historical data for {symbol}.")

    return data


def build_features(data):
    """Build features using only current and past market data."""
    close = data["Close"]
    volume = data["Volume"]

    features = pd.DataFrame(index=data.index)

    # Historical price momentum
    for period in (1, 2, 5, 10, 20):
        features[f"return_{period}d"] = close.pct_change(period)

    # Trend relative to moving averages
    for period in (5, 10, 20, 50):
        average = close.rolling(period).mean()
        features[f"ma_ratio_{period}d"] = close / average - 1

    # RSI, using rolling average gains and losses
    daily_change = close.diff()
    gains = daily_change.clip(lower=0)
    losses = -daily_change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    features["rsi_14"] = 100 - 100 / (1 + relative_strength)

    # Short- and medium-term volatility
    daily_returns = close.pct_change()

    features["volatility_5d"] = daily_returns.rolling(5).std()

    features["volatility_20d"] = daily_returns.rolling(20).std()

    # Volume behaviour
    features["volume_change_1d"] = volume.pct_change()

    average_volume = volume.rolling(20).mean()

    features["volume_ratio_20d"] = volume / average_volume.replace(0, np.nan)

    # Remove invalid values and rows without sufficient history
    features = features.replace([np.inf, -np.inf], np.nan)

    return features.dropna()


def train_horizon_model(features, close, horizon):
    """
    Predict whether the close in `horizon` trading sessions
    will exceed the current close.

    Uses chronological training and testing, with a small
    gap to reduce overlap between training and test labels.
    """
    future_close = close.shift(-horizon)

    # Unavailable future outcomes must remain missing,
    # not be incorrectly labelled as downward movements.
    target = (future_close > close).where(future_close.notna())

    dataset = features.copy()
    dataset["target"] = target
    dataset = dataset.dropna(subset=["target"])

    if len(dataset) < 300:
        raise ValueError(f"Not enough labelled data for the " f"{horizon}-day model.")

    dataset["target"] = dataset["target"].astype(int)

    split = int(len(dataset) * 0.80)

    # Purge the last few training rows because their future
    # outcome windows could overlap the test period.
    train_end = split - horizon

    train = dataset.iloc[:train_end]
    test = dataset.iloc[split:]

    if len(train) < 100 or len(test) < 30:
        raise ValueError(
            f"Insufficient training or test observations "
            f"for the {horizon}-day model."
        )

    feature_columns = list(features.columns)

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training data has only one outcome class " f"for the {horizon}-day model."
        )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate only on the later, held-out period.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, test_predictions)

    brier_score = brier_score_loss(y_test, test_probabilities)

    # Baseline: always predict the majority class in training.
    majority_class = int(y_train.mean() >= 0.5)

    baseline_predictions = np.full(len(y_test), majority_class)

    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # Generate a forecast from the most recent feature row.
    latest_features = features[feature_columns].iloc[[-1]]

    latest_probability = float(
        model.predict_proba(latest_features)[0][list(model.classes_).index(1)]
    )

    return {
        "horizon": horizon,
        "up_probability": latest_probability,
        "down_probability": 1 - latest_probability,
        "test_accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "brier_score": float(brier_score),
        "test_observations": int(len(test)),
        "training_observations": int(len(train)),
        "forecast_date": str(features.index[-1].date()),
    }


def generate_forecasts(symbol):
    """Generate independent 1-, 3- and 5-day forecasts."""
    history = load_history(symbol)
    features = build_features(history)

    results = []

    for horizon in HORIZONS:
        result = train_horizon_model(features, history["Close"], horizon)

        results.append(result)

    return results
