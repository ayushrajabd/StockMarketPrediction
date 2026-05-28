import numpy as np
import pandas as pd


def add_indicators(data):
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    for column in ["Close", "High", "Low", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["SMA"] = data["Close"].rolling(10).mean()
    data["Momentum"] = data["Close"] - data["Close"].shift(10)

    close_delta = data["Close"].diff()
    gain = close_delta.clip(lower=0).rolling(14).mean()
    loss = (-close_delta.clip(upper=0)).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + relative_strength))

    data["EMA"] = data["Close"].ewm(span=14, adjust=False).mean()
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26

    rolling_mean = data["Close"].rolling(20).mean()
    rolling_std = data["Close"].rolling(20).std()
    data["BB_upper"] = rolling_mean + (2 * rolling_std)
    data["BB_lower"] = rolling_mean - (2 * rolling_std)

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()

    direction = np.sign(data["Close"].diff()).fillna(0)
    data["OBV"] = (direction * data["Volume"]).cumsum()

    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

    return data.dropna()
