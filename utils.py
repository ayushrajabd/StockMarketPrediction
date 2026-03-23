import ta
import pandas as pd
import numpy as np

def add_indicators(data):

    data.columns=data.columns.get_level_values(0)

    data['Close']=pd.to_numeric(data['Close'],errors='coerce')
    data['High']=pd.to_numeric(data['High'],errors='coerce')
    data['Low']=pd.to_numeric(data['Low'],errors='coerce')
    data['Volume']=pd.to_numeric(data['Volume'],errors='coerce')

    data['SMA'] = data['Close'].rolling(10).mean()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)

    data['RSI'] = ta.momentum.rsi(data['Close'])
    data['EMA'] = ta.trend.ema_indicator(data['Close'])
    data['MACD'] = ta.trend.macd(data['Close'])

    data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
    data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])

    data['ATR'] = ta.volatility.average_true_range(
        data['High'], data['Low'], data['Close']
    )

    data['OBV'] = ta.volume.on_balance_volume(
        data['Close'], data['Volume']
    )

    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'],1,0)

    return data.dropna()
