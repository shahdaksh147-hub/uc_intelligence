import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

def build_features(df):
    df = df.copy()

    # 🔴 FIX: Flatten columns if yfinance returns MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 🔴 ENSURE 1-D Series
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    volume = df["Volume"].astype(float)

    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["EMA20"] = EMAIndicator(close, window=20).ema_indicator()

    df["Price_Change"] = close.pct_change() * 100
    df["Volume_Ratio"] = volume / volume.rolling(20).mean()
    df["Near_High"] = (close >= 0.99 * high.rolling(20).max()).astype(int)

    return df.dropna()
