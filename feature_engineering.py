import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

def build_features(df):
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    df["EMA20"] = EMAIndicator(df["Close"], 20).ema_indicator()

    df["Price_Change"] = df["Close"].pct_change() * 100
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Near_High"] = (df["Close"] >= 0.99 * df["High"].rolling(20).max()).astype(int)

    return df.dropna()
