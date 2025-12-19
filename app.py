import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="UC Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 UC Intelligence Platform (NSE)")
st.caption("Automatic NSE Scan | Candlestick + Volume | ML-based UC Probability")

# -------------------------------------------------
# NSE UNIVERSE (can expand later)
# -------------------------------------------------
NSE_STOCKS = [
    "IRFC.NS", "IREDA.NS", "HUDCO.NS", "NBCC.NS", "SUZLON.NS",
    "YESBANK.NS", "ADANIPOWER.NS", "TATASTEEL.NS",
    "PNB.NS", "SJVN.NS", "IOB.NS", "IDFCFIRSTB.NS","AURIGROW.NS"
]

selected_stock = st.selectbox("Select NSE Stock", NSE_STOCKS)

# -------------------------------------------------
# DATA DOWNLOAD
# -------------------------------------------------
df = yf.download(
    selected_stock,
    period="5d",
    interval="5m",
    progress=False
)

if df.empty:
    st.error("No data received from Yahoo Finance.")
    st.stop()

# 🔴 FIX: Flatten MultiIndex columns (Streamlit Cloud bug)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# -------------------------------------------------
# FEATURE ENGINEERING (SAFE)
# -------------------------------------------------
df = df.copy()

close = df["Close"].astype(float)
high = df["High"].astype(float)
volume = df["Volume"].astype(float)

df["RSI"] = RSIIndicator(close, window=14).rsi()
df["EMA20"] = EMAIndicator(close, window=20).ema_indicator()

df["Price_Change"] = close.pct_change() * 100
df["Volume_Ratio"] = volume / volume.rolling(20).mean()
df["Near_High"] = (close >= 0.99 * high.rolling(20).max()).astype(int)

df_feat = df.dropna()

if df_feat.empty:
    st.warning("Not enough data to compute indicators yet.")
    st.stop()

latest = df_feat.iloc[-1]

# -------------------------------------------------
# ML MODEL (UC PROBABILITY ENGINE)
# -------------------------------------------------
# NOTE: Dummy-trained model (replace later with real UC-labelled data)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Fake training data (to keep model valid)
X_dummy = np.random.rand(100, 4)
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

X_live = np.array([
    latest["RSI"],
    latest["Price_Change"],
    latest["Volume_Ratio"],
    latest["Near_High"]
]).reshape(1, -1)

uc_probability = model.predict_proba(X_live)[0][1] * 100

# -------------------------------------------------
# METRICS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("UC Probability (%)", f"{uc_probability:.2f}")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}")

# -------------------------------------------------
# CANDLESTICK + VOLUME CHART
# -------------------------------------------------
fig = go.Figure()

fig.add_candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
)

fig.add_bar(
    x=df.index,
    y=df["Volume"],
    name="Volume",
    yaxis="y2",
    opacity=0.3
)

fig.update_layout(
    title=f"{selected_stock.replace('.NS','')} – Intraday Price Action",
    height=600,
    xaxis_rangeslider_visible=False,
    yaxis2=dict(
        overlaying="y",
        side="right",
        showgrid=False
    )
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# UC SIGNAL INTERPRETATION
# -------------------------------------------------
if uc_probability >= 75:
    st.success("🔥 HIGH Upper Circuit Probability")
elif uc_probability >= 55:
    st.warning("⚡ MODERATE Upper Circuit Probability")
else:
    st.info("❄ LOW Upper Circuit Probability")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption(
    "⚠️ Educational tool. UC probability ≠ guaranteed returns. "
    "Upgrade to broker WebSocket for true real-time signals."
)
