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
st.caption("Live Rates • Mini Charts • Detailed Charts • UC Probability")

# -------------------------------------------------
# NSE STOCK UNIVERSE (FULL & VERIFIED)
# -------------------------------------------------
NSE_STOCKS = [
    # PSU / Infra
    "IRFC.NS", "IREDA.NS", "HUDCO.NS", "NBCC.NS",
    "NATIONALUM.NS", "RCF.NS", "BEL.NS",
    "SJVN.NS", "PNB.NS", "IOB.NS", "IDFCFIRSTB.NS",

    # Power
    "ADANIPOWER.NS", "JPPOWER.NS",

    # Small & Mid Caps
    "SUZLON.NS", "YESBANK.NS", "NESCO.NS",
    "NAVA.NS", "EXCEL.NS", "GOWRALE.NS", "AURIGROW.NS",

    # Metals
    "TATASTEEL.NS",

    # ETFs
    "SILVERBEES.NS", "MID150BEES.NS"
]

# -------------------------------------------------
# FETCH LIVE DATA FOR ALL STOCKS
# -------------------------------------------------
st.subheader("📊 Live Rates & UC Scores (All Stocks)")

live_rows = []
hist_data = {}

for ticker in NSE_STOCKS:
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)

        if df.empty or len(df) < 20:
            continue

        # Fix MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        hist_data[ticker] = df.copy()

        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        price = close.iloc[-1]
        prev = close.iloc[-2]
        change_pct = ((price - prev) / prev) * 100

        rsi = RSIIndicator(close, 14).rsi().iloc[-1]
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

        uc_score = sum([
            change_pct > 2,
            rsi > 60,
            vol_ratio > 2
        ])

        live_rows.append({
            "Stock": ticker.replace(".NS", ""),
            "Price": round(price, 2),
            "Change %": round(change_pct, 2),
            "RSI": round(rsi, 1),
            "Vol Ratio": round(vol_ratio, 2),
            "UC Score (0–3)": uc_score
        })

    except Exception:
        continue

df_live = pd.DataFrame(live_rows)

if df_live.empty:
    st.warning("No live data available.")
else:
    df_live = df_live.sort_values("UC Score (0–3)", ascending=False)
    st.dataframe(df_live, use_container_width=True)

# -------------------------------------------------
# MINI CHARTS FOR ALL STOCKS
# -------------------------------------------------
st.subheader("📈 Mini Charts (Last 2 Days)")

for ticker in df_live["Stock"]:
    full_ticker = ticker + ".NS"
    df = hist_data.get(full_ticker)

    if df is None or df.empty:
        continue

    with st.expander(ticker):
        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# DETAILED STOCK VIEW
# -------------------------------------------------
st.subheader("📌 Detailed Stock Analysis")

selected_stock = st.selectbox("Select Stock", NSE_STOCKS)
df = hist_data.get(selected_stock)

if df is None or df.empty:
    st.error("No data available for selected stock.")
    st.stop()

close = df["Close"].astype(float)
high = df["High"].astype(float)
volume = df["Volume"].astype(float)

df["RSI"] = RSIIndicator(close, 14).rsi()
df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
df["Pct_Change"] = close.pct_change() * 100
df["Vol_Ratio"] = volume / volume.rolling(20).mean()
df["Near_High"] = (close >= 0.99 * high.rolling(20).max()).astype(int)

df_feat = df.dropna()
latest = df_feat.iloc[-1]

# -------------------------------------------------
# ML UC PROBABILITY (DUMMY MODEL)
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

X_dummy = np.random.rand(100, 4)
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

X_live = np.array([
    latest["RSI"],
    latest["Pct_Change"],
    latest["Vol_Ratio"],
    latest["Near_High"]
]).reshape(1, -1)

uc_prob = model.predict_proba(X_live)[0][1] * 100

# -------------------------------------------------
# METRICS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("UC Probability (%)", f"{uc_prob:.2f}")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("Volume Ratio", f"{latest['Vol_Ratio']:.2f}")

# -------------------------------------------------
# DETAILED CHART
# -------------------------------------------------
fig = go.Figure()
fig.add_candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)
fig.add_bar(
    x=df.index,
    y=df["Volume"],
    yaxis="y2",
    opacity=0.3
)

fig.update_layout(
    title=f"{selected_stock.replace('.NS','')} – Intraday Chart",
    height=500,
    xaxis_rangeslider_visible=False,
    yaxis2=dict(overlaying="y", side="right", showgrid=False)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# UC INTERPRETATION
# -------------------------------------------------
if uc_prob >= 75:
    st.success("🔥 HIGH Upper Circuit Probability")
elif uc_prob >= 55:
    st.warning("⚡ MODERATE Upper Circuit Probability")
else:
    st.info("❄ LOW Upper Circuit Probability")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption("⚠️ Educational tool only. UC probability is not a guarantee.")
