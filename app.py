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
st.caption("Automatic NSE Scan | Live Rates | Charts | ML-based UC Probability")

# -------------------------------------------------
# NSE UNIVERSE
# -------------------------------------------------
NSE_STOCKS = [
    "IRFC.NS", "IREDA.NS", "HUDCO.NS", "NBCC.NS", "SUZLON.NS",
    "YESBANK.NS", "ADANIPOWER.NS", "TATASTEEL.NS",
    "PNB.NS", "SJVN.NS", "IOB.NS", "IDFCFIRSTB.NS",
    "AURIGROW.NS"
]

# -------------------------------------------------
# FETCH LIVE RATES FOR ALL
# -------------------------------------------------
st.subheader("📊 Live Rates & UC Scores (All Stocks)")

live_data = []

for ticker in NSE_STOCKS:
    try:
        info = yf.Ticker(ticker).history(period="2d", interval="5m")
        if info.empty:
            continue

        # Flatten MultiIndex
        if isinstance(info.columns, pd.MultiIndex):
            info.columns = info.columns.get_level_values(0)

        # last row → latest data
        latest = info.iloc[-1]
        prev_close = info["Close"].iloc[-2]

        price = latest["Close"]
        change_pct = ((price - prev_close) / prev_close) * 100

        # RSI
        rsi = RSIIndicator(info["Close"].astype(float), 14).rsi().iloc[-1]

        # Volume ratio
        vol_ratio = (
            latest["Volume"] /
            info["Volume"].rolling(20).mean().iloc[-1]
        )

        # SIMPLE UC Score
        uc_score = sum([
            change_pct > 2,
            rsi > 60,
            vol_ratio > 2
        ])

        live_data.append({
            "Stock": ticker.replace(".NS", ""),
            "Price": round(price, 2),
            "Change %": round(change_pct, 2),
            "RSI": round(rsi, 1),
            "Vol Ratio": round(vol_ratio, 2),
            "UC Score (0–3)": uc_score
        })

    except Exception as e:
        pass

df_live = pd.DataFrame(live_data)

if df_live.empty:
    st.warning("No live data available right now.")
else:
    df_live = df_live.sort_values("UC Score (0–3)", ascending=False)
    st.dataframe(df_live, use_container_width=True)

# -------------------------------------------------
# SELECT STOCK FOR CHART
# -------------------------------------------------
selected_stock = st.selectbox("Select Stock for Chart", NSE_STOCKS)

df_chart = yf.download(
    selected_stock,
    period="5d",
    interval="5m",
    progress=False
)

if df_chart.empty:
    st.error("No data received for the selected stock.")
    st.stop()

# Flatten columns if needed
if isinstance(df_chart.columns, pd.MultiIndex):
    df_chart.columns = df_chart.columns.get_level_values(0)

# -------------------------------------------------
# FEATURE ENGINEERING FOR SELECTED STOCK
# -------------------------------------------------
close = df_chart["Close"].astype(float)
high = df_chart["High"].astype(float)
volume = df_chart["Volume"].astype(float)

df_chart["RSI"] = RSIIndicator(close, window=14).rsi()
df_chart["EMA20"] = EMAIndicator(close, window=20).ema_indicator()

df_chart["Price_Change"] = close.pct_change() * 100
df_chart["Vol_Ratio"] = volume / volume.rolling(20).mean()
df_chart["Near_High"] = (close >= 0.99 * high.rolling(20).max()).astype(int)

df_feat = df_chart.dropna()

if df_feat.empty:
    st.warning("Not enough data to compute indicators yet.")
    st.stop()

latest = df_feat.iloc[-1]

# -------------------------------------------------
# ML MODEL TRAIN & PREDICT (DUMMY)
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# train dummy model (placeholder)
X_dummy = np.random.rand(100, 4)
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

X_live = np.array([
    latest["RSI"],
    latest["Price_Change"],
    latest["Vol_Ratio"],
    latest["Near_High"]
]).reshape(1, -1)

uc_probability = model.predict_proba(X_live)[0][1] * 100

# -------------------------------------------------
# METRICS FOR SELECTED STOCK
# -------------------------------------------------
st.subheader(f"📈 {selected_stock.replace('.NS','')} – Detailed Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("UC Probability (%)", f"{uc_probability:.2f}")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("Vol Ratio", f"{latest['Vol_Ratio']:.2f}")

# -------------------------------------------------
# CANDLESTICK + VOLUME
# -------------------------------------------------
fig = go.Figure()

fig.add_candlestick(
    x=df_chart.index,
    open=df_chart["Open"],
    high=df_chart["High"],
    low=df_chart["Low"],
    close=df_chart["Close"],
    name="Price"
)

fig.add_bar(
    x=df_chart.index,
    y=df_chart["Volume"],
    name="Volume",
    yaxis="y2",
    opacity=0.3
)

fig.update_layout(
    title=f"{selected_stock.replace('.NS','')} – Price Action",
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
    st.success("🔥 HIGH UC Probability")
elif uc_probability >= 55:
    st.warning("⚡ Moderate UC Probability")
else:
    st.info("❄ Low UC Probability")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption(
    "⚠️ Live rates + UC indication tool. UC probability ≠ guaranteed outcome."
)
