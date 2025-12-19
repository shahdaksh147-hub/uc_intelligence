import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import zipfile
import io
import datetime

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier

# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(
    page_title="UC Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 UC Intelligence Platform (NSE)")
st.caption("Auto NSE Scan • Live Rates • Charts • UC Probability")

# =================================================
# NSE BHAVCOPY UNIVERSE (AUTO SCAN)
# =================================================
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_nse_universe(min_turnover_cr=5):
    today = datetime.date.today()
    df = None

    for i in range(7):  # fallback for holidays/weekends
        date = today - datetime.timedelta(days=i)
        date_str = date.strftime("%d%b%Y").upper()

        url = (
            "https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{date.year}/{date.strftime('%b').upper()}/"
            f"cm{date_str}bhav.csv.zip"
        )

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue

            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            df = pd.read_csv(z.open(csv_name))
            break
        except Exception:
            continue

    if df is None or df.empty:
        return []

    df.columns = df.columns.str.strip()

    # Keep equity only
    df = df[df["SERIES"] == "EQ"]

    # Turnover filter
    df["TURNOVER_CR"] = df["TOTTRDQTY"] * df["CLOSE"] / 1e7
    df = df[df["TURNOVER_CR"] >= min_turnover_cr]

    symbols = sorted(df["SYMBOL"].unique())
    return [s + ".NS" for s in symbols]

# =================================================
# SIDEBAR SETTINGS
# =================================================
st.sidebar.header("⚙ NSE Universe Settings")

min_turnover = st.sidebar.slider(
    "Min Daily Turnover (₹ Crores)",
    min_value=1,
    max_value=100,
    value=5
)

NSE_STOCKS = get_nse_universe(min_turnover)

st.sidebar.success(f"Universe loaded: {len(NSE_STOCKS)} stocks")

# Limit scan for Streamlit Cloud safety
MAX_STOCKS = 50
SCAN_STOCKS = NSE_STOCKS[:MAX_STOCKS]

# =================================================
# LIVE DATA SCAN
# =================================================
st.subheader("📊 Live Rates & UC Scores")

live_rows = []
hist_data = {}

for ticker in SCAN_STOCKS:
    try:
        df = yf.download(
            ticker,
            period="2d",
            interval="5m",
            progress=False
        )

        if df is None or df.empty or len(df) < 20:
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

# 🔴 SAFE DataFrame (schema enforced)
df_live = pd.DataFrame(
    live_rows,
    columns=["Stock", "Price", "Change %", "RSI", "Vol Ratio", "UC Score (0–3)"]
)

if df_live.empty:
    st.warning("No stocks passed filters (holiday / low liquidity). Try lowering turnover.")
    st.stop()

df_live = df_live.sort_values("UC Score (0–3)", ascending=False)
st.dataframe(df_live, use_container_width=True)

# =================================================
# MINI CHARTS
# =================================================
st.subheader("📈 Mini Charts (Expandable)")

for stock in df_live["Stock"].dropna().tolist():
    ticker = stock + ".NS"
    df = hist_data.get(ticker)

    if df is None or df.empty:
        continue

    with st.expander(stock):
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
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# =================================================
# DETAILED STOCK VIEW
# =================================================
st.subheader("📌 Detailed Stock Analysis")

selected = st.selectbox(
    "Select Stock",
    df_live["Stock"].apply(lambda x: x + ".NS").tolist()
)

df = hist_data.get(selected)

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

# =================================================
# ML UC PROBABILITY (PLACEHOLDER)
# =================================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Dummy training
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

# =================================================
# METRICS
# =================================================
col1, col2, col3 = st.columns(3)
col1.metric("UC Probability (%)", f"{uc_prob:.2f}")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("Volume Ratio", f"{latest['Vol_Ratio']:.2f}")

# =================================================
# DETAILED CHART
# =================================================
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
    title=f"{selected.replace('.NS','')} – Intraday Chart",
    height=500,
    xaxis_rangeslider_visible=False,
    yaxis2=dict(overlaying="y", side="right", showgrid=False)
)

st.plotly_chart(fig, use_container_width=True)

# =================================================
# UC INTERPRETATION
# =================================================
if uc_prob >= 75:
    st.success("🔥 HIGH Upper Circuit Probability")
elif uc_prob >= 55:
    st.warning("⚡ MODERATE Upper Circuit Probability")
else:
    st.info("❄ LOW Upper Circuit Probability")

# =================================================
# FOOTER
# =================================================
st.caption(
    "⚠️ Educational tool only. Uses NSE bhavcopy + Yahoo Finance. "
    "UC probability is probabilistic, not guaranteed."
)
