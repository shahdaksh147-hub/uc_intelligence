import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

from nse_universe import get_nse_universe
from feature_engineering import build_features
from ml_model import train_dummy_model, predict_uc_probability

# --------------------------------
st.set_page_config(page_title="UC Intelligence", layout="wide")
st.title("🚀 UC Intelligence Platform (NSE)")

# --------------------------------
stocks = get_nse_universe()
selected_stock = st.selectbox("Select Stock", stocks)

# --------------------------------
df = yf.download(
    selected_stock,
    period="5d",
    interval="5m",
    progress=False
)

df_feat = build_features(df)

latest = df_feat.iloc[-1]

# --------------------------------
# ML Probability
model = train_dummy_model()

X = latest[[
    "RSI",
    "Price_Change",
    "Volume_Ratio",
    "Near_High"
]].values.reshape(1, -1)

uc_prob = predict_uc_probability(model, X)

# --------------------------------
# METRICS
col1, col2, col3 = st.columns(3)

col1.metric("UC Probability", f"{uc_prob} %")
col2.metric("RSI", round(latest["RSI"], 1))
col3.metric("Volume Ratio", round(latest["Volume_Ratio"], 2))

# --------------------------------
# CANDLESTICK + VOLUME CHART
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
    height=600,
    yaxis2=dict(
        overlaying="y",
        side="right",
        showgrid=False
    )
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# UC SIGNAL
if uc_prob >= 70:
    st.success("🔥 HIGH Upper Circuit Probability")
elif uc_prob >= 50:
    st.warning("⚡ Moderate UC Probability")
else:
    st.info("❄ Low UC Probability")
