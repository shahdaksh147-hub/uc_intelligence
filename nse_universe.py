import pandas as pd
import requests
import zipfile
import io
import datetime
import streamlit as st

# -------------------------------------------------
# DOWNLOAD NSE BHAVCOPY (CSV)
# -------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_nse_universe(min_turnover_cr=1):
    """
    Returns list of NSE symbols (Yahoo-compatible)
    Filters illiquid stocks using turnover
    """

    # NSE bhavcopy URL (Equity)
    today = datetime.date.today()

    # Try last 5 days (handles holidays)
    for i in range(5):
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
            df = None

    if df is None:
        return []

    # -------------------------------------------------
    # CLEAN + FILTER
    # -------------------------------------------------
    df.columns = df.columns.str.strip()

    # Keep only EQ series
    df = df[df["SERIES"] == "EQ"]

    # Turnover in Crores
    df["TURNOVER_CR"] = df["TOTTRDQTY"] * df["CLOSE"] / 1e7

    # Illiquid filter
    df = df[df["TURNOVER_CR"] >= min_turnover_cr]

    # Convert to Yahoo symbols
    symbols = sorted(df["SYMBOL"].unique())
    yahoo_symbols = [sym + ".NS" for sym in symbols]

    return yahoo_symbols
