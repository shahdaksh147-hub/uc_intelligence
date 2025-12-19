import yfinance as yf

def get_nse_universe():
    # Liquid + UC-prone universe (can expand later)
    stocks = [
        "IRFC.NS","IREDA.NS","HUDCO.NS","NBCC.NS","SUZLON.NS",
        "YESBANK.NS","ADANIPOWER.NS","TATASTEEL.NS",
        "PNB.NS","SJVN.NS","IOB.NS","IDFCFIRSTB.NS"
    ]
    return stocks
