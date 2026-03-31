import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import os

fred_api_key = os.environ.get("FRED_API_KEY")
if fred_api_key is None: raise RuntimeError("Fred API key not set.")
D0     = "1954-01-03"  # first date
D1     = "2026-01-01"  # final date
D0_BTC = "2017-09-07"  # according to Kasper's paper
AUTO_ADJUST = True

# "IS3S.DE"
universe = [
    "SPY", "QQQ", "IWM", # market
    "AGG", "TLT", "LQD", "TIP",  # bonds
    "GLD", "SLV", "CPER", "DBB",  # precious metals
    "XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB", # sectors
    "IBB", "IYR",  # Extra
    "EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWD", "EWL", "EWC",  # international "EEM", 
    "DBC", "DBA", "CORN", "SOYB", "USO", "WEAT", "CANE",  # commodities COTN.L
    "BTC-USD",  # crypto
]


# risk free data (Fed Fund Rate - FFR)
fred = Fred(api_key=fred_api_key)

ffr: pd.Series = fred.get_series("DFF", observation_start=D0)
ffr_y = ffr / 100
ffr_d = (1.0+ffr_y) ** (1/252) - 1.0

pd.DataFrame(ffr_d, columns=["DFF"]).to_parquet("./data/return/ffr_d.parquet")
pd.DataFrame(ffr_y, columns=["DFF"]).to_parquet("./data/return/ffr_y.parquet")


# risky asset data
ohlcv: pd.DataFrame = yf.download(universe, start=D0, end=D1, auto_adjust=AUTO_ADJUST)

trading_days: pd.DatetimeIndex = ohlcv["Close"]["SPY"].dropna().index
close: pd.DataFrame  = ohlcv["Close"].reindex(trading_days)
volume: pd.DataFrame = ohlcv["Volume"].reindex(trading_days)

close.loc[:D0_BTC, "BTC-USD"] = np.nan
volume.loc[:D0_BTC, "BTC-USD"] = np.nan

close.to_parquet("./data/ohlcv/close.parquet")
volume.to_parquet("./data/ohlcv/volume.parquet")

return_d = close.pct_change()
return_d.to_parquet("./data/return/return_d.parquet")
