#%% imports
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import os

from popt.config import FIRST_DATE, FINAL_DATE, FIRST_DATE_BTC, AUTO_ADJUST, universe

#%% risk free data (Fed Fund Rate - FFR)
fred_api_key = os.environ.get("FRED_API_KEY")
if fred_api_key is None: raise RuntimeError("Fred API key not set.")
fred = Fred(api_key=fred_api_key)

ffr: pd.Series = fred.get_series("DFF", observation_start=FIRST_DATE)
ffr_y = ffr / 100
ffr_d = (1.0+ffr_y) ** (1/252) - 1.0

pd.DataFrame(ffr_d, columns=["DFF"]).to_parquet("./data/return/ffr_d.parquet")
pd.DataFrame(ffr_y, columns=["DFF"]).to_parquet("./data/return/ffr_y.parquet")

#%% risky asset data
ohlcv: pd.DataFrame = yf.download(
    universe, 
    start=FIRST_DATE, 
    end=FINAL_DATE, 
    auto_adjust=AUTO_ADJUST,
    progress=True,  # silence download progress bar
    )

trading_days: pd.DatetimeIndex = ohlcv["Close"]["SPY"].dropna().index
close: pd.DataFrame  = ohlcv["Close"].reindex(trading_days)
volume: pd.DataFrame = ohlcv["Volume"].reindex(trading_days)

if "BTC-USD" in universe:
    close.loc[:FIRST_DATE_BTC, "BTC-USD"] = np.nan
    volume.loc[:FIRST_DATE_BTC, "BTC-USD"] = np.nan
return_d = close.pct_change()

close.to_parquet("./data/ohlcv/close.parquet")
volume.to_parquet("./data/ohlcv/volume.parquet")
return_d.to_parquet("./data/return/return_d.parquet")
