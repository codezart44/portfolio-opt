# %%
import pandas as pd
import numpy as np
from popt.backtest.modules.riskmodel import Riskmodel

# %%
ffr_d = pd.read_parquet("./data/return/ffr_d.parquet")
ret_d = pd.read_parquet("./data/return/return_d.parquet")
ffr_d.shape, ret_d.shape

# %%
_2W = 10
_2M = 42
_6M = 126

# saving new
rm = Riskmodel(k=6, lookback=_6M, halflife_cov=_6M, halflife_vol=_2M)
rm.fit(ret_d)
rm.save_to_npz(file_path="./data/riskmodel/k6_lb6m_hc6m_hv2m.npz", verbose=True)

rm = Riskmodel(k=8, lookback=_6M, halflife_cov=_6M, halflife_vol=_2M)
rm.fit(ret_d)
rm.save_to_npz(file_path="./data/riskmodel/k8_lb6m_hc6m_hv2m.npz", verbose=True)

rm = Riskmodel(k=10, lookback=_6M, halflife_cov=_6M, halflife_vol=_2M)
rm.fit(ret_d)
rm.save_to_npz(file_path="./data/riskmodel/k10_lb6m_hc6m_hv2m.npz", verbose=True)

rm = Riskmodel(k=6, lookback=_6M, halflife_cov=_2W, halflife_vol=_2M)  # reactive riskmodel
rm.fit(ret_d)
rm.save_to_npz(file_path="./data/riskmodel/k6_lb6m_hc2w_hv2m.npz", verbose=True)
