import numpy as np
import pandas as pd
from typing import Literal
from popt.backtest.modules.riskmodel import RiskModel

class DataBuilder:
    def __init__(
            self,
            universe: list[str],
            first_date: str,
            final_date: str,
            alpha: pd.DataFrame, 
            rd: pd.DataFrame,
            rf: pd.DataFrame,
            riskmodel: RiskModel,
            rebal_freq: Literal["D", "W", "M", "Q", "Y", None] = "M",
        ):
        _, U, k = riskmodel.F_cov.shape
        assert len(universe) == U
        assert riskmodel.d_var.shape[1] == U
        assert alpha.shape[1] == U
        assert rd.shape[1] == U

        self.d0 = first_date
        self.d1 = final_date
        timeline = self._dates_intersection([riskmodel.timeline, rd.index, rf.index])
        timeline = self._dates_truncated(timeline)

        self.universe = universe
        self.timeline = timeline
        self.alpha    = alpha.loc[timeline].values
        self.ret      = rd.loc[timeline].values + 1.0  # use simple returns
        self.rf       = rf.loc[timeline].values.ravel() + 1.0  # use simple returns
        self.F_cov    = riskmodel.F_cov[np.isin(riskmodel.timeline, timeline)]
        self.d_var    = riskmodel.d_var[np.isin(riskmodel.timeline, timeline)]
        self.asset_mask = ~np.isnan(self.F_cov).any(axis=2)
        self.trade_flag = self._trade_flag(timeline, rebal_freq)
        T, = self.trade_flag.shape

        self.ret = np.nan_to_num(self.ret, nan=0.0)
        self.F_cov = np.nan_to_num(self.F_cov, nan=0.0)
        self.d_var = np.nan_to_num(self.d_var, nan=0.0)

        assert np.any(np.isnan(self.timeline))   == False
        assert np.any(np.isnan(self.alpha))      == False
        assert np.any(np.isnan(self.ret))        == False
        assert np.any(np.isnan(self.rf))         == False
        assert np.any(np.isnan(self.F_cov))      == False
        assert np.any(np.isnan(self.d_var))      == False
        assert np.any(np.isnan(self.asset_mask)) == False
        assert np.any(np.isnan(self.trade_flag)) == False

        assert self.timeline.shape   == (T, )
        assert self.alpha.shape      == (T, U)
        assert self.ret.shape        == (T, U)
        assert self.rf.shape         == (T,)
        assert self.F_cov.shape      == (T, U, k)  # F
        assert self.d_var.shape      == (T, U)     # D
        assert self.asset_mask.shape == (T, U)
        assert self.trade_flag.shape == (T,)

    def __repr__(self):
        return f"{self.d0} : {self.d1}\n" + \
               f" :a  - {self.alpha.shape}, {type(self.alpha)}\n" + \
               f" :r  - {self.ret.shape}, {type(self.ret)}\n" + \
               f" :rf - {self.rf.shape}, {type(self.rf)}\n" + \
               f" :F  - {self.F_cov.shape}, {type(self.F_cov)}\n" + \
               f" :d  - {self.d_var.shape}, {type(self.d_var)}\n" + \
               f" :am - {self.asset_mask.shape}, {type(self.asset_mask)}\n" + \
               f" :tf - {self.trade_flag.shape}, {type(self.trade_flag)}\n" + \
               f" :universe - {self.universe}"
        
    def _dates_intersection(self, indices: list) -> pd.DatetimeIndex:
        index_overlap = pd.DatetimeIndex(indices[0])
        for index in indices[1:]:
            index_overlap = index_overlap.intersection(pd.DatetimeIndex(index))
        return index_overlap
    
    def _dates_truncated(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        assert self.d0 in index  # only allow exact date selection, forces explicitness
        assert self.d1 in index
        index: pd.DatetimeIndex = index[(self.d0 <= index) & (index <= self.d1)]
        return index
    
    def _trade_flag(self, timeline: pd.DatetimeIndex, rebal_freq: str) -> np.ndarray:
        months, quarters, years = timeline.month, timeline.quarter, timeline.year
        match rebal_freq:
            case "D":  rebal_flag = np.ones(timeline.shape[0], dtype=bool)
            case "W":  rebal_flag = (timeline.weekday == 4)  # 4 extra rebal on red fridays over 20 years, negligible
            case "M":  rebal_flag = months != np.roll(months, shift=-1)
            case "Q":  rebal_flag = quarters != np.roll(quarters, shift=-1)
            case "Y":  rebal_flag = years != np.roll(years, shift=-1)
            case None: rebal_flag = np.zeros(timeline.shape[0], dtype=bool)
            case _: raise ValueError("Invalid rebal frequency")
        rebal_flag[1] = True  # first day is initial state, second day we trade according to strategy
        return rebal_flag


# Convention: Data at day t is t inclusive always.
class DataLoader:
    def __init__(
            self,
            db: DataBuilder,
            tickers: list[str],
        ):
        T, U, _ = db.F_cov.shape
        N = len(tickers)
        assert np.isin(tickers, db.universe).all()
        assert len(db.universe) == U, len(db.universe)

        t2i = {t: i for i, t in enumerate(db.universe)}
        i_N  = np.array([t2i[t] for t in tickers], dtype=int)
        self.T = T
        self.U = U
        self.N = N
        self.tickers  = tickers
        self.universe = db.universe
        self.timeline = db.timeline
        self._alpha   = db.alpha[:, i_N]
        self._ret     = db.ret[:, i_N]
        self._rf      = db.rf
        self._F_cov   = db.F_cov[:, i_N, :]
        self._d_var   = db.d_var[:, i_N]
        self._asset_mask = db.asset_mask[:, i_N]
        self._trade_flag = db.trade_flag
    
    def get_alpha(self, t:int) -> np.ndarray:
        return self._alpha[t]
    
    def get_return(self, t:int) -> np.ndarray:
        return self._ret[t]
    
    def get_rf(self, t:int) -> np.ndarray:
        return self._rf[t]
    
    def get_F_cov(self, t:int) -> np.ndarray:
        return self._F_cov[t]  # [N, k]
    
    def get_d_var(self, t:int) -> np.ndarray:
        return self._d_var[t]  # [N,]

    def get_asset_mask(self, t: int) -> np.ndarray:
        return self._asset_mask[t]
    
    def get_trade_flag(self, t: int) -> int:
        return self._trade_flag[t]
