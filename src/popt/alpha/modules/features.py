import numpy as np
import pandas as pd
from typing import Callable

def standard_scale(x: np.ndarray, axis=1) -> np.ndarray:
    numer = x - x.mean(axis=axis, keepdims=True)
    denom = x.std(axis=axis, keepdims=True) + 1e-8
    return numer / denom

def momentum(r: np.ndarray) -> np.ndarray:
    return (1.0 + r).prod(axis=0) - 1.0

def drawdown(r: np.ndarray) -> np.ndarray:
    r_cum = (1.0 + r).cumprod(axis=0)
    r_max = np.maximum.accumulate(r_cum, axis=0)
    dd = 1.0 - r_cum / (r_max + 1e-8)
    return dd.max(axis=0)

def volatility(r: np.ndarray) -> np.ndarray:
    return r.std(axis=0, ddof=1)

def volatility_downside(r: np.ndarray) -> np.ndarray:
    r_neg: np.ndarray = np.minimum(r, 0.0)
    return np.sqrt((r_neg**2).mean(axis=0))

def mean_geom(r: np.ndarray) -> np.ndarray:
    return (1.0 + r).prod(axis=0) ** (1 / r.shape[0]) - 1.0

def sharpe_like(r: np.ndarray) -> np.ndarray:
    return mean_geom(r) / (volatility(r) + 1e-8)

def skewness(r: np.ndarray) -> np.ndarray:
    return ((r - mean_geom(r))**3).mean(axis=0) / (volatility(r) + 1e-8)**3

def idiosyncratic_returns(
        rr: np.ndarray,  # assets
        xx: np.ndarray,  # factors
    ) -> np.ndarray:
    # R = X @ B + E
    T = rr.shape[0]
    xx = np.c_[np.ones(T), xx]
    r, rt = rr[:T-1], rr[T-1]  # NOTE try including last day, non causal regression
    x, xt = xx[:T-1], xx[T-1]
    beta = np.linalg.solve(x.T @ x, x.T @ r)
    eps = rt - xt @ beta
    return eps

def rolling_regression(
        rr: np.ndarray,  # assets
        xx: np.ndarray,  # factors
        lookback: int,
    ) -> np.ndarray:
    T, N = rr.shape
    r_eps = np.full((T, N), fill_value=np.nan)
    for t in range(lookback-1, T):
        r = rr[t+1-lookback:t+1]
        x = xx[t+1-lookback:t+1]
        r_eps[t] = idiosyncratic_returns(r, x)
    return r_eps

def rolling_feature(
        rr: np.ndarray,  # assets
        lookback: int,
        callback: Callable[..., np.ndarray]
    ) -> np.ndarray:
    T, N = rr.shape
    feature = np.full((T, N), fill_value=np.nan)
    for t in range(lookback-1, T):
        r = rr[t+1-lookback:t+1]
        feature[t] = callback(r)
    return feature

class FeatureBuilder:
    def __init__(
            self,
            ret_d: pd.DataFrame,
            tickers: list[str],
            factors: list[str],
            lookback: int,
            first_date: str,
            final_date: str,
        ):
        rd_ = ret_d.loc[first_date:final_date]
        columns = rd_.columns
        assert np.isin(tickers, columns).all()
        assert np.isin(factors, columns).all()

        self.d0 = first_date
        self.d1 = final_date
        self.T = rd_.shape[0]
        self.U = len(tickers)
        self.K = len(factors)
        self.lookback_regression = lookback
        self.rr = rd_[tickers].values
        self.xx = rd_[factors].values
        self.ee = rolling_regression(self.rr, self.xx, lookback)
        self.timeline = rd_.index                
        self.tickers = tickers
        self.factors = factors

        self.F: int = None
        self.features:   np.ndarray = None
        self.__features: list[np.ndarray] = []
        self.names: list[str] = []
        self.lookbacks: list[int] = []
        self.callbacks: list[Callable[..., np.ndarray]] = []
        self.regressed: list[bool] = []
        self.z_scaled: list[bool] = []

    def add_feature(
            self,
            name: str,
            regress: bool,
            z_scale: bool,
            lookback: int,
            callback: Callable[..., np.ndarray],
        ) -> None:
        assert lookback > 0, lookback
        assert name not in self.names
        assert (lookback, callback) not in zip(self.lookbacks, self.callbacks)

        rr_ = self.rr if regress == False else self.ee
        feature = rolling_feature(rr_, lookback, callback)
        feature = feature if z_scale == False else standard_scale(feature)

        self.__features.append(feature)
        self.names.append(name)
        self.lookbacks.append(lookback)
        self.callbacks.append(callback)
        self.regressed.append(regress)
        self.z_scaled.append(z_scale)
    
    def consolidate(self) -> None:
        features = np.stack(self.__features, axis=2)
        T, U, F = self.T, self.U, len(self.__features)
        assert features.shape == (T, U, F)
        assert len(self.names) == F
        assert len(self.lookbacks) == F
        assert len(self.callbacks) == F
        assert len(self.regressed) == F
        assert len(self.z_scaled) == F
        self.F = F
        self.features = features

class FeatureView:
    def __init__(
            self,
            fb: FeatureBuilder,
            subset: list[str] = None,
        ):
        assert fb.F is not None
        assert fb.features is not None
        subset = subset if subset is not None else fb.tickers
        assert np.isin(subset, fb.tickers).all()
        T, _, F = fb.features.shape
        N = len(subset)
        t2i = {t: i for i, t in enumerate(fb.tickers)}
        i_N = np.array([t2i[t] for t in subset], dtype=int)
        self.T = T
        self.N = N
        self.F = F
        self.names = fb.names.copy()
        self.tickers = subset
        self.__features = fb.features[:, i_N, :]
        self.features = self.__features.copy()
        self.__mask = np.full((self.N, self.F), fill_value=False, dtype=bool)
        self.mask = self.__mask.copy()

    def apply_masking(self) -> None:       
        mask = self.mask[None, :, :].repeat(self.T, axis=0)  # [T, N, F]
        
        features = self.__features.copy()
        features[mask] = np.nan
        self.features = features

        self.mask = self.__mask.copy()

    # for given tickers:
    # exclude=True:  excludes choosen feature_names, includes rest
    # exclude=False: includes choosen feature_names, excludes rest
    def add_mask(
            self,
            tickers: list[str],
            names: list[str],
            exclude=True
        ) -> None:
        assert np.isin(tickers, self.tickers).all()
        assert np.isin(names, self.names).all()

        mask_tick = np.isin(self.tickers, tickers)
        mask_name = np.isin(self.names, names)
        
        mask = mask_tick[:,None] & mask_name[None,:]  # [N, F]
        if exclude == False:  # include, only selected tickers
            mask[mask_tick, :] = ~mask[mask_tick, :]

        self.mask |= mask
