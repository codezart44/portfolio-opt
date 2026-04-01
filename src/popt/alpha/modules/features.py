import numpy as np
from typing import Callable

def idiosyncratic_returns(
        rr: np.ndarray,  # assets
        rx: np.ndarray,  # factors
    ) -> np.ndarray:
    # R = X @ B + E
    T = rr.shape[0]
    rx = np.c_[np.ones(T), rx]
    r, rt = rr[:T-1], rr[T-1]  # NOTE include last day, non causal regression
    x, xt = rx[:T-1], rx[T-1]
    beta = np.linalg.solve(x.T @ x, x.T @ r)
    eps = rt - xt @ beta
    return eps

def rolling_target(
        rr: np.ndarray,  # assets
        rx: np.ndarray,  # factors
        lookback: int,
    ) -> np.ndarray:
    T, N = rr.shape
    r_eps = np.full((T, N), fill_value=np.nan)
    for t in range(lookback-1, T):
        r = rr[t+1-lookback:t+1]
        x = rx[t+1-lookback:t+1]
        r_eps[t] = idiosyncratic_returns(r, x)
    return r_eps



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
