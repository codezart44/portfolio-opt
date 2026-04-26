import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from popt.backtest.modules.backtestdata import DataLoader, DataBuilder
from abc import ABC, abstractmethod
from typing import Sequence

class BacktestStrategy(ABC):
    def __init__(self, dl: DataLoader):
        super().__init__()
        assert isinstance(dl, DataLoader), type(dl)
        self.dl = dl
        self.name = self.__class__.__name__.lower()

    @staticmethod
    def normalize_weights(w: np.ndarray, lev: float) -> np.ndarray:
        w = w.copy()
        w_sum = w.sum()
        cap = 1.0 + lev
        if w_sum > cap + 1e-8:
            w *= cap / (w_sum + 1e-8)
        return w
    
    @abstractmethod
    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_trade_flag(self, t: int) -> bool:
        pass


class MetaStrategy(BacktestStrategy):
    def __init__(
            self,
            db: DataBuilder,
            strategies: list[BacktestStrategy],
            w_blend: np.ndarray,
        ):
        tickers = np.unique([t for s in strategies for t in s.dl.tickers]).tolist()
        super().__init__(DataLoader(db=db, tickers=tickers))
        
        S = len(strategies)
        assert w_blend.shape[0] == S, S
        assert w_blend.sum() == 1.0, w_blend.sum()

        self.strategies = strategies
        self.w_blend = w_blend
        self.S = S

        self.indicies = []
        for s in range(self.S):
            t2i = {t: i for i, t in enumerate(self.dl.tickers)}
            i_T = np.array([t2i[t] for t in self.strategies[s].dl.tickers])
            self.indicies.append(i_T)

        # self.pv = ...  # record performance over time, to compute trailing vol
    
    def get_trade_flag(self, t: int):
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, t: int, w_prev: np.ndarray):
        w = np.zeros(self.dl.N)
        for s in range(self.S):
            b = self.w_blend[s]
            i_T = self.indicies[s]
            w[i_T] += b * self.strategies[s].get_weights(t, w_prev[i_T]/b)
        return w


class Markowitz(BacktestStrategy):
    def __init__(
            self,
            dl: DataLoader,
            gamma: float,
            lev: float,
            w_max: np.ndarray,
            vc_lim: float,
        ):
        super().__init__(dl)
        assert lev >= 0.0, lev
        assert len(w_max) == dl.N, len(w_max)
        assert ((0.0 <= w_max) & (w_max <= 1.0)).all()

        self.gamma = gamma
        self.lev = lev
        self.w_max = w_max
        self.vc_lim = vc_lim / np.sqrt(252)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)

    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        assert w_prev.shape == (self.dl.N,), w_prev.shape
        
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation
        w_max = self.w_max.copy()
        w_max[~asset_mask] = 0.0

        w: np.ndarray = markowitz(
            at     = self.dl.get_alpha(t-1),
            Ft     = self.dl.get_F_cov(t-1),
            dt     = self.dl.get_d_var(t-1),
            w_prev = w_prev, 
            w_max  = w_max, 
            gamma  = self.gamma, 
            lev    = self.lev, 
            vc_lim = self.vc_lim
        )
        w = self.normalize_weights(w, self.lev)
        return w


class FixedWeights(BacktestStrategy):
    def __init__(
            self,
            dl: DataLoader,
            w_rebal: np.ndarray,
            lev: float,
            vc_lim: float = None,
        ):
        super().__init__(dl)
        assert w_rebal.sum().round(4) == 1.0, w_rebal.sum()
        assert (w_rebal >= 0).all(), w_rebal
        assert w_rebal.shape == (dl.N,), w_rebal.shape

        self.w_rebal = w_rebal
        self.lev = lev
        self.vc_lim = vc_lim / np.sqrt(252) if vc_lim is not None else None

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)

    def get_weights(self, t:int, w_prev:np.ndarray) -> np.ndarray:
        assert w_prev.shape       == (self.dl.N,), w_prev.shape
        assert self.w_rebal.shape == (self.dl.N,), self.w_rebal.shape
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation
        Ft = self.dl.get_F_cov(t-1)
        dt = self.dl.get_d_var(t-1)
        w = self.w_rebal.copy()
        
        if not asset_mask.all():
            w[~asset_mask] = 0.0
            w = w / (w.sum() + 1e-8)

        if self.vc_lim is not None:
            vol = np.linalg.norm(np.hstack(( Ft.T @ w, dt**0.5 * w )), ord=2)
            k = self.vc_lim / (vol + 1e-8)
            k = np.minimum(k, 1.0 + self.lev)
            w = k * w  # scaling for vol control

        w = self.normalize_weights(w, self.lev)
        return w


class InverseVolatility(BacktestStrategy):
    def __init__(self, dl: DataLoader):
        super().__init__(dl)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, t:int, w_prev: np.ndarray) -> np.ndarray:
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation
        Ft = self.dl.get_F_cov(t-1)
        dt = self.dl.get_d_var(t-1)
        sigma = Ft @ Ft.T + np.diag(dt)  # covariance matrix sigma
        vol = np.sqrt(np.diag(sigma))

        w = np.zeros_like(w_prev)
        vol_inv = 1.0 / (vol[asset_mask] + 1e-8)
        w[asset_mask] = vol_inv / (vol_inv.sum() + 1e-8)

        return w


class MinimumVolatility(BacktestStrategy):
    def __init__(
            self, 
            dl: DataLoader,
            w_max: np.ndarray,
            ):
        super().__init__(dl)
        assert w_max.shape == (dl.N,), w_max.shape
        assert ((0.0 <= w_max) & (w_max <= 1.0)).all()
        self.w_max = w_max

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, t:int, w_prev: np.ndarray) -> np.ndarray:
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation
        mask = asset_mask.astype(float)

        if mask.sum() == 0:
            return np.zeros(self.dl.N)
        
        Ft = self.dl.get_F_cov(t-1)
        dt = self.dl.get_d_var(t-1)
        sigma = Ft @ Ft.T + np.diag(dt)  # covariance matrix sigma

        w = cp.Variable(self.dl.N, nonneg=True)
        prob = cp.Problem(
            objective=cp.Minimize(cp.quad_form(w, sigma)),
            constraints=[
                cp.sum(w) == 1.0, 
                w <= mask * self.w_max
            ]
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if w.value is None:
            raise RuntimeError(f"Solver failed with status: {prob.status}")
        return w.value


class MaximumDiversification(BacktestStrategy):
    def __init__(self, dl: DataLoader):
        super().__init__(dl)

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, t:int, w_prev: np.ndarray) -> np.ndarray:
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation
        mask = asset_mask.astype(float)

        if mask.sum() == 0:
            return np.zeros(self.dl.N)
        
        Ft = self.dl.get_F_cov(t-1)
        dt = self.dl.get_d_var(t-1)
        sigma = Ft @ Ft.T + np.diag(dt)  # covariance matrix sigma
        vol = np.diag(sigma) ** 0.5
        
        # QP reformulation
        Rt = sigma / (np.outer(vol, vol) + 1e-8)

        y = cp.Variable(self.dl.N, nonneg=True)
        prob = cp.Problem(
            objective=cp.Minimize(cp.quad_form(y, Rt)),
            constraints=[
                cp.sum(y) == 1.0,
                y <= mask,
            ]
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if y.value is None:
            raise RuntimeError(f"Solver failed with status: {prob.status}")
        w = y.value / (vol + 1e-8)
        w[~asset_mask] = 0.0
        w /= w.sum()
        return w


class RiskParity(BacktestStrategy):
    def __init__(
            self, 
            dl: DataLoader,
            kappa: float,
            ):
        super().__init__(dl)
        self.kappa = kappa

    def get_trade_flag(self, t: int) -> bool:
        return self.dl.get_trade_flag(t)
    
    def get_weights(self, t:int, w_prev: np.ndarray) -> np.ndarray:
        asset_mask  = self.dl.get_asset_mask(t-1)
        asset_mask &= self.dl.get_asset_mask(t)  # check for discontinuation

        w = np.zeros(self.dl.N)
        N = int(asset_mask.sum())
        if N == 0: return w
        
        Ft_n = self.dl.get_F_cov(t-1)[asset_mask]
        dt_n = self.dl.get_d_var(t-1)[asset_mask]
        sigma_n = Ft_n @ Ft_n.T + np.diag(dt_n)  # covariance matrix sigma

        w_n = cp.Variable(N, pos=True)
        prob = cp.Problem(
            objective=cp.Minimize(0.5 * cp.quad_form(w_n, sigma_n) - self.kappa * cp.sum(cp.log(w_n))),
            constraints=[cp.sum(w_n) == 1.0]
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if w_n.value is None:
            raise RuntimeError(f"Solver failed with status: {prob.status}")
        w[asset_mask] = w_n.value
        return w



def asset_plot(
        dl:DataLoader,
        d0:str=None,
        d1:str=None,
        figsize=(12,3)
    ) -> None:
    timeline = dl.timeline.copy()
    d0 = d0 if d0 is not None else timeline.min()
    d1 = d1 if d1 is not None else timeline.max()
    window = (d0 <= timeline) & (timeline <= d1)
    timeline_ = timeline[window]

    ret = dl._ret.copy()
    ret[~dl._asset_mask] = 1.0
    ret_ = ret[window]

    plt.figure(figsize=figsize)
    plt.plot(timeline_, ret_.cumprod(axis=0)-1.0, label=dl.tickers)
    plt.legend()
    plt.show()


def markowitz(
        at: np.ndarray,
        Ft: np.ndarray,
        dt: np.ndarray,
        w_prev: np.ndarray,
        w_max: np.ndarray,
        gamma: float,
        lev: float,
        vc_lim: float
    ) -> np.ndarray:
    w = cp.Variable(at.shape[0])
    vol = cp.norm2(cp.hstack(( Ft.T @ w, cp.multiply(dt**0.5, w) )))
    prob = cp.Problem(
        objective=cp.Maximize(at @ w - gamma * cp.norm1(w - w_prev)),
        constraints=[
            vol <= vc_lim,         # volatility control
            0.0 <= w,              # no shorting
            w <= w_max,            # capped positions
            cp.sum(w) <= 1 + lev,  # leverage
        ],
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        raise RuntimeError(f"Solver failed with status: {prob.status}")
    return w.value
