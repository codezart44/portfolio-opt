import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from popt.backtest.modules.strategies import BacktestStrategy

class BacktestSimulator:
    def __init__(
            self,
            spread: float = 5e-4,  # 5 basis points
        ):
        self.spread: float = spread
        self.pw: np.ndarray | None = None  # portfolio weights
        self.pv: np.ndarray | None = None  # portfolio value
        self.strategy: BacktestStrategy | None = None
        self.time = -1

    def run(self, strategy: BacktestStrategy, verbose=False) -> None:
        t0 = time.time()
        dl = strategy.dl
        T, N = dl.T, dl.N  # T timesteps, N etfs
        
        portfolio_weights = np.empty((T, N+1), dtype=float)
        portfolio_value = np.empty((T, 1), dtype=float)
        v = 1.0  # running value
        w = np.ones(N) / N
        w_cash = 0.0

        portfolio_weights[0, :N] = w
        portfolio_weights[0,  N] = w_cash
        portfolio_value[0] = v

        for t in range(1, T):
            asset_mask = dl.get_asset_mask(t)  # discontinuation known beginning of day t
            liq = w[~asset_mask].sum()
            if liq > 0.0:
                w_cash += liq  # converted to cash, no turnover
                w[~asset_mask] = 0.0

            # pre market open - configuring the portfolio setup for today, t
            if strategy.get_trade_flag(t) == True:
                w_prev = w.copy()
                w = strategy.get_weights(t, w_prev)
                w_cash = 1.0 - w.sum()
                turnover = 0.5 * np.abs(w - w_prev).sum()  # pay half of full spread
                v -= self.spread * turnover * v

            # market opens - realizing returns, today t
            r, rf = dl.get_return(t), dl.get_rf(t)
            g = w @ r + w_cash * rf
            v *= g

            # weights have drifted
            w *= r / g
            w_cash *= rf / g

            portfolio_weights[t, :N] = w
            portfolio_weights[t,  N] = w_cash
            portfolio_value[t] = v
        
        self.pw = portfolio_weights
        self.pv = portfolio_value
        self.strategy = strategy
        self.time = time.time() - t0
        if verbose == True: print_simulator_results(self)

    @property
    def ann_sharpe(self) -> float:
        assert self.pv is not None
        assert self.strategy is not None
        pv = self.pv
        rp = pv[1:] / pv[:-1] - 1.0
        rf = self.strategy.dl._rf[1:].ravel() - 1.0
        return sharpe_geom(rp, rf)

    @property
    def ann_vol(self) -> float:
        assert self.pv is not None
        pv = self.pv
        rp = pv[1:] / pv[:-1] - 1.0
        return np.std(rp) * np.sqrt(252)

    @property
    def ann_ret(self) -> float:
        assert self.pv is not None
        pv = self.pv
        last = pv[-1, 0]
        return last ** (252 / (pv.shape[0])) - 1.0
    
    @property
    def tot_ret(self) -> float:
        assert self.pv is not None
        last = self.pv[-1, 0]
        return last - 1.0
    
    @property
    def max_drawdown(self) -> float:
        assert self.pv is not None
        pv = self.pv.ravel()
        upper = np.maximum.accumulate(pv)
        return np.max((upper - pv) / upper)

    @property
    def timeline(self) -> pd.DatetimeIndex:
        assert self.strategy is not None
        return self.strategy.dl.timeline

def print_simulator_results(sim: BacktestSimulator) -> None:
    print(f"Backtest Runtime: {round(sim.time*1000)} ms")
    print(f"Ann Sharpe: {sim.ann_sharpe.round(4)}")
    print(f"Ann Ret:    {sim.ann_ret.round(4)}")
    print(f"Ann Vol:    {sim.ann_vol.round(4)}")
    print(f"Max DD :    {sim.max_drawdown.round(4)}")
    print(f"Tot Ret:    {sim.tot_ret.round(4)}")

def sharpe_arit(rp: np.ndarray, rf: np.ndarray) -> float:
    return (rp - rf).mean() / rp.std(ddof=1) * np.sqrt(252)

def sharpe_geom(rp: np.ndarray, rf: np.ndarray) -> float:
    ann_ret = ((1.0+rp).prod() / (1.0+rf).prod()) ** (252 / rp.shape[0]) - 1.0
    ann_vol = rp.std(ddof=1) * np.sqrt(252)
    return ann_ret / ann_vol

def wealth_plot(sim: BacktestSimulator, figsize=(12,3)) -> None:
    dl = sim.strategy.dl
    holdings = sim.pw * (sim.pv-1.0)
    plt.figure(figsize=figsize)
    plt.stackplot(dl.timeline, holdings.T, labels=dl.tickers+["Cash"])
    plt.legend()
    plt.show()
