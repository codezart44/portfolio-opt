import numpy as np
import pandas as pd
import time
from typing import Literal
from popt.alpha.modules.predictor import AlphaPredictor
from popt.alpha.modules.features import FeatureView
from popt.alpha.modules.utils import ic_score, rank_cs

class AlphaSimulator:
    def __init__(self, fv: FeatureView):
        self.fv = fv
        self.thetas: np.ndarray | None = None
        self.prd: np.ndarray | None = None
        self.ref: np.ndarray | None = None
        self.predictor: AlphaPredictor | None = None
        self.time = -1

    def run(self, predictor: AlphaPredictor, verbose=False, permute=False):
        t0 = time.time()
        fv = self.fv
        h = fv.horizon
        L = predictor.lookback
        T, N, F = fv.T, fv.N, fv.F
        thetas = np.full((T, F), fill_value=np.nan, dtype=float)
        alphas = np.full((T, N), fill_value=np.nan, dtype=float)
        gtruth = np.full((T, N), fill_value=np.nan, dtype=float)
        for t in range(h+L, T-h):
            x_trn = fv.get_x(t-h, L)       # [ t+1-h-l : t+1-h ]
            y_trn = fv.get_y(t  , L)       # [ t+1  -l : t+1   ]
            x_tst = fv.get_x(t  , 1)       # [ t       : t+1   ]
            y_tst = fv.get_y(t+h, 1)       # [ t  +h   : t+1+h ]

            if permute == True:
                index = np.argsort(np.random.rand(*y_trn.shape), axis=1)
                y_trn = np.take_along_axis(y_trn, index, axis=1)

            if np.isnan(x_trn).all(axis=2).any(): continue
            if np.isnan(x_tst).all(axis=2).any(): continue
            if np.isnan(y_trn).any(): continue
            if np.isnan(y_tst).any(): continue
            x_trn = np.nan_to_num(x_trn, nan=0.0)
            x_tst = np.nan_to_num(x_tst, nan=0.0)

            predictor.fit(x_trn, y_trn)
            thetas[t] = predictor.theta
            alphas[t] = predictor.predict(x_tst)[0]
            gtruth[t] = y_tst[0]
        
        self.thetas = thetas
        self.prd = alphas
        self.ref = gtruth
        self.time = time.time() - t0
        if verbose == True: print_simulator_results(self)

    def get_alpha(self, universe: list[str], cheat=False, rank=False) -> pd.DataFrame:
        tickers = self.fv.tickers
        timeline = self.fv.timeline
        T = timeline.shape[0]
        U = len(universe)
        i_N = np.array([universe.index(t) for t in tickers], dtype=int)
        alpha = np.full((T, U), fill_value=np.nan, dtype=float)
        alpha[:,i_N] = self.prd if cheat==False else self.ref
        alpha[:,i_N] = alpha[:,i_N] if rank==False else rank_cs(alpha[:,i_N], axis=1)
        alpha = pd.DataFrame(data=alpha, columns=universe, index=timeline)
        alpha = alpha.fillna(1.0)
        return alpha

    @property
    def ic_spearman(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return ic_score(self.prd, self.ref, method="spearman")

    @property
    def ic_pearson(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return ic_score(self.prd, self.ref, method="pearson")
    
    @property
    def timeline(self) -> pd.DatetimeIndex:
        return self.fv.timeline

def print_simulator_results(sim: AlphaSimulator) -> None:
    print(f"Backtest Runtime: {round(sim.time*1000)} ms")
    print(f"ic sprm:    {np.nanmean(sim.ic_spearman).round(4)}")
    print(f"ic prsn:    {np.nanmean(sim.ic_pearson).round(4)}")
