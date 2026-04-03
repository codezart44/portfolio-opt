import numpy as np
import pandas as pd
import time
from typing import Literal
from .predictor import AlphaPredictor
from .features import FeatureView

class AlphaSimulator:
    def __init__(self, fv: FeatureView):
        self.fv = fv
        self.thetas: np.ndarray | None = None
        self.prd: np.ndarray | None = None
        self.ref: np.ndarray | None = None
        self.predictor: AlphaPredictor | None = None
        self.time = -1

    def run(self, predictor: AlphaPredictor, verbose=False):
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



def rank(y: np.ndarray, axis=1) -> np.ndarray:
    return np.argsort(np.argsort(y, axis=axis), axis=axis)

def ic_score(
        a1: np.ndarray, 
        a2: np.ndarray, 
        method: Literal["spearman", "pearson"] = "spearman"
    ) -> np.ndarray:
    assert a1.shape == a2.shape
    T = a1.shape[0]
    ic = np.full((T,), fill_value=np.nan, dtype=float)
    notna = ~np.isnan(a1).any(axis=1) & ~np.isnan(a2).any(axis=1)
    match method:
        case "spearman":
            a1_ = rank(a1[notna])
            a2_ = rank(a2[notna])
        case "pearson":
            a1_ = a1[notna]
            a2_ = a2[notna]
        case _:
            raise ValueError
    a1_ = a1_ - a1_.mean(axis=1, keepdims=True)
    a2_ = a2_ - a2_.mean(axis=1, keepdims=True)
    
    cov = (a1_ * a2_).sum(axis=1)
    std = np.sqrt((a1_**2).sum(axis=1) * (a2_**2).sum(axis=1)) + 1e-8
    ic[notna] = cov / std
    return ic
