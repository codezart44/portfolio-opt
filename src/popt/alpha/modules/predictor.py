import numpy as np
from abc import ABC, abstractmethod
from popt.alpha.modules.utils import signed_square, tail_mask

class AlphaPredictor(ABC):
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = lookback
        self.theta = None
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def fit(self, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, xt: np.ndarray) -> np.ndarray:
        pass

class RidgeRanker(AlphaPredictor):
    def __init__(
            self,
            lookback: int,
            halflife: int,
            gamma: float,
            ):
        super().__init__(lookback)
        self.gamma = gamma
        self.lookback = lookback
        self.halflife = halflife

        w_ewma = 0.5 ** (np.arange(lookback) / halflife)
        w_ewma = (w_ewma[::-1] / w_ewma.sum()) ** 0.5
        self.w_ewma = w_ewma

    def fit(self, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
        T, N, F = xt.shape
        TN = T*N
        assert yt.shape == (T, N), yt.shape
        
        xw = xt * self.w_ewma[:,None,None]
        yw = yt * self.w_ewma[:,None]

        yw = signed_square(yw) # NOTE
        
        # yw *= tail_mask(yw, n_keep=3)

        # wt = 1.0 + 1.0 * np.abs(yt) ** 2   # (T, N)
        # sw = np.sqrt(wt)
        # yw *= sw
        # xw *= sw[:,:,None]

        xw = xw.reshape(TN, F)
        yw = yw.reshape(TN,)

        # NOTE dont regularize the constant term!! I_f[0, 0] = 0.0
        theta = np.linalg.solve(xw.T @ xw + self.gamma * np.eye(F), xw.T @ yw)
        self.theta = theta
    
    def predict(self, xt: np.ndarray) -> np.ndarray:
        _, _, F = xt.shape
        assert self.theta is not None
        assert self.theta.shape == (F,)
        pt = xt @ self.theta
        return pt
