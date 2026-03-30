import numpy as np
import pandas as pd

class Riskmodel:
    def __init__(
            self,
            k: int,             # 10
            lookback: int,      # 126
            halflife_cov: int,  # 126
            halflife_vol: int,  # 42
        ):
        self.k = k
        self.lookback = lookback
        self.halflife_cov = halflife_cov
        self.halflife_vol = halflife_vol

        self.F_cov = None
        self.d_var = None
        self.assets = None
        self.timeline = None

    def fit(self, ret_d: pd.DataFrame):
        Xx = ret_d.values

        k = self.k
        Ik = np.eye(k)
        lookback = self.lookback
        halflife_cov = self.halflife_cov
        halflife_vol = self.halflife_vol
        eps = 1e-8
        tol = 1e-4
        max_iter = 10

        T, N = ret_d.shape  # T timesteps, N assets

        ewma_cov = 0.5 ** (np.arange(lookback-1, -1, -1) / halflife_cov)[:, None]
        ewma_vol = 0.5 ** (np.arange(lookback-1, -1, -1) / halflife_vol)[:, None]
        ewma_cov = ewma_cov / ewma_cov.sum()
        ewma_vol = ewma_vol / ewma_vol.sum()
        F_cov = np.full((T, N, k), np.nan)  # factor loading matrix of covariance
        d_var = np.full((T, N), np.nan)     # idiosyncratic variance of covariance

        for t in range(lookback, T):
            Xt = Xx[t-lookback:t]  # returns t-L:t-1 (inclusive)
            mask = ~np.isnan(Xt).any(axis=0)
            if mask.sum() < k+1:  # not enough assets for k factors
                continue
            
            # whiten returns
            Xt = Xt[:, mask]  # reduce to viable universe
            vol = np.sqrt(ewma_vol.T @ Xt**2)
            vol = np.maximum(vol, eps)  # avoid division by zero
            Xt = np.clip(Xt / vol, a_min=-3.0, a_max=3.0)  # avoid extreme scaling

            Mt = Xt.T @ (ewma_cov * Xt)  # second moment
            scale = np.sqrt(np.diag(Mt))[:,None]
            M_norm: np.ndarray = Mt / (scale @ scale.T)  # normalized second moment
            M_norm = 0.5 * (M_norm + M_norm.T)

            # Eigendecomp to init F and D for EM
            d,  Qt = np.linalg.eigh(M_norm)
            dk, Qk = d[-k:], Qt[:,-k:]
            Ft = Qk * np.sqrt(dk)[None,:]
            dt = np.maximum( np.diag(M_norm - Ft @ Ft.T), eps )

            # EM to regularize F and D after PCA
            for i in range(max_iter):
                F_prev = Ft
                d_prev = dt
                
                # E step
                D_inv = np.diag(1.0/dt)
                Cov_z = np.linalg.solve(Ft.T @ D_inv @ Ft + Ik, Ik)  # Cov[z|x]
                Beta = Cov_z @ Ft.T @ D_inv                          # E[z|x] = Beta @ x
                M_xz = M_norm @ Beta.T                               # E[xz']
                M_zz = Beta @ M_norm @ Beta.T + Cov_z                # E[zz']

                # M step
                Ft = M_xz @ np.linalg.solve(M_zz, Ik)
                dt = np.diag(M_norm - 2 * M_xz @ Ft.T + Ft @ M_zz @ Ft.T)
                dt = np.maximum(dt, eps)

                # Convergence
                dF = np.linalg.norm(Ft - F_prev) / np.linalg.norm(F_prev)
                dD = np.linalg.norm(dt - d_prev) / np.linalg.norm(d_prev)
                if dF < tol and dD < tol:
                    break
            
            F_cov[t, mask] = Ft * vol.T
            d_var[t, mask] = dt * vol.ravel()**2

        self.F_cov = F_cov
        self.d_var = d_var
        self.assets = ret_d.columns.to_numpy()
        self.timeline = ret_d.index.to_numpy()

    def save_to_npz(self, file_path: str) -> None:
        assert self.F_cov is not None
        assert self.d_var is not None
        assert self.assets is not None
        assert self.timeline is not None
        np.savez(
            file = file_path,
            F_cov = self.F_cov,
            d_var = self.d_var,
            assets = self.assets,
            timeline = self.timeline,
            k = self.k,
            lookback = self.lookback,
            halflife_cov = self.halflife_cov,
            halflife_vol = self.halflife_vol
        )
    
    @classmethod
    def load_from_npz(cls, file_path: str):
        with np.load(file=file_path, allow_pickle=True) as riskmodel:
            rm = cls(
                k = riskmodel["k"].item(),
                lookback = riskmodel["lookback"].item(),
                halflife_cov = riskmodel["halflife_cov"].item(),
                halflife_vol = riskmodel["halflife_vol"].item(),
            )
            rm.F_cov = riskmodel["F_cov"]
            rm.d_var = riskmodel["d_var"]
            rm.assets = riskmodel["assets"]
            rm.timeline = riskmodel["timeline"]
        return rm
