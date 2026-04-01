import numpy as np
import pandas as pd
from scipy import linalg
from typing import Literal, Sequence

def train_shared(
        X_trn: np.ndarray,
        y_trn: np.ndarray,
        w_root: np.ndarray,
        G_f: np.ndarray, # Block Diagonal Interleaved Gamma*Identity
        T: int,
        N: int,
        F: int,
    ) -> np.ndarray:
    TN, NF = T*N, N*F
    assert X_trn.shape == (T, NF), X_trn.shape
    assert y_trn.shape == (T, N), y_trn.shape
    assert w_root.shape == (T,), w_root.shape
    assert G_f.shape == (F, F), G_f.shape

    # EWMA over residuals
    Xw = X_trn * w_root[:,None]       # [T, NF] * [T, 1] -> [T, NF]
    yw = y_trn * w_root[:,None]       # [T, N]  * [T, 1] -> [T, N]

    Xw = Xw.reshape(TN, F)
    yw = yw.reshape(TN)

    Aw = Xw.T @ Xw + G_f
    bw = Xw.T @ yw

    Cw, low = linalg.cho_factor(Aw, lower=True, overwrite_a=True, check_finite=False)  # A = LL^T
    theta = linalg.cho_solve((Cw, low), bw, overwrite_b=True, check_finite=False)
    return theta


def train(
        X_trn: np.ndarray,
        y_trn: np.ndarray,
        w_root: np.ndarray,
        L_nf: np.ndarray, # Block Diagonal Interleaved Laplacian Graph
        G_nf: np.ndarray, # Block Diagonal Interleaved Gamma*Identity
        T: int,
        N: int,
        F: int,
    ) -> np.ndarray:
    NF = N*F
    assert X_trn.shape == (T, NF), X_trn.shape
    assert y_trn.shape == (T, N), y_trn.shape
    assert w_root.shape == (T,), w_root.shape
    assert L_nf.shape == (NF, NF), L_nf.shape
    assert G_nf.shape == (NF, NF), G_nf.shape

    # EWMA over residuals
    Xw = X_trn * w_root[:,None]       # [T, NF] * [T, 1] -> [T, NF]
    yw = y_trn * w_root[:,None]       # [T, N]  * [T, 1] -> [T, N]

    Xw = Xw.reshape(T, N, F).transpose(1, 0, 2) # [T, NF] -> [N, T, F]
    XwT = Xw.transpose(0, 2, 1)
    Xw2 = XwT @ Xw  # [N, F, F]
    assert Xw2.shape == (N, F, F), Xw2.shape

    # Aw @ B = bw
    # Xw.T @ Xw + gamma*I_nf + L_nf
    Aw = G_nf + L_nf  # [NF, NF]
    index = np.arange(N)
    Aw = Aw.reshape(N,F,N,F)
    Aw[index,:,index,:] += Xw2  # Add blocks diagonally
    Aw = Aw.reshape(NF, NF)

    bw = (XwT @ yw.T[:,:,None]).reshape(NF)  # [NF,] !stacked X.T @ y and then flattened
    assert Aw.shape == (NF, NF), Aw.shape
    assert bw.shape == (NF,), bw.shape

    Cw, low = linalg.cho_factor(Aw, lower=True, overwrite_a=True, check_finite=False)  # A = LL^T
    Theta = linalg.cho_solve((Cw, low), bw, overwrite_b=True, check_finite=False)
    Theta = Theta.reshape(N, F)
    return Theta


def alpha_predictor(
        X: pd.DataFrame,
        y: pd.DataFrame,
        lap_graph: np.ndarray,  # includes edge weights
        gamma: float = 0.0,
        halflife: int = 126,
        lookback: int = 504,    # 2 yrs lookback for training
        horizon: int = 21,      # 21 days ahead, 1 trading month
        pooled: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """ Backtest for alpha predictor - Eval different strategies """
    # T train samples, N assets, F features, L iterations
    timeloss = lookback+horizon
    T = lookback
    _, N = y.shape
    F = X.shape[1] // N
    L = y.shape[0] - timeloss

    # Precomputed matricies
    I_n = np.eye(N)
    I_f = np.eye(F)
    # XXX This is contigent on the const feature being the first feature
    I_f[0, 0] = 0.0  # NOTE no regularization for bias term, thus 0.0
    L_nf = np.kron(lap_graph, I_f) # kronecker product [N, N] (x) [F, F] -> [NF, NF]
    G_nf = gamma * np.kron(I_n, I_f)
    G_f = gamma * I_f

    beta: int = (0.5) ** (1.0 / halflife)
    w_root = np.array([beta ** (T-i-1) for i in range(T)])
    w_root = np.sqrt(w_root)
    predictions = np.empty((L, N), dtype=np.float32)
    thetas = np.empty((L, N, F), dtype=np.float32)

    for t in range(timeloss, y.shape[0]):              # y[t-1]: t:t+H-1  <- The prediction we seek
        X_trn = X.iloc[t-timeloss:t-horizon].values    # [T, NF]      r: t-L-H:t-H-1
        y_trn = y.iloc[t-timeloss:t-horizon].values    # [T, N]       r: t-L:t-1
        X_tst = X.iloc[t].values                       # [NF,]        r: t-H+1:t
        y_tst = y.iloc[t].values                       # [N,]         r: t+1:t+H

        mu = X_trn.mean(axis=0)
        sd = X_trn.std(axis=0, ddof=1)
        mu[0::F] = 0.0  # avoid scaling bias term
        sd[0::F] = 1.0
        X_trn = (X_trn - mu) / sd
        X_tst = (X_tst - mu) / sd

        match pooled:
            case True:
                Theta = train_shared(X_trn, y_trn, w_root, G_f, T, N, F)
                y_prd = X_tst.reshape(N, F) @ Theta
            case False:
                Theta = train(X_trn, y_trn, w_root, L_nf, G_nf, T, N, F)  # Training            
                y_prd = (X_tst.reshape(N, F) * Theta).sum(axis=1) # Testing - sum([N, F] * [N, F]) over F
            case _: 
                raise ValueError("invalid pooled argument.")

        assert y_prd.shape == y_tst.shape, (y_prd.shape, y_tst.shape)  # [N,]
        predictions[t-timeloss] = y_prd
        thetas[t-timeloss] = Theta
    
    df_ref = y.iloc[timeloss:]
    df_prd = pd.DataFrame(
        data=predictions,
        columns=df_ref.columns,
        index=df_ref.index
    )

    assert df_ref.shape == df_prd.shape, (df_ref.shape, df_prd.shape)
    return df_prd, df_ref, thetas



def laplacian_graph(
        file_path: str,
        assets: list[str],
        attributes: list[str],
        w_edges: Sequence[float] | None = None,
    ) -> np.ndarray:  

    df_cat = pd.read_csv(file_path, index_col="Ticker").drop(columns="Name")
    df_cat = df_cat.loc[assets][attributes]
    ohe = pd.get_dummies(df_cat).astype(float).values

    _, C = ohe.shape
    w_edges = np.ones(ohe.shape[1]) if w_edges == None else w_edges
    assert w_edges.shape == (C,), (w_edges.shape, C)
    ohe = ohe * np.sqrt(w_edges[None, :])
    Wm = ohe @ ohe.T
    Dm = np.diag(Wm.sum(axis=1))
    L_graph = Dm - Wm

    return L_graph


# def preprocess_alpha_data(
#         data_ret: pd.DataFrame,
#         data_mac: pd.DataFrame,
#         assets: list[str],
#         features_ret: list[str],
#         features_mac: list[str],
#         target: str,
#     ) -> tuple[pd.DataFrame, list[str]]:
#     data_ret: pd.DataFrame = data_ret.loc[:, pd.IndexSlice[assets, features_ret+[target]]].copy()  # filter by selection
#     has_nans = ~data_ret.isna().any(axis=0).groupby(level=0).any()
#     assets_remaining = has_nans.index[has_nans]
#     data_ret = data_ret[assets_remaining]  # filter away etfs with features containing nans
#     assert data_ret.isna().any().any() == False

#     data_mac = data_mac.reindex(data_ret.index)
#     assert data_mac.isna().any().any() == False
#     md_bc = pd.concat(objs=[data_mac[features_mac]]*len(assets_remaining), keys=assets_remaining, axis=1)
#     assert md_bc.isna().any().any() == False

#     const = pd.DataFrame(1.0, columns=pd.MultiIndex.from_product([assets_remaining, ["Const"]]), index=data_ret.index)
#     data = pd.concat([const, data_ret, md_bc], axis=1)[assets_remaining]
#     assert data.isna().any().any() == False

#     return data, assets_remaining

# def xy_split(
#         data: pd.DataFrame,
#         target: str,
#         features_ret: list[str],
#         horizon: int,
#     ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     assert data.isna().any().any() == False
#     ydata = data.xs(target, axis=1, level=1).shift(-horizon).dropna().copy()   # For T1M at position t : t+1, ... , t+21
#     to_drop = [target] if target not in features_ret else []
#     xdata = data.drop(columns=to_drop, level=1).reindex(ydata.index).copy()    # For T1M at position t : t-20, ... , t
#     assert xdata.shape[0] == ydata.shape[0], (xdata.shape, ydata.shape)
#     return xdata, ydata
