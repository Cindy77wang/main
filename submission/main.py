from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------------------------------------------------------
# Frozen configuration. Every constant is fixed in advance (Rules 1, 2, 18).
# ----------------------------------------------------------------------------
SEED = 42
TARGET = "ret_exc_lead1m"

# Sample design
EARLIEST_OOS_YEAR = 1975      # walk-forward starts here, giving 15 years of
                              # out-of-sample history before the 1990 test start
MIN_TRAIN_MONTHS = 36         # still runs on the 123-month validation panel
GAP_MONTHS = 1                # extra month between training data and use

# Target
TARGET_CLIP = 3.0

# Ridge
RIDGE_GRID = tuple(float(v) for v in 10.0 ** np.arange(-3.0, 2.01, 0.5))
VAL_MONTHS = 60

# Gradient-boosted trees
GBM_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.03,
    max_leaf_nodes=127,
    l2_regularization=1.0,
    max_bins=255,
    max_features=0.5,
    early_stopping=False,
)
GBM_LEAF_FRAC = 0.001         # min leaf = 0.1% of training rows if that is larger ...
GBM_MIN_LEAF = 2000           # ... than 2,000. A smaller floor let early, small samples overfit:
                              # the 1975 validation chose the fewest trees on the grid.
GBM_MAX_ROWS = 1_500_000      # seeded row subsample to bound run time
GBM_MAX_ITER = 600
GBM_ITER_STEP = 50
GBM_DEFAULT_ITER = 300
GBM_TUNE_EVERY = 5
GBM_BAGS = 2                  # boosted ensembles averaged, each on its own seeded row subsample
GBM_BAG_FRAC = 0.7            # share of training rows in each bag (when GBM_BAGS > 1)

# Feed-forward network: NN3 of Gu, Kelly and Xiu (2020), seed ensemble, NumPy only
NN_HIDDEN = (32, 16, 8)
NN_SEEDS = 3
NN_MAX_ROWS = 500_000         # seeded row subsample per seed to bound run time
NN_MAX_EPOCHS = 15
NN_DEFAULT_EPOCHS = 5
NN_BATCH = 4096
NN_LR = 1e-3
NN_L2 = 1e-5

# Risk model
N_CLUSTERS = 13               # number of themes in Jensen, Kelly, Pedersen (2023)
CLUSTER_WINDOW = 60           # months of characteristics used for clustering
RISK_WINDOW_DAYS = 756
HL_VOL = 84
HL_CORR = 504
MIN_FACTOR_DAYS = 60
MIN_IDIO_OBS = 63
IDIO_FLOOR = 0.1              # idio variance floored at 10% of cross-sectional median
DAYS_PER_MONTH = 21
MIN_INDUSTRY = 20             # 2-digit SIC industries smaller than this share an "other" factor
MIN_LW_OBS = 250              # daily observations a stock needs for the sample-covariance book
BOOK_WEIGHT_LW = 0.5          # weight of the sample-covariance book; 1 - this on the factor book

# Portfolio and configuration selection
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)       # weight on the machine-learning (trees + NN) forecast
K_GRID = (0.0, 0.25, 1.0, 4.0, np.inf)     # ridge on Sigma, multiples of avg variance
DEFAULT_COMBO = (2, 2)                     # a = 0.5, k = 1 until history exists
SELECT_WINDOW = 120
SELECT_MIN = 24
TOP_K = 3
VOL_TARGET = 0.10 / np.sqrt(12.0)
RV_DAYS = 126                 # volatility timing: trailing days of the book's synthetic returns
RV_HL = 21                    # EWMA half-life (days) of the realized-variance forecast
RV_WEIGHT = 0.5               # weight on realized variance; 1 - this on the risk-model variance
MIN_NAMES = 30


def _log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def _flag(s: pd.Series) -> np.ndarray:
    """ctff_test as boolean. Bool, 0/1 and text encodings all occur."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t"]).to_numpy()


def _rank_block(block: np.ndarray) -> np.ndarray:
    """Within-month ranks mapped to (-0.5, 0.5). NaN stays NaN."""
    r = rankdata(block, axis=0, nan_policy="omit")
    cnt = np.sum(np.isfinite(block), axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r - 0.5) / cnt - 0.5
    return out.astype(np.float32)


def _prepare(chars: pd.DataFrame, features: pd.DataFrame) -> dict:
    reserved = {"id", "eom", TARGET, "ctff_test"}
    feats, seen = [], set()
    for f in sorted(features["features"].astype(str).tolist()):  # canonical order (Rule 18)
        if f in seen or f in reserved or f not in chars.columns:
            continue
        seen.add(f)
        if pd.api.types.is_numeric_dtype(chars[f]):
            feats.append(f)

    df = pd.DataFrame({
        "eom_out": chars["eom"].to_numpy(),
        "_id": pd.to_numeric(chars["id"]).to_numpy().astype(np.int64),
        "_eom": pd.to_datetime(chars["eom"]).to_numpy().astype("datetime64[ns]"),
        "_ret": pd.to_numeric(chars[TARGET], errors="coerce").to_numpy(dtype=np.float64),
        "_test": _flag(chars["ctff_test"]) if "ctff_test" in chars.columns
        else np.zeros(len(chars), dtype=bool),
    })
    order = np.lexsort((df["_id"].to_numpy(), df["_eom"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    keep = ~df.duplicated(subset=["_eom", "_id"], keep="first").to_numpy()
    rows = order[keep]
    df = df.loc[keep].reset_index(drop=True)

    eom = pd.to_datetime(df["_eom"])
    mi = (eom.dt.year.to_numpy() * 12 + eom.dt.month.to_numpy() - 1).astype(np.int64)
    months, starts = np.unique(mi, return_index=True)
    ends = np.append(starts[1:], len(df))
    month_dt = df["_eom"].to_numpy()[starts].astype("datetime64[ns]")

    raw = chars[feats].to_numpy(dtype=np.float32)[rows]
    raw[~np.isfinite(raw)] = np.nan
    x0 = np.empty_like(raw)
    ret = df["_ret"].to_numpy()
    y = np.full(len(df), np.nan)
    for lo, hi in zip(starts, ends):
        x0[lo:hi] = np.nan_to_num(_rank_block(raw[lo:hi].astype(np.float64)), nan=0.0)
        r = ret[lo:hi]
        ok = np.isfinite(r)
        if ok.sum() >= 3:
            sd = r[ok].std()
            if sd > 0:
                yy = np.full(hi - lo, np.nan)
                yy[ok] = np.clip((r[ok] - r[ok].mean()) / sd, -TARGET_CLIP, TARGET_CLIP)
                y[lo:hi] = yy
    # 2-digit SIC industry of record at month s, -1 if unknown (risk model only)
    ind = None
    if "sic" in chars.columns:
        sic = pd.to_numeric(chars["sic"], errors="coerce").to_numpy(dtype=np.float64)[rows]
        ind = np.where(np.isfinite(sic), np.floor(sic / 100.0), -1).astype(np.int64)

    p = len(feats)
    G = np.zeros((len(months), p, p))
    c = np.zeros((len(months), p))
    nobs = np.zeros(len(months), dtype=np.int64)
    for m, (lo, hi) in enumerate(zip(starts, ends)):
        ok = np.isfinite(y[lo:hi])
        if ok.sum() == 0:
            continue
        xm = x0[lo:hi][ok].astype(np.float64)
        G[m] = xm.T @ xm
        c[m] = xm.T @ y[lo:hi][ok]
        nobs[m] = ok.sum()

    return dict(df=df, feats=feats, months=months, starts=starts, ends=ends,
                month_dt=month_dt, x_raw=raw, x0=x0, ind=ind, y=y, ret=ret,
                G=G, c=c, nobs=nobs)


def _daily_matrix(daily_ret: pd.DataFrame, panel_ids: np.ndarray, start: np.datetime64):
    """Dense (days x stocks) matrix of daily excess returns, NaN where absent."""
    rcol = "ret_exc" if "ret_exc" in daily_ret.columns else "ret"
    ids = pd.to_numeric(daily_ret["id"], errors="coerce").to_numpy(dtype=np.float64)
    dates = pd.to_datetime(daily_ret["date"]).to_numpy().astype("datetime64[ns]")
    rets = pd.to_numeric(daily_ret[rcol], errors="coerce").to_numpy(dtype=np.float32)
    uids = np.unique(panel_ids)
    good = np.isfinite(ids) & (dates >= start) & np.isfinite(rets)
    ids_i = ids[good].astype(np.int64)
    d_k, r_k = dates[good], rets[good]
    inpanel = np.isin(ids_i, uids)
    ids_i, d_k, r_k = ids_i[inpanel], d_k[inpanel], r_k[inpanel]
    udates, di = np.unique(d_k, return_inverse=True)
    ci = np.searchsorted(uids, ids_i)
    M = np.full((len(udates), len(uids)), np.nan, dtype=np.float32)
    M[di, ci] = r_k
    return udates, uids, M


# ----------------------------------------------------------------------------
# Forecasting models
# ----------------------------------------------------------------------------
def _monthly_ic(P: np.ndarray, y: np.ndarray, bounds) -> np.ndarray:
    """Mean over months of the cross-sectional Pearson correlation, per column of P."""
    acc, n = np.zeros(P.shape[1]), 0
    for lo, hi in bounds:
        yy = y[lo:hi]
        ok = np.isfinite(yy)
        if ok.sum() < 10:
            continue
        pp = P[lo:hi][ok] - P[lo:hi][ok].mean(axis=0)
        yc = yy[ok] - yy[ok].mean()
        den = np.sqrt((pp ** 2).sum(axis=0) * (yc ** 2).sum())
        acc += np.where(den > 0, (pp * yc[:, None]).sum(axis=0) / np.where(den > 0, den, 1), 0.0)
        n += 1
    return acc / max(n, 1)


def _ridge_betas(G: np.ndarray, c: np.ndarray, grid) -> np.ndarray:
    p = G.shape[0]
    scale = max(np.trace(G) / p, 1e-12)
    ev, V = eigh(G)
    proj = V.T @ c
    lam = np.asarray(grid)[None, :] * scale
    return V @ (proj[:, None] / (np.maximum(ev, 0)[:, None] + lam))


def _fit_ridge(D: dict, train_m: np.ndarray, val_m: np.ndarray):
    """Pick the penalty on validation months, then refit on all training months."""
    lam = RIDGE_GRID[len(RIDGE_GRID) // 2]
    if len(val_m) >= 6 and len(train_m) > len(val_m):
        fit_m = np.setdiff1d(train_m, val_m)
        B = _ridge_betas(D["G"][fit_m].sum(0), D["c"][fit_m].sum(0), RIDGE_GRID)
        lo0 = D["starts"][val_m[0]]
        hi0 = D["ends"][val_m[-1]]
        P = D["x0"][lo0:hi0].astype(np.float64) @ B
        bounds = [(D["starts"][m] - lo0, D["ends"][m] - lo0) for m in val_m]
        ic = _monthly_ic(P, D["y"][lo0:hi0], bounds)
        lam = RIDGE_GRID[int(np.argmax(ic))]
    beta = _ridge_betas(D["G"][train_m].sum(0), D["c"][train_m].sum(0), (lam,))[:, 0]
    return beta, lam


def _rows_with_target(D: dict, month_ids: np.ndarray) -> np.ndarray:
    idx = np.concatenate([np.arange(D["starts"][m], D["ends"][m]) for m in month_ids])
    return idx[np.isfinite(D["y"][idx])]


def _gbm(n_iter: int, n_rows: int, seed: int) -> HistGradientBoostingRegressor:
    leaf = max(GBM_MIN_LEAF, int(round(GBM_LEAF_FRAC * n_rows)))
    return HistGradientBoostingRegressor(max_iter=int(n_iter), min_samples_leaf=leaf,
                                         random_state=seed, **GBM_PARAMS)


def _subsample(idx: np.ndarray, seed: int) -> np.ndarray:
    if len(idx) <= GBM_MAX_ROWS:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=GBM_MAX_ROWS, replace=False))


def _observed_cols(X: np.ndarray) -> np.ndarray:
    """Columns with at least one finite value. sklearn's histogram binning fails on a column
    that is entirely missing, which happens for characteristics that start late in the sample."""
    return np.flatnonzero(np.isfinite(X).any(axis=0))


def _fit_gbm(D: dict, rows: np.ndarray, n_iter: int, seed: int) -> list:
    """GBM_BAGS boosted ensembles, each fit on its own seeded row subsample."""
    models = []
    for b in range(GBM_BAGS):
        n = len(rows) if GBM_BAGS == 1 else int(round(GBM_BAG_FRAC * len(rows)))
        n = min(n, GBM_MAX_ROWS)
        rng = np.random.default_rng(seed + 1_000 * b)
        idx = rows if n >= len(rows) else np.sort(rng.choice(rows, size=n, replace=False))
        X = D["x_raw"][idx]
        cols = _observed_cols(X)
        if len(cols) < X.shape[1]:
            X = X[:, cols]
        model = _gbm(n_iter, len(idx), seed + b)
        model.fit(X, D["y"][idx])
        models.append((cols, model))
    return models


def _predict_gbm(models: list, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X[:, c] if len(c) < X.shape[1] else X) for c, m in models], axis=0)


def _tune_gbm(D: dict, train_m: np.ndarray, val_m: np.ndarray, seed: int) -> int:
    """Number of boosting rounds that maximises validation IC."""
    if len(val_m) < 6 or len(train_m) <= len(val_m):
        return GBM_DEFAULT_ITER
    fit_m = np.setdiff1d(train_m, val_m)
    idx = _subsample(_rows_with_target(D, fit_m), seed)
    X = D["x_raw"][idx]
    cols = _observed_cols(X)
    model = _gbm(GBM_MAX_ITER, len(idx), seed)
    model.fit(X[:, cols], D["y"][idx])
    lo0, hi0 = D["starts"][val_m[0]], D["ends"][val_m[-1]]
    grid = list(range(GBM_ITER_STEP, GBM_MAX_ITER + 1, GBM_ITER_STEP))
    preds = []
    for it, p in enumerate(model.staged_predict(D["x_raw"][lo0:hi0][:, cols]), start=1):
        if it in grid:
            preds.append(p)
    grid = grid[:len(preds)]
    P = np.column_stack(preds)
    bounds = [(D["starts"][m] - lo0, D["ends"][m] - lo0) for m in val_m]
    ic = _monthly_ic(P, D["y"][lo0:hi0], bounds)
    return grid[int(np.argmax(ic))]


def _nn_forward(params: list, X: np.ndarray) -> list:
    hs, h = [X], X
    n_layers = len(params) // 2
    for k in range(n_layers):
        h = h @ params[2 * k] + params[2 * k + 1]
        if k < n_layers - 1:
            h = np.maximum(h, 0.0)
        hs.append(h)
    return hs


def _nn_predict(nets: list, X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    return np.mean([_nn_forward(p, X)[-1][:, 0] for p in nets], axis=0).astype(np.float64)


def _train_nn(X: np.ndarray, y: np.ndarray, epochs: int, rng, val=None):
    """NN3 trained by Adam on squared error. With val = (Xv, yv, bounds), also returns the
    validation IC after every epoch."""
    sizes = [X.shape[1], *NN_HIDDEN, 1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
        params.append(np.zeros(b, dtype=np.float32))
    m1 = [np.zeros_like(q) for q in params]
    m2 = [np.zeros_like(q) for q in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    l2 = np.float32(NN_L2)
    y = y.astype(np.float32)
    n, ics = len(X), []
    for _ in range(int(epochs)):
        perm = rng.permutation(n)
        for s in range(0, n, NN_BATCH):
            idx = perm[s:s + NN_BATCH]
            hs = _nn_forward(params, X[idx])
            g = (2.0 / len(idx)) * (hs[-1][:, 0] - y[idx])[:, None]
            grads = [None] * len(params)
            for k in range(len(params) // 2 - 1, -1, -1):
                W = params[2 * k]
                grads[2 * k] = hs[k].T @ g + l2 * W
                grads[2 * k + 1] = g.sum(axis=0)
                if k > 0:
                    g = (g @ W.T) * (hs[k] > 0)
            t += 1
            c1, c2 = 1.0 - b1 ** t, 1.0 - b2 ** t
            for q, gq, a1, a2 in zip(params, grads, m1, m2):
                a1 *= b1
                a1 += (1 - b1) * gq
                a2 *= b2
                a2 += (1 - b2) * gq * gq
                q -= (NN_LR * (a1 / c1) / (np.sqrt(a2 / c2) + eps)).astype(np.float32)
        if val is not None:
            Xv, yv, bounds = val
            ics.append(_monthly_ic(_nn_forward(params, Xv)[-1], yv, bounds)[0])
    return params, ics


def _nn_rows(D: dict, month_ids: np.ndarray, rng) -> np.ndarray:
    idx = _rows_with_target(D, month_ids)
    if len(idx) > NN_MAX_ROWS:
        idx = np.sort(rng.choice(idx, size=NN_MAX_ROWS, replace=False))
    return idx


def _tune_nn(D: dict, train_m: np.ndarray, val_m: np.ndarray, seed: int) -> int:
    """Number of epochs that maximises the validation IC of the seed ensemble."""
    if len(val_m) < 6 or len(train_m) <= len(val_m):
        return NN_DEFAULT_EPOCHS
    fit_m = np.setdiff1d(train_m, val_m)
    lo0, hi0 = D["starts"][val_m[0]], D["ends"][val_m[-1]]
    bounds = [(D["starts"][m] - lo0, D["ends"][m] - lo0) for m in val_m]
    val = (D["x0"][lo0:hi0], D["y"][lo0:hi0], bounds)
    curve = np.zeros(NN_MAX_EPOCHS)
    for s in range(NN_SEEDS):
        rng = np.random.default_rng(seed + 100 * s)
        idx = _nn_rows(D, fit_m, rng)
        _, ics = _train_nn(D["x0"][idx], D["y"][idx], NN_MAX_EPOCHS, rng, val)
        curve += np.asarray(ics)
    return int(np.argmax(curve)) + 1


def _fit_nn(D: dict, train_m: np.ndarray, epochs: int, seed: int) -> list:
    nets = []
    for s in range(NN_SEEDS):
        rng = np.random.default_rng(seed + 100 * s)
        idx = _nn_rows(D, train_m, rng)
        nets.append(_train_nn(D["x0"][idx], D["y"][idx], epochs, rng)[0])
    return nets


# ----------------------------------------------------------------------------
# Risk model
# ----------------------------------------------------------------------------
def _cluster_loadings(Gw: np.ndarray):
    """Hierarchical clustering of characteristics; returns (p x K) loading matrix."""
    p = Gw.shape[0]
    d = np.diag(Gw).copy()
    labels_full = np.zeros(p, dtype=np.int64)
    if p == 0 or d.max() <= 0:
        return np.zeros((p, 0)), labels_full
    idx = np.flatnonzero(d > 1e-10 * d.max())
    s = np.sqrt(d[idx])
    C = np.clip(Gw[np.ix_(idx, idx)] / np.outer(s, s), -1.0, 1.0)
    if len(idx) == 1:
        labels = np.array([1])
    else:
        dist = 1.0 - np.abs(C)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(np.maximum(dist, 0.0), checks=False), method="average")
        labels = fcluster(Z, t=min(N_CLUSTERS, len(idx)), criterion="maxclust")
    uniq = np.unique(labels)
    L = np.zeros((p, len(uniq)))
    for j, lab in enumerate(uniq):
        mem = np.flatnonzero(labels == lab)
        if len(mem) == 1:
            sign = np.ones(1)
        else:
            _, vecs = eigh(C[np.ix_(mem, mem)])
            v = vecs[:, -1]
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            sign = np.where(v >= 0, 1.0, -1.0)
        L[idx[mem], j] = sign / len(mem)
        labels_full[idx[mem]] = j + 1
    return L, labels_full


def _industry_set(ind_block: np.ndarray) -> np.ndarray:
    """Industries with at least MIN_INDUSTRY stocks get their own factor; the rest share one."""
    codes, counts = np.unique(ind_block[ind_block >= 0], return_counts=True)
    return codes[counts >= MIN_INDUSTRY]


def _exposures(x0_block: np.ndarray, L: np.ndarray, ind_block=None, inds=None) -> np.ndarray:
    """Industry dummies (or a market column if no industries) plus standardised cluster composites."""
    C = x0_block.astype(np.float64) @ L
    sd = C.std(axis=0)
    Z = (C - C.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    if inds is None or len(inds) == 0:
        return np.column_stack([np.ones(len(C)), Z])
    pos = np.searchsorted(inds, ind_block)
    hit = (pos < len(inds)) & (inds[np.minimum(pos, len(inds) - 1)] == ind_block)
    dummies = np.zeros((len(C), len(inds) + 1))
    dummies[np.arange(len(C)), np.where(hit, pos, len(inds))] = 1.0   # last column = "other"
    return np.column_stack([dummies, Z])


def _ewma_w(age: np.ndarray, hl: float) -> np.ndarray:
    return np.power(0.5, age / hl)


class _FactorRisk:
    """Sigma = E F E' + diag(d0); inverted with the Woodbury identity in O(n k^2)."""

    def __init__(self, E, F, d0):
        self.E, self.F, self.d0 = E, F, d0
        self.avg_var = (np.einsum("ij,jk,ik->i", E, F, E).sum() + d0.sum()) / len(d0)

    def mul(self, V):
        return self.E @ (self.F @ (self.E.T @ V)) + self.d0[:, None] * V

    def solve(self, V, lam):
        Di = 1.0 / (self.d0 + lam)
        DiV, DiE = V * Di[:, None], self.E * Di[:, None]
        k = self.E.shape[1]
        inner = np.linalg.solve(np.eye(k) + self.F @ (self.E.T @ DiE), self.F @ (self.E.T @ DiV))
        return DiV - DiE @ inner

    def var(self, W):
        X = self.E.T @ W
        return np.einsum("ij,ik,kj->j", X, self.F, X) + (self.d0[:, None] * W ** 2).sum(axis=0)


class _SampleRisk:
    """Dense Ledoit-Wolf shrunk sample covariance; solved by Cholesky."""

    def __init__(self, S):
        self.S = S
        self.avg_var = float(np.trace(S)) / S.shape[0]

    def mul(self, V):
        return self.S @ V

    def solve(self, V, lam):
        A = self.S.copy()
        A[np.diag_indices(A.shape[0])] += lam
        return cho_solve(cho_factor(A, lower=True, check_finite=False), V, check_finite=False)

    def var(self, W):
        return np.einsum("ij,ij->j", W, self.S @ W)


def _scale(W, risk):
    """Scale each portfolio (columns after the first axis) to the volatility target."""
    shp = W.shape
    Wf = W.reshape(shp[0], -1)
    if risk is not None:
        s = VOL_TARGET / np.sqrt(np.maximum(risk.var(Wf), 1e-20))
    else:
        g = np.abs(Wf).sum(axis=0)
        s = 2.0 / np.where(g > 0, g, 1.0)
    return (Wf * s[None, :]).reshape(shp)


def _portfolios(risk, MU):
    """Weights for every (blend, ridge) configuration under one risk model, shape (n, nA, nK).

    Maximises mu'w - (g/2) w' Sigma_k w subject to 1'w = 0 and beta'w = 0, where beta is
    the risk model's beta on the equal-weighted market. With risk=None only the dollar
    constraint is imposed and weights are scaled to unit gross on each side.
    """
    n, nA = MU.shape
    W = np.zeros((n, nA, len(K_GRID)))
    one = np.ones(n)
    if risk is not None:
        s1 = risk.mul(one[:, None])[:, 0]
        A = np.column_stack([one, n * s1 / (one @ s1)])
    else:
        A = one[:, None]
    for ik, kap in enumerate(K_GRID):
        if np.isinf(kap) or risk is None:
            use = A if np.linalg.cond(A.T @ A) < 1e8 else A[:, :1]
            gam = np.linalg.lstsq(use, MU, rcond=None)[0]
            Wk = MU - use @ gam
        else:
            lam = max(kap, 1e-8) * risk.avg_var
            S = risk.solve(np.column_stack([MU, A]), lam)
            SM, SA, Au = S[:, :nA], S[:, nA:], A
            M2 = Au.T @ SA
            if np.linalg.cond(M2) > 1e8:
                SA, Au = SA[:, :1], A[:, :1]
                M2 = Au.T @ SA
            gam = np.linalg.solve(M2, Au.T @ SM)
            Wk = SM - SA @ gam
        W[:, :, ik] = Wk
    return _scale(W, risk)


def _round_sig(w: np.ndarray, digits: int = 7) -> np.ndarray:
    """Round to `digits` significant digits so the CSV stays well under 50 MB (Rule 12)."""
    w = np.asarray(w, dtype=np.float64)
    out = np.zeros_like(w)
    nz = np.isfinite(w) & (w != 0)
    p = digits - 1 - np.floor(np.log10(np.abs(w[nz])))
    out[nz] = np.round(w[nz] * 10.0 ** p) / 10.0 ** p
    return out


def _zs(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def _select(hist_ret: np.ndarray):
    """Top-K configurations by trailing out-of-sample Sharpe ratio."""
    if hist_ret.shape[0] < SELECT_MIN:
        return [DEFAULT_COMBO]
    H = hist_ret[-SELECT_WINDOW:]
    R = H.reshape(H.shape[0], -1)
    sd = R.std(axis=0)
    sr = np.where(sd > 0, R.mean(axis=0) / np.where(sd > 0, sd, 1.0), -np.inf)
    order = np.argsort(-sr, kind="stable")[:TOP_K]
    return [tuple(int(v) for v in np.unravel_index(o, H.shape[1:])) for o in order]


# ----------------------------------------------------------------------------
# Risk estimates for one month
# ----------------------------------------------------------------------------
def _month_risk(FR, RES, age, cols, k, n):
    """Factor covariance F (monthly) and idiosyncratic variances d0 (monthly)."""
    vrow = np.isfinite(FR).all(axis=1)
    if vrow.sum() < MIN_FACTOR_DAYS:
        return None
    fv, ag = FR[vrow], age[vrow]
    wv, wc = _ewma_w(ag, HL_VOL), _ewma_w(ag, HL_CORR)
    var = (wv[:, None] * fv ** 2).sum(0) / wv.sum()
    Cc = (fv * wc[:, None]).T @ fv / wc.sum()
    sd = np.sqrt(np.maximum(np.diag(Cc), 0.0))
    nz = sd > 0
    corr = np.eye(k)
    corr[np.ix_(nz, nz)] = Cc[np.ix_(nz, nz)] / np.outer(sd[nz], sd[nz])
    vs = np.sqrt(var)
    F = np.outer(vs, vs) * corr * DAYS_PER_MONTH

    Rw = RES[:, cols].astype(np.float64)
    obs = np.isfinite(Rw)
    wv_all = _ewma_w(age, HL_VOL)
    num = wv_all @ np.where(obs, Rw ** 2, 0.0)
    den = wv_all @ obs
    cnt = obs.sum(axis=0)
    good = (cnt >= MIN_IDIO_OBS) & (den > 0)
    if good.sum() < MIN_NAMES:
        return None
    iv = np.full(n, np.nan)
    iv[good] = num[good] / den[good]
    med = np.median(iv[good])
    iv = np.where(np.isfinite(iv), iv, med)
    iv = np.maximum(iv, IDIO_FLOOR * med)
    return F, iv * DAYS_PER_MONTH


def _sample_risk(M, end, cols):
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity of the trailing daily covariance.

    Returns (eligible mask over the month's stocks, _SampleRisk in monthly units) or None.
    Missing days count as zero returns after the eligibility screen, then columns are demeaned.
    """
    lo = max(0, end - RISK_WINDOW_DAYS)
    X = M[lo:end][:, cols].astype(np.float64)
    obs = np.isfinite(X)
    enough = obs.sum(axis=0) >= MIN_LW_OBS
    if enough.sum() < MIN_NAMES:
        return None
    X = np.where(obs[:, enough], X[:, enough], 0.0)
    X -= X.mean(axis=0, keepdims=True)
    T, n = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / n
    norm2 = float(np.square(S).sum())
    d2 = norm2 - 2.0 * mu * np.trace(S) + n * mu ** 2
    b2 = float(np.square(np.square(X).sum(axis=1)).sum()) / T ** 2 - norm2 / T
    delta = min(max(b2, 0.0), d2) / d2 if d2 > 0 else 1.0
    S *= 1.0 - delta
    S[np.diag_indices(n)] += delta * mu
    return enough, _SampleRisk(S * DAYS_PER_MONTH)


def _realized_var(M, end, cols, w):
    """EWMA variance (monthly units) of the book's synthetic daily returns: today's weights
    applied to the trailing RV_DAYS of daily returns dated at or before the formation date."""
    lo = max(0, end - RV_DAYS)
    if end - lo < DAYS_PER_MONTH:
        return np.nan
    r = np.nan_to_num(M[lo:end][:, cols].astype(np.float64), nan=0.0) @ w
    wt = _ewma_w(np.arange(len(r))[::-1].astype(np.float64), RV_HL)
    return float(wt @ r ** 2 / wt.sum()) * DAYS_PER_MONTH


# ----------------------------------------------------------------------------
# Walk-forward engine
# ----------------------------------------------------------------------------
def build_portfolio(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame,
                    keep_diagnostics: bool = False):
    """Weights for every out-of-sample month, plus optional diagnostics."""
    np.random.seed(SEED)
    t0 = time.perf_counter()
    D = _prepare(chars, features)
    df, months, starts, ends = D["df"], D["months"], D["starts"], D["ends"]
    ids_all, eom_all, test_all = df["_id"].to_numpy(), df["eom_out"].to_numpy(), df["_test"].to_numpy()
    _log(f"[prep] {len(df):,} rows, {len(D['feats'])} features, {len(months)} months "
         f"({time.perf_counter() - t0:.0f}s)")

    first_y = int(np.ceil((months[0] + MIN_TRAIN_MONTHS + GAP_MONTHS + 1) / 12.0))
    oos_start = max(EARLIEST_OOS_YEAR, first_y)
    years = list(range(oos_start, int(months[-1] // 12) + 1))
    if not years:
        _log("[warn] sample too short for walk-forward; returning equal weights")
        n_per = pd.Series(df["_eom"]).map(pd.Series(df["_eom"]).value_counts()).to_numpy()
        out = pd.DataFrame({"id": ids_all, "eom": eom_all, "w": 1.0 / n_per, "_test": test_all})
        return out, None

    udates, uids, M = _daily_matrix(daily_ret, ids_all, np.datetime64(f"{oos_start - 4}-01-01", "ns"))
    colmap = np.searchsorted(uids, ids_all)
    _log(f"[daily] {M.shape[0]:,} days x {M.shape[1]:,} ids ({time.perf_counter() - t0:.0f}s)")

    nA, nK = len(A_GRID), len(K_GRID)
    combo_hist, hist_mi = [], []
    out_frames, mon_rec, yr_rec, beta_rec = [], [], [], []
    prev_ids = prev_w = None
    gbm_iter = GBM_DEFAULT_ITER
    nn_epochs = NN_DEFAULT_EPOCHS
    last_clusters = None

    for Y in years:
        ty = time.perf_counter()
        cut = Y * 12 - 1 - GAP_MONTHS
        train_m = np.flatnonzero((months <= cut) & (D["nobs"] > 0))
        year_m = np.flatnonzero(months // 12 == Y)
        if len(year_m) == 0 or len(train_m) == 0:
            continue
        val_m = train_m[-min(VAL_MONTHS, max(12, len(train_m) // 4)):]

        # 1. Ridge forecast
        beta, lam = _fit_ridge(D, train_m, val_m)

        # 2. Tree forecast
        tuned = (Y - oos_start) % GBM_TUNE_EVERY == 0
        if tuned:
            gbm_iter = _tune_gbm(D, train_m, val_m, SEED + Y)
        idx = _rows_with_target(D, train_m)
        gbm = _fit_gbm(D, idx, gbm_iter, SEED + 10_000 + Y)

        # 2b. Neural-network forecast (seed ensemble)
        if tuned:
            nn_epochs = _tune_nn(D, train_m, val_m, SEED + 20_000 + Y)
        nets = _fit_nn(D, train_m, nn_epochs, SEED + 30_000 + Y)

        # 3. Risk factors from clustered characteristics
        L, labels = _cluster_loadings(D["G"][train_m[-CLUSTER_WINDOW:]].sum(0))
        inds = None
        if D["ind"] is not None:
            last = train_m[-1]
            inds = _industry_set(D["ind"][starts[last]:ends[last]])
        k = L.shape[1] + (len(inds) + 1 if inds is not None and len(inds) else 1)

        # 4. Configuration choice from out-of-sample returns realised before year Y
        H = np.asarray(combo_hist)[np.asarray(hist_mi) <= cut] if combo_hist else np.zeros((0, nA, nK))
        chosen = _select(H)

        # 5. Daily cross-sectional factor regressions for this year's risk windows
        end_first = int(np.searchsorted(udates, D["month_dt"][year_m[0]], side="right"))
        span_hi = int(np.searchsorted(udates, D["month_dt"][year_m[-1]], side="right"))
        span_lo = max(0, end_first - RISK_WINDOW_DAYS)
        nspan = max(span_hi - span_lo, 0)
        FR = np.full((nspan, k), np.nan)
        RES = np.full((nspan, M.shape[1]), np.nan, dtype=np.float32)
        cache: dict = {}

        def expo(m: int) -> np.ndarray:
            if m not in cache:
                ib = D["ind"][starts[m]:ends[m]] if D["ind"] is not None else None
                cache[m] = _exposures(D["x0"][starts[m]:ends[m]], L, ib, inds)
            return cache[m]

        if nspan > 0:
            # returns on day d are explained by exposures from the last month-end before d
            em = np.searchsorted(D["month_dt"], udates[span_lo:span_hi], side="left") - 1
            for g in np.unique(em):
                if g < 0:
                    continue
                js = np.flatnonzero(em == g)
                E = expo(int(g))
                cols = colmap[starts[g]:ends[g]]
                R = M[span_lo + js][:, cols]
                for jj, j in enumerate(js):
                    r = R[jj].astype(np.float64)
                    ok = np.isfinite(r)
                    if ok.sum() < max(3 * k, MIN_NAMES):
                        continue
                    Eo = E[ok]
                    f = np.linalg.solve(Eo.T @ Eo + 1e-10 * np.eye(k), Eo.T @ r[ok])
                    FR[j] = f
                    RES[j, cols[ok]] = r[ok] - Eo @ f

        # 6. Monthly portfolios
        for m in year_m:
            lo, hi = starts[m], ends[m]
            n = hi - lo
            if n < MIN_NAMES:
                continue
            zr = _zs(D["x0"][lo:hi].astype(np.float64) @ beta)
            zb = _zs(_predict_gbm(gbm, D["x_raw"][lo:hi]))
            zn = _zs(_nn_predict(nets, D["x0"][lo:hi]))
            zg = _zs(zb + zn)
            MU = np.column_stack([a * zg + (1 - a) * zr for a in A_GRID])

            E = expo(int(m))
            end = int(np.searchsorted(udates, D["month_dt"][m], side="right"))
            wlo = max(span_lo, end - RISK_WINDOW_DAYS)
            fac = None
            if end > wlo:
                age = (end - 1 - np.arange(wlo, end)).astype(np.float64)
                fac = _month_risk(FR[wlo - span_lo:end - span_lo], RES[wlo - span_lo:end - span_lo],
                                  age, colmap[lo:hi], k, n)
            fm_ok = fac is not None
            risk_f = _FactorRisk(E, *fac) if fm_ok else None

            # Book A: factor risk model. Book B: Ledoit-Wolf sample covariance.
            W_f = _portfolios(risk_f, MU)
            W_s, lw = None, (_sample_risk(M, end, colmap[lo:hi]) if fm_ok else None)
            if lw is not None:
                eligible, risk_s = lw
                W_s = np.zeros_like(W_f)
                W_s[eligible] = _portfolios(risk_s, MU[eligible])
                W = _scale((1 - BOOK_WEIGHT_LW) * W_f + BOOK_WEIGHT_LW * W_s, risk_f)
            else:
                W = W_f
            rr = np.nan_to_num(D["ret"][lo:hi], nan=0.0)
            combo_ret = np.einsum("i,iak->ak", rr, W)
            combo_hist.append(combo_ret)
            hist_mi.append(int(months[m]))

            w = np.mean([W[:, ia, ik] for ia, ik in chosen], axis=0)
            w = _scale(w[:, None], risk_f)[:, 0]
            w = np.where(np.isfinite(w), w, 0.0)
            # volatility timing: blend the risk model's variance with the book's realized one
            rv = _realized_var(M, end, colmap[lo:hi], w) if fm_ok else np.nan
            if np.isfinite(rv) and rv > 0:
                w = w * (VOL_TARGET / np.sqrt((1 - RV_WEIGHT) * VOL_TARGET ** 2 + RV_WEIGHT * rv))
            ids_t = ids_all[lo:hi]
            out_frames.append(pd.DataFrame({"id": ids_t, "eom": eom_all[lo:hi], "w": w,
                                            "_test": test_all[lo:hi]}))

            if keep_diagnostics:
                ics = _monthly_ic(np.column_stack([zr, zb, zn, MU[:, 2]]), D["y"][lo:hi], [(0, n)])
                traded = np.nan
                if prev_ids is not None:
                    u = np.union1d(prev_ids, ids_t)
                    a_, b_ = np.zeros(len(u)), np.zeros(len(u))
                    a_[np.searchsorted(u, ids_t)] = w
                    b_[np.searchsorted(u, prev_ids)] = prev_w
                    traded = float(np.abs(a_ - b_).sum())
                rec = dict(eom=pd.Timestamp(D["month_dt"][m]), year=Y, n=n,
                           ic_ridge=ics[0], ic_gbm=ics[1], ic_nn=ics[2], ic_ens=ics[3],
                           rv_book=rv,
                           ret=float(w @ rr), gross=float(np.abs(w).sum()), net=float(w.sum()),
                           traded=traded, n_long=int((w > 0).sum()), n_short=int((w < 0).sum()),
                           ew_mkt=float(np.nanmean(D["ret"][lo:hi])), fm_ok=fm_ok,
                           test=bool(test_all[lo:hi].any()),
                           r_book_factor=float(rr @ W_f[:, 2, 2]),
                           r_book_sample=float(rr @ W_s[:, 2, 2]) if W_s is not None else np.nan,
                           n_lw=int(lw[0].sum()) if lw is not None else 0, n_factors=k)
                for ia in range(nA):
                    for ik in range(nK):
                        rec[f"r_a{ia}_k{ik}"] = float(combo_ret[ia, ik])
                mon_rec.append(rec)
                prev_ids, prev_w = ids_t, w

        if keep_diagnostics:
            yr_rec.append(dict(year=Y, ridge_lambda=lam, gbm_iter=gbm_iter, gbm_tuned=tuned,
                               nn_epochs=nn_epochs,
                               n_train_rows=int(len(idx)), n_clusters=L.shape[1],
                               n_industries=int(len(inds)) if inds is not None else 0,
                               chosen=";".join(f"{A_GRID[a]}|{K_GRID[kk]}" for a, kk in chosen),
                               secs=time.perf_counter() - ty))
            beta_rec.append(pd.Series(beta, index=D["feats"], name=Y))
            last_clusters = (labels, L)
        _log(f"[{Y}] train months {len(train_m)}, ridge lambda {lam:g}, trees {gbm_iter}"
             f"{' (tuned)' if tuned else ''}, nn epochs {nn_epochs}, clusters {L.shape[1]}, "
             f"industries {len(inds) if inds is not None else 0}, "
             f"configs {[(A_GRID[a], K_GRID[kk]) for a, kk in chosen]}, "
             f"{time.perf_counter() - ty:.0f}s")

    out = pd.concat(out_frames, ignore_index=True)
    out["id"] = out["id"].astype(np.int64)
    out["w"] = _round_sig(out["w"].to_numpy())
    _log(f"[done] {len(out):,} weights in {(time.perf_counter() - t0) / 60:.1f} min")

    diag = None
    if keep_diagnostics:
        clus = None
        if last_clusters is not None:
            labels, L = last_clusters
            clus = pd.DataFrame({"feature": D["feats"], "cluster": labels,
                                 "sign": np.sign(L.sum(axis=1)).astype(int)})
        diag = dict(monthly=pd.DataFrame(mon_rec), yearly=pd.DataFrame(yr_rec),
                    ridge_betas=pd.DataFrame(beta_rec), clusters=clus)
    return out, diag


def _restrict_to_test(weights: pd.DataFrame) -> pd.DataFrame:
    """ctff_test rows only; if the flag is absent or never true, every row."""
    if weights["_test"].any():
        weights = weights[weights["_test"]]
    return weights[["id", "eom", "w"]].reset_index(drop=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Monthly portfolio weights with columns id, eom, w (Rules 11-12)."""
    np.random.seed(SEED)
    weights, _ = build_portfolio(chars, features, daily_ret, keep_diagnostics=False)
    out = _restrict_to_test(weights)
    assert len(out) > 0 and out.notna().all().all()
    return out


if __name__ == "__main__":
    # Local use only; the CTF harness imports main() and never runs this block.
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    chars = pd.read_parquet(f"{folder}/ctff_chars.parquet")
    features = pd.read_parquet(f"{folder}/ctff_features.parquet")
    daily_ret = pd.read_parquet(f"{folder}/ctff_daily_ret.parquet")
    pf = main(chars, features, daily_ret)
    pf.to_csv("output.csv", index=False)
    print(pf.head())
