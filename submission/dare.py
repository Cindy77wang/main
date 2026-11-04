"""
DARE: Disagreement-Adaptive Risk-Aware Ensemble
===============================================

Core idea
  DARE uses the disagreement between two very different return forecasters as a signal
  of forecast reliability. When a linear and a non-linear model agree on a stock, its
  forecast is robust to the modelling choice; when they disagree, at least one of them is
  wrong. DARE learns from past out-of-sample errors how much to trust each model at each
  level of disagreement, and sizes positions accordingly.

Forecasts
  Two learners are refit every January on all data available at that time: a pooled ridge
  regression on within-month characteristic ranks, and gradient-boosted trees
  (scikit-learn HistGradientBoosting) on raw characteristics. Both forecasts are
  standardised across stocks, giving z_tree and z_ridge.

Disagreement-adaptive weighting
  Each month, stocks are split into terciles of disagreement |z_tree - z_ridge|. Each
  January, within each tercile, realised returns are regressed on the two forecasts using
  the past ten years of matured out-of-sample forecasts; the weights are shrunk half-way
  towards the pooled weights and floored at zero. The forecast for a stock is
  b_tree(g) * z_tree + b_ridge(g) * z_ridge for its tercile g, so the data decide whether
  the trees, the ridge or neither deserve trust where the models disagree, and stocks on
  which neither model has been reliable get small forecasts.

Risk
  Two covariance models: a factor model (2-digit SIC industries plus 13 characteristic
  clusters with exponentially weighted covariances) and a Ledoit-Wolf shrunk covariance of
  daily returns. Each candidate portfolio maximises expected return with predicted risk at
  most the 10% target under both models, subject to zero net investment and zero beta.

Allocation and trading
  Twelve candidate portfolios (the disagreement-adaptive forecast, a 50/50 blend and the
  trees alone, each under four risk ridges) are weighted by exp(eta * trailing Sharpe
  ratio). Holdings move a fraction tau of the way towards this target each month;
  (eta, tau) is chosen each January by trailing Sharpe ratio net of an assumed cost of
  10 bp per dollar traded.

Timing and validation
  A month's returns are used only after all of them are realised (eom_ret, never before
  the next month-end). Risk clusters use the characteristics of every stock known before
  the year. No setting is tuned on fewer than 36 validation or 60 fitting months, and
  every choice uses past data only. The walk-forward starts in 1975, or as early as
  possible for samples that end before then.

Source (note added by the CTF administrators):
  This model is derived from the CTF entry "Disagreement-Adaptive Risk-Aware Ensemble"
  by Cindy Wang (2026), published at
  jkpfactors.com/ctf/models/disagreement-adaptive-risk-aware-ensemble-wmz88vxvcq

CTF Admin Modifications (2026-10-08):
--------------------------------------
1. Added the source note above. The code is otherwise unchanged.
   Reason: a derived entry names the entry it builds on.

2. Added a requirements.txt that pins scikit-learn to 1.8.0.
   Reason: the submission had no dependency file, so the run would use
   scikit-learn 1.9.1. That version stops with "window shape cannot be
   larger than input array shape" when a characteristic is missing for
   every training row, which happens in the early training windows of the
   full dataset. Versions 1.6 to 1.8 handle that case. The version the
   authors used is not known.

"""

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
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)       # weight on the tree forecast
K_GRID = (0.0, 0.25, 1.0, 4.0, np.inf)     # ridge on Sigma, multiples of avg variance
DEFAULT_COMBO = (2, 2)                     # a = 0.5, k = 1 until history exists
SELECT_WINDOW = 120
SELECT_MIN = 36                            # months of matured history before any selection
MIN_VAL_MONTHS = 36                        # never tune on fewer validation months ...
MIN_FIT_MONTHS = 60                        # ... or fewer fitting months; use defaults instead
TOP_K = 3
VOL_TARGET = 0.10 / np.sqrt(12.0)
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
    reserved = {"id", "eom", "eom_ret", TARGET, "ctff_test"}
    feats, seen = [], set()
    for f in features["features"].astype(str).tolist():
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

    # Label maturity: a row's target is known once its return is realised (eom_ret), and never
    # before the next month-end. A month's labels are usable only when all of them are known.
    nxt = (pd.to_datetime(df["_eom"]) + pd.offsets.MonthEnd(1)).to_numpy().astype("datetime64[ns]")
    if "eom_ret" in chars.columns:
        er = pd.to_datetime(chars["eom_ret"], errors="coerce").to_numpy().astype("datetime64[ns]")[rows]
        mat = np.where(np.isnat(er), nxt, np.maximum(er, nxt))
    else:
        mat = nxt
    month_mature = np.array([mat[lo:hi].max() for lo, hi in zip(starts, ends)], dtype="datetime64[ns]")

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
                G=G, c=c, nobs=nobs, month_mature=month_mature)


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
    if len(val_m) >= MIN_VAL_MONTHS and len(train_m) - len(val_m) >= MIN_FIT_MONTHS:
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


def _tune_gbm(D: dict, train_m: np.ndarray, val_m: np.ndarray, seed: int) -> int:
    """Number of boosting rounds that maximises validation IC."""
    if len(val_m) < MIN_VAL_MONTHS or len(train_m) - len(val_m) < MIN_FIT_MONTHS:
        return GBM_DEFAULT_ITER
    fit_m = np.setdiff1d(train_m, val_m)
    idx = _subsample(_rows_with_target(D, fit_m), seed)
    model = _gbm(GBM_MAX_ITER, len(idx), seed)
    model.fit(D["x_raw"][idx], D["y"][idx])
    lo0, hi0 = D["starts"][val_m[0]], D["ends"][val_m[-1]]
    grid = list(range(GBM_ITER_STEP, GBM_MAX_ITER + 1, GBM_ITER_STEP))
    preds = []
    for it, p in enumerate(model.staged_predict(D["x_raw"][lo0:hi0]), start=1):
        if it in grid:
            preds.append(p)
    grid = grid[:len(preds)]
    P = np.column_stack(preds)
    bounds = [(D["starts"][m] - lo0, D["ends"][m] - lo0) for m in val_m]
    ic = _monthly_ic(P, D["y"][lo0:hi0], bounds)
    return grid[int(np.argmax(ic))]


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


def _char_gram(D: dict, month_ids) -> np.ndarray:
    """Characteristic cross-products over every stock in the given months, labelled or not."""
    p = D["x0"].shape[1]
    G = np.zeros((p, p))
    for m in month_ids:
        xm = D["x0"][D["starts"][m]:D["ends"][m]].astype(np.float64)
        G += xm.T @ xm
    return G


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


# ----------------------------------------------------------------------------
# Walk-forward engine
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Version 3: reliability-aware forecasts, joint risk control, smooth allocation, costs
# ----------------------------------------------------------------------------
F_NAMES = ("learned", "blend", "trees")       # forecast candidates
K3_GRID = (0.0, 0.25, 1.0, np.inf)            # risk ridge; 4 was never selected in 1978-1989
OMEGA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)      # weight on the factor covariance in the joint problem
STACK_GROUPS = 3                              # disagreement terciles
STACK_WINDOW = 120                            # months of matured out-of-sample forecasts
STACK_MIN = 36                                # before this, the learned forecast is the 50/50 blend
STACK_SHRINK = 1.0                            # group weights shrunk half-way to the pooled weights
ETA_GRID = (1.0, 3.0, 10.0)                   # softmax temperature on trailing Sharpe ratios
TAU_GRID = (1.0, 0.5, 0.25)                   # trading speed towards the target portfolio
COST = 0.001                                  # assumed trading cost: 10 bp per dollar traded
DEFAULT_PATH = (1, 1)                         # eta = 3, tau = 0.5 until history exists


def _dis_groups(zg: np.ndarray, zr: np.ndarray) -> np.ndarray:
    """Within-month disagreement terciles of |z_tree - z_ridge| (0 = agree most)."""
    d = np.abs(zg - zr)
    r = (rankdata(d) - 0.5) / len(d)
    return np.minimum((r * STACK_GROUPS).astype(np.int64), STACK_GROUPS - 1)


def _stack_fit(zg: np.ndarray, zr: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Weights on (z_tree, z_ridge) by disagreement group, learned from realised outcomes.

    Each group's least-squares weights are shrunk towards the pooled weights (STACK_SHRINK)
    and floored at zero, so a group in which neither model has predicted returns gets a
    small forecast: 'both unreliable'.
    """
    X = np.column_stack([zg, zr])
    n = len(y)
    pooled = np.linalg.solve(X.T @ X + 1e-8 * n * np.eye(2), X.T @ y)
    B = np.tile(pooled, (STACK_GROUPS, 1))
    for k in range(STACK_GROUPS):
        s = g == k
        nk = int(s.sum())
        if nk < 50:
            continue
        lam = STACK_SHRINK * nk
        B[k] = np.linalg.solve(X[s].T @ X[s] + lam * np.eye(2), X[s].T @ y[s] + lam * pooled)
    return np.maximum(B, 0.0)


class _JointRisk:
    """Factor covariance (A) and Ledoit-Wolf covariance (B) on a common universe."""

    def __init__(self, risk_f, lw, n):
        if lw is not None:
            self.el, rs = lw
            SB = rs.S
        else:
            self.el, SB = np.ones(n, dtype=bool), None
        E = risk_f.E[self.el]
        SA = E @ risk_f.F @ E.T
        SA[np.diag_indices(len(SA))] += risk_f.d0[self.el]
        self.SA, self.SB = SA, (SB if SB is not None else SA)
        ne = len(SA)
        one = np.ones(ne)
        bA = ne * (SA @ one) / (one @ SA @ one)
        bB = ne * (self.SB @ one) / (one @ self.SB @ one)
        self.beta = 0.5 * (bA + bB)                 # one beta, applied to the final weights
        self.avg = 0.5 * (np.trace(SA) + np.trace(self.SB)) / ne

    def var_max(self, W):
        vA = np.einsum("ij,ij->j", W, self.SA @ W)
        vB = np.einsum("ij,ij->j", W, self.SB @ W)
        return np.maximum(vA, vB)

    def scale(self, W):
        """Scale columns so that the larger of the two predicted variances hits the target."""
        v = self.var_max(W)
        return W * (VOL_TARGET / np.sqrt(np.maximum(v, 1e-20)))[None, :]


def _joint_portfolios(jr: _JointRisk, MU: np.ndarray):
    """Robust mean-variance weights for every (forecast, kappa), plus the omega chosen.

    For each kappa and omega the problem is solved under omega*Sigma_A + (1-omega)*Sigma_B
    (+ ridge), subject to 1'w = 0 and beta'w = 0. Each solution is scaled so its risk is at
    target under BOTH models; the omega with the highest expected return mu'w is kept, which
    approximates the portfolio that maximises expected return subject to both risk limits.
    """
    ne, nF = MU.shape
    one = np.ones(ne)
    A = np.column_stack([one, jr.beta])
    if np.linalg.cond(A.T @ A) > 1e8:
        A = one[:, None]
    W = np.zeros((ne, nF, len(K3_GRID)))
    omega = np.full((nF, len(K3_GRID)), np.nan)
    extremes = {}
    for ik, kap in enumerate(K3_GRID):
        if np.isinf(kap):
            gam = np.linalg.lstsq(A, MU, rcond=None)[0]
            W[:, :, ik] = jr.scale(MU - A @ gam)
            continue
        best_val = np.full(nF, -np.inf)
        for om in OMEGA_GRID:
            S = om * jr.SA + (1.0 - om) * jr.SB
            S[np.diag_indices(ne)] += max(kap, 1e-8) * jr.avg
            cf = cho_factor(S, lower=True, check_finite=False)
            SM = cho_solve(cf, MU, check_finite=False)
            SA_ = cho_solve(cf, A, check_finite=False)
            gam = np.linalg.solve(A.T @ SA_, A.T @ SM)
            Wo = jr.scale(SM - SA_ @ gam)
            val = np.einsum("ij,ij->j", MU, Wo)
            if ik == 2 and om in (0.0, 1.0):
                extremes[om] = Wo[:, 0].copy()          # for the 'averaged books' ablation
            better = val > best_val
            W[:, better, ik] = Wo[:, better]
            omega[better, ik] = om
            best_val = np.where(better, val, best_val)
    return W, omega, extremes


def _softmax_weights(H: np.ndarray, eta: float) -> np.ndarray:
    """Configuration weights proportional to exp(eta * trailing Sharpe ratio)."""
    R = H[-SELECT_WINDOW:]
    sd = R.std(axis=0)
    sr = np.where(sd > 0, R.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(12.0)
    z = eta * (sr - sr.max())
    w = np.exp(z)
    return w / w.sum()


def _trade_amount(prev, ids_t, w_new, prev_aligned):
    """Dollars traded: changes in held names plus full exits from names that left."""
    traded = np.abs(w_new - prev_aligned).sum()
    if prev is not None:
        pids, pw = prev
        gone = ~np.isin(pids, ids_t)
        traded += np.abs(pw[gone]).sum()
    return float(traded)


def build_portfolio(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame,
                    keep_diagnostics: bool = False):
    """Weights for every out-of-sample month, plus optional diagnostics."""
    np.random.seed(SEED)
    t0 = time.perf_counter()
    D = _prepare(chars, features)
    df, months, starts, ends = D["df"], D["months"], D["starts"], D["ends"]
    ids_all, eom_all, test_all = df["_id"].to_numpy(), df["eom_out"].to_numpy(), df["_test"].to_numpy()
    mature = D["month_mature"]
    _log(f"[prep] {len(df):,} rows, {len(D['feats'])} features, {len(months)} months "
         f"({time.perf_counter() - t0:.0f}s)")

    first_y = int(np.ceil((months[0] + MIN_TRAIN_MONTHS + GAP_MONTHS + 1) / 12.0))
    # The 1975 floor only bounds run time on long samples; a sample that ends before 1975
    # (e.g. a 123-month validation panel) starts its walk-forward as early as it can.
    last_y = int(months[-1] // 12)
    oos_start = max(EARLIEST_OOS_YEAR, first_y) if last_y >= EARLIEST_OOS_YEAR else first_y
    years = list(range(oos_start, int(months[-1] // 12) + 1))
    if not years:
        _log("[warn] sample too short for walk-forward; returning equal weights")
        n_per = pd.Series(df["_eom"]).map(pd.Series(df["_eom"]).value_counts()).to_numpy()
        out = pd.DataFrame({"id": ids_all, "eom": eom_all, "w": 1.0 / n_per, "_test": test_all})
        return out, None

    udates, uids, M = _daily_matrix(daily_ret, ids_all, np.datetime64(f"{oos_start - 4}-01-01", "ns"))
    colmap = np.searchsorted(uids, ids_all)
    _log(f"[daily] {M.shape[0]:,} days x {M.shape[1]:,} ids ({time.perf_counter() - t0:.0f}s)")

    nF, nK, nE, nT = len(F_NAMES), len(K3_GRID), len(ETA_GRID), len(TAU_GRID)
    zr_hist = np.full(len(df), np.nan)
    zg_hist = np.full(len(df), np.nan)
    cfg_hist, path_net_hist, hist_m = [], [], []
    paths = {}                       # (eta, tau) -> last (ids, weights)
    final_prev = None
    out_frames, mon_rec, yr_rec, beta_rec = [], [], [], []
    gbm_iter = GBM_DEFAULT_ITER
    last_clusters = None
    chosen_path = DEFAULT_PATH

    for Y in years:
        ty = time.perf_counter()
        year_m = np.flatnonzero(months // 12 == Y)
        if len(year_m) == 0:
            continue
        t_first = D["month_dt"][year_m[0]]
        # Fix 1: only months whose labels were all realised before the first formation date
        train_m = np.flatnonzero((mature < t_first) & (D["nobs"] > 0))
        if len(train_m) == 0:
            continue
        val_m = train_m[-min(VAL_MONTHS, max(MIN_VAL_MONTHS, len(train_m) // 4)):]

        beta, lam = _fit_ridge(D, train_m, val_m)
        tuned = (Y - oos_start) % GBM_TUNE_EVERY == 0
        if tuned:
            gbm_iter = _tune_gbm(D, train_m, val_m, SEED + Y)
        idx = _subsample(_rows_with_target(D, train_m), SEED + 10_000 + Y)
        gbm = _gbm(gbm_iter, len(idx), SEED + Y)
        gbm.fit(D["x_raw"][idx], D["y"][idx])

        # Fix 3: clusters from characteristics known before the year, labelled or not
        known_m = np.flatnonzero(D["month_dt"] < t_first)
        L, labels = _cluster_loadings(_char_gram(D, known_m[-CLUSTER_WINDOW:]))
        inds = None
        if D["ind"] is not None and len(known_m):
            last = known_m[-1]
            inds = _industry_set(D["ind"][starts[last]:ends[last]])
        k = L.shape[1] + (len(inds) + 1 if inds is not None and len(inds) else 1)

        # Matured out-of-sample history (Fix 1 applies here too)
        hm = np.asarray(hist_m)
        ok_hist = (mature[hm] < t_first) if len(hm) else np.zeros(0, dtype=bool)
        H_cfg = np.asarray(cfg_hist)[ok_hist] if len(hm) else np.zeros((0, nF * nK))
        H_path = np.asarray(path_net_hist)[ok_hist] if len(hm) else np.zeros((0, nE * nT))

        # Change 1: learn when disagreement matters, from matured out-of-sample forecasts
        stack_B = np.full((STACK_GROUPS, 2), 0.5)
        stack_months = hm[ok_hist][-STACK_WINDOW:] if len(hm) else np.zeros(0, dtype=np.int64)
        if len(stack_months) >= STACK_MIN:
            rows = np.concatenate([np.arange(starts[m], ends[m]) for m in stack_months])
            good = np.isfinite(D["y"][rows]) & np.isfinite(zg_hist[rows]) & np.isfinite(zr_hist[rows])
            rows = rows[good]
            g_hist = np.concatenate([_dis_groups(zg_hist[starts[m]:ends[m]], zr_hist[starts[m]:ends[m]])
                                     for m in stack_months])[good]
            stack_B = _stack_fit(zg_hist[rows], zr_hist[rows], D["y"][rows], g_hist)

        # Smooth allocation and trading speed: chosen by trailing net-of-cost Sharpe (past only)
        if len(H_path) >= SELECT_MIN:
            Rp = H_path[-SELECT_WINDOW:]
            sd = Rp.std(axis=0)
            srp = np.where(sd > 0, Rp.mean(axis=0) / np.where(sd > 0, sd, 1.0), -np.inf)
            chosen_path = tuple(int(v) for v in np.unravel_index(int(np.argmax(srp)), (nE, nT)))
        else:
            chosen_path = DEFAULT_PATH
        alloc = [(_softmax_weights(H_cfg, eta) if len(H_cfg) >= SELECT_MIN
                  else np.full(nF * nK, 1.0 / (nF * nK))) for eta in ETA_GRID]

        # Daily factor regressions (unchanged)
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
            em = np.searchsorted(D["month_dt"], udates[span_lo:span_hi], side="left") - 1
            for gm in np.unique(em):
                if gm < 0:
                    continue
                js = np.flatnonzero(em == gm)
                E = expo(int(gm))
                cols = colmap[starts[gm]:ends[gm]]
                Rd = M[span_lo + js][:, cols]
                for jj, j in enumerate(js):
                    r = Rd[jj].astype(np.float64)
                    okr = np.isfinite(r)
                    if okr.sum() < max(3 * k, MIN_NAMES):
                        continue
                    Eo = E[okr]
                    f = np.linalg.solve(Eo.T @ Eo + 1e-10 * np.eye(k), Eo.T @ r[okr])
                    FR[j] = f
                    RES[j, cols[okr]] = r[okr] - Eo @ f

        for m in year_m:
            lo, hi = starts[m], ends[m]
            n = hi - lo
            if n < MIN_NAMES:
                continue
            ids_t = ids_all[lo:hi]
            zr = _zs(D["x0"][lo:hi].astype(np.float64) @ beta)
            zg = _zs(gbm.predict(D["x_raw"][lo:hi]))
            zr_hist[lo:hi], zg_hist[lo:hi] = zr, zg
            grp = _dis_groups(zg, zr)
            mu_learned = stack_B[grp, 0] * zg + stack_B[grp, 1] * zr
            MU = np.column_stack([mu_learned, 0.5 * zg + 0.5 * zr, zg])

            E = expo(int(m))
            end = int(np.searchsorted(udates, D["month_dt"][m], side="right"))
            wlo = max(span_lo, end - RISK_WINDOW_DAYS)
            fac = None
            if end > wlo:
                age = (end - 1 - np.arange(wlo, end)).astype(np.float64)
                fac = _month_risk(FR[wlo - span_lo:end - span_lo], RES[wlo - span_lo:end - span_lo],
                                  age, colmap[lo:hi], k, n)
            rr = np.nan_to_num(D["ret"][lo:hi], nan=0.0)

            if fac is not None:
                risk_f = _FactorRisk(E, *fac)
                lw = _sample_risk(M, end, colmap[lo:hi])
                jr = _JointRisk(risk_f, lw, n)
                el = jr.el
                We, omega, extremes = _joint_portfolios(jr, MU[el])
                W = np.zeros((n, nF, nK))
                W[el] = We
                scale_full = lambda w, jr=jr, el=el: _embed(jr.scale(w[el][:, None])[:, 0], el)
            else:                                       # no risk model yet: dollar-neutral, unit gross
                jr, el, omega, extremes = None, np.ones(n, dtype=bool), np.full((nF, nK), np.nan), {}
                P = MU - MU.mean(axis=0, keepdims=True)
                W = np.repeat((P / np.maximum(np.abs(P).sum(axis=0), 1e-12) * 2.0)[:, :, None], nK, axis=2)
                scale_full = lambda w: w * (2.0 / max(np.abs(w).sum(), 1e-12))

            Wf = W.reshape(n, nF * nK)
            cfg_ret = rr @ Wf

            # Allocation (per eta) and turnover-aware paths (per eta, tau)
            path_net = np.zeros((nE, nT))
            path_gross = np.zeros((nE, nT))
            path_traded = np.zeros((nE, nT))
            targets = []
            for ie in range(nE):
                tgt = scale_full(Wf @ alloc[ie])
                targets.append(tgt)
                for it, tau in enumerate(TAU_GRID):
                    prev = paths.get((ie, it))
                    pa = _align(prev, ids_t)
                    h = scale_full(pa + tau * (tgt - pa))
                    h = np.where(np.isfinite(h), h, 0.0)
                    tr = _trade_amount(prev, ids_t, h, pa)
                    paths[(ie, it)] = (ids_t, h)
                    g_ret = float(rr @ h)
                    path_gross[ie, it], path_traded[ie, it] = g_ret, tr
                    path_net[ie, it] = g_ret - COST * tr
            cfg_hist.append(cfg_ret)
            path_net_hist.append(path_net.ravel())
            hist_m.append(int(m))

            # Final holdings: their own path, using the currently chosen (eta, tau)
            ie, it = chosen_path
            pa = _align(final_prev, ids_t)
            w = scale_full(pa + TAU_GRID[it] * (targets[ie] - pa))
            w = np.where(np.isfinite(w), w, 0.0)
            traded = _trade_amount(final_prev, ids_t, w, pa)
            final_prev = (ids_t, w)
            out_frames.append(pd.DataFrame({"id": ids_t, "eom": eom_all[lo:hi], "w": w,
                                            "_test": test_all[lo:hi]}))

            if keep_diagnostics:
                ics = _monthly_ic(np.column_stack([zr, zg, mu_learned]), D["y"][lo:hi], [(0, n)])
                rec = dict(eom=pd.Timestamp(D["month_dt"][m]), year=Y, n=n,
                           ic_ridge=ics[0], ic_gbm=ics[1], ic_learned=ics[2],
                           ret=float(w @ rr), traded=traded, gross=float(np.abs(w).sum()),
                           net=float(w.sum()), n_eligible=int(el.sum()),
                           ew_mkt=float(np.nanmean(D["ret"][lo:hi])), fm_ok=fac is not None,
                           test=bool(test_all[lo:hi].any()), omega_learned_k1=float(omega[0, 2]),
                           eta=ETA_GRID[ie], tau=TAU_GRID[it])
                for f in range(nF):
                    for kk in range(nK):
                        rec[f"r_{F_NAMES[f]}_k{kk}"] = float(cfg_ret[f * nK + kk])
                for e2 in range(nE):
                    for t2 in range(nT):
                        rec[f"p_e{e2}_t{t2}_ret"] = path_gross[e2, t2]
                        rec[f"p_e{e2}_t{t2}_traded"] = path_traded[e2, t2]
                order = np.argsort(-np.asarray(alloc[1]), kind="stable")[:3]
                top3 = scale_full(Wf[:, order].mean(axis=1))
                rec["r_top3"] = float(rr @ top3)
                if 0.0 in extremes and 1.0 in extremes and jr is not None:
                    avg = 0.5 * (_embed(extremes[0.0], el) + _embed(extremes[1.0], el))
                    fac_only = _FactorRisk(E, *fac)
                    rec["r_avgbooks"] = float(rr @ _scale(avg[:, None], fac_only)[:, 0])
                mon_rec.append(rec)

        if keep_diagnostics:
            sizes = np.bincount(labels[labels > 0]) if (labels > 0).any() else np.array([0])
            yr_rec.append(dict(year=Y, ridge_lambda=lam, gbm_iter=gbm_iter, gbm_tuned=tuned,
                               n_train_rows=int(len(idx)), n_clusters=L.shape[1],
                               largest_cluster=int(sizes.max()),
                               n_industries=int(len(inds)) if inds is not None else 0,
                               eta=ETA_GRID[chosen_path[0]], tau=TAU_GRID[chosen_path[1]],
                               **{f"b_g{gk}_{nm}": float(stack_B[gk, j])
                                  for gk in range(STACK_GROUPS) for j, nm in enumerate(("tree", "ridge"))},
                               secs=time.perf_counter() - ty))
            beta_rec.append(pd.Series(beta, index=D["feats"], name=Y))
            last_clusters = (labels, L)
        _log(f"[{Y}] train months {len(train_m)}, ridge lambda {lam:g}, trees {gbm_iter}"
             f"{' (tuned)' if tuned else ''}, clusters {L.shape[1]}, "
             f"stack tree/ridge by disagreement {np.round(stack_B, 2).tolist()}, "
             f"eta {ETA_GRID[chosen_path[0]]:g}, tau {TAU_GRID[chosen_path[1]]:g}, "
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


def _align(prev, ids_t: np.ndarray) -> np.ndarray:
    """Last month's weights mapped onto this month's (sorted) ids; 0 for new names."""
    if prev is None or len(prev[0]) == 0:
        return np.zeros(len(ids_t))
    pids, pw = prev
    pos = np.clip(np.searchsorted(pids, ids_t), 0, len(pids) - 1)
    return np.where(pids[pos] == ids_t, pw[pos], 0.0)


def _embed(w_el: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Expand weights on the eligible universe to the full cross-section (zeros elsewhere)."""
    out = np.zeros(len(el))
    out[el] = w_el
    return out


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
