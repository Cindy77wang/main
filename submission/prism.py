"""PRISM: a Precision-Rotated, Integrated SDF Machine for the JKP Common Task Framework.

Author: Cindy Wang (Yale SOM, MGT 924 Statistical Foundations).

The model maps the CTF data into monthly portfolio weights in six steps. Every step uses
only information available at the portfolio formation date t (Rule 1): characteristics
observed at t, monthly returns realized at or before t, and daily returns dated at or
before t. All randomness is seeded, inputs are put in a canonical order first, and every
refit happens on a fixed calendar schedule (each December), so truncating the data at any
month leaves earlier weights unchanged.

1. Features. Each characteristic is ranked within its month and mapped to [-0.5, 0.5];
   missing values become 0 (the cross-sectional median). All provided features are used;
   there is no hand selection (Rule 2).
2. Risk model. A Barra-style factor model, re-specified every December. Its factors are
   the market, the Fama-French 12 industries, and every ranked characteristic
   (standardized within the month), as in the organizers' Markowitz-ML Barra model.
   Characteristic "themes" (hierarchical clustering of the average cross-sectional
   correlation matrix of the ranked characteristics) are reported as diagnostics and can
   replace the characteristics as factors (risk_char_factors = False). Daily ridge
   cross-sectional regressions give factor returns and residuals. The factor covariance
   is an EWMA (half-lives of 504 days for correlations and 84 for variances, as in MSCI
   Barra USE4S). Specific variances are an EWMA of squared residuals with an 84-day
   half-life; stocks without enough history are imputed from their exposures.
3. Return learners, refit every December on a rolling 20-year window with a time-ordered
   validation block: ridge regression, XGBoost, a feed-forward network (NumPy), and an
   LSTM over each stock's 12-month trajectory of characteristic principal components
   (NumPy). XGBoost learns GLS residual returns: each month's returns net of the risk
   model's factor exposures, per unit of specific volatility, i.e. the standardized
   diversifiable alpha that inv(Sigma) can harvest; the others learn z-scored raw returns.
4. Sleeves, i.e. candidate efficient portfolios, each scaled to unit ex-ante volatility:
   * B: the Markowitz portfolio inv(Sigma_t) mu_hat_t on the equal-weight combination of
     the learners' z-scored forecasts; a standardized-alpha forecast is first put back in
     return units by the specific volatility (alpha = IC x volatility x score).
   * A0 / A1: random-feature SDFs. Random Fourier features of the characteristics define
     thousands of managed portfolios, used raw (A0) or rotated by inv(Sigma_t) (A1).
     Their weights come from ridge-regularized Markowitz on past managed-portfolio
     returns, with the shrinkage chosen by blocked cross-validation of the Sharpe ratio.
5. Meta-combination. Non-negative Sharpe-maximizing weights on the sleeves, estimated by
   NNLS on their past out-of-sample returns and shrunk toward equal weights.
6. Volatility timing. The book is scaled to a 10% annual volatility target. The forecast
   is an EWMA (lambda = 0.97) of the book's synthetic daily returns, i.e. today's weights
   applied to the past year of daily returns.
"""

import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import nnls
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

# Configuration. Every value is fixed a priori, from the cited literature or as a
# conventional default; none was tuned on the test period. Hyperparameters that are
# data-driven (ridge penalties, XGBoost depth and trees, network early stopping, SDF
# shrinkage, sleeve weights) are chosen inside the code from past data only.
CONFIG = {
    "seed": 20251006,
    "n_threads": 0,  # 0 = min(32, cpu count); the CTF provides 32 cores (Rule 9)
    "learners": ("ridge", "xgb", "mlp", "lstm"),
    # Risk model (MSCI Barra USE4S half-lives; Menchero, Orr and Wang, 2011)
    "risk_cov_days": 2520,
    "risk_hl_corr": 504.0,
    "risk_hl_var": 84.0,
    "risk_hl_idio": 84.0,
    "risk_idio_days": 252,
    "risk_idio_min_obs": 63,
    "risk_min_hist_days": 63,
    "risk_ridge": 1e-4,
    "risk_min_industry": 5,
    "risk_theme_months": 60,
    "risk_max_themes": 13,  # the 13 themes of Jensen, Kelly and Pedersen (2023)
    "risk_char_factors": True,  # every characteristic is a factor (else the themes are)
    "risk_idio_floor_q": 0.05,
    "risk_stat_factors": 0,  # statistical factors from residuals (hybrid model); 0 = off
    # Return learners
    "learn_window": 240,
    "learn_min_months": 60,
    "learn_val_months": 36,
    "learn_val_frac": 0.25,
    "target_winsor": 0.01,
    # Learners trained on GLS residual returns (see _residual_returns); the others learn
    # z-scored raw returns
    "learn_resid": ("xgb",),
    "b_sleeves": "ens",  # "ens": one B sleeve on the combined forecast; "each": one per learner
    "ridge_grid": (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3),
    "xgb_depths": (3, 6),
    "xgb_eta": 0.1,
    "xgb_rounds": 500,
    "xgb_early_stop": 30,
    "xgb_subsample": 0.5,
    "xgb_colsample": 0.5,
    "xgb_min_child_frac": 5e-4,
    "xgb_lambda": 1.0,
    "xgb_max_bin": 64,
    "mlp_hidden": (32, 16, 8),  # NN3 of Gu, Kelly and Xiu (2020)
    "mlp_seeds": 3,
    "mlp_epochs": 20,
    "mlp_patience": 4,
    "mlp_batch": 4096,
    "mlp_lr": 1e-3,
    "mlp_l2": 1e-5,
    "lstm_hidden": 16,
    "lstm_inputs": 12,
    "lstm_len": 12,
    "lstm_seeds": 2,
    "lstm_epochs": 15,
    "lstm_patience": 3,
    "lstm_batch": 2048,
    "lstm_lr": 2e-3,
    "lstm_subsample": 0.5,
    # Random-feature SDF sleeves (Didisheim, Ke, Kelly and Malamud, 2024)
    "sdf_rff": 3000,
    "sdf_gammas": (0.5, 1.0, 2.0, 3.0),  # spans the bandwidths of JKMP (2024) and DKKM (2024)
    "sdf_window": 360,
    "sdf_min_months": 60,
    "sdf_folds": 5,
    "sdf_zeta_grid": tuple(10.0 ** np.arange(-5.0, 4.01, 0.5)),
    # Meta-combination and volatility timing
    "meta_window": 240,
    "meta_min_months": 36,
    "meta_shrink": 0.3,
    "vol_target": 0.10,
    "vol_days": 252,
    "vol_lambda": 0.97,  # RiskMetrics (1996) decay for monthly horizons
    "vol_min_cov": 0.5,
    "vol_cap_mult": 4.0,
    "vol_cap_months": 60,
}

ID_COLS = ("id", "eom", "eom_ret", "excntry", "sic", "size_grp", "ret_exc_lead1m", "ctff_test")
FF12 = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
        "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
DAYS_PER_MONTH = 21.0
_T0 = [time.time()]


def _log(msg):
    print(f"[PRISM {time.time() - _T0[0]:8.1f}s] {msg}", flush=True)


def _seed(cfg, *keys):
    """Independent, reproducible random stream for one model component."""
    return np.random.default_rng(np.random.SeedSequence([int(cfg["seed"])] + [int(k) for k in keys]))


# ----------------------------------------------------------------------------------------
# Section 1: data preparation
# ----------------------------------------------------------------------------------------

def _to_datetime(values):
    """Parse dates given as date objects, strings, datetime64 or yyyymmdd integers."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        return pd.to_datetime(num.round().astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _month_index(values):
    """Calendar month index year*12 + month - 1."""
    dt = _to_datetime(values)
    if dt.isna().any():
        raise ValueError("unparseable dates in input")
    return (dt.dt.year * 12 + dt.dt.month - 1).to_numpy(dtype=np.int64)


def _month_end_day(m):
    """Days since 1970-01-01 of the last calendar day of month index m."""
    mm = np.asarray(m, dtype=np.int64) - 1970 * 12
    first_next = (mm + 1).astype("datetime64[M]").astype("datetime64[D]")
    return first_next.astype(np.int64) - 1


def _month_end_date(m):
    """Month index -> month-end dates (datetime64[ns])."""
    return pd.to_datetime(_month_end_day(np.asarray(m, dtype=np.int64)).astype("datetime64[D]"))


def _as_bool(series):
    s = pd.Series(series).reset_index(drop=True)
    if s.dtype == bool:
        return s.to_numpy()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.to_numpy() != 0
    return s.astype(str).str.strip().str.lower().isin(("true", "t", "1", "1.0", "yes")).to_numpy()


def ff12_codes(sic):
    """Fama-French 12 industry code (index into FF12) from SIC codes given as text or numbers."""
    s = pd.to_numeric(pd.Series(sic).reset_index(drop=True), errors="coerce").to_numpy(dtype=np.float64)
    s = np.where(np.isfinite(s), s, -1).astype(np.int64)

    def isin(ranges):
        out = np.zeros(len(s), dtype=bool)
        for lo, hi in ranges:
            out |= (s >= lo) & (s <= hi)
        return out

    rules = [
        [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
        [(2500, 2519), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
         (3792, 3792), (3900, 3939), (3990, 3999)],
        [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
         (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
         (3830, 3839), (3860, 3899)],
        [(1200, 1399), (2900, 2999)],
        [(2800, 2829), (2840, 2899)],
        [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
        [(4800, 4899)],
        [(4900, 4949)],
        [(5000, 5999), (7200, 7299), (7600, 7699)],
        [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
        [(6000, 6999)],
    ]
    code = np.full(len(s), 11, dtype=np.int64)
    assigned = np.zeros(len(s), dtype=bool)
    for k, ranges in enumerate(rules):  # first match wins, as in the R case_when
        hit = isin(ranges) & ~assigned
        code[hit] = k
        assigned |= hit
    return code


def _rank_block(block):
    """Within-month ranks of each column mapped to [-0.5, 0.5]; NaN -> 0."""
    n = block.shape[0]
    if n == 0:
        return np.zeros(block.shape, dtype=np.float32)
    valid = ~np.isnan(block)
    cnt = valid.sum(axis=0)
    ranks = rankdata(block, axis=0, method="average", nan_policy="omit")
    denom = np.where(cnt > 1, cnt - 1, 1).astype(np.float64)
    out = (ranks - 1.0) / denom - 0.5
    out[~valid] = 0.0
    out[:, cnt <= 1] = 0.0
    return out.astype(np.float32)


class Panel:
    """Monthly stock panel sorted by (month, id), with ranked features."""

    def __init__(self, chars, features):
        t0 = time.time()
        if isinstance(features, pd.DataFrame):
            col = "features" if "features" in features.columns else features.columns[0]
            names = features[col]
        else:
            names = pd.Series(features)
        names = sorted({str(f) for f in names.dropna()} - set(ID_COLS))
        self.features = [f for f in names if f in chars.columns]
        missing = len(names) - len(self.features)
        if missing:
            _log(f"warning: {missing} listed features are not columns of chars")
        if not self.features:
            raise ValueError("no usable features")

        if chars.columns.duplicated().any():
            chars = chars.loc[:, ~chars.columns.duplicated()]
        ids = pd.to_numeric(chars["id"].reset_index(drop=True)).to_numpy().astype(np.int64)
        month = _month_index(chars["eom"])
        y = pd.to_numeric(chars["ret_exc_lead1m"].reset_index(drop=True), errors="coerce")
        y = y.to_numpy(dtype=np.float64, na_value=np.nan)
        # Canonical order: month, then id
        order = np.lexsort((ids, month))
        dup = (np.diff(month[order]) == 0) & (np.diff(ids[order]) == 0)
        if dup.any():
            # Duplicated (id, eom) rows (absent from the CTF data): keep one, chosen by the
            # content of its characteristics (never by the return), so the choice does not
            # depend on the input order
            involved = np.zeros(len(order), dtype=bool)
            involved[1:] |= dup
            involved[:-1] |= dup
            sub = chars.iloc[order[involved]][self.features].apply(pd.to_numeric, errors="coerce")
            sub = sub.to_numpy(dtype=np.float64, na_value=np.nan)
            keys = np.zeros((3, len(chars)))
            keys[0, order[involved]] = np.isnan(sub).sum(axis=1)
            keys[1, order[involved]] = np.nansum(sub, axis=1)
            keys[2, order[involved]] = np.nansum(sub * sub, axis=1)
            order = np.lexsort((keys[2], keys[1], keys[0], ids, month))
            keep = np.ones(len(order), dtype=bool)
            keep[1:] = ~((np.diff(month[order]) == 0) & (np.diff(ids[order]) == 0))
            order = order[keep]
            _log(f"warning: dropped {(~keep).sum()} duplicated (id, eom) rows")

        self.src = order  # positions in the original chars
        self.month = month[order]
        self.ids = ids[order]
        self.y = np.where(np.isfinite(y[order]), y[order], 0.0)
        self.test = _as_bool(chars["ctff_test"])[order]
        sic = chars["sic"] if "sic" in chars.columns else pd.Series(np.full(len(chars), np.nan))
        self.ind = ff12_codes(sic)[order].astype(np.int8)
        self.uid = np.unique(self.ids)
        self.code = np.searchsorted(self.uid, self.ids).astype(np.int64)
        self.months, first = np.unique(self.month, return_index=True)
        self.starts = np.append(first, len(self.month)).astype(np.int64)
        self.pos = {int(m): p for p, m in enumerate(self.months)}

        # Ranked features, built in column chunks to bound memory
        n, d = len(order), len(self.features)
        self.X = np.empty((n, d), dtype=np.float32)
        chunk = 48
        for c0 in range(0, d, chunk):
            cols = self.features[c0:c0 + chunk]
            try:
                raw = chars[cols].to_numpy(dtype=np.float64, na_value=np.nan)[order]
            except (TypeError, ValueError):  # non-numeric storage: coerce column by column
                raw = chars[cols].apply(pd.to_numeric, errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan)[order]
            raw[~np.isfinite(raw)] = np.nan
            for p in range(len(self.months)):
                a, b = self.starts[p], self.starts[p + 1]
                self.X[a:b, c0:c0 + len(cols)] = _rank_block(raw[a:b])
            del raw
        _log(f"panel: {n:,} rows, {len(self.months)} months, {d} features, "
             f"{len(self.uid):,} stocks ({time.time() - t0:.0f}s)")

    def rows(self, m):
        p = self.pos[int(m)]
        return slice(int(self.starts[p]), int(self.starts[p + 1]))


class Daily:
    """Dense (trading day x stock) matrix of daily excess returns for the panel's stocks."""

    def __init__(self, daily_ret, panel):
        t0 = time.time()
        ids = pd.to_numeric(daily_ret["id"].reset_index(drop=True), errors="coerce").to_numpy()
        days = _to_datetime(daily_ret["date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
        ret = pd.to_numeric(daily_ret["ret_exc"].reset_index(drop=True), errors="coerce")
        ret = ret.to_numpy(dtype=np.float64, na_value=np.nan)
        first_day = int(_month_end_day(panel.months[0])) - 400
        ok = np.isfinite(ids) & np.isfinite(ret) & (days >= first_day)
        ids, days, ret = ids[ok].astype(np.int64), days[ok], ret[ok]
        # Trading-day calendar from all stocks (not only the panel's), so that removing later
        # months of chars never changes the day index of earlier dates
        self.dates = np.unique(days)
        pos = np.searchsorted(panel.uid, ids)
        pos = np.minimum(pos, len(panel.uid) - 1)
        ok = panel.uid[pos] == ids
        pos, days, ret = pos[ok], days[ok], ret[ok]
        # Duplicated (id, date) rows (absent from the CTF data): keep one, order-independently
        srt = np.lexsort((ret, days, pos))
        pos, days, ret = pos[srt], days[srt], ret[srt]
        first = np.ones(len(pos), dtype=bool)
        first[1:] = (np.diff(pos) != 0) | (np.diff(days) != 0)
        pos, days, ret = pos[first], days[first], ret[first]
        di = np.searchsorted(self.dates, days)
        self.R = np.full((len(self.dates), len(panel.uid)), np.nan, dtype=np.float32)
        self.R[di, pos] = ret.astype(np.float32)
        # Last trading day at or before each month end (-1 if none)
        me = _month_end_day(panel.months)
        self.day_end = np.searchsorted(self.dates, me, side="right") - 1
        self.me = dict(zip((int(m) for m in panel.months), (int(x) for x in self.day_end)))
        _log(f"daily: {len(self.dates):,} trading days x {len(panel.uid):,} stocks, "
             f"{ok.sum():,} returns ({time.time() - t0:.0f}s)")

    def last_day_on_or_before(self, day):
        return int(np.searchsorted(self.dates, day, side="right") - 1)


# ----------------------------------------------------------------------------------------
# Section 2: risk model
# ----------------------------------------------------------------------------------------

class MonthRisk:
    """Sigma = E L L' E' + diag(spec), in daily units, for the stocks of one month."""

    def __init__(self, E, L, spec, ok):
        self.E, self.L, self.spec, self.ok = E, L, spec, ok
        self.EL = E @ L

    def solve(self, B):
        """inv(Sigma) @ B by the Woodbury identity; B is (n,) or (n, P)."""
        dinv = 1.0 / self.spec
        vec = B.ndim == 1
        Bm = B[:, None] if vec else B
        DB = Bm * dinv[:, None]
        DEL = self.EL * dinv[:, None]
        M = np.eye(self.EL.shape[1]) + self.EL.T @ DEL
        out = DB - DEL @ np.linalg.solve(M, self.EL.T @ DB)
        return out[:, 0] if vec else out

    def var(self, B):
        """Daily variance B' Sigma B of each column of B (or of the vector B)."""
        vec = B.ndim == 1
        Bm = B[:, None] if vec else B
        v = (Bm * Bm * self.spec[:, None]).sum(axis=0) + ((self.EL.T @ Bm) ** 2).sum(axis=0)
        return float(v[0]) if vec else v


def _ewma_weights(n, halflife):
    age = np.arange(n - 1, -1, -1, dtype=np.float64)
    return 0.5 ** (age / halflife)


def _weighted_cov(Fm, w):
    w = w / w.sum()
    mu = w @ Fm
    Z = Fm - mu
    return (Z * w[:, None]).T @ Z / max(1.0 - (w ** 2).sum(), 1e-12)


def _month_corr(X):
    n = X.shape[0]
    if n < 3:
        return np.zeros((X.shape[1], X.shape[1]))
    Xd = X.astype(np.float64)
    Xd = Xd - Xd.mean(axis=0)
    cov = Xd.T @ Xd / n
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    inv = np.where(sd > 1e-12, 1.0 / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return cov * inv[:, None] * inv[None, :]


def _themes(C, n_themes, min_avail=0.5):
    """Hierarchical clustering of the features available in the window; returns (labels, signs).

    The diagonal of the window-average correlation matrix is the share of months in which a
    feature varies across stocks. Features available in fewer than half of the months (e.g.
    before a data item exists) get label -1 and no theme, so they neither occupy theme slots
    nor, when numerous, force all features into a single cluster.
    """
    d = C.shape[0]
    live = np.flatnonzero(np.clip(np.diag(C), 0.0, 1.0) >= min_avail)
    if len(live) < 2:
        live = np.arange(d)
    n_eff = int(min(n_themes, max(1, len(live) // 3)))
    lab_l, sg_l = _cluster(C[np.ix_(live, live)], n_eff)
    labels = np.full(d, -1, dtype=np.int64)
    signs = np.ones(d)
    labels[live], signs[live] = lab_l, sg_l
    return labels, signs


def _cluster(C, n_themes):
    """Average-linkage clustering on 1 - |corr|, canonical labels, sign alignment."""
    d = C.shape[0]
    if d == 1 or n_themes <= 1:
        signs = np.ones(d)
        if d > 1:
            v = np.linalg.eigh(C)[1][:, -1]
            signs = np.where(v >= 0, 1.0, -1.0) * (1.0 if v[0] >= 0 else -1.0)
        return np.zeros(d, dtype=np.int64), signs
    D = 1.0 - np.abs(C)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    raw = fcluster(linkage(squareform(D, checks=False), method="average"),
                   t=n_themes, criterion="maxclust")
    # Canonical labels: clusters ordered by their first (alphabetical) feature
    first = {}
    for j, lab in enumerate(raw):
        first.setdefault(lab, j)
    relabel = {lab: k for k, lab in enumerate(sorted(first, key=first.get))}
    labels = np.array([relabel[lab] for lab in raw], dtype=np.int64)
    signs = np.ones(d)
    for k in range(labels.max() + 1):
        J = np.flatnonzero(labels == k)
        if len(J) > 1:
            v = np.linalg.eigh(C[np.ix_(J, J)])[1][:, -1]
            s = np.where(v >= 0, 1.0, -1.0)
            signs[J] = s * s[0]  # orient so the cluster's first feature enters positively
    return labels, signs


def _theme_scores(X, labels, signs):
    k = labels.max() + 1
    T = np.zeros((X.shape[0], k))
    Xs = X.astype(np.float64) * signs[None, :]
    for c in range(k):
        J = labels == c
        T[:, c] = Xs[:, J].mean(axis=1)
    sd = T.std(axis=0)
    return np.where(sd > 1e-12, (T - T.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def _zcols(T):
    """Columns standardized to mean 0 and unit variance; constant columns become 0."""
    T = T - T.mean(axis=0)
    sd = T.std(axis=0)
    return np.where(sd > 1e-12, T / np.where(sd > 1e-12, sd, 1.0), 0.0)


def _day_regressions(E, Rm, lam, min_n):
    """Daily cross-sectional ridge regressions of returns Rm (days x n) on exposures E (n x K).

    Returns factor returns (days x K) and residuals (days x n), NaN where unavailable. Days
    with missing returns are fitted on the stocks with returns if there are at least min_n."""
    K = E.shape[1]
    F = np.full((Rm.shape[0], K), np.nan)
    Res = np.full(Rm.shape, np.nan)
    full = ~np.isnan(Rm).any(axis=1)
    if full.any():
        G = E.T @ E + lam * E.shape[0] * np.eye(K)
        F[full] = np.linalg.solve(G, E.T @ Rm[full].T).T
        Res[full] = Rm[full] - F[full] @ E.T
    for j in np.flatnonzero(~full):
        v = ~np.isnan(Rm[j])
        if v.sum() < min_n:
            continue
        Ev = E[v]
        F[j] = np.linalg.solve(Ev.T @ Ev + lam * v.sum() * np.eye(K), Ev.T @ Rm[j, v])
        Res[j, v] = Rm[j, v] - Ev @ F[j]
    return F, Res


def build_risk_model(panel, daily, cfg, diag):
    """Monthly factor risk model; returns {month: MonthRisk} for every panel month."""
    t0 = time.time()
    months = [int(m) for m in panel.months]
    d = len(panel.features)
    n_themes = int(min(cfg["risk_max_themes"], max(1, d // 3)))
    starts = [months[0]] + [m for m in months[1:] if m % 12 == 11]
    block_of = {}
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else months[-1] + 1
        for m in months:
            if s <= m < e:
                block_of[m] = b
    corr_cache = {}
    reg_cache = {}  # (month, industries) -> daily regressions, reused across blocks
    char_f = bool(cfg["risk_char_factors"])
    risk = {}
    theme_rows = []
    hl_c, hl_v, hl_s = cfg["risk_hl_corr"], cfg["risk_hl_var"], cfg["risk_hl_idio"]
    n_cov, n_idio = int(cfg["risk_cov_days"]), int(cfg["risk_idio_days"])

    for b, s in enumerate(starts):
        block_months = [m for m in months if block_of[m] == b]
        # Themes from the average correlation over the trailing theme window
        hist = [m for m in months if s - cfg["risk_theme_months"] < m <= s]
        for m in hist:
            if m not in corr_cache:
                corr_cache[m] = _month_corr(panel.X[panel.rows(m)])
        for m in list(corr_cache):
            if m <= s - cfg["risk_theme_months"]:
                del corr_cache[m]
        C = np.mean([corr_cache[m] for m in hist], axis=0)
        labels, signs = _themes(C, n_themes)
        theme_rows.append(pd.DataFrame({"refit_eom": s, "feature": panel.features, "cluster": labels}))
        # Industries with enough stocks at the block start get a dummy (others load on the market only)
        cnt = np.bincount(panel.ind[panel.rows(s)], minlength=12)
        inds = [k for k in range(12) if cnt[k] >= cfg["risk_min_industry"]]

        n_style = d if char_f else labels.max() + 1

        def exposures(m):
            r = panel.rows(m)
            n = r.stop - r.start
            E = np.empty((n, 1 + len(inds) + n_style))
            E[:, 0] = 1.0
            ind = panel.ind[r]
            for j, k in enumerate(inds):
                E[:, 1 + j] = ind == k
            if char_f:
                E[:, 1 + len(inds):] = _zcols(panel.X[r].astype(np.float64))
            else:
                E[:, 1 + len(inds):] = _theme_scores(panel.X[r], labels, signs)
            return E

        # Daily regressions over the covariance window of the block's first month
        de_first = daily.me[s]
        de_last = daily.me[block_months[-1]]
        day_lo = max(0, de_first - n_cov + 1)
        K = 1 + len(inds) + n_style
        # Ridge identifies the characteristic factors, so a partial day needs enough stocks
        # for the market and industry factors only
        min_n = 1 + len(inds) + 5 if char_f else K + 5
        n_days = max(de_last - day_lo + 1, 0)
        FR = np.full((n_days, K), np.nan)
        # Residuals are needed only for the specific-risk window of the block's months
        res_lo = max(day_lo, de_first - n_idio + 1)
        RES = np.full((max(de_last - res_lo + 1, 0), len(panel.uid)), np.nan, dtype=np.float32)
        exp_cache = {}
        first_e = None
        for e in months:
            if e > block_months[-1]:
                break
            if len(daily.dates) == 0 or daily.me[e] + 1 > de_last \
                    or _month_end_day(e + 1) < daily.dates[day_lo]:
                continue
            a = max(daily.me[e] + 1, day_lo)
            z = min(daily.last_day_on_or_before(int(_month_end_day(e + 1))), de_last)
            if z < a:
                continue
            r = panel.rows(e)
            codes = panel.code[r]
            first_e = e if first_e is None else first_e
            key = (e, tuple(inds))
            if char_f and key in reg_cache and e not in block_months:
                a0, Fm, Resm = reg_cache[key]
            else:
                E = exposures(e)
                if e in block_months:
                    exp_cache[e] = E
                # With characteristic factors the exposures of month e do not depend on the
                # block, so the whole return month is regressed once and reused by later blocks
                a0 = daily.me[e] + 1 if char_f else a
                z0 = daily.last_day_on_or_before(int(_month_end_day(e + 1))) if char_f else z
                Rm = daily.R[a0:z0 + 1][:, codes].astype(np.float64)
                Fm, Resm = _day_regressions(E, Rm, cfg["risk_ridge"], min_n)
                if char_f:
                    reg_cache[key] = (a0, Fm, Resm.astype(np.float32))
            FR[a - day_lo:z - day_lo + 1] = Fm[a - a0:z - a0 + 1]
            k0 = max(a, res_lo)
            if z >= k0:
                RES[k0 - res_lo:z - res_lo + 1, codes] = Resm[k0 - a0:z - a0 + 1]
        for key in [k for k in reg_cache if first_e is None or k[0] < first_e]:
            del reg_cache[key]  # later blocks start their windows later

        for t in block_months:
            r = panel.rows(t)
            E = exp_cache.get(t)
            if E is None:
                E = exposures(t)
            dt_end = daily.me[t]
            hi = dt_end - day_lo + 1
            lo = max(0, hi - n_cov)
            Fw = FR[lo:hi]
            Fw = Fw[~np.isnan(Fw).any(axis=1)] if hi > lo else np.zeros((0, K))
            ok = len(Fw) >= cfg["risk_min_hist_days"]
            if len(Fw) >= 2:
                wc, wv = _ewma_weights(len(Fw), hl_c), _ewma_weights(len(Fw), hl_v)
                cc = _weighted_cov(Fw, wc)
                sd_c = np.sqrt(np.clip(np.diag(cc), 1e-20, None))
                cor = cc / np.outer(sd_c, sd_c)
                var = np.clip(np.diag(_weighted_cov(Fw, wv)), 1e-20, None)
                n_eff = wc.sum() ** 2 / (wc ** 2).sum()
                shrink = min(1.0, K / max(n_eff, 1.0))
                cor = (1.0 - shrink) * cor + shrink * np.eye(K)
                Om = cor * np.outer(np.sqrt(var), np.sqrt(var))
            else:
                Om = np.eye(K) * 1e-6
            lam_, V = np.linalg.eigh(0.5 * (Om + Om.T))
            L = V * np.sqrt(np.clip(lam_, 0.0, None))[None, :]
            # Specific variance: EWMA of squared residuals over the past year
            codes = panel.code[r]
            rhi = dt_end - res_lo + 1
            rlo = max(0, rhi - n_idio)
            Rw = RES[rlo:rhi][:, codes].astype(np.float64) if rhi > rlo else np.zeros((0, len(codes)))
            valid = ~np.isnan(Rw)
            w = _ewma_weights(Rw.shape[0], hl_s)
            num = w @ np.where(valid, Rw * Rw, 0.0)
            den = w @ valid
            nobs = valid.sum(axis=0)
            good = (nobs >= cfg["risk_idio_min_obs"]) & (den > 0)
            spec = np.full(len(codes), np.nan)
            spec[good] = num[good] / den[good]
            spec = _impute_spec(spec, E, cfg)
            if cfg["risk_stat_factors"] > 0 and Rw.shape[0] >= cfg["risk_idio_min_obs"]:
                E, L, spec = _add_stat_factors(E, L, spec, Rw, valid, w, cfg)
            risk[t] = MonthRisk(E, L, spec, ok)
        if (b + 1) % 12 == 0:
            _log(f"risk model: block {b + 1}/{len(starts)} ({time.time() - t0:.0f}s)")
    diag["themes"] = pd.concat(theme_rows, ignore_index=True)
    _log(f"risk model: {len(starts)} blocks, {n_themes} themes ({time.time() - t0:.0f}s)")
    return risk


def _add_stat_factors(E, L, spec, Rw, valid, w, cfg):
    """Hybrid risk model: statistical factors from the past year of specific returns.

    The characteristic factors miss part of the common variation (heterogeneous market
    betas, measurement error in the exposures). The leading principal components of the
    EWMA-weighted, standardized residuals capture it. Only components above the
    Marchenko-Pastur noise edge are kept, their variance is shrunk by the edge, and the
    specific variances are reduced by the variance they explain, so total risk is unchanged.
    """
    n = len(spec)
    sd = np.sqrt(spec)
    Z = np.where(valid, Rw, 0.0) / sd[None, :]
    wn = w / w.sum()
    Zw = Z * np.sqrt(wn)[:, None]
    t_eff = 1.0 / (wn ** 2).sum()
    try:
        _, sv, Vt = np.linalg.svd(Zw, full_matrices=False)
    except np.linalg.LinAlgError:
        return E, L, spec
    lam = sv ** 2  # variance of each component in standardized units (unit noise level)
    edge = (1.0 + np.sqrt(n / t_eff)) ** 2
    k = int(min(cfg["risk_stat_factors"], np.sum(lam > edge)))
    if k == 0:
        return E, L, spec
    lam_s = lam[:k] - edge  # shrink the spiked eigenvalues by the noise edge
    V = Vt[:k].T  # n x k loadings on standardized residuals
    load = V * sd[:, None]  # loadings in return units
    explained = (load ** 2 * lam_s[None, :]).sum(axis=1)
    floor = np.quantile(spec, cfg["risk_idio_floor_q"])
    spec_new = np.maximum(spec - explained, np.maximum(0.25 * spec, floor))
    K0 = L.shape[1]
    L_new = np.zeros((K0 + k, K0 + k))
    L_new[:K0, :K0] = L
    L_new[K0:, K0:] = np.diag(np.sqrt(lam_s))
    return np.column_stack([E, load]), L_new, spec_new


def _impute_spec(spec, E, cfg):
    """Fill missing specific variances from a ridge regression of log variance on exposures; floor."""
    good = np.isfinite(spec) & (spec > 0)
    if good.sum() >= max(10, E.shape[1] + 2):
        y = np.log(spec[good])
        Eg = E[good]
        beta = np.linalg.solve(Eg.T @ Eg + 1e-4 * good.sum() * np.eye(E.shape[1]), Eg.T @ y)
        fill = np.exp(E @ beta)
    elif good.any():
        fill = np.full(len(spec), np.median(spec[good]))
    else:
        fill = np.full(len(spec), 0.02 ** 2)
    out = np.where(good, spec, fill)
    floor = np.quantile(spec[good], cfg["risk_idio_floor_q"]) if good.sum() >= 2 else 1e-8
    return np.maximum(out, max(floor, 1e-8))


# ----------------------------------------------------------------------------------------
# Section 3: return learners
# ----------------------------------------------------------------------------------------

def _zscore_target(y, month, months, starts, q):
    """Within-month winsorized, demeaned, unit-variance returns (the learning target)."""
    z = np.zeros_like(y)
    for p in range(len(months)):
        a, b = starts[p], starts[p + 1]
        v = y[a:b]
        if len(v) < 3:
            continue
        lo, hi = np.quantile(v, [q, 1.0 - q])
        v = np.clip(v, lo, hi)
        v = v - v.mean()
        sd = v.std()
        z[a:b] = v / sd if sd > 0 else 0.0
    return z


def _wls_residual(v, rk, lam):
    """Residual of v after a ridge regression on the exposures weighted by 1/specific variance."""
    E = rk.E
    n, K = E.shape
    wv = 1.0 / rk.spec
    Ew = E * (wv / wv.mean())[:, None]
    f = np.linalg.solve(Ew.T @ E + lam * n * np.eye(K), Ew.T @ v)
    return v - E @ f


def _residual_returns(panel, risk, cfg):
    """GLS learning target: month-m returns net of the month-m factor exposures, per unit of
    specific volatility.

    The cross-section of raw returns is dominated by common factor shocks (market, industry,
    style), and the priced part of a raw-return forecast loads on factors that inv(Sigma)
    hedges away in the B sleeve. Regressing each month's realized returns on the risk model's
    exposures E_m (weighted least squares with weights 1/spec and the risk model's ridge) and
    dividing the residual by the specific volatility leaves the diversifiable part of returns
    on a homoskedastic scale, so a learner fits the standardized alpha alpha_i / sigma_i (the
    residual-return / information-ratio framework of Grinold and Kahn, 2000). E_m and spec_m
    are known at m, and the residual of month m uses the return realized at m+1, exactly like
    the raw label; every month is computed on its own, so truncation leaves it unchanged.
    """
    out = np.zeros(len(panel.y))
    lam = cfg["risk_ridge"]
    for m in panel.months:
        r = panel.rows(m)
        rk = risk[int(m)]
        if r.stop - r.start < rk.E.shape[1] + 5:
            continue
        out[r] = _wls_residual(panel.y[r], rk, lam) / np.sqrt(DAYS_PER_MONTH * rk.spec)
    return out


def _month_ic(pred, y, bounds):
    ics = []
    for a, b in bounds:
        p, t = pred[a:b], y[a:b]
        if b - a >= 3 and p.std() > 0 and t.std() > 0:
            ics.append(np.corrcoef(p, t)[0, 1])
    return float(np.mean(ics)) if ics else -np.inf


class _Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            x -= self.lr * (m / c1) / (np.sqrt(v / c2) + eps)


def _mlp_forward(params, X):
    hs = [X]
    h = X
    n_layers = len(params) // 2
    for k in range(n_layers):
        W, b = params[2 * k], params[2 * k + 1]
        h = h @ W + b
        if k < n_layers - 1:
            h = np.maximum(h, 0.0)
        hs.append(h)
    return hs


def _train_mlp(Xtr, ytr, Xva, yva, cfg, rng):
    sizes = [Xtr.shape[1]] + list(cfg["mlp_hidden"]) + [1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
        params.append(np.zeros(b, dtype=np.float32))
    opt = _Adam(params, cfg["mlp_lr"])
    l2 = np.float32(cfg["mlp_l2"])
    best, best_loss, bad = [x.copy() for x in params], np.inf, 0
    n = len(Xtr)
    bs = int(min(cfg["mlp_batch"], max(256, n // 100)))  # >= ~100 updates per epoch
    for _ in range(int(cfg["mlp_epochs"])):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            hs = _mlp_forward(params, Xtr[idx])
            g = (2.0 / len(idx)) * (hs[-1][:, 0] - ytr[idx])[:, None].astype(np.float32)
            grads = [None] * len(params)
            for k in range(len(params) // 2 - 1, -1, -1):
                W = params[2 * k]
                grads[2 * k] = hs[k].T @ g + l2 * W
                grads[2 * k + 1] = g.sum(axis=0)
                if k > 0:
                    g = (g @ W.T) * (hs[k] > 0)
            opt.step(grads)
        loss = float(np.mean((_mlp_forward(params, Xva)[-1][:, 0] - yva) ** 2))
        if loss < best_loss - 1e-7:
            best_loss, best, bad = loss, [x.copy() for x in params], 0
        else:
            bad += 1
            if bad >= cfg["mlp_patience"]:
                break
    return best


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _LSTM:
    """Single-layer LSTM regressor on (batch, time, inputs), trained with BPTT and Adam."""

    def __init__(self, n_in, H, rng):
        s = 1.0 / np.sqrt(H)
        self.Wx = rng.uniform(-s, s, (n_in, 4 * H)).astype(np.float32)
        self.Wh = rng.uniform(-s, s, (H, 4 * H)).astype(np.float32)
        self.b = np.zeros(4 * H, dtype=np.float32)
        self.b[H:2 * H] = 1.0  # forget-gate bias
        self.Wo = rng.uniform(-s, s, (H, 1)).astype(np.float32)
        self.bo = np.zeros(1, dtype=np.float32)
        self.H = H

    @property
    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wo, self.bo]

    def forward(self, U, keep=False):
        B, T, _ = U.shape
        H = self.H
        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)
        cache = []
        for t in range(T):
            z = U[:, t] @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(z[:, :H])
            f = _sigmoid(z[:, H:2 * H])
            o = _sigmoid(z[:, 2 * H:3 * H])
            g = np.tanh(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            if keep:
                cache.append((h, c, i, f, o, g, tc))
            h, c = o * tc, c_new
        return (h @ self.Wo + self.bo)[:, 0], h, cache

    def grads(self, U, y):
        B, T, _ = U.shape
        H = self.H
        pred, hT, cache = self.forward(U, keep=True)
        dy = ((2.0 / B) * (pred - y))[:, None].astype(np.float32)
        gWo, gbo = hT.T @ dy, dy.sum(axis=0)
        dh = dy @ self.Wo.T
        dc = np.zeros((B, H), dtype=np.float32)
        gWx, gWh, gb = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        for t in range(T - 1, -1, -1):
            h_prev, c_prev, i, f, o, g, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc * tc)
            dz = np.concatenate([dc * g * i * (1 - i), dc * c_prev * f * (1 - f),
                                 do * o * (1 - o), dc * i * (1 - g * g)], axis=1)
            gWx += U[:, t].T @ dz
            gWh += h_prev.T @ dz
            gb += dz.sum(axis=0)
            dh = dz @ self.Wh.T
            dc = dc * f
        grads = [gWx, gWh, gb, gWo, gbo]
        norm = np.sqrt(sum(float((x * x).sum()) for x in grads))
        if norm > 1.0:
            grads = [x / norm for x in grads]
        return grads


def _train_lstm(Utr, ytr, Uva, yva, cfg, rng):
    net = _LSTM(Utr.shape[2], int(cfg["lstm_hidden"]), rng)
    opt = _Adam(net.params, cfg["lstm_lr"])
    best, best_loss, bad = [x.copy() for x in net.params], np.inf, 0
    n = len(Utr)
    bs = int(min(cfg["lstm_batch"], max(256, n // 100)))
    for _ in range(int(cfg["lstm_epochs"])):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.step(net.grads(Utr[idx], ytr[idx]))
        loss = float(np.mean((net.forward(Uva)[0] - yva) ** 2))
        if loss < best_loss - 1e-7:
            best_loss, best, bad = loss, [x.copy() for x in net.params], 0
        else:
            bad += 1
            if bad >= cfg["lstm_patience"]:
                break
    for x, bst in zip(net.params, best):
        x[...] = bst
    return net


def _sequences(panel, rows, U, lag_index, L):
    """(len(rows), L, k+1) array: past L months of U for each row's stock, plus a presence flag."""
    k = U.shape[1]
    out = np.zeros((len(rows), L, k + 1), dtype=np.float32)
    for j in range(L):  # j = 0 is the oldest month, L-1 the current month
        src = lag_index[L - 1 - j][rows]
        has = src >= 0
        out[has, j, :k] = U[src[has]]
        out[has, j, k] = 1.0
    return out


def _lag_index(panel, L):
    """lag_index[l][row] = row of the same stock l months earlier, or -1."""
    pos_of_row = np.repeat(np.arange(len(panel.months)), np.diff(panel.starts))
    grid = np.full((len(panel.months), len(panel.uid)), -1, dtype=np.int64)
    grid[pos_of_row, panel.code] = np.arange(len(panel.month))
    out = []
    for lag in range(L):
        prev_pos = pos_of_row - lag
        # Calendar check: the earlier position must be exactly `lag` months back
        ok = prev_pos >= 0
        idx = np.full(len(panel.month), -1, dtype=np.int64)
        cand = np.where(ok, grid[np.maximum(prev_pos, 0), panel.code], -1)
        same_gap = np.zeros(len(panel.month), dtype=bool)
        same_gap[ok] = panel.months[prev_pos[ok]] == panel.month[ok] - lag
        idx[same_gap] = cand[same_gap]
        out.append(idx)
    return out


def fit_learners(panel, cfg, diag, n_threads, risk=None):
    """Out-of-sample predictions of each learner, refit every December; dict name -> (n_rows,) array.

    Learners in cfg["learn_resid"] predict standardized alphas (see _residual_returns)."""
    t0 = time.time()
    months = [int(m) for m in panel.months]
    yz = _zscore_target(panel.y, panel.month, panel.months, panel.starts, cfg["target_winsor"])
    yr = None
    if any(k in cfg["learn_resid"] for k in cfg["learners"]):
        if risk is None:
            raise ValueError("learn_resid needs the risk model")
        yr = _zscore_target(_residual_returns(panel, risk, cfg), panel.month, panel.months,
                            panel.starts, cfg["target_winsor"])
    preds = {k: np.full(len(panel.month), np.nan, dtype=np.float64) for k in cfg["learners"]}
    lag_index = _lag_index(panel, int(cfg["lstm_len"])) if "lstm" in cfg["learners"] else None
    refits = [m for m in months if m % 12 == 11]
    timing = {k: 0.0 for k in cfg["learners"]}
    for R in refits:
        win = [m for m in months if R - cfg["learn_window"] <= m <= R - 1]
        if len(win) < cfg["learn_min_months"]:
            continue
        pred_m = [m for m in months if R <= m < R + 12]
        n_val = int(min(cfg["learn_val_months"], max(6, round(cfg["learn_val_frac"] * len(win)))))
        tr_m, va_m = win[:-n_val], win[-n_val:]
        tr = slice(panel.rows(tr_m[0]).start, panel.rows(tr_m[-1]).stop)
        va = slice(panel.rows(va_m[0]).start, panel.rows(va_m[-1]).stop)
        al = slice(tr.start, va.stop)
        pr = slice(panel.rows(pred_m[0]).start, panel.rows(pred_m[-1]).stop)
        pr_blocks = [panel.rows(m) for m in pred_m]
        va_bounds = [(panel.rows(m).start - va.start, panel.rows(m).stop - va.start) for m in va_m]
        X = panel.X
        year = R // 12
        step = {}
        for name in cfg["learners"]:
            t1 = time.time()
            rng = _seed(cfg, 100 + list(cfg["learners"]).index(name), year)
            y = yr if name in cfg["learn_resid"] else yz
            if name == "ridge":
                p = _ridge_fit_predict(X, y, tr, va, al, pr_blocks, va_bounds, cfg)
            elif name == "xgb":
                p = _xgb_fit_predict(X, y, tr, va, al, pr_blocks, cfg, rng, n_threads)
            elif name == "mlp":  # small mini-batch products: single-threaded BLAS is faster
                with threadpool_limits(limits=1, user_api="blas"):
                    p = _mlp_fit_predict(X, y, tr, va, al, pr_blocks, cfg, rng)
            elif name == "lstm":
                with threadpool_limits(limits=1, user_api="blas"):
                    p = _lstm_fit_predict(panel, y, tr, va, al, pr_blocks, lag_index, cfg, rng)
            else:
                raise ValueError(name)
            preds[name][pr] = p
            step[name] = time.time() - t1
            timing[name] += step[name]
        _log(f"learners refit {R // 12}-12: train {len(tr_m)}m ({tr.stop - tr.start:,} rows), "
             f"val {len(va_m)}m, predict {len(pred_m)}m | "
             + " ".join(f"{k} {v:.0f}s" for k, v in step.items()) + f" | {time.time() - t0:.0f}s total")
    rows = []
    for name in cfg["learners"]:
        for m in months:
            r = panel.rows(m)
            p, t = preds[name][r], panel.y[r]
            if np.isfinite(p).all() and len(p) >= 3 and p.std() > 0 and t.std() > 0:
                rows.append((m, name, float(np.corrcoef(p, t)[0, 1])))
    diag["learner_ic"] = pd.DataFrame(rows, columns=["eom", "learner", "ic"])
    diag["_timing_learners"] = timing
    _log(f"learners done ({time.time() - t0:.0f}s): " + ", ".join(f"{k} {v:.0f}s" for k, v in timing.items()))
    return preds


def _per_month(fn, blocks):
    """Apply fn to each month's row block and concatenate. Matrix products then have the same
    shape whether or not later months exist, so results are bit-identical under truncation."""
    return np.concatenate([np.asarray(fn(b), dtype=np.float64).reshape(-1) for b in blocks])


def _ridge_fit_predict(X, y, tr, va, al, pr_blocks, va_bounds, cfg):
    def gram(sl):
        Xs = X[sl].astype(np.float64)
        return Xs.T @ Xs, Xs.T @ y[sl]

    G_tr, b_tr = gram(tr)
    G_va, b_va = gram(va)
    s, V = np.linalg.eigh(G_tr)
    s = np.clip(s, 0.0, None)
    scale = max(np.trace(G_tr) / G_tr.shape[0], 1e-12)
    Vb = V.T @ b_tr
    Xva = X[va].astype(np.float64)
    best_z, best_ic = cfg["ridge_grid"][0], -np.inf
    for z in cfg["ridge_grid"]:
        beta = V @ (Vb / (s + z * scale))
        ic = _month_ic(Xva @ beta, y[va], va_bounds)
        if ic > best_ic + 1e-12:
            best_z, best_ic = z, ic
    G, bvec = G_tr + G_va, b_tr + b_va  # train and validation blocks are adjacent: al = tr + va
    scale = max(np.trace(G) / G.shape[0], 1e-12)
    beta = np.linalg.solve(G + best_z * scale * np.eye(G.shape[0]), bvec)
    return _per_month(lambda b: X[b].astype(np.float64) @ beta, pr_blocks)


def _xgb_fit_predict(X, y, tr, va, al, pr_blocks, cfg, rng, n_threads):
    # Threads scale with the training size (OpenMP overhead dominates on small samples). The
    # count depends only on the training window, so it is the same in truncated reruns.
    n_threads = int(min(n_threads, max(1, (tr.stop - tr.start) // 10000)))
    dtr = xgb.DMatrix(X[tr], label=y[tr], nthread=n_threads)
    dva = xgb.DMatrix(X[va], label=y[va], nthread=n_threads)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    base = {
        "objective": "reg:squarederror", "tree_method": "hist", "eta": cfg["xgb_eta"],
        "subsample": cfg["xgb_subsample"], "colsample_bytree": cfg["xgb_colsample"],
        "min_child_weight": max(1.0, cfg["xgb_min_child_frac"] * (tr.stop - tr.start)),
        "lambda": cfg["xgb_lambda"], "max_bin": int(cfg["xgb_max_bin"]), "base_score": 0.0,
        "nthread": n_threads, "seed": seed, "verbosity": 0,
    }
    best = None
    for depth in cfg["xgb_depths"]:
        params = dict(base, max_depth=int(depth))
        bst = xgb.train(params, dtr, num_boost_round=int(cfg["xgb_rounds"]),
                        evals=[(dva, "val")], early_stopping_rounds=int(cfg["xgb_early_stop"]),
                        verbose_eval=False)
        score = float(bst.best_score)
        if best is None or score < best[0] - 1e-12:
            best = (score, depth, bst.best_iteration + 1)
    _, depth, n_trees = best
    params = dict(base, max_depth=int(depth))
    dal = xgb.DMatrix(X[al], label=y[al], nthread=n_threads)
    bst = xgb.train(params, dal, num_boost_round=max(int(n_trees), 10))
    return _per_month(lambda b: bst.predict(xgb.DMatrix(X[b], nthread=n_threads)), pr_blocks)


def _mlp_fit_predict(X, y, tr, va, al, pr_blocks, cfg, rng):
    Xtr, ytr = X[tr], y[tr].astype(np.float32)
    Xva, yva = X[va], y[va].astype(np.float32)
    out = 0.0
    for s in range(int(cfg["mlp_seeds"])):
        params = _train_mlp(Xtr, ytr, Xva, yva, cfg, np.random.default_rng(rng.integers(0, 2 ** 63)))
        out = out + _per_month(lambda b: _mlp_forward(params, X[b])[-1][:, 0], pr_blocks)
    return out / cfg["mlp_seeds"]


def _lstm_fit_predict(panel, y, tr, va, al, pr_blocks, lag_index, cfg, rng):
    L, k = int(cfg["lstm_len"]), int(cfg["lstm_inputs"])
    Xtr = panel.X[tr].astype(np.float64)
    cov = Xtr.T @ Xtr / max(len(Xtr), 1)
    _, V = np.linalg.eigh(cov)
    V = V[:, ::-1][:, :min(k, V.shape[1])]
    V = V * np.where(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])] >= 0, 1.0, -1.0)[None, :]
    # Scores for all rows that can appear in a sequence (training window and prediction rows)
    first_m = int(panel.month[tr.start]) - L
    last_m = int(panel.month[pr_blocks[-1].start])
    blocks = [panel.rows(m) for m in panel.months if first_m <= m <= last_m]
    U = np.zeros((len(panel.month), V.shape[1]), dtype=np.float32)
    for b in blocks:  # per month, so shapes do not depend on later data
        U[b] = (panel.X[b].astype(np.float64) @ V).astype(np.float32)
    sd = U[tr].std(axis=0)
    for b in blocks:
        U[b] /= np.where(sd > 1e-12, sd, 1.0)[None, :].astype(np.float32)
    tr_rows = np.arange(tr.start, tr.stop)
    n_sub = int(round(cfg["lstm_subsample"] * len(tr_rows)))
    if 0 < n_sub < len(tr_rows):
        tr_rows = np.sort(rng.choice(tr_rows, n_sub, replace=False))
    va_rows = np.arange(va.start, va.stop)
    Utr, ytr = _sequences(panel, tr_rows, U, lag_index, L), y[tr_rows].astype(np.float32)
    Uva, yva = _sequences(panel, va_rows, U, lag_index, L), y[va_rows].astype(np.float32)
    Upr = [_sequences(panel, np.arange(b.start, b.stop), U, lag_index, L) for b in pr_blocks]
    out = 0.0
    for s in range(int(cfg["lstm_seeds"])):
        net = _train_lstm(Utr, ytr, Uva, yva, cfg, np.random.default_rng(rng.integers(0, 2 ** 63)))
        out = out + np.concatenate([net.forward(u)[0] for u in Upr])
    return out / cfg["lstm_seeds"]


# ----------------------------------------------------------------------------------------
# Section 4-6: sleeves, meta-combination and volatility timing (one chronological pass)
# ----------------------------------------------------------------------------------------

def _rm_alpha(Kmat, zeta):
    """Dual ridge-Markowitz coefficients: b = F' alpha with alpha = (FF' + z I)^-1 1."""
    T = Kmat.shape[0]
    scale = max(np.trace(Kmat) / T, 1e-18)
    return np.linalg.solve(Kmat + zeta * scale * np.eye(T), np.ones(T))


def _rm_cv(Kmat, grid, folds):
    """Blocked K-fold CV of the ridge-Markowitz shrinkage: maximize the pooled OOS Sharpe."""
    T = Kmat.shape[0]
    parts = np.array_split(np.arange(T), folds)
    oos = np.zeros((len(grid), T))
    for idx in parts:
        trn = np.setdiff1d(np.arange(T), idx)
        Ktr = Kmat[np.ix_(trn, trn)]
        s, V = np.linalg.eigh(Ktr)
        scale = max(np.trace(Ktr) / len(trn), 1e-18)
        Vt1 = V.T @ np.ones(len(trn))
        cross = Kmat[np.ix_(idx, trn)] @ V
        for g, z in enumerate(grid):
            oos[g, idx] = cross @ (Vt1 / (s + z * scale))
    sr = oos.mean(axis=1) / np.maximum(oos.std(axis=1), 1e-18)
    return float(grid[int(np.argmax(sr))])


def _signals(X, Wg):
    """Characteristics plus random Fourier features, demeaned across stocks, plus a constant.

    Wg holds the random directions already multiplied by their bandwidths."""
    n, d = X.shape
    P = Wg.shape[1]
    Xd = X.astype(np.float64)
    Z = Xd @ Wg
    S = np.empty((n, d + 2 * P + 1))
    S[:, :d] = Xd
    np.sin(Z, out=S[:, d:d + P])
    np.cos(Z, out=S[:, d + P:d + 2 * P])
    S[:, :-1] -= S[:, :-1].mean(axis=0, keepdims=True)
    S[:, -1] = 1.0
    return S


def _unit_vol(w, risk_t):
    v = risk_t.var(w) * DAYS_PER_MONTH
    return w / np.sqrt(v) if np.isfinite(v) and v > 1e-300 else None


def _learner_mus(preds, r, rk, cfg, learners):
    """Expected-return vectors for inv(Sigma), one per learner with a usable forecast in rows r.
    Each forecast is z-scored across stocks; a standardized-alpha forecast (learn_resid) is put
    back in return units by the specific volatility."""
    mus = {}
    for k in learners:
        p = preds[k][r]
        if np.isfinite(p).all() and p.std() > 0:
            z = (p - p.mean()) / p.std()
            mus[k] = z * np.sqrt(rk.spec) if k in cfg["learn_resid"] else z
    return mus


def _meta_weights(R, cfg):
    K = R.shape[1]
    R = R[np.isfinite(R).all(axis=1)]
    if len(R) == 0:
        return np.full(K, 1.0 / K)
    th, _ = nnls(R, np.ones(R.shape[0]))
    th = th / th.sum() if th.sum() > 0 else np.full(K, 1.0 / K)
    a = cfg["meta_shrink"]
    return (1.0 - a) * th + a / K


def build_portfolios(panel, daily, risk, preds, cfg, diag):
    t0 = time.time()
    months = [int(m) for m in panel.months]
    d = len(panel.features)
    rng = _seed(cfg, 7)
    P = int(cfg["sdf_rff"])
    W = rng.standard_normal((d, P))
    gam = np.array([cfg["sdf_gammas"][p % len(cfg["sdf_gammas"])] for p in range(P)])
    zscale = np.sqrt(d / 12.0)  # ranks have variance 1/12, so omega'x / zscale has unit scale
    Wg = W * (gam / zscale)[None, :]
    learners = list(cfg["learners"])
    ens = cfg["b_sleeves"] == "ens"
    sleeves = ["A0", "A1"] + (["B_ens"] if ens else [f"B_{k}" for k in learners])
    F_hist = {"A0": [], "A1": []}  # (month, managed-portfolio return vector)
    zeta = {"A0": None, "A1": None}
    sret = {s: {} for s in sleeves}  # month -> realized sleeve return
    out_w = np.zeros(len(panel.month))
    zeta_rows, meta_rows, book_rows, ret_rows = [], [], [], []
    scales = []
    target_m = cfg["vol_target"] / np.sqrt(12.0)
    lam_v = cfg["vol_lambda"]

    for i_t, t in enumerate(months):
        r = panel.rows(t)
        rk = risk[t]
        y = panel.y[r]
        X = panel.X[r]
        n = X.shape[0]
        sw = {}  # sleeve -> unit-vol weights at t
        if rk.ok and n >= 5:
            S = _signals(X, Wg)
            var0 = rk.var(S) * DAYS_PER_MONTH
            G = rk.solve(S)
            var1 = (S * G).sum(axis=0) * DAYS_PER_MONTH
            mats = {}
            for name, M, v in (("A0", S, var0), ("A1", G, var1)):
                good = np.isfinite(v) & (v > 1e-300)
                M = M * np.where(good, 1.0 / np.sqrt(np.where(good, v, 1.0)), 0.0)[None, :]
                mats[name] = M
                hist = [(m, f) for m, f in F_hist[name] if t - cfg["sdf_window"] <= m < t]
                if len(hist) >= cfg["sdf_min_months"]:
                    Fm = np.array([f for _, f in hist])
                    Kmat = Fm @ Fm.T
                    if zeta[name] is None or t % 12 == 11:
                        zeta[name] = _rm_cv(Kmat, np.array(cfg["sdf_zeta_grid"]), int(cfg["sdf_folds"]))
                        zeta_rows.append((t, name, zeta[name]))
                    b = Fm.T @ _rm_alpha(Kmat, zeta[name])
                    w = _unit_vol(M @ b, rk)
                    if w is not None:
                        sw[name] = w
            mus = _learner_mus(preds, r, rk, cfg, learners)
            if ens and mus:  # forecast combination: equal-weight average of the standardized forecasts
                mus = {"ens": np.mean([m / m.std() for m in mus.values()], axis=0)}
            for k, mu in mus.items():
                w = _unit_vol(rk.solve(mu), rk)
                if w is not None:
                    sw[f"B_{k}"] = w
            # Managed-portfolio returns realized over (t, t+1]: used only from t+1 on
            for name in ("A0", "A1"):
                F_hist[name].append((t, mats[name].T @ y))
            del S, G, mats

        # Meta-combination from sleeve returns realized before t
        avail = [s for s in sleeves if s in sw]
        book = None
        if avail:
            hist_m = [m for m in months if t - cfg["meta_window"] <= m < t
                      and all(m in sret[s] for s in avail)]
            if len(hist_m) >= cfg["meta_min_months"] and len(avail) > 1:
                Rh = np.array([[sret[s][m] for s in avail] for m in hist_m])
                th = _meta_weights(Rh, cfg)
            else:
                th = np.full(len(avail), 1.0 / len(avail))
            book = sum(th[j] * sw[s] for j, s in enumerate(avail))
            meta_rows.extend((t, s, float(th[j])) for j, s in enumerate(avail))
        for s, w in sw.items():
            sret[s][t] = float(w @ y)
            ret_rows.append((t, s, sret[s][t]))
        if book is None or not np.isfinite(book).all() or np.abs(book).sum() == 0:
            book = _fallback_weights(preds, r, n)
            v = rk.var(book) * DAYS_PER_MONTH  # same units as the sleeves: unit ex-ante vol
            if np.isfinite(v) and v > 0:
                book = book / np.sqrt(v)
        # Volatility timing: forecast from today's book applied to the past year of daily returns
        scale, vs, vf = _vol_scale(book, panel, daily, rk, t, cfg, target_m, lam_v, scales)
        w_final = book * scale
        if not np.isfinite(w_final).all():  # degenerate month: never emit non-finite weights
            _log(f"warning: non-finite weights in month {t // 12}-{t % 12 + 1:02d}; using the fallback book")
            book = _fallback_weights(preds, r, n)
            v = rk.var(book) * DAYS_PER_MONTH
            book = book / np.sqrt(v) if np.isfinite(v) and v > 0 else book
            w_final = np.nan_to_num(book * target_m, nan=0.0, posinf=0.0, neginf=0.0)
        out_w[r] = w_final
        if (i_t + 1) % 60 == 0:
            _log(f"portfolios: through {t // 12}-{t % 12 + 1:02d} ({time.time() - t0:.0f}s)")
        book_rows.append((t, scale, vs, vf, float(np.abs(w_final).sum()), float(w_final.sum()),
                          int((w_final != 0).sum()), float(w_final @ y)))

    diag["sleeve_returns"] = pd.DataFrame(ret_rows, columns=["eom", "sleeve", "ret"])
    diag["meta_weights"] = pd.DataFrame(meta_rows, columns=["eom", "sleeve", "theta"])
    diag["sdf_zeta"] = pd.DataFrame(zeta_rows, columns=["eom", "sleeve", "zeta"])
    diag["book"] = pd.DataFrame(book_rows, columns=["eom", "scale", "vol_synth", "vol_fm", "gross",
                                                    "net", "n_stocks", "ret"])
    _log(f"portfolios: {len(months)} months ({time.time() - t0:.0f}s)")
    return out_w


def _fallback_weights(preds, r, n):
    """Average of available learner forecasts as a long-short book; equal weights if none."""
    mus = [preds[k][r] for k in preds if np.isfinite(preds[k][r]).all() and preds[k][r].std() > 0]
    if mus:
        z = np.mean([(m - m.mean()) / m.std() for m in mus], axis=0)
        if np.abs(z).sum() > 0:
            return z / np.abs(z).sum()
    return np.full(n, 1.0 / max(n, 1))


def _vol_scale(book, panel, daily, rk, t, cfg, target_m, lam_v, scales):
    hi = daily.me[t]
    vf = rk.var(book) * DAYS_PER_MONTH if rk is not None else np.nan
    vs = np.nan
    if hi >= 0:
        lo = max(0, hi - int(cfg["vol_days"]) + 1)
        Rd = daily.R[lo:hi + 1][:, panel.code[panel.rows(t)]].astype(np.float64)
        valid = ~np.isnan(Rd)
        nobs = valid.sum(axis=0)
        gross = np.abs(book).sum()
        covered = np.abs(book)[nobs >= int(cfg["vol_days"]) // 2].sum() / gross if gross > 0 else 0.0
        if covered >= cfg["vol_min_cov"] and Rd.shape[0] >= 21:
            br = np.where(valid, Rd, 0.0) @ book
            wts = lam_v ** np.arange(len(br) - 1, -1, -1, dtype=np.float64)
            vs = DAYS_PER_MONTH * float(wts @ (br * br) / wts.sum())
    v = vs if np.isfinite(vs) and vs > 0 else vf
    if not np.isfinite(v) or v <= 0:
        v = float(np.abs(book).sum()) ** 2 * 1e-3
    scale = target_m / np.sqrt(v)
    past = [s for m, s in scales if t - cfg["vol_cap_months"] <= m < t]
    if len(past) >= 12:
        scale = min(scale, cfg["vol_cap_mult"] * float(np.median(past)))
    scales.append((t, scale))
    return scale, (np.sqrt(vs) * np.sqrt(12.0) if np.isfinite(vs) else np.nan), \
        (np.sqrt(vf) * np.sqrt(12.0) if np.isfinite(vf) else np.nan)


# ----------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------

def run_model(chars, features, daily_ret, config=None, diagnostics=False):
    """Full pipeline. Returns (weights DataFrame, diagnostics dict)."""
    _T0[0] = time.time()
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    np.random.seed(int(cfg["seed"]) % (2 ** 32))
    cpus = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    n_threads = int(cfg["n_threads"]) or int(min(32, cpus or 1))
    diag = {}
    timing = []
    t = time.time()
    panel = Panel(chars, features)
    timing.append(("panel", time.time() - t))
    t = time.time()
    daily = Daily(daily_ret, panel)
    timing.append(("daily", time.time() - t))
    t = time.time()
    # BLAS threads are set explicitly per stage: many small solves are fastest single-threaded,
    # and an uncapped pool oversubscribes when the container quota is below the host CPU count
    with threadpool_limits(limits=1, user_api="blas"):
        risk = build_risk_model(panel, daily, cfg, diag)
    timing.append(("risk_model", time.time() - t))
    t = time.time()
    with threadpool_limits(limits=n_threads, user_api="blas"):
        preds = fit_learners(panel, cfg, diag, n_threads, risk)
    timing.append(("learners", time.time() - t))
    t = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        w = build_portfolios(panel, daily, risk, preds, cfg, diag)
    timing.append(("portfolios", time.time() - t))

    test = panel.test
    eom_src = chars["eom"].reset_index(drop=True).to_numpy()[panel.src[test]]
    out = pd.DataFrame({"id": panel.ids[test].astype(np.int64), "eom": eom_src,
                        "w": np.round(w[test].astype(np.float64), 12)})  # 12 decimals: smaller CSV
    if len(out) == 0:
        raise ValueError("no ctff_test rows in chars")
    for key in ("sleeve_returns", "meta_weights", "sdf_zeta", "book", "learner_ic"):
        if key in diag and len(diag[key]):
            diag[key]["eom"] = _month_end_date(diag[key]["eom"].to_numpy())
    if "themes" in diag:
        diag["themes"]["refit_eom"] = _month_end_date(diag["themes"]["refit_eom"].to_numpy())
    diag.pop("_timing_learners", None)
    diag["timing"] = pd.DataFrame(timing, columns=["step", "seconds"])
    diag["config"] = pd.DataFrame([(k, str(v)) for k, v in cfg.items()], columns=["key", "value"])
    gross = out.groupby(panel.month[test])["w"].apply(lambda s: s.abs().sum())
    _log(f"output: {len(out):,} rows, {out['eom'].nunique()} months, mean gross leverage "
         f"{gross.mean():.2f}; total {time.time() - _T0[0]:.0f}s")
    return out, diag


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """CTF entry point: monthly portfolio weights (id, eom, w) for every ctff_test row."""
    np.random.seed(CONFIG["seed"] % (2 ** 32))
    weights, _ = run_model(chars, features, daily_ret, diagnostics=False)
    return weights


if __name__ == "__main__":
    chars = pd.read_parquet("ctff_chars.parquet")
    features = pd.read_parquet("ctff_features.parquet")
    daily_ret = pd.read_parquet("ctff_daily_ret.parquet")
    pf = main(chars, features, daily_ret)
    pf.to_csv("output.csv", index=False)
