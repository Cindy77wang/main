"""Question 1: ARMA models for four JKP factor returns (1975-2025).

Steps (each written from scratch with numpy; scipy only for the optimiser and
chi-square / normal tail probabilities):
  1. Sample ACF and PACF (Durbin-Levinson) for lags 1-12, with Bartlett / 1/sqrt(T)
     bands, individual z-tests and Ljung-Box Q(m) tests; plot them.
     A rule-of-thumb picks the ARMA(p,q) "of choice" from the ACF/PACF cut-offs
     (can be overridden in CHOICE below).
  2. Estimate AR(1), ARMA(1,1) and the chosen ARMA(p,q) by
       (a) regression: OLS for pure AR; Hannan-Rissanen two-stage regression when
           there is an MA part (long AR -> residuals -> OLS on lagged x and residuals)
       (b) exact Gaussian MLE via the prediction-error decomposition (Kalman filter
           with the stationary initial state covariance); stationarity and
           invertibility are imposed with the Jones/Monahan partial-autocorrelation
           transform. For AR(1) the closed-form exact likelihood is used as a check.
  3. Re-estimate on 1975-2015, produce one-step-ahead forecasts for 2016-2025 with
     the parameters held fixed, and compare MSE, out-of-sample R^2 versus the
     training-sample mean and Diebold-Mariano tests.
  4. Next-month expected return from the preferred full-sample model.

DATA. Download the factor returns from https://jkpfactors.com/data (region: USA,
frequency: monthly, weighting: e.g. capped value-weighted) and save the CSV file(s)
in inputs/jkp/. Both the long JKP layout (columns ... name, date, ret ...) and a
wide layout (a 'date' column plus one column per factor) are read.
Set JKP_DIR / Q1_RESULTS environment variables to use other folders.

Run:  python q1_arma.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from common import (HERE, INK, MUTED, SERIES_COLORS, newey_west, plt, style_axes,
                    write_markdown_table)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FACTORS = {
    "ret_1_0": "Short-term reversal",
    "ret_12_1": "Momentum (12-1)",
    "be_me": "Value (book-to-market)",
    "betabab_1260d": "Betting against beta",
}
START, END = "1975-01-01", "2025-12-31"
TRAIN_END = "2015-12-31"
NLAGS = 12
CHOICE: dict[str, tuple[int, int]] = {}  # e.g. {"ret_12_1": (1, 0)} to override the rule
JKP_DIR = Path(os.environ.get("JKP_DIR", HERE / "inputs" / "jkp"))
OUT = Path(os.environ.get("Q1_RESULTS", HERE / "results"))
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_jkp() -> pd.DataFrame:
    files = sorted(glob.glob(str(JKP_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {JKP_DIR}. Download monthly USA factor returns from "
            "https://jkpfactors.com/data and save them there.")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        if {"name", "date", "ret"}.issubset(d.columns):
            if "location" in d.columns:
                d = d[d["location"].str.lower() == "usa"]
            if "freq" in d.columns:
                d = d[d["freq"].str.lower() == "monthly"]
            w = d.pivot_table(index="date", columns="name", values="ret", aggfunc="first")
        else:
            w = d.set_index("date")
        w.index = pd.to_datetime(w.index)
        frames.append(w)
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()].sort_index()
    missing = [f for f in FACTORS if f not in df.columns]
    if missing:
        raise KeyError(f"Factors {missing} not found; available: {list(df.columns)[:20]} ...")
    df = df[list(FACTORS)].loc[START:END].dropna()
    return df


# ---------------------------------------------------------------------------
# ACF / PACF / Ljung-Box
# ---------------------------------------------------------------------------
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = x @ x
    return np.array([1.0] + [x[k:] @ x[:-k] / d for k in range(1, nlags + 1)])


def pacf_dl(r, nlags):
    """Durbin-Levinson recursion: partial autocorrelations from autocorrelations."""
    pac = np.zeros(nlags + 1)
    pac[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - phi[k - 1, 1:k] @ r[1:k][::-1]
        den = 1 - phi[k - 1, 1:k] @ r[1:k]
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        pac[k] = phi[k, k]
    return pac


def correlogram(x, nlags=NLAGS):
    T = len(x)
    r = acf(x, nlags)
    p = pacf_dl(r, nlags)
    lags = np.arange(1, nlags + 1)
    se_acf_bartlett = np.sqrt((1 + 2 * np.cumsum(np.r_[0, r[1:nlags] ** 2])) / T)
    se_white = 1 / np.sqrt(T)
    Q = T * (T + 2) * np.cumsum(r[1:] ** 2 / (T - lags))
    return pd.DataFrame({
        "lag": lags, "acf": r[1:], "acf_z": r[1:] / se_white,
        "acf_se_bartlett": se_acf_bartlett,
        "acf_p": 2 * stats.norm.sf(np.abs(r[1:]) / se_white),
        "pacf": p[1:], "pacf_z": p[1:] / se_white,
        "pacf_p": 2 * stats.norm.sf(np.abs(p[1:]) / se_white),
        "ljung_box_Q": Q, "ljung_box_p": stats.chi2.sf(Q, lags),
    }).set_index("lag")


def choose_order(cg: pd.DataFrame, maxlag=4):
    """Rule of thumb from the correlogram: the last significant PACF lag among the first
    `maxlag` lags gives p*, the last significant ACF lag gives q*. Pick the more
    parsimonious of AR(p*) and MA(q*). If nothing is significant but Ljung-Box(12)
    rejects, use ARMA(1,1) (slowly decaying low-level dependence)."""
    sig_a = cg.index[(cg.acf_p < 0.05) & (cg.index <= maxlag)]
    sig_p = cg.index[(cg.pacf_p < 0.05) & (cg.index <= maxlag)]
    q_star = int(sig_a.max()) if len(sig_a) else 0
    p_star = int(sig_p.max()) if len(sig_p) else 0
    if p_star == 0 and q_star == 0:
        if cg.ljung_box_p.iloc[-1] < 0.05:
            return (1, 1), "no single low-order lag is significant but Ljung-Box(12) rejects"
        return (0, 0), "no significant autocorrelation; returns look like white noise"
    if p_star and (p_star <= q_star or q_star == 0):
        return (p_star, 0), f"PACF cuts off after lag {p_star} while the ACF decays"
    return (0, q_star), f"ACF cuts off after lag {q_star} while the PACF decays"


# ---------------------------------------------------------------------------
# Parameter transforms (stationarity / invertibility)
# ---------------------------------------------------------------------------
def pacf_to_ar(u):
    """Unconstrained u -> partial autocorrelations tanh(u) -> stationary AR coefs."""
    r = np.tanh(np.asarray(u, float))
    k = len(r)
    a = np.zeros(k)
    for j in range(k):
        a_new = a.copy()
        a_new[j] = r[j]
        for i in range(j):
            a_new[i] = a[i] - r[j] * a[j - 1 - i]
        a = a_new
    return a


def ar_to_pacf(a):
    """Inverse of pacf_to_ar (step-down recursion)."""
    a = np.array(a, float)
    k = len(a)
    r = np.zeros(k)
    for j in range(k - 1, -1, -1):
        r[j] = a[j]
        if j == 0:
            break
        den = 1 - r[j] ** 2
        a = np.array([(a[i] + r[j] * a[j - 1 - i]) / den for i in range(j)])
    return np.arctanh(np.clip(r, -0.999, 0.999))


def is_stationary(phi):
    if len(phi) == 0:
        return True
    roots = np.roots(np.r_[1, -np.asarray(phi)][::-1])
    return np.all(np.abs(roots) > 1.0 + 1e-8)


# ---------------------------------------------------------------------------
# Exact Gaussian likelihood via the Kalman filter
# ---------------------------------------------------------------------------
def ss_matrices(phi, theta):
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    Tm = np.zeros((r, r))
    Tm[:p, 0] = phi
    Tm[:-1, 1:] += np.eye(r - 1) if r > 1 else 0
    R = np.zeros(r)
    R[0] = 1.0
    R[1:q + 1] = theta
    return Tm, R


def kalman(x, mu, phi, theta, sigma2=1.0):
    """Prediction-error decomposition. Returns one-step predictions xhat_t = E[x_t|x_{1:t-1}],
    innovations v_t and their variances F_t (in units of sigma2)."""
    Tm, R = ss_matrices(phi, theta)
    r = len(R)
    Q = np.outer(R, R) * sigma2
    vecP = np.linalg.solve(np.eye(r * r) - np.kron(Tm, Tm), Q.ravel())
    P = vecP.reshape(r, r)
    a = np.zeros(r)
    n = len(x)
    xhat, v, F = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        xhat[t] = mu + a[0]
        F[t] = P[0, 0]
        v[t] = x[t] - xhat[t]
        K = P[:, 0] / F[t]
        a = Tm @ (a + K * v[t])
        P = Tm @ (P - np.outer(K, P[0, :])) @ Tm.T + Q
    return xhat, v, F


def exact_loglik(x, mu, phi, theta, sigma2):
    """l = -T/2 log(2 pi) - 1/2 sum log(F_t) - 1/2 sum v_t^2/F_t
       = -T/2 log(2 pi) - 1/2 log|Sigma| - 1/2 (x - mu)' Sigma^{-1} (x - mu)."""
    _, v, F = kalman(x, mu, phi, theta, sigma2)
    return -0.5 * (len(x) * np.log(2 * np.pi) + np.sum(np.log(F)) + np.sum(v**2 / F))


def ar1_exact_loglik(x, mu, phi, sigma2):
    """Closed form exact AR(1) log-likelihood (check on the Kalman filter)."""
    T = len(x)
    e = (x[1:] - mu) - phi * (x[:-1] - mu)
    return (-0.5 * np.log(2 * np.pi * sigma2 / (1 - phi**2))
            - (1 - phi**2) * (x[0] - mu) ** 2 / (2 * sigma2)
            - (T - 1) / 2 * np.log(2 * np.pi * sigma2) - e @ e / (2 * sigma2))


def unpack(z, p, q):
    mu = z[0]
    phi = pacf_to_ar(z[1:1 + p]) if p else np.zeros(0)
    theta = -pacf_to_ar(z[1 + p:1 + p + q]) if q else np.zeros(0)
    return mu, phi, theta


def mle(x, p, q, start=None):
    """Exact Gaussian MLE; sigma2 is concentrated out of the likelihood."""
    x = np.asarray(x, float)
    T = len(x)

    def negll(z):
        mu, phi, theta = unpack(z, p, q)
        _, v, F = kalman(x, mu, phi, theta, 1.0)
        s2 = np.mean(v**2 / F)
        return 0.5 * (T * (np.log(2 * np.pi) + 1 + np.log(s2)) + np.sum(np.log(F)))

    starts = []
    if start is not None:
        mu0, phi0, th0 = start
        z = [mu0]
        z += list(ar_to_pacf(phi0)) if p and is_stationary(phi0) else [0.0] * p
        z += list(ar_to_pacf(-np.asarray(th0))) if q and is_stationary(-np.asarray(th0)) else [0.0] * q
        starts.append(np.array(z, float))
    starts.append(np.r_[x.mean(), np.zeros(p + q)])
    starts.append(np.r_[x.mean(), 0.2 * np.ones(p), -0.2 * np.ones(q)])
    best = None
    for z0 in starts:
        res = optimize.minimize(negll, z0, method="BFGS", options=dict(gtol=1e-8, maxiter=5000))
        res2 = optimize.minimize(negll, res.x, method="Nelder-Mead",
                                 options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
        r = res2 if res2.fun < res.fun else res
        if best is None or r.fun < best.fun:
            best = r
    mu, phi, theta = unpack(best.x, p, q)
    _, v, F = kalman(x, mu, phi, theta, 1.0)
    s2 = np.mean(v**2 / F)
    params = np.r_[mu, phi, theta, s2]
    ll = exact_loglik(x, mu, phi, theta, s2)
    # standard errors: numerical Hessian of the exact log-likelihood in natural parameters
    def ll_nat(th):
        ph, tt = th[1:1 + p], th[1 + p:1 + p + q]
        if th[-1] <= 0 or not is_stationary(ph):
            return -1e12
        return exact_loglik(x, th[0], ph, tt, th[-1])
    H = num_hessian(ll_nat, params)
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(params), np.nan)
    return dict(params=params, se=se, loglik=ll, mu=mu, phi=phi, theta=theta, sigma2=s2,
                aic=-2 * ll + 2 * len(params), bic=-2 * ll + np.log(T) * len(params))


def num_hessian(f, x, rel=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                 + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


# ---------------------------------------------------------------------------
# Regression estimators
# ---------------------------------------------------------------------------
def ols_fit(y, X):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(s2 * XtXi)), e, s2


def lagmat(x, lags, start):
    return np.column_stack([x[start - k:len(x) - k] for k in range(1, lags + 1)]) if lags else \
        np.zeros((len(x) - start, 0))


def regression(x, p, q, long_ar=None):
    """OLS for AR(p); Hannan-Rissanen for ARMA(p,q).
    Returns intercept c, phi, theta, sigma2 and OLS standard errors, plus mu = c/(1-sum phi)."""
    x = np.asarray(x, float)
    if q == 0:
        s = p
        X = np.column_stack([np.ones(len(x) - s), lagmat(x, p, s)])
        b, se, e, s2 = ols_fit(x[s:], X)
        phi, theta = b[1:], np.zeros(0)
    else:
        m = long_ar or max(12, int(np.ceil(np.log(len(x)) ** 2)))
        Xl = np.column_stack([np.ones(len(x) - m), lagmat(x, m, m)])
        bl, _, el, _ = ols_fit(x[m:], Xl)
        ehat = np.r_[np.zeros(m), el]
        s = m + max(p, q)
        X = np.column_stack([np.ones(len(x) - s), lagmat(x, p, s), lagmat(ehat, q, s)])
        b, se, e, s2 = ols_fit(x[s:], X)
        phi, theta = b[1:1 + p], b[1 + p:]
    c = b[0]
    mu = c / (1 - phi.sum()) if p else c
    return dict(c=c, phi=phi, theta=theta, sigma2=s2, b=b, se=se, mu=mu, nobs=len(x) - s)


def recursive_forecasts(x, c, phi, theta):
    """One-step forecasts xhat_t = c + sum phi_i x_{t-i} + sum theta_j e_{t-j},
    e_t = x_t - xhat_t, pre-sample errors and values set to their means."""
    x = np.asarray(x, float)
    p, q = len(phi), len(theta)
    mu = c / (1 - phi.sum()) if p and abs(1 - phi.sum()) > 1e-8 else c
    xs = np.r_[np.full(p, mu), x]
    e = np.zeros(q + len(x))
    xhat = np.empty(len(x))
    for t in range(len(x)):
        ar = sum(phi[i] * xs[p + t - 1 - i] for i in range(p))
        ma = sum(theta[j] * e[q + t - 1 - j] for j in range(q))
        xhat[t] = c + ar + ma
        e[q + t] = x[t] - xhat[t]
    return xhat


# ---------------------------------------------------------------------------
# Forecast evaluation
# ---------------------------------------------------------------------------
def diebold_mariano(e1, e2):
    """DM test of equal MSE; d_t = e1^2 - e2^2, HAC variance. Negative => model 1 better."""
    d = e1**2 - e2**2
    T = len(d)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lrv = newey_west(d[:, None], lags)[0, 0]
    stat = d.mean() / np.sqrt(lrv / T)
    return stat, 2 * stats.norm.sf(abs(stat))


def clark_west(y, f_bench, f_model):
    """Clark-West (2007) adjusted MSPE test for nested models (one-sided)."""
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    T = len(adj)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    stat = adj.mean() / np.sqrt(newey_west(adj[:, None], lags)[0, 0] / T)
    return stat, stats.norm.sf(stat)


def label(p, q):
    if q == 0 and p == 0:
        return "ARMA(0,0)"
    if q == 0:
        return f"AR({p})"
    if p == 0:
        return f"MA({q})"
    return f"ARMA({p},{q})"


# ---------------------------------------------------------------------------
def main():
    df = load_jkp()
    df.to_csv(OUT / "q1_factor_returns_used.csv")
    print(f"Sample {df.index.min():%Y-%m} to {df.index.max():%Y-%m}, T = {len(df)}")
    summ = df.describe().T[["count", "mean", "std", "min", "max"]]
    summ["ann. Sharpe"] = df.mean() / df.std() * np.sqrt(12)
    write_markdown_table(summ, OUT / "q1_summary_stats.md")

    # 1. Correlograms ---------------------------------------------------------
    choices, reasons, cgs = {}, {}, {}
    for f in FACTORS:
        cg = correlogram(df[f].to_numpy())
        cgs[f] = cg
        cg.to_csv(OUT / f"q1_correlogram_{f}.csv")
        (pq, why) = choose_order(cg)
        choices[f] = CHOICE.get(f, pq)
        reasons[f] = why if f not in CHOICE else "manual choice"
    cg_all = pd.concat({f: cgs[f][["acf", "acf_p", "pacf", "pacf_p", "ljung_box_Q", "ljung_box_p"]]
                        for f in FACTORS}, axis=1)
    cg_all.to_csv(OUT / "q1_correlograms.csv")
    with open(OUT / "q1_correlograms.md", "w") as fh:
        for f in FACTORS:
            fh.write(f"\n### {FACTORS[f]} (`{f}`)\n\n")
            t = cgs[f][["acf", "acf_z", "acf_p", "pacf", "pacf_z", "pacf_p", "ljung_box_Q", "ljung_box_p"]]
            fh.write(write_markdown_table(t, OUT / f"q1_correlogram_{f}.md", ".3f"))
        fh.write("\n### Model expected from the correlogram\n\n")
        for f in FACTORS:
            fh.write(f"- **{FACTORS[f]}**: {label(*choices[f])} - {reasons[f]}.\n")

    T = len(df)
    band = 1.96 / np.sqrt(T)
    fig, axes = plt.subplots(2, len(FACTORS), figsize=(13, 5.2), sharey="row")
    for j, f in enumerate(FACTORS):
        for i, (col, ttl) in enumerate((("acf", "ACF"), ("pacf", "PACF"))):
            ax = axes[i, j]
            vals = cgs[f][col].to_numpy()
            ax.bar(cgs[f].index, vals, width=0.6, color=SERIES_COLORS[j])
            ax.axhspan(-band, band, color="#e4e3dd", alpha=0.7, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(1, NLAGS + 1))
            ax.set_title(f"{FACTORS[f]}: {ttl}", loc="left", fontsize=9)
            style_axes(ax)
    axes[1, 0].set_xlabel("lag (months)")
    fig.text(0.01, 0.005, "Shaded band: ±1.96/√T", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q1_acf_pacf.png", dpi=160)

    # 2. Full-sample estimation ----------------------------------------------
    rows = []
    fits_full = {}
    for f in FACTORS:
        x = df[f].to_numpy()
        models = [(1, 0), (1, 1)]
        if choices[f] not in models:
            models.append(choices[f])
        for (p, q) in models:
            reg = regression(x, p, q)
            ml = mle(x, p, q, start=(reg["mu"], reg["phi"], reg["theta"]))
            fits_full[(f, p, q)] = (reg, ml)
            names = ["mu"] + [f"phi{i+1}" for i in range(p)] + [f"theta{j+1}" for j in range(q)] + ["sigma2"]
            reg_vals = np.r_[reg["mu"], reg["phi"], reg["theta"], reg["sigma2"]]
            reg_se = np.r_[np.nan, reg["se"][1:], np.nan]
            for k, nm in enumerate(names):
                rows.append(dict(factor=f, model=label(p, q), param=nm,
                                 regression=reg_vals[k], reg_se=reg_se[k],
                                 mle=ml["params"][k], mle_se=ml["se"][k]))
            rows.append(dict(factor=f, model=label(p, q), param="logL / BIC",
                             regression=np.nan, reg_se=np.nan, mle=ml["loglik"], mle_se=ml["bic"]))
            if (p, q) == (1, 0):  # closed-form exact AR(1) check of the Kalman filter
                chk = ar1_exact_loglik(x, ml["mu"], ml["phi"][0], ml["sigma2"])
                assert abs(chk - ml["loglik"]) < 1e-6, (chk, ml["loglik"])
    est = pd.DataFrame(rows)
    est.to_csv(OUT / "q1_estimates.csv", index=False)
    e2 = est.copy()
    e2.index = e2.pop("factor") + " " + e2.pop("model") + " " + e2.pop("param")
    e2.index.name = "factor model param"
    write_markdown_table(e2, OUT / "q1_estimates.md", ".4f")

    # 3. Out-of-sample -------------------------------------------------------
    oos_rows, cum = [], {}
    for f in FACTORS:
        x_all = df[f]
        x_tr = x_all.loc[:TRAIN_END].to_numpy()
        test_mask = x_all.index > pd.Timestamp(TRAIN_END)
        y_te = x_all[test_mask].to_numpy()
        f_mean = np.full(len(y_te), x_tr.mean())
        models = [(1, 0), (1, 1)] + ([choices[f]] if choices[f] not in [(1, 0), (1, 1)] else [])
        preds = {"historical mean": f_mean}
        for (p, q) in models:
            reg = regression(x_tr, p, q)
            ml = mle(x_tr, p, q, start=(reg["mu"], reg["phi"], reg["theta"]))
            xh_ml, _, _ = kalman(x_all.to_numpy(), ml["mu"], ml["phi"], ml["theta"])
            xh_reg = recursive_forecasts(x_all.to_numpy(), reg["c"], reg["phi"], reg["theta"])
            preds[f"{label(p, q)} regression"] = xh_reg[test_mask]
            preds[f"{label(p, q)} MLE"] = xh_ml[test_mask]
        mse_bench = np.mean((y_te - f_mean) ** 2)
        e_ar1 = y_te - preds["AR(1) MLE"]
        for nm, fc in preds.items():
            e = y_te - fc
            mse = np.mean(e**2)
            rec = dict(factor=f, model=nm, chosen=nm.startswith(label(*choices[f]) + " "),
                       MSE=mse, RMSE=np.sqrt(mse), MAE=np.mean(np.abs(e)),
                       R2_oos=1 - mse / mse_bench)
            if nm != "historical mean":
                rec["DM vs mean"], rec["DM p"] = diebold_mariano(e, y_te - f_mean)
                rec["CW vs mean"], rec["CW p"] = clark_west(y_te, f_mean, fc)
                if not nm.startswith("AR(1) MLE"):
                    rec["DM vs AR(1) MLE"], rec["DM p (AR1)"] = diebold_mariano(e, e_ar1)
            oos_rows.append(rec)
            cum[(f, nm)] = np.cumsum((y_te - f_mean) ** 2 - e**2)
    oos = pd.DataFrame(oos_rows)
    oos.to_csv(OUT / "q1_oos.csv", index=False)
    o2 = oos.copy()
    o2.index = o2.pop("factor") + " | " + o2.pop("model")
    o2.index.name = "factor | model"
    write_markdown_table(o2, OUT / "q1_oos.md", ".4f")

    test_dates = df.index[df.index > pd.Timestamp(TRAIN_END)]
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(13, 3.4), sharey=False)
    for ax, f in zip(axes, FACTORS):
        names = [nm for (ff, nm) in cum if ff == f and nm.endswith("MLE")]
        for k, nm in enumerate(names):
            ax.plot(test_dates, cum[(f, nm)], lw=2, color=SERIES_COLORS[k], label=nm.replace(" MLE", ""))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(FACTORS[f], loc="left", fontsize=9)
        style_axes(ax)
    axes[0].set_ylabel("cum. SSE(mean) − SSE(model)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Out-of-sample 2016-2025: above zero = model beats the training mean",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "q1_oos_cumulative.png", dpi=160)

    # 4. Next month's expected return ----------------------------------------
    nm_rows = []
    nxt = df.index.max() + pd.offsets.MonthBegin(1)
    for f in FACTORS:
        x = df[f].to_numpy()
        for (p, q) in sorted({(1, 0), (1, 1), choices[f]}):
            _, ml = fits_full[(f, p, q)]
            fc = one_step_ahead(x, ml)
            nm_rows.append(dict(factor=f, model=label(p, q), chosen=(p, q) == choices[f],
                                last_return=x[-1], uncond_mean=ml["mu"], forecast=fc[0],
                                forecast_sd=fc[1], lo95=fc[0] - 1.96 * fc[1],
                                hi95=fc[0] + 1.96 * fc[1]))
    nm_df = pd.DataFrame(nm_rows)
    nm_df.to_csv(OUT / "q1_next_month.csv", index=False)
    n2 = nm_df.copy()
    n2.index = n2.pop("factor") + " | " + n2.pop("model")
    n2.index.name = f"forecast for {nxt:%Y-%m}"
    write_markdown_table(n2, OUT / "q1_next_month.md", ".5f")

    print(open(OUT / "q1_correlograms.md").read().split("### Model expected")[1])
    print(o2.to_string())
    print(n2.to_string())


def one_step_ahead(x, ml):
    """E[x_{T+1} | x_{1:T}] and its standard deviation from the Kalman filter."""
    x_ext = np.r_[x, 0.0]  # dummy last value; prediction at T+1 doesn't use it
    xh, _, F = kalman(x_ext, ml["mu"], ml["phi"], ml["theta"], ml["sigma2"])
    return xh[-1], np.sqrt(F[-1])


if __name__ == "__main__":
    main()
