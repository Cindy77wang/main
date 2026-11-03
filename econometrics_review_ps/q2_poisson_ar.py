"""Question 2: autoregressive model for WSJ phrase counts, OLS vs. Poisson MLE.

Model: E_{t-1}[c_t] = lambda_t = delta0 + delta1 * c_{t-1}

(a) OLS of c_t on (1, c_{t-1}).
(b) Exact Poisson MLE: c_t | c_{t-1} ~ Poisson(lambda_t), with the conditional mean
    used as the Poisson intensity. Log-likelihood conditional on c_1:
        l(d) = sum_{t=2}^T [ c_t log(lambda_t) - lambda_t - log(c_t!) ].
    Robustness: we also add a term for c_1 ~ Poisson(delta0 / (1 - delta1)), the
    stationary mean, so that every observation enters the likelihood.
Tests of H0: delta1 = 0 (no dynamics) and H0: delta1 = 1 (unit-root mean):
    OLS t / Wald (classical and White), ML Wald, LR, LM (plus robust versions).

Run:  python q2_poisson_ar.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

from common import (FIGURES, RESULTS, SERIES_COLORS, INK, MUTED, load_wsj, ols,
                    plt, style_axes, write_markdown_table)

EPS = 1e-10


# ---------------------------------------------------------------------------
# Poisson autoregression: likelihood, score, Hessian
# ---------------------------------------------------------------------------
def design(c: np.ndarray):
    y = c[1:].astype(float)
    X = np.column_stack([np.ones(len(y)), c[:-1].astype(float)])
    return y, X


def loglik(d, y, X, c1=None):
    """Exact Poisson log-likelihood. If c1 is given, include the term for the first
    observation from Poisson(delta0/(1-delta1)) (only defined when delta1 < 1)."""
    lam = X @ d
    if np.any(lam <= 0):
        return -np.inf
    ll = np.sum(y * np.log(lam) - lam - gammaln(y + 1))
    if c1 is not None:
        if d[1] >= 1:
            return -np.inf
        mu = d[0] / (1 - d[1])
        ll += c1 * np.log(mu) - mu - gammaln(c1 + 1)
    return ll


def score_obs(d, y, X):
    """Per-observation score s_t = (c_t/lambda_t - 1) x_t (T x 2)."""
    lam = X @ d
    return X * (y / lam - 1)[:, None]


def hessian(d, y, X):
    """Observed Hessian: -sum c_t/lambda_t^2 x_t x_t'."""
    lam = X @ d
    return -(X * (y / lam**2)[:, None]).T @ X


def info_expected(d, X):
    """Expected (Fisher) information: sum x_t x_t' / lambda_t."""
    lam = X @ d
    return (X / lam[:, None]).T @ X


def poisson_mle(y, X, start=None, c1=None):
    """Maximise the Poisson likelihood subject to lambda_t > 0 for all t."""
    if start is None:
        start = ols(y, X)["b"]
    start = np.array([max(start[0], 0.5), min(max(start[1], 0.01), 0.99)])

    def nll(d):
        v = loglik(d, y, X, c1)
        return 1e12 if not np.isfinite(v) else -v

    def ngrad(d):
        lam = X @ d
        if np.any(lam <= 0):
            return np.zeros(2)
        g = score_obs(d, y, X).sum(axis=0)
        if c1 is not None and d[1] < 1:
            mu = d[0] / (1 - d[1])
            dmu = np.array([1 / (1 - d[1]), d[0] / (1 - d[1]) ** 2])
            g += (c1 / mu - 1) * dmu
        return -g

    # lambda_t >= delta0 + delta1 * min(c) > 0; c >= 0 so delta0 > 0, delta1 >= 0 suffices
    # whenever min(c_{t-1}) = 0 (true for 'climate change'); otherwise we allow any delta1
    # with positive intensities, which the -inf barrier enforces.
    bounds = [(EPS, None), (0.0 if X[:, 1].min() == 0 else None, 0.999999 if c1 is not None else None)]
    res = optimize.minimize(nll, start, jac=ngrad, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=10_000, ftol=1e-15, gtol=1e-10))
    d = res.x
    if c1 is None:  # polish with Newton steps using the analytic Hessian
        for _ in range(50):
            g = score_obs(d, y, X).sum(axis=0)
            H = hessian(d, y, X)
            step = np.linalg.solve(H, g)
            dn = d - step
            if np.any(X @ dn <= 0):
                break
            d = dn
            if np.max(np.abs(step)) < 1e-12:
                break
    return d, loglik(d, y, X, c1)


def restricted_mle(y, X, delta1_0):
    """MLE of delta0 when delta1 is fixed at delta1_0."""
    if delta1_0 == 0:
        d0 = y.mean()  # closed form
    else:
        lo = EPS - delta1_0 * X[:, 1].min()
        f = lambda a: -loglik(np.array([a, delta1_0]), y, X)
        r = optimize.minimize_scalar(f, bounds=(lo + 1e-8, y.max() + 100), method="bounded",
                                     options=dict(xatol=1e-12))
        d0 = r.x
    d = np.array([d0, delta1_0])
    return d, loglik(d, y, X)


def lm_test(d_r, y, X, R):
    """LM statistic at the restricted estimate. Classical: s' I^{-1} s with expected
    information; robust (Wooldridge): uses the OPG as the meat."""
    s = score_obs(d_r, y, X)
    S = s.sum(axis=0)
    A = info_expected(d_r, X)
    Ai = np.linalg.inv(A)
    lm = S @ Ai @ S
    B = s.T @ s
    RA = R @ Ai
    mid = np.linalg.inv(RA @ B @ RA.T)
    lm_rob = (RA @ S) @ mid @ (RA @ S)
    return float(lm), float(lm_rob)


# ---------------------------------------------------------------------------
def analyse(name, c):
    y, X = design(c)
    T = len(y)
    out = {"phrase": name, "T": T}

    # (a) OLS
    o = ols(y, X)
    se_c, se_w = np.sqrt(np.diag(o["V"])), np.sqrt(np.diag(o["V_white"]))
    out.update(ols_d0=o["b"][0], ols_d1=o["b"][1], ols_se_d0=se_c[0], ols_se_d1=se_c[1],
               ols_wse_d0=se_w[0], ols_wse_d1=se_w[1], ols_r2=o["r2"])

    # (b) Poisson MLE
    d, ll = poisson_mle(y, X, start=o["b"])
    H = hessian(d, y, X)
    Hi = np.linalg.inv(-H)
    I_exp_i = np.linalg.inv(info_expected(d, X))
    s = score_obs(d, y, X)
    V_sand = Hi @ (s.T @ s) @ Hi
    lam = X @ d
    pearson = np.sum((y - lam) ** 2 / lam) / (T - 2)  # dispersion; =1 under Poisson
    out.update(mle_d0=d[0], mle_d1=d[1], mle_loglik=ll,
               mle_se_d0=np.sqrt(Hi[0, 0]), mle_se_d1=np.sqrt(Hi[1, 1]),
               mle_ise_d0=np.sqrt(I_exp_i[0, 0]), mle_ise_d1=np.sqrt(I_exp_i[1, 1]),
               mle_rse_d0=np.sqrt(V_sand[0, 0]), mle_rse_d1=np.sqrt(V_sand[1, 1]),
               pearson_dispersion=pearson,
               max_abs_score=float(np.abs(s.sum(axis=0)).max()))

    # Full-sample likelihood including the first observation (stationary mean term)
    try:
        d_full, ll_full = poisson_mle(y, X, start=d, c1=float(c[0]))
        out.update(mle_full_d0=d_full[0], mle_full_d1=d_full[1], mle_full_loglik=ll_full)
    except Exception:  # pragma: no cover
        out.update(mle_full_d0=np.nan, mle_full_d1=np.nan, mle_full_loglik=np.nan)

    # OLS vs ML as weighted least squares: ML = WLS with weights 1/lambda_t
    w = 1 / lam
    Xw = X * w[:, None]
    b_wls = np.linalg.solve(Xw.T @ X, Xw.T @ y)
    out.update(wls_check_d0=b_wls[0], wls_check_d1=b_wls[1])

    # Tests
    R = np.array([[0.0, 1.0]])
    tests = []
    for h0 in (0.0, 1.0):
        rec = {"phrase": name, "H0": f"delta1 = {h0:g}"}
        t_c = (o["b"][1] - h0) / se_c[1]
        t_w = (o["b"][1] - h0) / se_w[1]
        rec["OLS t (classical)"] = t_c
        rec["OLS Wald (classical)"] = t_c**2
        rec["OLS Wald (White)"] = t_w**2
        rec["ML Wald (Hessian)"] = (d[1] - h0) ** 2 / Hi[1, 1]
        rec["ML Wald (sandwich)"] = (d[1] - h0) ** 2 / V_sand[1, 1]
        d_r, ll_r = restricted_mle(y, X, h0)
        rec["restricted delta0"] = d_r[0]
        rec["LR"] = 2 * (ll - ll_r)
        lm, lm_r = lm_test(d_r, y, X, R)
        rec["LM"] = lm
        rec["LM (robust)"] = lm_r
        rec["LR / dispersion"] = rec["LR"] / pearson  # quasi-LR scaled by Pearson phi
        tests.append(rec)
    return out, tests, (y, X, o["b"], d)


def main():
    df = load_wsj()
    rows, test_rows, fits = [], [], {}
    for name in df.columns:
        r, t, f = analyse(name, df[name].to_numpy())
        rows.append(r)
        test_rows += t
        fits[name] = f

    est = pd.DataFrame(rows).set_index("phrase")
    tests = pd.DataFrame(test_rows).set_index(["phrase", "H0"])
    est.to_csv(RESULTS / "q2_estimates.csv")
    tests.to_csv(RESULTS / "q2_tests.csv")

    # Pretty tables
    tab = pd.DataFrame({
        "OLS d0": est.ols_d0, "(se)": est.ols_se_d0, "(White se)": est.ols_wse_d0,
        "OLS d1": est.ols_d1, "(se) ": est.ols_se_d1, "(White se) ": est.ols_wse_d1,
        "MLE d0": est.mle_d0, "(ML se)": est.mle_se_d0, "(robust se)": est.mle_rse_d0,
        "MLE d1": est.mle_d1, "(ML se) ": est.mle_se_d1, "(robust se) ": est.mle_rse_d1,
        "log L": est.mle_loglik, "Pearson phi": est.pearson_dispersion,
    })
    write_markdown_table(tab, RESULTS / "q2_estimates.md")
    full = est[["mle_d0", "mle_d1", "mle_full_d0", "mle_full_d1", "wls_check_d0", "wls_check_d1"]]
    write_markdown_table(full, RESULTS / "q2_mle_variants.md")
    pv = tests.copy()
    for c in ["OLS Wald (classical)", "OLS Wald (White)", "ML Wald (Hessian)",
              "ML Wald (sandwich)", "LR", "LM", "LM (robust)", "LR / dispersion"]:
        pv[c + " p"] = stats.chi2.sf(tests[c], 1)
    pv.to_csv(RESULTS / "q2_tests_with_pvalues.csv")
    tt = tests.drop(columns=["restricted delta0"]).reset_index()
    tt.index = tt.pop("phrase") + ", " + tt.pop("H0")
    tt.index.name = "phrase, H0"
    write_markdown_table(tt, RESULTS / "q2_tests.md", ".2f")

    # Figure: data with fitted conditional means
    fig, axes = plt.subplots(len(fits), 1, figsize=(9, 8), sharex=True)
    dates = df.index[1:]
    for ax, (name, (y, X, b_ols, d)) in zip(axes, fits.items()):
        ax.plot(dates, y, color="#b9b8b0", lw=1.0, label="count $c_t$")
        ax.plot(dates, X @ b_ols, color=SERIES_COLORS[0], lw=1.2, label="OLS fitted mean")
        ax.plot(dates, X @ d, color=SERIES_COLORS[1], lw=1.2, label="Poisson MLE fitted mean")
        ax.set_title(f"'{name}'", loc="left", fontsize=10)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.suptitle("WSJ phrase counts and fitted $E_{t-1}[c_t]$", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_fitted_means.png", dpi=160)

    # Figure: residual variance vs fitted mean (shows Poisson heteroskedasticity/overdispersion)
    fig, axes = plt.subplots(1, len(fits), figsize=(11, 3.4))
    for ax, (name, (y, X, b_ols, d)), col in zip(axes, fits.items(), SERIES_COLORS):
        lam = X @ d
        ax.scatter(lam, (y - lam) ** 2, s=10, color=col, alpha=0.6, edgecolor="none")
        g = np.linspace(lam.min(), lam.max(), 50)
        ax.plot(g, g, color=MUTED, lw=1, ls="--", label="Var = mean (Poisson)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("fitted $\\lambda_t$"); ax.set_ylabel("$(c_t-\\lambda_t)^2$")
        ax.set_title(f"'{name}'", loc="left", fontsize=10)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_overdispersion.png", dpi=160)

    print(tab.T.to_string())
    print()
    print(full.T.to_string())
    print()
    print(pv.T.to_string())
    (RESULTS / "q2_summary.json").write_text(json.dumps(
        {"estimates": est.reset_index().to_dict("records")}, indent=1, default=float))


if __name__ == "__main__":
    main()
