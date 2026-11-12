"""Problem Set: Volatility (MGT 924) -- all computations and figures.

Usage:  python ps_volatility/volatility_ps.py --data PATH/data_crsp_msft_1990_2020.csv

The CSV (CRSP daily, from WRDS) has columns date, vwretd, mktrf, rf, msft_ret. It is
licensed data and is not committed. Returns are used in percent. The market series is
the CRSP value-weighted return (vwretd); MSFT is its CRSP holding-period return.

Every likelihood is coded by hand and maximized with scipy; the `arch` package is used
only as an independent cross-check of the point estimates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
H_MONTH = 21  # trading days in a month


# ---------------------------------------------------------------------------
# Variance recursions
# ---------------------------------------------------------------------------
def garch_filter(eps, omega, alpha, beta, gamma=0.0, extra=None, s2_0=None):
    """sigma2_t = omega + (alpha + gamma*1[eps_{t-1}<0]) eps_{t-1}^2 + beta sigma2_{t-1} (+ extra_t).

    sigma2_1 is initialized at the sample variance of eps (a standard backcast)."""
    T = eps.size
    s2 = np.empty(T)
    s2[0] = np.mean(eps**2) if s2_0 is None else s2_0
    for t in range(1, T):
        e = eps[t - 1]
        s2[t] = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2[t - 1]
        if extra is not None:
            s2[t] += extra[t]
    return s2


# ---------------------------------------------------------------------------
# Log likelihoods (per observation). Parameter vectors:
#   GARCH-N  : (mu, omega, alpha, beta)
#   GJR-t    : (mu, omega, alpha, gamma, beta, nu)
#   GARCH-M-N: (mu, lam, omega, alpha, beta), r_t = mu + lam*sigma2_t + eps_t
# ---------------------------------------------------------------------------
def ll_garch_n(theta, r):
    mu, omega, alpha, beta = theta
    eps = r - mu
    s2 = garch_filter(eps, omega, alpha, beta)
    return -0.5 * (np.log(2 * np.pi) + np.log(s2) + eps**2 / s2)


def ll_gjr_t(theta, r):
    mu, omega, alpha, gamma, beta, nu = theta
    eps = r - mu
    s2 = garch_filter(eps, omega, alpha, beta, gamma)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return c - 0.5 * np.log(s2) - 0.5 * (nu + 1) * np.log1p(eps**2 / ((nu - 2) * s2))


def ll_garchm_n(theta, r):
    mu, lam, omega, alpha, beta = theta
    T = r.size
    s2 = np.empty(T)
    eps = np.empty(T)
    s2[0] = np.var(r)
    for t in range(T):
        if t > 0:
            s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = r[t] - mu - lam * s2[t]
    return -0.5 * (np.log(2 * np.pi) + np.log(s2) + eps**2 / s2)


# ---------------------------------------------------------------------------
# MLE with robust (Bollerslev-Wooldridge sandwich) standard errors
# ---------------------------------------------------------------------------
def num_scores(f, theta, r, rel=1e-5):
    k = theta.size
    G = np.empty((r.size, k))
    for i in range(k):
        h = rel * max(abs(theta[i]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        G[:, i] = (f(tp, r) - f(tm, r)) / (2 * h)
    return G


def num_hessian(F, theta, rel=1e-4):
    k = theta.size
    H = np.empty((k, k))
    hs = rel * np.maximum(np.abs(theta), 1e-3)
    for i in range(k):
        for j in range(i, k):
            def shift(a, b):
                t = theta.copy()
                t[i] += a
                t[j] += b
                return F(t)
            hi, hj = hs[i], hs[j]
            H[i, j] = H[j, i] = (shift(hi, hj) - shift(hi, -hj) - shift(-hi, hj) + shift(-hi, -hj)) / (4 * hi * hj)
    return H


def fit(f, x0, bounds, r, cons=()):
    negll = lambda th: -np.sum(f(th, r))
    res = optimize.minimize(negll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 2000, "ftol": 1e-12})
    th = res.x
    H = num_hessian(lambda t: np.sum(f(t, r)), th)
    G = num_scores(f, th, r)
    Hinv = np.linalg.inv(-H)
    V_rob = Hinv @ (G.T @ G) @ Hinv
    V_mle = Hinv
    return {"theta": th, "se": np.sqrt(np.diag(V_rob)), "se_mle": np.sqrt(np.diag(V_mle)),
            "ll": -res.fun, "success": bool(res.success), "msg": res.message}


def fit_garch_n(r):
    v = np.var(r)
    x0 = np.array([r.mean(), 0.05 * v, 0.08, 0.90])
    bnds = [(None, None), (1e-8, None), (0, 1), (0, 1)]
    cons = ({"type": "ineq", "fun": lambda th: 0.99999 - th[2] - th[3]},)
    return fit(ll_garch_n, x0, bnds, r, cons)


def fit_gjr_t(r):
    v = np.var(r)
    x0 = np.array([r.mean(), 0.03 * v, 0.03, 0.08, 0.90, 7.0])
    bnds = [(None, None), (1e-8, None), (0, 1), (-1, 1), (0, 1), (2.05, 200)]
    cons = ({"type": "ineq", "fun": lambda th: 0.99999 - th[2] - th[3] / 2 - th[4]},
            {"type": "ineq", "fun": lambda th: th[2] + th[3]})  # positivity after a negative shock
    return fit(ll_gjr_t, x0, bnds, r, cons)


def fit_garchm_n(r, start):
    mu, om, a, b = start
    x0 = np.array([mu, 0.0, om, a, b])
    bnds = [(None, None), (None, None), (1e-8, None), (0, 1), (0, 1)]
    cons = ({"type": "ineq", "fun": lambda th: 0.99999 - th[3] - th[4]},)
    return fit(ll_garchm_n, x0, bnds, r, cons)


# ---------------------------------------------------------------------------
def acf(x, nlags):
    x = x - x.mean()
    d = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / d for k in range(1, nlags + 1)])


def acf_robust_se(x, nlags):
    """Heteroskedasticity-robust s.e. of the lag-k autocorrelation (Taylor 1984; Lo-MacKinlay):
    se_k = sqrt(sum_t x_t^2 x_{t-k}^2) / sum_t x_t^2, x demeaned. Valid under GARCH-type
    dependence in the variance, where the i.i.d. band 1/sqrt(T) is too narrow."""
    x = x - x.mean()
    d = np.dot(x, x)
    return np.array([np.sqrt(np.sum(x[k:] ** 2 * x[:-k] ** 2)) / d for k in range(1, nlags + 1)])


def ljung_box(rho, T):
    k = np.arange(1, rho.size + 1)
    q = T * (T + 2) * np.sum(rho**2 / (T - k))
    return q, stats.chi2.sf(q, rho.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True, parents=True)

    d = pd.read_csv(args.data, parse_dates=["date"])
    d = d[(d.date >= "1990-01-01") & (d.date <= "2020-12-31")].reset_index(drop=True)
    series = {"Market (CRSP VW)": 100 * d.vwretd.to_numpy(), "MSFT": 100 * d.msft_ret.to_numpy()}
    keys = {"Market (CRSP VW)": "mkt", "MSFT": "msft"}
    T = len(d)
    out = {"T": T, "start": str(d.date.iloc[0].date()), "end": str(d.date.iloc[-1].date())}

    # ---------------- Descriptive statistics ----------------
    for name, r in series.items():
        out[keys[name] + "_desc"] = {
            "mean": r.mean(), "sd": r.std(ddof=1), "ann_vol": r.std(ddof=1) * np.sqrt(252),
            "skew": stats.skew(r), "exkurt": stats.kurtosis(r), "min": r.min(), "max": r.max()}

    # ---------------- Q1(a) autocorrelations ----------------
    L = 20
    lags = np.arange(1, L + 1)
    band = 1.96 / np.sqrt(T)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    colors = {"Market (CRSP VW)": "#1f4e79", "MSFT": "#c0504d"}
    for j, (name, r) in enumerate(series.items()):
        k = keys[name]
        x = r - r.mean()
        rho_r, rho_r2 = acf(r, L), acf(x**2, L)   # squared *demeaned* returns = eps_t^2
        rho_abs = acf(np.abs(x), L)
        se_rob = acf_robust_se(r, L)
        out[k + "_acf_r_robust_se"] = se_rob.tolist()
        out[k + "_acf_r_out_iid"] = int(np.sum(np.abs(rho_r) > band))
        out[k + "_acf_r_out_robust"] = int(np.sum(np.abs(rho_r) > 1.96 * se_rob))
        out[k + "_acf_r2_out_iid"] = int(np.sum(np.abs(rho_r2) > band))
        out[k + "_acf_r2_mean"] = float(rho_r2.mean())
        out[k + "_acf_r2_lag20_over_lag1"] = float(rho_r2[-1] / rho_r2[0])
        out[k + "_acf_r"] = rho_r.tolist()
        out[k + "_acf_r2"] = rho_r2.tolist()
        out[k + "_acf_abs"] = rho_abs.tolist()
        out[k + "_lb_r"] = ljung_box(rho_r, T)
        out[k + "_lb_r2"] = ljung_box(rho_r2, T)
        out[k + "_lb_abs"] = ljung_box(rho_abs, T)
        for i, (rho, lab) in enumerate([(rho_r, r"$r_t$"), (rho_r2, r"$(r_t-\bar r)^2$")]):
            ax = axes[i, j]
            ax.bar(lags, rho, color=colors[name], width=0.65)
            ax.axhspan(-band, band, color="0.85", zorder=0)
            if i == 0:
                ax.plot(lags, 1.96 * se_rob, "k--", lw=0.7)
                ax.plot(lags, -1.96 * se_rob, "k--", lw=0.7)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{name}: ACF of {lab}", fontsize=10)
            ax.set_ylim(-0.1, 0.5)
            ax.grid(axis="y", alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("lag (trading days)")
        ax.set_xticks([1, 5, 10, 15, 20])
    fig.text(0.5, 0.005, "Grey band: ±1.96/√T (95%, i.i.d.). Dashed: 95% heteroskedasticity-robust band. Daily data, 1990–2020.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "acf.png", dpi=200)
    plt.close(fig)

    # Squared-return ACF side by side, plus |r|
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, r in series.items():
        k = keys[name]
        ax[0].plot(lags, out[k + "_acf_r2"], "o-", color=colors[name], label=name, ms=4)
        ax[1].plot(lags, out[k + "_acf_abs"], "o-", color=colors[name], label=name, ms=4)
    ax[0].set_title(r"ACF of $(r_t-\bar r)^2$", fontsize=10)
    ax[1].set_title(r"ACF of $|r_t|$ (robust to outliers)", fontsize=10)
    for a in ax:
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel("lag")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "acf_compare.png", dpi=200)
    plt.close(fig)

    # ---------------- Q1(b) GARCH(1,1) ----------------
    fits = {}
    for name, r in series.items():
        k = keys[name]
        g = fit_garch_n(r)
        mu, om, a, b = g["theta"]
        pers = a + b
        g.update({"persistence": pers, "half_life": np.log(0.5) / np.log(pers),
                  "uncond_var": om / (1 - pers), "uncond_ann_vol": np.sqrt(252 * om / (1 - pers))})
        g["s2"] = garch_filter(r - mu, om, a, b)
        z = (r - mu) / np.sqrt(g["s2"])
        rz2 = acf(z**2, L)
        g["z_kurtosis"] = float(stats.kurtosis(z, fisher=False))
        g["z2_lb"] = ljung_box(rz2, T)
        fits[k] = g
        # arch cross-check
        try:
            from arch import arch_model
            am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
            g["arch_check"] = am.params.to_numpy().tolist() + [am.loglikelihood]
        except Exception as e:  # pragma: no cover
            g["arch_check"] = str(e)
        out[k + "_garch"] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) and kk != "s2" else vv)
                             for kk, vv in g.items() if kk != "s2"}

    # conditional vol plot
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name in series:
        k = keys[name]
        ax.plot(d.date, np.sqrt(252 * fits[k]["s2"]), color=colors[name], lw=0.8, label=name)
    ax.set_ylabel("annualized conditional vol (%)")
    ax.set_title("GARCH(1,1) conditional volatility", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "garch_vol.png", dpi=200)
    plt.close(fig)

    # ---------------- Q1(c) Is MSFT vol just the market? ----------------
    rm, ri = series["Market (CRSP VW)"], series["MSFT"]
    X = np.column_stack([np.ones(T), rm])
    coef, *_ = np.linalg.lstsq(X, ri, rcond=None)
    resid = ri - X @ coef
    beta_full = coef[1]
    r2_full = 1 - resid.var() / ri.var()
    c = {"beta_full": beta_full, "alpha_full": coef[0], "R2_full": r2_full}

    # conditional systematic share with a rolling 1-year (252d) beta (only past data)
    beta_roll = pd.Series(ri).rolling(252).cov(pd.Series(rm)) / pd.Series(rm).rolling(252).var()
    beta_roll = beta_roll.shift(1).to_numpy()
    s2m, s2i = fits["mkt"]["s2"], fits["msft"]["s2"]
    share_const = beta_full**2 * s2m / s2i
    share_roll = beta_roll**2 * s2m / s2i
    c["share_const_mean"] = float(np.nanmean(share_const))
    c["share_roll_mean"] = float(np.nanmean(share_roll))
    c["share_roll_median"] = float(np.nanmedian(share_roll))
    c["share_by_year"] = pd.Series(share_roll).groupby(d.date.dt.year).mean().round(3).to_dict()
    c["share_roll_q10_q90"] = [float(np.nanquantile(share_roll, .1)), float(np.nanquantile(share_roll, .9))]

    # GARCH on idiosyncratic residual: does idiosyncratic vol cluster on its own?
    gi = fit_garch_n(resid)
    mu_i, om_i, a_i, b_i = gi["theta"]
    s2e = garch_filter(resid - mu_i, om_i, a_i, b_i)
    c["idio_garch"] = {"theta": gi["theta"].tolist(), "se": gi["se"].tolist(),
                       "persistence": a_i + b_i, "half_life": np.log(.5) / np.log(a_i + b_i),
                       "uncond_ann_vol": np.sqrt(252 * om_i / (1 - a_i - b_i))}
    rho_e2 = acf(resid**2, L)
    c["idio_acf_r2"] = rho_e2.tolist()

    # Additive decomposition: sigma2_MSFT,t = b^2 sigma2_m,t + sigma2_u,t (shares in [0,1])
    s2_sys = beta_full**2 * s2m
    share_add = s2_sys / (s2_sys + s2e)
    c["share_add_mean"] = float(share_add.mean())
    c["share_add_q10_q90"] = [float(np.quantile(share_add, .1)), float(np.quantile(share_add, .9))]
    c["mkt_part_vol_mean"] = float(np.mean(np.sqrt(252 * s2_sys)))
    c["firm_vol_mean"] = float(np.mean(np.sqrt(252 * s2e)))
    c["frac_days_share_above_half"] = float(np.mean(share_add > 0.5))
    eps_def = {"Full sample 1990-2020": ("1990-01-01", "2020-12-31"),
               "Dot-com and antitrust 2000-02": ("2000-01-01", "2002-12-31"),
               "Financial crisis Sep 08-Jun 09": ("2008-09-01", "2009-06-30"),
               "COVID crash Mar-Apr 2020": ("2020-03-01", "2020-04-30"),
               "Calm year 2017": ("2017-01-01", "2017-12-31")}
    c["episodes"] = []
    for lab, (lo, hi) in eps_def.items():
        m = ((d.date >= lo) & (d.date <= hi)).to_numpy()
        c["episodes"].append({"period": lab, "share": float(share_add[m].mean()),
                              "mkt_part_vol": float(np.mean(np.sqrt(252 * s2_sys[m]))),
                              "firm_vol": float(np.mean(np.sqrt(252 * s2e[m])))})
    yr = []
    for y in sorted(d.date.dt.year.unique()):
        m = (d.date.dt.year == y).to_numpy()
        Xy = np.column_stack([np.ones(m.sum()), rm[m]])
        cy, *_ = np.linalg.lstsq(Xy, ri[m], rcond=None)
        ey = ri[m] - Xy @ cy
        yr.append({"year": int(y), "beta": float(cy[1]), "R2": float(1 - ey.var() / ri[m].var())})
    c["yearly"] = yr
    c["idio_lb_r2"] = ljung_box(rho_e2, T)

    # How much of MSFT's conditional-variance *variation* does market variance explain?
    lm, li = np.log(s2m), np.log(s2i)
    Xr = np.column_stack([np.ones(T), lm])
    cr, *_ = np.linalg.lstsq(Xr, li, rcond=None)
    c["logvar_reg"] = {"slope": cr[1], "R2": 1 - np.var(li - Xr @ cr) / np.var(li),
                       "corr_vol": np.corrcoef(np.sqrt(s2m), np.sqrt(s2i))[0, 1]}

    # Does lagged market variance add to MSFT's own GARCH? (GARCH-X with Q=sigma2_mkt,t)
    def ll_garchx(theta, r, q=s2m):
        mu, omega, alpha, beta, delta = theta
        eps = r - mu
        s2 = garch_filter(eps, omega, alpha, beta, extra=np.r_[0, delta * q[1:]])
        return -0.5 * (np.log(2 * np.pi) + np.log(s2) + eps**2 / s2)
    gx = fit(ll_garchx, np.r_[fits["msft"]["theta"], 0.05],
             [(None, None), (1e-8, None), (0, 1), (0, 1), (0, None)], ri,
             ({"type": "ineq", "fun": lambda th: 0.99999 - th[2] - th[3]},))
    c["garchx"] = {"theta": gx["theta"].tolist(), "se": gx["se"].tolist(), "ll": gx["ll"],
                   "LR_vs_garch": 2 * (gx["ll"] - fits["msft"]["ll"])}
    # Note: sigma2_mkt,t is predetermined (a function of market returns through t-1).

    # by-decade variance decomposition
    dec = []
    for lo, hi in [(1990, 1999), (2000, 2009), (2010, 2020)]:
        m = (d.date.dt.year >= lo) & (d.date.dt.year <= hi)
        Xd = np.column_stack([np.ones(m.sum()), rm[m]])
        cd, *_ = np.linalg.lstsq(Xd, ri[m], rcond=None)
        ed = ri[m] - Xd @ cd
        dec.append({"period": f"{lo}-{hi}", "beta": cd[1], "R2": 1 - ed.var() / ri[m].var(),
                    "vol_msft": ri[m].std() * np.sqrt(252), "vol_mkt": rm[m].std() * np.sqrt(252),
                    "vol_idio": ed.std() * np.sqrt(252)})
    c["by_decade"] = dec
    out["q1c"] = c

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    ax[0].plot(d.date, np.sqrt(252 * s2e), color="#7f7f7f", lw=0.7, label=r"firm-specific: $\sigma_{u,t}$ (GARCH on residual)")
    ax[0].plot(d.date, np.sqrt(252 * s2_sys), color=colors["Market (CRSP VW)"], lw=0.7,
               label=r"market part: $b\,\sigma_{m,t}$")
    ax[0].set_ylabel("annualized vol (%)")
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[1].plot(d.date, pd.Series(share_add).rolling(21).mean(), color=colors["Market (CRSP VW)"], lw=0.8,
               label=r"$b^2\sigma^2_{m,t}/(b^2\sigma^2_{m,t}+\sigma^2_{u,t})$, 21-day avg")
    ys = pd.to_datetime([f"{r_['year']}-01-01" for r_ in yr] + ["2021-01-01"])
    ax[1].step(ys, [r_["R2"] for r_ in yr] + [yr[-1]["R2"]], where="post", color=colors["MSFT"], lw=1.2,
               label=r"yearly market-model $R^2$")
    ax[1].axhline(share_add.mean(), color="k", ls="--", lw=0.6)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("market share of\nMSFT variance")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "msft_decomp.png", dpi=200)
    plt.close(fig)

    # ---------------- Q1(d) one-month market-risk forecast ----------------
    g = fits["mkt"]
    mu, om, a, b = g["theta"]
    eps_last = rm[-1] - mu
    s2_next = om + a * eps_last**2 + b * g["s2"][-1]
    vbar = om / (1 - a - b)
    h = np.arange(1, H_MONTH + 1)
    path = vbar + (a + b) ** (h - 1) * (s2_next - vbar)
    month_var = path.sum()
    month_vol = np.sqrt(month_var)
    month_vol_lr = np.sqrt(H_MONTH * vbar)
    # Monthly VaR by simulation from the fitted GARCH (captures fat tails of aggregated returns)
    rng = np.random.default_rng(0)
    nsim = 200_000
    s2 = np.full(nsim, s2_next)
    cum = np.zeros(nsim)
    for _ in range(H_MONTH):
        e = np.sqrt(s2) * rng.standard_normal(nsim)
        cum += mu + e
        s2 = om + a * e**2 + b * s2
    out["q1d"] = {
        "last_date": str(d.date.iloc[-1].date()),
        "vol_today_ann": np.sqrt(252 * g["s2"][-1]), "vol_next_ann": np.sqrt(252 * s2_next),
        "vol_longrun_ann": np.sqrt(252 * vbar), "vol_path_day21_ann": np.sqrt(252 * path[-1]),
        "month_vol": month_vol, "month_vol_longrun": month_vol_lr,
        "month_var_ratio": month_var / (H_MONTH * vbar),
        "VaR95_normal": -(H_MONTH * mu - 1.645 * month_vol), "VaR95_sim": -np.quantile(cum, 0.05),
        "VaR99_sim": -np.quantile(cum, 0.01), "p_drop_10": float(np.mean(cum < -10)),
        "band68": [H_MONTH * mu - month_vol, H_MONTH * mu + month_vol],
        "band95": [H_MONTH * mu - 1.96 * month_vol, H_MONTH * mu + 1.96 * month_vol],
        "band95_sim": [float(np.quantile(cum, 0.025)), float(np.quantile(cum, 0.975))],
        "half_life": g["half_life"],
    }
    # what the forecast looked like at the March 2020 peak, for intuition
    ipk = int(np.argmax(g["s2"]))
    s2p = g["s2"][ipk]
    path_pk = vbar + (a + b) ** (h - 1) * (s2p - vbar)
    out["q1d"]["peak"] = {"date": str(d.date.iloc[ipk].date()), "vol_ann": np.sqrt(252 * s2p),
                          "month_vol": np.sqrt(path_pk.sum())}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(h, np.sqrt(252 * path), "o-", ms=3, color=colors["Market (CRSP VW)"], label="Dec-31-2020 forecast")
    ax.plot(h, np.sqrt(252 * path_pk), "s-", ms=3, color="#c0504d",
            label=f"forecast made at peak ({out['q1d']['peak']['date']})")
    ax.axhline(np.sqrt(252 * vbar), color="k", ls="--", lw=0.8, label="long-run level")
    ax.set_xlabel("trading days ahead")
    ax.set_ylabel("annualized vol (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("GARCH(1,1) market volatility term structure", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "forecast.png", dpi=200)
    plt.close(fig)

    # ---------------- Q2 illustration: GARCH-in-mean ----------------
    for name, r in series.items():
        k = keys[name]
        gm = fit_garchm_n(r, fits[k]["theta"])
        out[k + "_garchm"] = {"theta": gm["theta"].tolist(), "se": gm["se"].tolist(), "ll": gm["ll"]}

    # ---------------- Q3 GJR-GARCH(1,1,1) with Student t ----------------
    for name, r in series.items():
        k = keys[name]
        gj = fit_gjr_t(r)
        mu, om, a, gm_, b, nu = gj["theta"]
        pers = a + gm_ / 2 + b
        gj.update({"persistence": pers, "half_life": np.log(.5) / np.log(pers),
                   "uncond_ann_vol": np.sqrt(252 * om / (1 - pers)),
                   "news_ratio": (a + gm_) / a if a > 1e-6 else np.inf})
        s2j = garch_filter(r - mu, om, a, b, gm_)
        try:
            from arch import arch_model
            am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
            gj["arch_check"] = am.params.to_numpy().tolist() + [am.loglikelihood]
        except Exception as e:  # pragma: no cover
            gj["arch_check"] = str(e)
        # Gaussian GJR and t-GARCH as intermediate steps (which change drives the fit?)
        try:
            from arch import arch_model
            out[k + "_ll_ladder"] = {
                "garch_n": fits[k]["ll"],
                "garch_t": arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off").loglikelihood,
                "gjr_n": arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="normal").fit(disp="off").loglikelihood,
                "gjr_t": gj["ll"]}
        except Exception:
            pass
        out[k + "_gjr"] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in gj.items()}
        fits[k + "_gjr_s2"] = s2j

    # news impact curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    z = np.linspace(-5, 5, 201)
    for j, name in enumerate(series):
        k = keys[name]
        mu, om, a, b = fits[k]["theta"]
        vbar_g = om / (1 - a - b)
        nic_g = om + a * (z**2) * vbar_g + b * vbar_g
        mu, om2, a2, gm2, b2, nu = out[k + "_gjr"]["theta"]
        vbar_j = om2 / (1 - a2 - gm2 / 2 - b2)
        nic_j = om2 + (a2 + gm2 * (z < 0)) * z**2 * vbar_j + b2 * vbar_j
        ax[j].plot(z, nic_g / vbar_g, color="0.4", label="GARCH(1,1)-N")
        ax[j].plot(z, nic_j / vbar_j, color=colors[name], label="GJR-GARCH-t")
        ax[j].set_title(f"{name}: news impact curve", fontsize=10)
        ax[j].set_xlabel(r"shock $\varepsilon_{t-1}$ in units of long-run s.d.")
        ax[j].set_ylabel(r"$\sigma^2_t/\bar\sigma^2$")
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "news_impact.png", dpi=200)
    plt.close(fig)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    (OUT / "results.json").write_text(json.dumps(conv(out), indent=1))
    print(json.dumps(conv(out), indent=1))


if __name__ == "__main__":
    main()
