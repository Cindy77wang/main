"""Question 3: GMM estimation of a Poisson parameter lambda_i for each phrase.

Moment conditions (Poisson: mean = variance = lambda):
    g_t(lambda) = [ c_t - lambda ,  (c_t - lambda)^2 - lambda ]'
    E[g_t(lambda_0)] = 0  ->  2 moments, 1 parameter, 1 overidentifying restriction.

GMM:  lambda_hat = argmin_lambda  gbar(lambda)' W gbar(lambda),  gbar = (1/T) sum_t g_t.
  step 1: W = I
  step 2: W = S^{-1}, S = long-run covariance of g_t at the step-1 estimate
  iterated: repeat step 2 until lambda converges
  CUE: S re-evaluated at every lambda inside the objective
S is estimated two ways: iid (S = (1/T) sum g g', centered) and Newey-West HAC,
because the counts are strongly serially correlated.

Asymptotics: sqrt(T)(lambda_hat - lambda) -> N(0, (G' S^{-1} G)^{-1}),
G = E[d g_t / d lambda] = [-1, -1 - 2(cbar - lambda)]'.
Hansen J = T gbar' S^{-1} gbar -> chi2(1) under H0 (Poisson moments hold).

MLE under iid Poisson: lambda_hat = cbar, se = sqrt(cbar / T).

Run:  python q3_gmm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from common import (FIGURES, RESULTS, SERIES_COLORS, INK, MUTED, load_wsj, newey_west,
                    nw_lag_rule, plt, style_axes, write_markdown_table)


def moments(lam, c):
    return np.column_stack([c - lam, (c - lam) ** 2 - lam])


def gbar(lam, c):
    # closed form of the sample moments (fast; identical to moments(lam, c).mean(0))
    m1, m2 = c.mean(), np.mean(c**2)
    return np.array([m1 - lam, m2 - 2 * lam * m1 + lam**2 - lam])


def G_mat(lam, c):
    return np.array([-1.0, -1.0 - 2.0 * (c.mean() - lam)])


def S_hat(lam, c, lags):
    g = moments(lam, c)
    return newey_west(g, lags, center=True) if lags > 0 else np.cov(g.T, bias=True)


def minimise(obj, c):
    lo, hi = 1e-6, max(c.max(), 1.0) * 3
    grid = np.linspace(lo, hi, 600)
    vals = np.array([obj(x) for x in grid])
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = optimize.minimize_scalar(obj, bounds=(a, b), method="bounded", options=dict(xatol=1e-12))
    return r.x


def gmm(c, lags, W=None):
    """One GMM step with fixed W."""
    W = np.eye(2) if W is None else W
    return minimise(lambda l: gbar(l, c) @ W @ gbar(l, c), c)


def inference(lam, c, lags, W):
    T = len(c)
    G = G_mat(lam, c)
    S = S_hat(lam, c, lags)
    # general sandwich (valid for any W); equals (G'S^-1G)^-1 when W = S^-1
    bread = 1.0 / (G @ W @ G)
    var = bread * (G @ W @ S @ W @ G) * bread / T
    gb = gbar(lam, c)
    J = T * gb @ np.linalg.inv(S) @ gb
    return np.sqrt(var), J


def estimate(c, lags):
    out = {}
    l1 = gmm(c, lags)
    se1, _ = inference(l1, c, lags, np.eye(2))
    out["step1"] = (l1, se1, np.nan)
    W2 = np.linalg.inv(S_hat(l1, c, lags))
    l2 = gmm(c, lags, W2)
    se2, J2 = inference(l2, c, lags, W2)
    out["two-step"] = (l2, se2, J2)
    l_it = l2
    for _ in range(500):
        W = np.linalg.inv(S_hat(l_it, c, lags))
        l_new = gmm(c, lags, W)
        if abs(l_new - l_it) < 1e-10:
            l_it = l_new
            break
        l_it = l_new
    W = np.linalg.inv(S_hat(l_it, c, lags))
    se_it, J_it = inference(l_it, c, lags, W)
    out["iterated"] = (l_it, se_it, J_it)
    lcue = minimise(lambda l: gbar(l, c) @ np.linalg.inv(S_hat(l, c, lags)) @ gbar(l, c), c)
    Wc = np.linalg.inv(S_hat(lcue, c, lags))
    se_c, J_c = inference(lcue, c, lags, Wc)
    out["CUE"] = (lcue, se_c, J_c)
    return out


def main():
    df = load_wsj()
    T = len(df)
    L = nw_lag_rule(T)
    rows, jrows = [], []
    for name in df.columns:
        c = df[name].to_numpy(dtype=float)
        cbar, s2 = c.mean(), c.var()
        se_ml = np.sqrt(cbar / T)
        se_ml_hac = np.sqrt(newey_west(c - cbar, L)[0, 0] / T)
        rec = {"phrase": name, "mean": cbar, "variance": s2, "var/mean": s2 / cbar,
               "MLE": cbar, "MLE se": se_ml, "MLE HAC se": se_ml_hac}
        for lab, lags in (("iid", 0), ("HAC", L)):
            res = estimate(c, lags)
            for step, (lam, se, J) in res.items():
                rec[f"GMM {step} [{lab}]"] = lam
                rec[f"se {step} [{lab}]"] = se
                if np.isfinite(J):
                    jrows.append({"phrase": name, "S": lab, "estimator": step, "lambda": lam,
                                  "J": J, "p-value": stats.chi2.sf(J, 1)})
        # Exactly identified checks: each moment alone
        rec["GMM mean-only"] = cbar
        # variance-only: (c - l)^2 mean = l -> s2 + (cbar - l)^2 = l, take root nearest cbar
        disc = (2 * cbar + 1) ** 2 - 4 * (cbar**2 + s2)
        rec["GMM variance-only (nan = no real root)"] = ((2 * cbar + 1) + np.sqrt(disc)) / 2 if disc >= 0 else np.nan
        rows.append(rec)

    est = pd.DataFrame(rows).set_index("phrase")
    jt = pd.DataFrame(jrows)
    est.to_csv(RESULTS / "q3_estimates.csv")
    jt.to_csv(RESULTS / "q3_jtests.csv", index=False)
    write_markdown_table(est.T, RESULTS / "q3_estimates.md", ".3f")
    jt_md = jt.copy()
    jt_md.index = jt_md.pop("phrase") + " / " + jt_md.pop("S") + " / " + jt_md.pop("estimator")
    jt_md.index.name = "phrase / S / estimator"
    write_markdown_table(jt_md, RESULTS / "q3_jtests.md", ".3f")
    print(f"Newey-West lags: {L}")
    print(est.T.to_string())
    print(jt.to_string())

    # Figure: GMM objective (two-step HAC weighting) for each phrase
    fig, axes = plt.subplots(1, len(df.columns), figsize=(11, 3.3))
    for ax, name, col in zip(axes, df.columns, SERIES_COLORS):
        c = df[name].to_numpy(dtype=float)
        res = estimate(c, L)
        l1 = res["step1"][0]
        W2 = np.linalg.inv(S_hat(l1, c, L))
        lam_hat = res["two-step"][0]
        grid = np.linspace(0.3 * c.mean(), max(2.0 * lam_hat, 1.5 * c.mean()), 400)
        J = [T * gbar(l, c) @ W2 @ gbar(l, c) for l in grid]
        ax.plot(grid, J, color=col, lw=2)
        ax.axvline(c.mean(), color=MUTED, lw=1, ls="--")
        ax.text(c.mean(), ax.get_ylim()[1] * 0.92, " MLE = mean", color=MUTED, fontsize=8, ha="left")
        ax.axvline(lam_hat, color=INK, lw=1)
        ax.text(lam_hat, ax.get_ylim()[1] * 0.80, "2-step GMM ", color=INK, fontsize=8, ha="right")
        ax.set_title(f"'{name}'", loc="left", fontsize=10)
        ax.set_xlabel("$\\lambda$"); ax.set_ylabel("$T\\,\\bar g' W \\bar g$")
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_gmm_objective.png", dpi=160)


if __name__ == "__main__":
    main()
