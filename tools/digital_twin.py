"""Digital twin: where PRISM's Sharpe ratio is won and lost, in a simulated CTF market.

The CTF data cannot tell us how far a model is from the best achievable portfolio, because
the true expected returns and covariances are unknown. In a simulated market in the exact
CTF format (tools/make_synthetic_data.py --truth) they are known, so each layer of PRISM
can be measured against the truth:

    oracle                 inv(Sigma_true) mu_true            the conditional tangency portfolio
    true mu, PRISM risk    inv(Sigma_hat) mu_true             cost of estimating the risk model
    learner ensemble       inv(Sigma_hat) mu_hat              cost of estimating expected returns
    SDF sleeves A0 / A1    random-feature SDFs                what learning the SDF directly recovers
    book (before / after)  meta-combination and vol timing    the submitted portfolio

and compared with replicas of the organizers' benchmarks on the same data (Markowitz-ML:
inv(Sigma_hat) x XGBoost forecast of raw returns; Factor-ML: XGBoost decile long-short;
1/N). Every portfolio is scaled to unit ex-ante volatility, so the numbers are annualized
Sharpe ratios; the pre-test period is the validation period (portfolios formed before
1989-12) and the test period mirrors the CTF's.

Usage:
    python tools/make_synthetic_data.py --mode full --out data/synthetic/full --truth
    python tools/digital_twin.py --data data/synthetic/full [--data DIR2 ...] --out docs/figures
        [--preds DIR/preds.npz --raw-xgb DIR/preds.npz]   # optional: reuse learner forecasts

Writes <out>/digital_twin.csv, digital_twin.json and digital_twin.png/.pdf. Running the
learners takes ~10-30 minutes per world on a laptop; --preds reuses forecasts from an earlier
run of the same model and configuration (they are deterministic).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_synthetic_data as G  # noqa: E402

TEST_START = 1989 * 12 + 11  # month index of 1989-12, the first CTF portfolio month
ROWS = [  # key, label, group
    ("oracle", "Oracle: true $\\mu$, true $\\Sigma$", "ceiling"),
    ("true_mu_prism_risk", "True $\\mu$, PRISM risk model", "ceiling"),
    ("B_ens", "PRISM learner ensemble", "prism"),
    ("A0", "PRISM SDF sleeve, raw", "prism"),
    ("A1", "PRISM SDF sleeve, rotated", "prism"),
    ("book_raw", "PRISM book, no vol. timing", "prism"),
    ("book", "PRISM book", "prism_main"),
    ("markowitz_ml", "Markowitz-ML replica", "bench"),
    ("factor_ml", "Factor-ML replica", "bench"),
    ("one_over_n", "1/N market", "bench"),
]


def load_model(path):
    spec = importlib.util.spec_from_file_location("prism", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prism"] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.mean() / x.std() * np.sqrt(12)) if len(x) > 12 and x.std() > 0 else float("nan")


def true_risk(P, T, mkt_var):
    """True covariance (daily units) of a month's stocks from the generator's structure."""
    lr = G.MKT_LONG_RUN_VAR
    vs = (mkt_var / lr) ** 0.25
    ind = T["ff12"].map({n: i for i, n in enumerate(G.FF12_NAMES)}).to_numpy()
    B = np.column_stack([T["beta"], (ind[:, None] == np.arange(12)) * T["gamma"].to_numpy()[:, None],
                         T[["l_size", "l_value", "l_invest", "l_risk"]].to_numpy()])
    fv = np.concatenate([[mkt_var], np.full(12, G.IND_DAILY_VOL ** 2 * vs ** 2), G.STYLE_DAILY_VOL ** 2 * vs ** 2])
    dv = (T["sigma_idio"].to_numpy() * vs) ** 2 / 21.0
    return P.MonthRisk(B, np.diag(np.sqrt(fv)), dv, True)


def unit_ret(rk, w, y):
    v = rk.var(w) * 21.0
    return float(w @ y / np.sqrt(v)) if np.isfinite(v) and v > 0 else np.nan


def run_world(P, data, preds_file=None, raw_xgb_file=None, threads=2):
    t0 = time.time()
    chars = pd.read_parquet(data / "ctff_chars.parquet")
    feats = pd.read_parquet(data / "ctff_features.parquet")
    daily_ret = pd.read_parquet(data / "ctff_daily_ret.parquet")
    truth = pd.read_parquet(data / "truth_chars.parquet")
    mk = pd.read_parquet(data / "truth_market.parquet")
    cfg = dict(P.CONFIG)
    panel = P.Panel(chars, feats)
    daily = P.Daily(daily_ret, panel)
    with threadpool_limits(1, "blas"):
        risk = P.build_risk_model(panel, daily, cfg, {})
    with threadpool_limits(threads, "blas"):
        preds = dict(np.load(preds_file)) if preds_file else P.fit_learners(panel, cfg, {}, threads, risk)
        if raw_xgb_file:
            raw_xgb = dict(np.load(raw_xgb_file))["xgb"]
        else:  # the replica forecasts raw returns, as the organizers' XGBoost does
            raw_xgb = P.fit_learners(panel, dict(cfg, learners=("ridge", "xgb"), learn_resid=()),
                                     {}, threads, risk)["xgb"]
    diag = {}
    with threadpool_limits(1, "blas"):
        P.build_portfolios(panel, daily, risk, preds, cfg, diag)

    truth["m"] = P._month_index(truth["eom"])
    tr = pd.DataFrame({"id": panel.ids, "m": panel.month}).merge(truth, on=["id", "m"], how="left")
    mk["date"] = pd.to_datetime(mk["date"])
    mvar = (mk.groupby(mk["date"].dt.year * 12 + mk["date"].dt.month - 1)["mkt_vol"].last() ** 2).to_dict()
    series = {k: {} for k, _, _ in ROWS}
    for t in [int(m) for m in panel.months]:
        r = panel.rows(t)
        y = panel.y[r]
        series["one_over_n"][t] = y.mean()
        T = tr.iloc[r.start:r.stop]
        rk = risk[t]
        if not rk.ok or T["mu"].isna().any():
            continue
        mu = T["mu"].to_numpy().copy()
        mu -= mu.mean()
        rt = true_risk(P, T, mvar.get(t, G.MKT_LONG_RUN_VAR))
        series["oracle"][t] = unit_ret(rt, rt.solve(mu), y)
        series["true_mu_prism_risk"][t] = unit_ret(rt, rk.solve(mu), y)
        p = raw_xgb[r]
        if np.isfinite(p).all() and p.std() > 0:
            z = (p - p.mean()) / p.std()
            series["markowitz_ml"][t] = unit_ret(rk, rk.solve(z), y)
            q = pd.Series(p).rank(pct=True).to_numpy()
            series["factor_ml"][t] = y[q > 0.9].mean() - y[q <= 0.1].mean()
    sr = diag["sleeve_returns"]
    for s in ("B_ens", "A0", "A1"):
        g = sr[sr["sleeve"] == s]
        series[s] = dict(zip(g["eom"], g["ret"]))
    bk = diag["book"]
    series["book"] = dict(zip(bk["eom"], bk["ret"]))
    series["book_raw"] = dict(zip(bk["eom"], bk["ret"] / bk["scale"]))
    out = []
    for key, label, group in ROWS:
        s = pd.Series(series[key], dtype=float)
        idx = np.array([m if isinstance(m, (int, np.integer)) else
                        pd.Timestamp(m).year * 12 + pd.Timestamp(m).month - 1 for m in s.index])
        out.append({"key": key, "label": label, "group": group,
                    "sharpe_pre": sharpe(s.values[idx < TEST_START]),
                    "sharpe_test": sharpe(s.values[idx >= TEST_START]),
                    "months_test": int((idx >= TEST_START).sum())})
    print(f"{data.name}: done in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)


def figure(results, out):
    """Emphasis bar chart: PRISM's layers in the accent hue, ceilings and benchmarks in gray."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_report import AXIS, GRID, INK, INK_2, INK_MUTED, MPL_RC, OTHER, SLOTS

    accent, light_accent = SLOTS[0], "#9cc2ee"
    colors = {"ceiling": INK_MUTED, "prism": light_accent, "prism_main": accent, "bench": OTHER}
    worlds = list(results)
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(1, len(worlds), figsize=(6.5, 2.75), sharey=True, squeeze=False)
        for ax, w in zip(axes[0], worlds):
            df = results[w].iloc[::-1].reset_index(drop=True)
            y = np.arange(len(df))
            ax.barh(y, df["sharpe_test"], height=0.62, color=[colors[g] for g in df["group"]],
                    edgecolor="white", linewidth=1.0, zorder=2)
            ax.scatter(df["sharpe_pre"], y, s=16, facecolor="white", edgecolor=INK, linewidth=0.9,
                       zorder=3, label="pre-test (1957-1989)")
            for yi, v in zip(y, df["sharpe_test"]):
                ax.text(max(v, 0) + 0.04, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.5, color=INK_2)
            ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
            ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(w, fontsize=7.5, loc="left")
            ax.set_xlabel("Annualized Sharpe ratio, test period (bars) and pre-test (circles)", fontsize=6.5)
            ax.set_xlim(min(0, df[["sharpe_test", "sharpe_pre"]].min().min()) - 0.1,
                        df[["sharpe_test", "sharpe_pre"]].max().max() + 0.45)
            for sep in (2.5, 7.5):  # boundaries between benchmarks | PRISM | ceilings
                ax.axhline(sep, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
        axes[0][0].set_yticks(np.arange(len(ROWS)))
        axes[0][0].set_yticklabels([lab for _, lab, _ in ROWS][::-1], fontsize=6.8)
        axes[0][0].legend(loc="lower right", fontsize=6, frameon=False)
        fig.tight_layout(w_pad=1.2)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"digital_twin.{ext}", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, action="append", required=True, help="synthetic dataset with truth")
    ap.add_argument("--name", action="append", help="display name per --data (default: folder name)")
    ap.add_argument("--preds", type=Path, action="append", help="cached learner forecasts per --data")
    ap.add_argument("--raw-xgb", type=Path, action="append", help="cached raw-target XGBoost forecasts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "submission" / "prism.py")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    P = load_model(args.model)
    results = {}
    for i, d in enumerate(args.data):
        name = args.name[i] if args.name else d.name
        pf = args.preds[i] if args.preds else None
        rx = args.raw_xgb[i] if args.raw_xgb else None
        results[name] = run_world(P, d, pf, rx, args.threads)
    table = pd.concat([df.assign(world=w) for w, df in results.items()], ignore_index=True)
    table.to_csv(args.out / "digital_twin.csv", index=False)
    (args.out / "digital_twin.json").write_text(json.dumps(
        {w: df.set_index("key")[["sharpe_pre", "sharpe_test"]].round(3).to_dict("index")
         for w, df in results.items()}, indent=2))
    print(table.pivot(index="key", columns="world", values=["sharpe_pre", "sharpe_test"]).round(2))
    figure(results, args.out)
    print(f"figure: {args.out / 'digital_twin.png'}")


if __name__ == "__main__":
    main()
