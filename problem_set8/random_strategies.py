"""
Problem Set 8 - Monte Carlo of random strategies.

Each month, draw one of the 75 size / past-return portfolios uniformly at random
(integer 1..75) and take its return as the strategy's return that month. Doing
this every month gives one random strategy; repeat 5,000 times.

For each strategy compute the mean, standard deviation, Sharpe ratio and the
t-statistic of the mean excess return, then plot the histogram of t-stats.

Usage:
    python random_strategies.py path/to/Problem_Set8_2026.xlsx [--sims 5000] [--seed 42]

Data notes
- Sheet "125 Size-Past return portfolios": column 0 is the date (YYYYMM) and the
  75 portfolios are three 5x5 blocks (past 1-month, 2-12-month, 13-60-month
  return). Columns 26 and 52 repeat the date and are dropped.
- -99.99 marks a missing return. The 13-60-month sort needs 5 years of history,
  and a few earlier months have gaps, so the sample is the contiguous block in
  which all 75 portfolios exist (1938-03 onward), so every draw is valid.
- Returns are in percent per month. Excess returns subtract RF from the
  "FF factors" sheet, so the sample ends at the last month with RF.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_data(path):
    raw = pd.read_excel(path, sheet_name="125 Size-Past return portfolios", header=None)
    body = raw.iloc[9:].reset_index(drop=True)
    dates = body[0].astype(int).values
    port_cols = [c for c in body.columns if c not in (0, 26, 52)]
    rets = body[port_cols].apply(pd.to_numeric, errors="coerce")
    rets = rets.mask(rets <= -99)                      # -99.99 = missing
    rets.index = dates
    rets.columns = range(1, 76)                        # portfolio numbers 1..75
    assert rets.shape[1] == 75

    ff = pd.read_excel(path, sheet_name="FF factors", header=0, index_col=0)
    ff.index = ff.index.astype(int)
    rf = ff["RF"]

    # Keep the contiguous block after the last month with any missing portfolio
    # (a few months before 1938 have gaps), so every draw is valid and the
    # series has no holes.
    last_gap = rets.index[rets.isna().any(axis=1)].max()
    rets = rets.loc[rets.index > last_gap]
    common = rets.index.intersection(rf.index)
    return rets.loc[common], rf.loc[common]


def simulate(rets, rf, n_sims, seed):
    rng = np.random.default_rng(seed)
    R = rets.values                                    # T x 75
    T = R.shape[0]
    picks = rng.integers(1, 76, size=(n_sims, T))      # integer 1..75 each month
    strat = R[np.arange(T), picks - 1]                 # n_sims x T raw returns
    excess = strat - rf.values                         # subtract that month's RF

    mean = strat.mean(axis=1)
    sd = strat.std(axis=1, ddof=1)
    ex_mean = excess.mean(axis=1)
    ex_sd = excess.std(axis=1, ddof=1)
    sharpe = ex_mean / ex_sd                           # monthly Sharpe
    tstat = ex_mean / (ex_sd / np.sqrt(T))             # = Sharpe * sqrt(T)
    return pd.DataFrame({
        "mean": mean, "std": sd,
        "excess_mean": ex_mean, "excess_std": ex_sd,
        "sharpe_monthly": sharpe, "sharpe_annual": sharpe * np.sqrt(12),
        "t_stat": tstat,
    }), T


def plot_tstats(res, T, first, last, out_png):
    t = res["t_stat"].values
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.hist(t, bins=60, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(1.96, color="#52514e", linestyle="--", linewidth=1)
    ax.text(1.96, ax.get_ylim()[1] * 0.97, " t = 1.96", color="#52514e",
            va="top", ha="left", fontsize=9)
    ax.axvline(t.mean(), color="#0b0b0b", linewidth=1.5)
    ax.text(t.mean(), ax.get_ylim()[1] * 0.88, f" mean = {t.mean():.2f}",
            color="#0b0b0b", va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_title(f"t-statistics of mean excess return, {len(t):,} random strategies\n"
                 f"(one of 75 portfolios drawn each month, {first}-{last}, T = {T})",
                 fontsize=11, loc="left")
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Number of strategies")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xlsx")
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="random_strategies")
    a = p.parse_args()

    rets, rf = load_data(a.xlsx)
    res, T = simulate(rets, rf, a.sims, a.seed)
    first, last = rets.index[0], rets.index[-1]

    print(f"Sample: {first}-{last}, T = {T} months, {a.sims} strategies\n")
    print("Cross-strategy distribution (returns in % per month):")
    print(res.describe(percentiles=[0.025, 0.5, 0.975]).round(4).to_string())
    print(f"\nShare with t > 1.96: {(res.t_stat > 1.96).mean():.2%}")
    print(f"Share with |t| > 1.96: {(res.t_stat.abs() > 1.96).mean():.2%}")

    # Reference: the 75 underlying portfolios themselves
    ex = rets.sub(rf, axis=0)
    t75 = ex.mean() / (ex.std(ddof=1) / np.sqrt(T))
    print(f"\n75 actual portfolios: t-stat range {t75.min():.2f} to {t75.max():.2f}, "
          f"mean {t75.mean():.2f}")

    res.to_csv(f"{a.out}.csv", index_label="strategy")
    plot_tstats(res, T, first, last, f"{a.out}_tstat_hist.png")
    print(f"\nSaved {a.out}.csv and {a.out}_tstat_hist.png")


if __name__ == "__main__":
    main()
