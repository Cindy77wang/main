"""Shared helpers for the econometrics review problem set.

Everything statistical is written by hand with numpy. scipy is used only for
numerical optimisation (scipy.optimize) and for chi-square / normal tail
probabilities (scipy.stats), which the problem set allows.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Categorical palette, fixed order (blue, orange, aqua, yellow), never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dd"


def style_axes(ax):
    """Recessive grid and axes, thin spines."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def load_wsj() -> pd.DataFrame:
    df = pd.read_csv(INPUTS / "wsj-counts.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df.set_index("date")


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS from the normal equations with classical and White (HC1) covariances."""
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (T - k)
    V_classic = s2 * XtX_inv
    meat = (X * e[:, None] ** 2).T @ X
    V_white = XtX_inv @ meat @ XtX_inv * T / (T - k)
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return dict(b=b, resid=e, s2=s2, V=V_classic, V_white=V_white, r2=r2, T=T, k=k)


def newey_west(g: np.ndarray, lags: int, center: bool = True) -> np.ndarray:
    """Newey-West (Bartlett kernel) long-run covariance of the rows of g."""
    g = np.atleast_2d(g)
    if g.shape[0] < g.shape[1]:
        g = g.T
    T = g.shape[0]
    if center:
        g = g - g.mean(axis=0)
    S = g.T @ g / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = g[l:].T @ g[:-l] / T
        S += w * (G + G.T)
    return S


def nw_lag_rule(T: int) -> int:
    """Newey-West (1994) plug-in rule floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def fmt(x, d=4):
    return f"{x:.{d}f}"


def write_markdown_table(df: pd.DataFrame, path: Path, floatfmt: str = ".4f") -> str:
    """Write a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    idx_name = df.index.name or ""
    lines = ["| " + " | ".join([idx_name] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("")
                elif abs(v) >= 1e4 or (abs(v) < 1e-4 and v != 0):
                    cells.append(f"{v:.3e}")
                else:
                    cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text
