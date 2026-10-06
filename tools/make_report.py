"""Performance statistics, figures and the documentation PDF for a CTF portfolio model.

Turns the model output (the weights CSV plus, optionally, the diagnostics written by
tools/run_local.py) into the numbers, charts and the investor-style white paper that
accompany the submission (Rule 5, "Methodology Document").

Usage
-----
    python tools/make_report.py --data DIR --weights weights.csv [--diagnostics DIAG_DIR]
        --out docs/report [--doc-source docs/documentation.md] [--pdf PATH]
        [--model-name PRISM] [--max-pages 5] [--dpi 300] [--no-pdf] [--strict]

--data          folder with ctff_chars.parquet; only id, eom, eom_ret, ret_exc_lead1m,
                ctff_test and (if present) market_equity are read
--weights       CSV with columns id, eom, w (the CTF "Portfolio Weights" file)
--diagnostics   folder of <key>.parquet files (docs/DEV_SPEC.md section 5); every
                statistic, table and figure that needs a missing file is skipped
--doc-source    narrative source (syntax below); default docs/documentation.md if it
                exists, otherwise no PDF is built
--strict        exit with status 2 if there were warnings (missing placeholders,
                unavailable figures, TeX errors, more than --max-pages pages)

Outputs (under --out)
---------------------
    stats.json               every statistic (NaN -> null); source of {{stat:...}}
    performance_stats.md     markdown tables: the organizer's table, then extended,
                             decade, CAPM, sleeve and learner tables
    monthly_returns.csv      book / market returns, turnover, leverage per test month
    figures/<name>.pdf|png   standalone figures (vector PDF and PNG)
    documentation.pdf        the white paper (or --pdf PATH)
    _build/                  intermediates: equation images, document-sized figures

Statistics
----------
Semantics follow utils/R/performance_stats.R of the organizer repo exactly:
* ret_t = sum_i w_it r_it over the eoms flagged ctff_test (weights in other months are
  dropped with a warning). A held position (w != 0) without a realized return is an
  error, as in check_returns(); a zero weight without one contributes nothing.
* mean = 12 mean(ret); sd = sqrt(12) sd(ret) (ddof 1); sharpe = mean / sd.
* avg_stocks = mean over months of #{w != 0}; gross_leverage = mean of sum |w|;
  net_exposure = mean of sum w.
* turnover = mean over months 2..T of sum_i |w_it - w~_it|, where
  w~_it = w_i,t-1 (1 + r_i,t-1) / (1 + R_p,t-1); names entering or leaving count in full.
* max_dd = max_t (1 - W_t / max(1, max_{s<=t} W_s)) with W = cumprod(1 + ret);
  max_dd_scaled is the same for ret * 0.10 / sd (10% annual volatility).
Additions: t_stat = sqrt(T) mean/sd (monthly); bias-adjusted sample skewness and excess
kurtosis; hit rate; worst/best month; Sortino ratio. Sharpe standard error under i.i.d.
returns (Lo 2002; Jobson-Korkie): se = sqrt(12 (1 + SR_m^2 / 2) / T), 95% CI
SR +/- 1.96 se; the Mertens (2002) version with skewness and kurtosis is also reported.
Decades are calendar decades of the return month eom_ret (1990s = Jan 1990 - Dec 1999).
CAPM: OLS of ret_t on a market return with intercept, OLS t-statistics, alpha x 12. The
equal-weighted market is the average ret_exc_lead1m of all chars stocks at eom; the
value-weighted proxy weights stocks by the market_equity characteristic at eom (only if
that column exists and is a positive level).
Diagnostics: sleeve Sharpe pre-test vs test, sleeve correlations, average meta weights,
learner IC by period and year, chosen SDF shrinkage, leverage scale and ex-ante volatility
(and the bias statistic sd(ret_t / sigma_hat_t) of the volatility forecast).

Document source syntax (--doc-source)
-------------------------------------
A small Markdown dialect. Blocks are separated by blank lines.

    ---                          optional front matter at the top of the file:
    title: PRISM                 title, subtitle, author (default "Cindy Wang"), course
    subtitle: ...                (default "MGT 924 Statistical Foundations, Yale SOM"),
    ---                          date (default: current month), footer, max_pages (5)
    <!-- comment -->             removed (may span lines)
    # Heading {#sec:intro}       numbered section "1"; ## numbered "1.1"; ### unnumbered
                                 italic heading; a trailing {-} unnumbers # and ##
    text lines                   consecutive lines form one justified paragraph
    - item | * item | 1. item    bullet / numbered list (flat); indented lines continue
                                 an item; a blank line between items is allowed
    $$ tex $$ {#eq:name}         display equation, centred; numbered "(n)" if labelled;
                                 may span source lines; "\\" splits it into lines
    | a | b |                    pipe table; a |:--|--:| second row sets the alignment;
    |:--|--:|                    an optional "Table: caption {#tab:name}" line right
    Table: caption {#tab:x}      after it numbers and captions the table
    ::: abstract ... :::         fenced div: abstract (small, indented, run-in
                                 "Abstract."), box (shaded call-out), small (9 pt),
                                 references (one entry per line, hanging indent)
    [[figure:NAME[,NAME2] width=6.5in height=2.4in caption="..."]]
                                 generated figure, re-rendered at exactly this size so
                                 chart text stays 7-8 pt; several names are set side by
                                 side with (a), (b) labels (width is per panel; units in,
                                 cm, mm, pt or % of the text width); default captions
    [[table:NAME caption="..." width=100%]]
                                 generated table (booktabs style); perf also takes
                                 layout=wide (two column pairs, short labels),
                                 rows=key,key,... and labels=short|long
    [[image:relative/path.png width=6.5in caption="..."]]
                                 static image (path relative to the document source),
                                 numbered and captioned like a figure
    [[keyfacts stats="sharpe,mean:Return p.a.,max_dd_scaled"]]
                                 a row of headline numbers; KEY:Label overrides a label
    [[pagebreak]]  [[vspace 6pt]]  [[condbreak 2in]] (new page if less space is left)
Directives may span several lines; option values with spaces must be double-quoted.

Inline: **bold**, *italic*, `code`, [text](url), $tex$ inline math (pandoc rules: no
space after the opening or before the closing $, the closing $ not followed by a digit;
\\$ is a literal dollar), -- en dash, --- em dash; a hyphen before a digit that follows
a space or bracket is typeset as a true minus.
Math is matplotlib mathtext (a TeX subset): \\frac, \\dfrac (display-size fractions),
\\sqrt, \\sum, \\hat, \\mathbf, \\boldsymbol, \\operatorname, \\text, \\left( \\right)
work; environments (aligned, cases), \\Big, \\lvert, \\nolimits, \\tfrac and a bare %
do not. Unsupported TeX is printed as italic source with a warning.
Placeholders, substituted everywhere (also in captions, tables and math, where % is
escaped automatically):
    {{stat:key}}  {{stat:a.b.c}}  {{stat:key|fmt}}   value from stats.json; fmt is pct<d>,
        int, date, raw or a Python format spec (.3f); default pct1 for return, volatility,
        drawdown, turnover, hit-rate and weight keys, int for counts, .2f otherwise
    {{config:key}}  {{config:key|fmt}}                a value of the model configuration
    {{fig:name}} {{tab:name}} {{eq:name}} {{sec:name}}   number of a labelled object (a
        figure or table directive is labelled by its NAME, or by label=...)
Unknown keys render as "n/a" (references as "??") and raise a warning.

Figures: cumret, rolling_sharpe, drawdown, meta_weights, sleeve_corr, leverage,
learner_ic, decade_sharpe, zeta, sleeve_sharpe. Tables: perf, decades, capm, sleeves,
learners. Perf rows (rows=): period, mean, sd, sharpe, sharpe_ci, t_stat, sortino,
avg_stocks, gross_leverage, net_exposure, turnover, max_dd, max_dd_scaled, skewness,
kurtosis, hit_rate, worst_month, best_month, alpha_ew, beta_ew, alpha_vw, beta_vw.
A figure or table whose inputs are missing becomes a framed placeholder (numbering is
kept) and a warning.

Built with reportlab 5.0.1 (platypus) and matplotlib 3.11.2 (figures and mathtext); the
body font is STIX General (a Times design shipped with matplotlib, matching the mathtext
"stix" font set), falling back to the built-in Times-Roman.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import reportlab  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402
from reportlab.lib import colors as rl_colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm, inch, mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak,
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHOR = "Cindy Wang"
DEFAULT_COURSE = "MGT 924 Statistical Foundations, Yale SOM"
VOL_SCALE = 0.10  # annual volatility of the rescaled drawdown (performance_stats.R)
Z95 = 1.959963984540054
ROLL_MONTHS = 36
MIN_MONTHS_SHARPE = 12  # fewer months -> Sharpe statistics are reported as null

# --------------------------------------------------------------------------------------
# Visual identity: categorical slots in fixed order (validated colorblind-safe for
# adjacent pairs, dataviz reference palette), recessive chrome, a blue<->red diverging map.
# --------------------------------------------------------------------------------------

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b4b2a9"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MARKET = INK_2
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#a32626", "#e34948", "#f0efec", "#3987e5", "#104281"])
SLEEVE_LABELS = {"A0": "SDF-RF raw", "A1": "SDF-RF rotated", "B_ridge": "Ridge",
                 "B_xgb": "XGBoost", "B_mlp": "MLP", "B_lstm": "LSTM", "B_lgbm": "LightGBM",
                 "B_ens": "Learner ensemble"}
LEARNER_LABELS = {"ridge": "Ridge", "xgb": "XGBoost", "mlp": "MLP", "lstm": "LSTM",
                  "lgbm": "LightGBM"}
PCT_KEYS = {"mean", "sd", "max_dd", "max_dd_scaled", "turnover", "hit_rate", "worst_month",
            "best_month", "alpha", "resid_sd", "realized_vol", "vol_target", "avg_vol_synth",
            "avg_vol_fm", "exante_vol", "ic_pos_share", "avg_theta"}
DEFAULT_SPECS = {"avg_stocks": "int", "avg_universe": "int", "ic_mean": ".3f", "last": ".3g"}
MPL_RC = {
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
}

_WARNINGS: list[str] = []


def warn(msg: str) -> None:
    """Print a warning and remember it for --strict."""
    _WARNINGS.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def note(msg: str) -> None:
    """Informational message (does not count for --strict)."""
    print(f"note: {msg}")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def to_datetime(values: Any) -> pd.Series:
    """Dates as datetime64[ns] whatever the input (datetime.date objects, strings, ...)."""
    return pd.Series(pd.to_datetime(pd.Series(values))).astype("datetime64[ns]")


def load_returns(data_dir: Path) -> pd.DataFrame:
    """id, eom, eom_ret, r, test (bool) and, when usable, me from ctff_chars.parquet."""
    path = data_dir / "ctff_chars.parquet"
    names = pq.read_schema(path).names
    cols = ["id", "eom", "eom_ret", "ret_exc_lead1m", "ctff_test"]
    cols = [c for c in cols if c in names] + (["market_equity"] if "market_equity" in names else [])
    df = pq.read_table(path, columns=cols).to_pandas(date_as_object=False)
    out = pd.DataFrame({"id": df["id"].astype(np.int64), "eom": to_datetime(df["eom"]).to_numpy()})
    if "eom_ret" in df:
        out["eom_ret"] = to_datetime(df["eom_ret"]).to_numpy()
    else:
        out["eom_ret"] = out["eom"] + pd.offsets.MonthEnd(1)
    out["r"] = df["ret_exc_lead1m"].astype(np.float64).to_numpy()
    out["test"] = df["ctff_test"].astype(bool).to_numpy()
    if "market_equity" in df:
        me = df["market_equity"].astype(np.float64).to_numpy()
        finite = np.isfinite(me)
        if finite.any() and (me[finite] > 0).mean() >= 0.99:
            out["me"] = np.where(finite & (me > 0), me, np.nan)
        else:
            note("market_equity is not a positive level (transformed?); no value-weighted market")
    return out


def load_weights(path: Path) -> pd.DataFrame:
    """The weights CSV as id (int64), eom (datetime64), w (float64), checked like Rule 12."""
    pf = pd.read_csv(path)
    if list(pf.columns) != ["id", "eom", "w"]:
        warn(f"weights columns are {list(pf.columns)}, expected exactly id, eom, w")
    pf = pf[["id", "eom", "w"]].copy()
    if pf.isna().any().any():
        raise ValueError(f"{path}: weights contain missing values")
    pf["id"] = pf["id"].astype(np.int64)
    pf["eom"] = to_datetime(pf["eom"]).to_numpy()
    pf["w"] = pf["w"].astype(np.float64)
    if pf.duplicated(["id", "eom"]).any():
        raise ValueError(f"{path}: duplicated (id, eom) rows")
    return pf


def load_diagnostics(diag_dir: Path | None) -> dict[str, pd.DataFrame]:
    """All <key>.parquet files of diag_dir, with date columns as datetime64."""
    if diag_dir is None:
        return {}
    if not diag_dir.is_dir():
        warn(f"diagnostics folder {diag_dir} not found; diagnostics are skipped")
        return {}
    diag = {}
    for path in sorted(diag_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        for col in ("eom", "refit_eom"):
            if col in frame:
                frame[col] = to_datetime(frame[col]).to_numpy()
        for col in ("sleeve", "learner", "key", "step"):
            if col in frame:
                frame[col] = frame[col].astype(str)
        diag[path.stem] = frame
    return diag


# --------------------------------------------------------------------------------------
# Core statistics (utils/R/performance_stats.R)
# --------------------------------------------------------------------------------------

def attach_returns(pf: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Weights of test months with their realized return r (0 for unheld missing rows)."""
    test = rets.loc[rets["test"], ["id", "eom", "r"]]
    in_test = pf["eom"].isin(set(test["eom"]))
    if not in_test.all():
        warn(f"{int((~in_test).sum())} weight rows are outside the ctff_test months; dropped")
    m = pf.loc[in_test].merge(test, on=["id", "eom"], how="left", validate="one_to_one")
    n_miss = int((m["r"].isna() & (m["w"] != 0)).sum())
    if n_miss:
        raise ValueError(f"{n_miss} held positions have no realized return")
    m["r"] = m["r"].fillna(0.0)
    return m.sort_values(["eom", "id"], kind="stable").reset_index(drop=True)


def portfolio_returns(m: pd.DataFrame) -> pd.Series:
    """Monthly portfolio excess return sum_i w_i r_i, indexed by eom."""
    return (m["w"] * m["r"]).groupby(m["eom"]).sum().sort_index()


def monthly_turnover(m: pd.DataFrame) -> pd.Series:
    """sum_i |w_t - w~_t| for months 2..T, with w~ last month's weights after drifting."""
    eoms = np.sort(m["eom"].unique())
    held = m.loc[m["w"] != 0, ["id", "eom", "w", "r"]].copy()
    rp = (held["w"] * held["r"]).groupby(held["eom"]).transform("sum")
    held["w_drift"] = held["w"] * (1 + held["r"]) / (1 + rp)
    nxt = dict(zip(eoms[:-1], eoms[1:]))
    held["eom"] = held["eom"].map(nxt)
    prev = held.dropna(subset=["eom"])[["id", "eom", "w_drift"]]
    prev = prev.astype({"eom": "datetime64[ns]"})
    both = m[["id", "eom", "w"]].merge(prev, on=["id", "eom"], how="outer")
    both = both.loc[both["eom"] > eoms[0]].fillna({"w": 0.0, "w_drift": 0.0})
    return (both["w"] - both["w_drift"]).abs().groupby(both["eom"]).sum().sort_index()


def drawdown(ret: np.ndarray) -> np.ndarray:
    """1 - W_t / running peak of wealth, the peak starting at the initial wealth of 1."""
    wealth = np.cumprod(1.0 + np.asarray(ret, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return 1.0 - wealth / peak


def max_drawdown(ret: np.ndarray) -> float:
    return float(np.max(drawdown(ret))) if len(ret) else math.nan


def sharpe_block(ret: pd.Series | np.ndarray, min_months: int = 2) -> dict[str, float]:
    """Annualized mean, sd, Sharpe and its i.i.d. (Lo 2002) and Mertens (2002) SEs."""
    x = np.asarray(pd.Series(ret).dropna(), dtype=np.float64)
    t = len(x)
    out = {"months": t, "mean": math.nan, "sd": math.nan, "sharpe": math.nan,
           "sharpe_se": math.nan, "sharpe_ci_low": math.nan, "sharpe_ci_high": math.nan,
           "sharpe_se_mertens": math.nan}
    if t < max(2, min_months):
        return out
    mu, sd = x.mean(), x.std(ddof=1)
    out["mean"], out["sd"] = 12 * mu, math.sqrt(12) * sd
    if sd <= 0:
        return out
    sr = mu / sd
    se = math.sqrt((1 + 0.5 * sr * sr) / t)
    s = pd.Series(x)
    skew = float(s.skew()) if t > 2 else 0.0
    exk = float(s.kurt()) if t > 3 else 0.0
    var_m = (1 + 0.5 * sr * sr - skew * sr + 0.25 * exk * sr * sr) / t
    out.update(sharpe=sr * math.sqrt(12), sharpe_se=se * math.sqrt(12),
               sharpe_ci_low=(sr - Z95 * se) * math.sqrt(12),
               sharpe_ci_high=(sr + Z95 * se) * math.sqrt(12),
               sharpe_se_mertens=math.sqrt(max(var_m, 0.0)) * math.sqrt(12))
    return out


def ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """OLS of y on [1, x]: annualized alpha, beta, OLS t-stats, R^2, appraisal ratio."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    keys = ("alpha", "alpha_t", "beta", "beta_t", "r2", "resid_sd", "appraisal", "months")
    if n < 3 or np.var(x) == 0:
        return dict.fromkeys(keys, math.nan) | {"months": n}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * s2)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())) if np.var(y) > 0 else math.nan
    resid_sd = math.sqrt(s2 * 12)
    return {"alpha": 12 * coef[0], "alpha_t": coef[0] / se[0], "beta": coef[1],
            "beta_t": coef[1] / se[1], "r2": r2, "resid_sd": resid_sd,
            "appraisal": 12 * coef[0] / resid_sd if resid_sd > 0 else math.nan, "months": n}


def market_returns(rets: pd.DataFrame, eoms: pd.Index) -> pd.DataFrame:
    """Equal-weighted and (if possible) value-weighted proxy market returns per eom."""
    sub = rets.loc[rets["eom"].isin(set(eoms))]
    g = sub.groupby("eom")
    out = pd.DataFrame({"mkt_ew": g["r"].mean(), "n_universe": g["r"].size()})
    if "me" in sub:
        valid = sub.dropna(subset=["me"])
        num = (valid["me"] * valid["r"]).groupby(valid["eom"]).sum()
        den = valid["me"].groupby(valid["eom"]).sum()
        out["mkt_vw"] = num / den
    else:
        out["mkt_vw"] = np.nan
    return out.reindex(eoms)


def build_timeseries(m: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Per test month: eom_ret, ret, turnover, gross, net, n_stocks, markets."""
    ret = portfolio_returns(m)
    eoms = ret.index
    aw = m["w"].abs()
    ts = pd.DataFrame({"ret": ret})
    eom_ret = rets.drop_duplicates("eom").set_index("eom")["eom_ret"]
    ts.insert(0, "eom_ret", eom_ret.reindex(eoms).to_numpy())
    ts["turnover"] = monthly_turnover(m).reindex(eoms)
    ts["gross"] = aw.groupby(m["eom"]).sum()
    ts["net"] = m["w"].groupby(m["eom"]).sum()
    ts["n_stocks"] = (m["w"] != 0).groupby(m["eom"]).sum().astype(np.int64)
    ts = ts.join(market_returns(rets, eoms))
    ts.index.name = "eom"
    return ts


def performance_stats(ts: pd.DataFrame) -> dict[str, Any]:
    """The organizer's perf_stats() plus the extended book statistics."""
    r = ts["ret"]
    sb = sharpe_block(r)
    s: dict[str, Any] = {"start": ts.index.min(), "end": ts.index.max(), "months": len(ts),
                         "first_return_month": ts["eom_ret"].min(),
                         "last_return_month": ts["eom_ret"].max()}
    s.update({k: sb[k] for k in ("mean", "sd", "sharpe", "sharpe_se", "sharpe_ci_low",
                                 "sharpe_ci_high", "sharpe_se_mertens")})
    s["t_stat"] = s["sharpe"] / math.sqrt(12) * math.sqrt(len(r)) if len(r) > 1 else math.nan
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2) * 12)
    s["sortino"] = s["mean"] / downside if downside > 0 else math.nan
    s["avg_stocks"] = float(ts["n_stocks"].mean())
    s["gross_leverage"] = float(ts["gross"].mean())
    s["net_exposure"] = float(ts["net"].mean())
    s["turnover"] = float(ts["turnover"].mean()) if ts["turnover"].notna().any() else math.nan
    dd = drawdown(r.to_numpy())
    s["max_dd"] = float(dd.max())
    s["max_dd_date"] = ts["eom_ret"].iloc[int(dd.argmax())]
    s["max_dd_scaled"] = max_drawdown(r.to_numpy() * VOL_SCALE / sb["sd"]) if sb["sd"] > 0 else math.nan
    s["skewness"] = float(r.skew()) if len(r) > 2 else math.nan
    s["kurtosis"] = float(r.kurt()) if len(r) > 3 else math.nan
    s["hit_rate"] = float((r > 0).mean())
    s["worst_month"], s["worst_month_date"] = float(r.min()), ts.loc[r.idxmin(), "eom_ret"]
    s["best_month"], s["best_month_date"] = float(r.max()), ts.loc[r.idxmax(), "eom_ret"]
    return s


def decade_stats(ts: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Sharpe (with CI) of the book and the EW market per calendar decade of eom_ret."""
    decade = (pd.DatetimeIndex(ts["eom_ret"]).year // 10) * 10
    out = {}
    for d in sorted(set(decade)):
        sel = ts.loc[decade == d]
        b = sharpe_block(sel["ret"], MIN_MONTHS_SHARPE)
        mk = sharpe_block(sel["mkt_ew"], MIN_MONTHS_SHARPE)
        out[f"{d}s"] = b | {"mkt_ew_sharpe": mk["sharpe"], "mkt_ew_ci_low": mk["sharpe_ci_low"],
                            "mkt_ew_ci_high": mk["sharpe_ci_high"],
                            "max_dd": max_drawdown(sel["ret"].to_numpy())}
    return out


def market_stats(ts: pd.DataFrame, col: str) -> dict[str, float] | None:
    x = ts[col]
    if x.isna().all():
        return None
    b = sharpe_block(x)
    return {k: b[k] for k in ("mean", "sd", "sharpe", "sharpe_ci_low", "sharpe_ci_high")} | {
        "max_dd": max_drawdown(x.dropna().to_numpy()), "months": int(x.notna().sum())}


# --------------------------------------------------------------------------------------
# Diagnostics statistics
# --------------------------------------------------------------------------------------

def sleeve_order(diag: dict[str, pd.DataFrame]) -> list[str]:
    """Sleeves in order of first appearance (the model writes them in a fixed order)."""
    for key in ("sleeve_returns", "meta_weights"):
        if key in diag and "sleeve" in diag[key]:
            return list(dict.fromkeys(diag[key]["sleeve"]))
    return []


def learner_order(diag: dict[str, pd.DataFrame]) -> list[str]:
    if "learner_ic" in diag:
        return list(dict.fromkeys(diag["learner_ic"]["learner"]))
    return []


def wide(diag: dict[str, pd.DataFrame], key: str, col: str, value: str) -> pd.DataFrame | None:
    """Long diagnostics table -> eom x col matrix (None if unavailable)."""
    if key not in diag or diag[key].empty:
        return None
    d = diag[key]
    return d.pivot_table(index="eom", columns=col, values=value, aggfunc="last").sort_index()


def sleeve_stats(diag: dict[str, pd.DataFrame], ts: pd.DataFrame,
                 test_start: pd.Timestamp) -> dict[str, Any]:
    """Per-sleeve Sharpe pre-test vs test, meta weights, correlations; book pre vs test."""
    out: dict[str, Any] = {}
    sw = wide(diag, "sleeve_returns", "sleeve", "ret")
    mw = wide(diag, "meta_weights", "sleeve", "theta")
    order = sleeve_order(diag)
    if sw is not None:
        test_idx = sw.index.intersection(ts.index)
        pre = sw.loc[sw.index < test_start]
        per = {}
        for s in order:
            if s not in sw:
                continue
            a, b = sharpe_block(pre[s], MIN_MONTHS_SHARPE), sharpe_block(sw.loc[test_idx, s], MIN_MONTHS_SHARPE)
            corr = sw.loc[test_idx, s].corr(ts.loc[test_idx, "ret"]) if len(test_idx) > 2 else math.nan
            per[s] = {"label": SLEEVE_LABELS.get(s, s), "sharpe_pre": a["sharpe"],
                      "months_pre": a["months"], "sharpe_test": b["sharpe"],
                      "sharpe_test_ci_low": b["sharpe_ci_low"],
                      "sharpe_test_ci_high": b["sharpe_ci_high"], "months_test": b["months"],
                      "corr_book": corr}
            if mw is not None and s in mw:
                per[s]["avg_theta_test"] = float(mw.loc[mw.index >= test_start, s].fillna(0).mean())
                per[s]["avg_theta_pre"] = float(mw.loc[mw.index < test_start, s].fillna(0).mean())
        out["sleeves"] = per
        combo = sw[[s for s in order if s in sw]].mean(axis=1, skipna=True)
        out["sleeve_ew_combo"] = {
            "sharpe_pre": sharpe_block(combo[combo.index < test_start], MIN_MONTHS_SHARPE)["sharpe"],
            "sharpe_test": sharpe_block(combo.reindex(test_idx), MIN_MONTHS_SHARPE)["sharpe"]}
        cm_ = sw.loc[test_idx, [s for s in order if s in sw]].copy()
        cm_["Book"] = ts.loc[test_idx, "ret"]
        corr = cm_.corr(min_periods=12)
        out["sleeve_corr"] = {r: {c: corr.loc[r, c] for c in corr.columns} for r in corr.index}
        off = corr.drop(index="Book", columns="Book").to_numpy()
        iu = np.triu_indices_from(off, k=1)
        vals = off[iu][np.isfinite(off[iu])]
        out["sleeve_corr_avg"] = float(vals.mean()) if len(vals) else math.nan
    if "book" in diag and "ret" in diag["book"]:
        bk = diag["book"].set_index("eom").sort_index()
        out["book_pre"] = sharpe_block(bk.loc[bk.index < test_start, "ret"], MIN_MONTHS_SHARPE)
        if "scale" in bk:
            # The meta-combined book before the volatility-timing overlay (ablation)
            raw = bk["ret"] / bk["scale"].where(bk["scale"] > 0)
            out["book_unscaled"] = {
                "sharpe_pre": sharpe_block(raw[raw.index < test_start], MIN_MONTHS_SHARPE)["sharpe"],
                "sharpe_test": sharpe_block(raw[raw.index >= test_start], MIN_MONTHS_SHARPE)["sharpe"]}
    return out


def learner_stats(diag: dict[str, pd.DataFrame], test_start: pd.Timestamp) -> dict[str, Any] | None:
    """Mean OOS IC per learner pre-test and in the test period, with t-statistics."""
    icw = wide(diag, "learner_ic", "learner", "ic")
    if icw is None:
        return None
    out = {}
    for k in learner_order(diag):
        x = icw[k].dropna()
        pre, tst = x[x.index < test_start], x[x.index >= test_start]
        row: dict[str, Any] = {"label": LEARNER_LABELS.get(k, k)}
        for tag, v in (("pre", pre), ("test", tst)):
            n = len(v)
            row[f"ic_mean_{tag}"] = float(v.mean()) if n else math.nan
            row[f"ic_t_{tag}"] = float(v.mean() / v.std(ddof=1) * math.sqrt(n)) if n > 2 and v.std() > 0 else math.nan
            row[f"ic_pos_share_{tag}"] = float((v > 0).mean()) if n else math.nan
            row[f"months_{tag}"] = n
        out[k] = row
    return out


def book_diag_stats(diag: dict[str, pd.DataFrame], ts: pd.DataFrame,
                    test_start: pd.Timestamp, vol_target: float) -> dict[str, Any] | None:
    """Leverage scale, ex-ante vol and the bias statistic of the volatility forecast."""
    if "book" not in diag or diag["book"].empty:
        return None
    bk = diag["book"].set_index("eom").sort_index()
    t = bk.loc[bk.index >= test_start]
    vol_used = t["vol_synth"].where(t["vol_synth"] > 0, t["vol_fm"])
    exante = t["scale"] * vol_used  # annualized ex-ante vol of the final book
    z = t["ret"] / (exante / math.sqrt(12))
    common = t.index.intersection(ts.index)
    diff = (t.loc[common, "ret"] - ts.loc[common, "ret"]).abs()
    return {
        "avg_scale": float(t["scale"].mean()), "max_scale": float(t["scale"].max()),
        "avg_vol_synth": float(t["vol_synth"].mean()), "avg_vol_fm": float(t["vol_fm"].mean()),
        "exante_vol": float(exante.mean()), "realized_vol": float(t["ret"].std() * math.sqrt(12)),
        "vol_target": vol_target,
        "vol_bias": float(z.std()) if z.notna().sum() > 2 else math.nan,
        "corr_scale_vol": float(t["scale"].corr(vol_used)) if len(t) > 2 else math.nan,
        "book_ret_max_abs_diff": float(diff.max()) if len(diff) else math.nan,
    }


def zeta_stats(diag: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    if "sdf_zeta" not in diag or diag["sdf_zeta"].empty:
        return None
    out = {}
    for s, g in diag["sdf_zeta"].groupby("sleeve", sort=False):
        z = g.sort_values("eom")["zeta"]
        out[s] = {"median_log10": float(np.log10(z).median()), "last": float(z.iloc[-1]),
                  "refits": int(len(z))}
    return out


def config_dict(diag: dict[str, pd.DataFrame]) -> dict[str, str]:
    if "config" not in diag:
        return {}
    return dict(zip(diag["config"]["key"].astype(str), diag["config"]["value"].astype(str)))


def compute_all_stats(pf: pd.DataFrame, rets: pd.DataFrame, diag: dict[str, pd.DataFrame],
                      model: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Every statistic of the report, plus the monthly time series behind them."""
    m = attach_returns(pf, rets)
    if m.empty:
        raise ValueError("no weights in ctff_test months")
    ts = build_timeseries(m, rets)
    test_start = rets.loc[rets["test"], "eom"].min()
    stats: dict[str, Any] = {"model": model}
    stats.update(performance_stats(ts))
    stats["decades"] = decade_stats(ts)
    stats["market_ew"] = market_stats(ts, "mkt_ew")
    stats["market_vw"] = market_stats(ts, "mkt_vw")
    stats["corr_mkt_ew"] = float(ts["ret"].corr(ts["mkt_ew"])) if len(ts) > 2 else math.nan
    stats["capm_ew"] = ols(ts["ret"].to_numpy(), ts["mkt_ew"].to_numpy())
    stats["capm_vw"] = ols(ts["ret"].to_numpy(), ts["mkt_vw"].to_numpy()) if stats["market_vw"] else None
    stats["avg_universe"] = float(ts["n_universe"].mean())
    j = rets.loc[rets["test"], ["id", "eom"]].merge(pf[["id", "eom"]], how="outer", indicator=True)
    stats["checks"] = {"missing_test_rows": int((j["_merge"] == "left_only").sum()),
                       "rows_outside_test": int((j["_merge"] == "right_only").sum()),
                       "test_months_without_weights": int(
                           rets.loc[rets["test"], "eom"].nunique() - len(ts))}
    if stats["checks"]["missing_test_rows"]:
        warn(f"{stats['checks']['missing_test_rows']} ctff_test rows have no weight (Rule 5)")
    cfg = config_dict(diag)
    stats.update(sleeve_stats(diag, ts, test_start))
    stats["learners"] = learner_stats(diag, test_start)
    vol_target = _to_float(cfg.get("vol_target")) or VOL_SCALE
    stats["book_diag"] = book_diag_stats(diag, ts, test_start, vol_target)
    stats["zeta"] = zeta_stats(diag)
    if "timing" in diag:
        stats["runtime_minutes"] = float(diag["timing"]["seconds"].sum() / 60)
        stats["runtime_hours"] = stats["runtime_minutes"] / 60
    if "themes" in diag and len(diag["themes"]):
        th = diag["themes"]
        last = th[th["refit_eom"] == th["refit_eom"].max()]
        stats["n_features"] = int(th["feature"].nunique())
        stats["n_themes_last"] = int(last.loc[last["cluster"] >= 0, "cluster"].nunique())
    return stats, ts


def _to_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def json_clean(o: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python, dates -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, np.datetime64, date)):
        return pd.Timestamp(o).strftime("%Y-%m-%d")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


# --------------------------------------------------------------------------------------
# Value formatting and tables
# --------------------------------------------------------------------------------------

def lookup(stats: dict[str, Any], path: str) -> Any:
    cur: Any = stats
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def default_format(leaf: str, value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "raw"
    if isinstance(value, str):
        return "date" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else "raw"
    if isinstance(value, int):
        return "int"
    base = re.sub(r"_(pre|test|ew|vw)$", "", leaf)
    if base in DEFAULT_SPECS or leaf in DEFAULT_SPECS:
        return DEFAULT_SPECS.get(leaf) or DEFAULT_SPECS[base]
    return "pct1" if leaf in PCT_KEYS or base in PCT_KEYS else ".2f"


def format_value(value: Any, spec: str) -> str:
    """Format a statistic: pct<d>, int, date, raw or a Python format spec."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if spec == "raw":
        return str(value)
    if spec == "date":
        return pd.Timestamp(value).strftime("%b %Y")
    if spec == "int":
        return f"{int(round(float(value))):,}"
    if spec.startswith("pct"):
        d = int(spec[3:] or 1)
        return _no_neg_zero(f"{100 * float(value):.{d}f}%")
    try:
        return _no_neg_zero(format(float(value), spec))
    except ValueError:
        return format(int(round(float(value))), spec)


def _no_neg_zero(s: str) -> str:
    """'-0.00' -> '0.00' (a tiny negative number rounded to zero)."""
    return s[1:] if re.fullmatch(r"-0[.,]?0*%?", s) else s


def stat_text(stats: dict[str, Any], path: str, spec: str | None = None) -> str:
    """Formatted statistic for placeholders and tables ("n/a" if missing)."""
    try:
        value = json_clean(lookup(stats, path))
    except KeyError:
        warn(f"unknown statistic '{path}'")
        return "n/a"
    if isinstance(value, (dict, list)):
        warn(f"statistic '{path}' is a group, not a value")
        return "n/a"
    return format_value(value, spec or default_format(path.split(".")[-1], value))


# key -> (organizer-style label, short label for compact tables, value)
PERF_ROWS: dict[str, tuple[str, str, Callable[[dict[str, Any]], str]]] = {
    "period": ("Test period (portfolio formation dates)", "Test period", lambda s: (
        f"{stat_text(s, 'start')} to {stat_text(s, 'end')} ({s['months']} months)")),
    "mean": ("Average return (annualized)", "Excess return (ann.)", lambda s: stat_text(s, "mean")),
    "sd": ("Standard deviation (annualized)", "Volatility (ann.)", lambda s: stat_text(s, "sd")),
    "sharpe": ("Sharpe ratio (annualized)", "Sharpe ratio (ann.)", lambda s: stat_text(s, "sharpe")),
    "sharpe_ci": ("Sharpe ratio, 95% CI (i.i.d., Lo 2002)", "Sharpe 95% CI (Lo 2002)", lambda s: (
        f"[{stat_text(s, 'sharpe_ci_low')}, {stat_text(s, 'sharpe_ci_high')}]")),
    "t_stat": ("t-statistic of the mean", "t-statistic of mean", lambda s: stat_text(s, "t_stat")),
    "sortino": ("Sortino ratio (annualized)", "Sortino ratio", lambda s: stat_text(s, "sortino")),
    "avg_stocks": ("Average number of stocks", "Avg. number of stocks",
                   lambda s: stat_text(s, "avg_stocks", "int")),
    "gross_leverage": ("Gross leverage (average $\\Sigma_i |w_{i,t}|$)", "Gross leverage",
                       lambda s: stat_text(s, "gross_leverage")),
    "net_exposure": ("Net exposure (average $\\Sigma_i w_{i,t}$)", "Net exposure",
                     lambda s: stat_text(s, "net_exposure")),
    "turnover": ("Turnover (average monthly)", "Turnover (monthly)", lambda s: stat_text(s, "turnover")),
    "max_dd": ("Maximum drawdown", "Max drawdown", lambda s: stat_text(s, "max_dd")),
    "max_dd_scaled": ("Maximum drawdown, scaled to 10% volatility", "Max drawdown at 10% vol.",
                      lambda s: stat_text(s, "max_dd_scaled")),
    "skewness": ("Skewness (monthly)", "Skewness", lambda s: stat_text(s, "skewness")),
    "kurtosis": ("Excess kurtosis (monthly)", "Excess kurtosis", lambda s: stat_text(s, "kurtosis")),
    "hit_rate": ("Hit rate (months with positive return)", "Hit rate",
                 lambda s: stat_text(s, "hit_rate")),
    "worst_month": ("Worst month", "Worst month", lambda s: (
        f"{stat_text(s, 'worst_month')} ({stat_text(s, 'worst_month_date')})")),
    "best_month": ("Best month", "Best month", lambda s: (
        f"{stat_text(s, 'best_month')} ({stat_text(s, 'best_month_date')})")),
    "alpha_ew": ("CAPM alpha vs. EW market (annualized, t)", "Alpha vs. EW market (t)",
                 lambda s: _alpha_text(s, "capm_ew")),
    "beta_ew": ("CAPM beta vs. EW market", "Beta vs. EW market", lambda s: _beta_text(s, "capm_ew")),
    "alpha_vw": ("CAPM alpha vs. VW-proxy market (annualized, t)", "Alpha vs. VW market (t)",
                 lambda s: _alpha_text(s, "capm_vw")),
    "beta_vw": ("CAPM beta vs. VW-proxy market", "Beta vs. VW market",
                lambda s: _beta_text(s, "capm_vw")),
}
ORGANIZER_ROWS = ["period", "mean", "sd", "sharpe", "avg_stocks", "gross_leverage", "turnover",
                  "max_dd", "max_dd_scaled"]
EXTENDED_ROWS = ["sharpe_ci", "t_stat", "sortino", "net_exposure", "skewness", "kurtosis",
                 "hit_rate", "worst_month", "best_month", "alpha_ew", "beta_ew", "alpha_vw",
                 "beta_vw"]


def _alpha_text(s: dict[str, Any], key: str) -> str:
    if not s.get(key):
        return "n/a"
    return f"{stat_text(s, key + '.alpha')} ({stat_text(s, key + '.alpha_t')})"


def _beta_text(s: dict[str, Any], key: str) -> str:
    return stat_text(s, key + ".beta") if s.get(key) else "n/a"


@dataclass
class TableData:
    """A generated table: header row, body rows (inline markup allowed), alignments."""
    header: list[str]
    rows: list[list[str]]
    align: list[str]  # "l", "r" or "c" per column
    caption: str


def perf_table(stats: dict[str, Any], rows: list[str] | None = None, short: bool = False) -> TableData:
    keys = rows or (ORGANIZER_ROWS + EXTENDED_ROWS)
    body = []
    for k in keys:
        if k not in PERF_ROWS:
            warn(f"unknown perf table row '{k}'")
            continue
        if k.endswith("_vw") and not stats.get("capm_vw"):
            continue
        label, short_label, fn = PERF_ROWS[k]
        body.append([short_label if short else label, fn(stats)])
    return TableData(["Statistic", "Value"], body, ["l", "r"],
                     f"Out-of-sample performance of the {stats['model']} portfolio, test period.")


def decades_table(stats: dict[str, Any]) -> TableData:
    body = []
    for d, v in stats["decades"].items():
        cell = {k: stat_text(stats, f"decades.{d}.{k}") for k in
                ("mean", "sd", "sharpe", "sharpe_ci_low", "sharpe_ci_high", "mkt_ew_sharpe", "max_dd")}
        body.append([d, str(v["months"]), cell["mean"], cell["sd"], cell["sharpe"],
                     f"[{cell['sharpe_ci_low']}, {cell['sharpe_ci_high']}]", cell["mkt_ew_sharpe"],
                     cell["max_dd"]])
    s = sharpe_full_row(stats)
    body.append(["Full sample", str(stats["months"]), *s])
    return TableData(["Decade", "Months", "Mean", "Vol.", "Sharpe", "95% CI", "EW mkt SR", "Max DD"],
                     body, ["l", "r", "r", "r", "r", "r", "r", "r"],
                     "Performance by calendar decade of the return month.")


def sharpe_full_row(stats: dict[str, Any]) -> list[str]:
    mk = stats.get("market_ew") or {}
    return [stat_text(stats, "mean"), stat_text(stats, "sd"), stat_text(stats, "sharpe"),
            f"[{stat_text(stats, 'sharpe_ci_low')}, {stat_text(stats, 'sharpe_ci_high')}]",
            format_value(json_clean(mk.get("sharpe")), ".2f"), stat_text(stats, "max_dd")]


def capm_table(stats: dict[str, Any]) -> TableData:
    body = []
    for key, name in (("capm_ew", "Equal-weighted market"), ("capm_vw", "Value-weighted proxy")):
        c = stats.get(key)
        if not c:
            continue
        body.append([name, stat_text(stats, f"{key}.alpha"), stat_text(stats, f"{key}.alpha_t"),
                     stat_text(stats, f"{key}.beta"), stat_text(stats, f"{key}.beta_t"),
                     stat_text(stats, f"{key}.r2"), stat_text(stats, f"{key}.appraisal")])
    return TableData(["Benchmark", "$\\alpha$ (ann.)", "t($\\alpha$)", "$\\beta$", "t($\\beta$)",
                      "$R^2$", "Appraisal"], body, ["l"] + ["r"] * 6,
                     "Market regressions of the monthly book return (OLS t-statistics).")


def sleeves_table(stats: dict[str, Any]) -> TableData | None:
    sl = stats.get("sleeves")
    if not sl:
        return None
    fmt = lambda v, spec: format_value(json_clean(v), spec)  # noqa: E731
    body = [[v["label"], fmt(v["sharpe_pre"], ".2f"), fmt(v["sharpe_test"], ".2f"),
             fmt(v.get("avg_theta_test"), "pct0"), fmt(v["corr_book"], ".2f")] for v in sl.values()]
    combo = stats.get("sleeve_ew_combo", {})
    body.append(["Equal-weight sleeves", fmt(combo.get("sharpe_pre"), ".2f"),
                 fmt(combo.get("sharpe_test"), ".2f"), "", ""])
    raw = stats.get("book_unscaled", {})
    if raw:
        body.append(["Meta book, no vol. timing", fmt(raw.get("sharpe_pre"), ".2f"),
                     fmt(raw.get("sharpe_test"), ".2f"), "", ""])
    pre = (stats.get("book_pre") or {}).get("sharpe")
    body.append([f"{stats['model']} book", fmt(pre, ".2f"), stat_text(stats, "sharpe"), "100%", "1.00"])
    return TableData(["Sleeve", "SR pre-test", "SR test", "Avg. weight", "Corr. book"], body,
                     ["l", "r", "r", "r", "r"],
                     "Sleeves: annualized Sharpe ratios before and during the test period, "
                     "average meta weight and correlation with the final book (test period).")


def learners_table(stats: dict[str, Any]) -> TableData | None:
    ls = stats.get("learners")
    if not ls:
        return None
    fmt = lambda v, spec: format_value(json_clean(v), spec)  # noqa: E731
    body = [[v["label"], fmt(v["ic_mean_pre"], ".3f"), fmt(v["ic_mean_test"], ".3f"),
             fmt(v["ic_t_test"], ".2f"), fmt(v["ic_pos_share_test"], "pct0")] for v in ls.values()]
    return TableData(["Learner", "IC pre-test", "IC test", "t(IC) test", "IC > 0"], body,
                     ["l", "r", "r", "r", "r"],
                     "Out-of-sample cross-sectional information coefficients of the return learners.")


TABLES: dict[str, Callable[[dict[str, Any]], TableData | None]] = {
    "perf": perf_table, "decades": decades_table, "capm": capm_table,
    "sleeves": sleeves_table, "learners": learners_table,
}


def markdown_table(t: TableData, math_to_md: bool = True) -> list[str]:
    sep = {"l": ":--", "r": "--:", "c": ":-:"}
    lines = ["| " + " | ".join(t.header) + " |", "|" + "|".join(sep[a] for a in t.align) + "|"]
    lines += ["| " + " | ".join(c.replace("|", "\\vert ") if math_to_md and "$" in c else c
                                for c in row) + " |" for row in t.rows]
    return lines


def write_markdown(stats: dict[str, Any], path: Path) -> None:
    """Organizer-format table first (as scripts/performance_stats.R), then the rest."""
    org = perf_table(stats, ORGANIZER_ROWS)
    org.rows = [[r[0].replace("$\\Sigma_i |w_{i,t}|$", "$\\sum_i \\lvert w_{i,t} \\rvert$"), r[1]]
                for r in org.rows]
    lines = markdown_table(org, math_to_md=False) + [""]
    lines += ["### Extended statistics", ""] + markdown_table(perf_table(stats, EXTENDED_ROWS)) + [""]
    for name in ("decades", "capm", "sleeves", "learners"):
        t = TABLES[name](stats)
        if t is not None and t.rows:
            lines += [f"### {t.caption}", ""] + markdown_table(t) + [""]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

@dataclass
class Ctx:
    """Everything the figure functions need."""
    model: str
    ts: pd.DataFrame
    stats: dict[str, Any]
    diag: dict[str, pd.DataFrame]
    test_start: pd.Timestamp
    sleeves: list[str] = field(default_factory=list)
    learners: list[str] = field(default_factory=list)
    colors: dict[str, str] = field(default_factory=dict)

    def sleeve_wide(self) -> pd.DataFrame | None:
        return wide(self.diag, "sleeve_returns", "sleeve", "ret")


def assign_colors(sleeves: list[str], learners: list[str]) -> dict[str, str]:
    """Book = slot 1, sleeves the next slots in fixed order; a learner shares its sleeve's hue."""
    colors = {"__book__": SLOTS[0]}
    free = iter(SLOTS[1:])
    for s in sleeves:
        colors[s] = next(free, OTHER)
    if len(sleeves) > len(SLOTS) - 1:
        warn(f"{len(sleeves)} sleeves exceed the {len(SLOTS) - 1} categorical slots; extras are gray")
    for k in learners:
        colors[f"L_{k}"] = colors.get(f"B_{k}") or next(free, OTHER)
    return colors


def _date_axis(ax: Any, start: pd.Timestamp, end: pd.Timestamp, narrow: bool) -> None:
    years = (end - start).days / 365.25
    if years < 2:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        return
    base = next(b for b, lim in ((1, 4), (2, 10), (5, 40), (10, 1e9)) if years < lim)
    if narrow and base < 10:
        base *= 2
    ax.xaxis.set_major_locator(mdates.YearLocator(base=base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(start, end)


def _mark_test_start(ax: Any, ctx: Ctx, xmin: pd.Timestamp) -> None:
    if xmin < ctx.test_start:
        _vline_label(ax, mdates.date2num(ctx.test_start), "test period")


def _vline_label(ax: Any, x: float, text: str) -> None:
    """Vertical hairline with a label at the top, on whichever side has room."""
    ax.axvline(x, color=INK_MUTED, lw=0.6, zorder=1)
    lo, hi = ax.get_xlim()
    right = (x - lo) / (hi - lo) > 0.8 if hi > lo else False
    ax.annotate(text, xy=(x, 1), xycoords=("data", "axes fraction"),
                xytext=(-3 if right else 3, -2), textcoords="offset points",
                ha="right" if right else "left", va="top", fontsize=6.5, color=INK_2,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})


def _pct_axis(ax: Any, decimals: int = 0) -> None:
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=decimals))


def _legend_top(fig: Figure, ncol: int) -> None:
    """Shared legend above the plot (below it when a figure title occupies the top)."""
    if fig.axes and fig.axes[0].get_legend_handles_labels()[0]:
        loc = "outside lower center" if getattr(fig, "_has_title", False) else "outside upper center"
        fig.legend(*_all_handles(fig), loc=loc, ncol=ncol, frameon=False)


def _all_handles(fig: Figure) -> tuple[list, list]:
    hs, ls = [], []
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in ls and not lab.startswith("_"):
                hs.append(h)
                ls.append(lab)
    return hs, ls


def _cum_log(r: pd.Series, x0: pd.Timestamp, x: pd.Series) -> tuple[list, np.ndarray]:
    y = np.concatenate([[0.0], np.log1p(r.fillna(0.0).to_numpy()).cumsum()])
    return [x0, *list(x)], y


def fig_cumret(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Cumulative log excess return of the book, the sleeves and the EW market (vol-matched)."""
    ts = ctx.ts
    if len(ts) < 2:
        return False
    ax = fig.add_subplot()
    target = ts["ret"].std()
    x0 = ts.index[0]
    sw = ctx.sleeve_wide()
    if sw is not None:
        sw = sw.reindex(ts.index)
        for s in ctx.sleeves:
            r = sw.get(s)
            if r is None or r.notna().sum() < 12 or not r.std() > 0:
                continue
            ax.plot(*_cum_log(r * target / r.std(), x0, ts["eom_ret"]), color=ctx.colors[s],
                    lw=0.8, label=SLEEVE_LABELS.get(s, s))
    if ts["mkt_ew"].std() > 0:
        mk = ts["mkt_ew"] * target / ts["mkt_ew"].std()
        ax.plot(*_cum_log(mk, x0, ts["eom_ret"]), color=MARKET, lw=0.9, label="EW market")
    ax.plot(*_cum_log(ts["ret"], x0, ts["eom_ret"]), color=ctx.colors["__book__"], lw=1.7,
            label=f"{ctx.model} book", zorder=5)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_ylabel("Cumulative log excess return")
    _date_axis(ax, x0, ts["eom_ret"].iloc[-1], narrow)
    _legend_top(fig, 3 if narrow else 4 if len(ctx.sleeves) > 2 else 3)
    return True


def fig_rolling_sharpe(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Rolling 36-month annualized Sharpe ratio of the book and the EW market."""
    ts = ctx.ts
    if len(ts) < ROLL_MONTHS + 2:
        return False
    ax = fig.add_subplot()

    def roll(r: pd.Series) -> pd.Series:
        return r.rolling(ROLL_MONTHS).mean() / r.rolling(ROLL_MONTHS).std() * math.sqrt(12)

    x = ts["eom_ret"]
    ax.plot(x, roll(ts["mkt_ew"]).to_numpy(), color=MARKET, lw=0.9, label="EW market")
    ax.plot(x, roll(ts["ret"]).to_numpy(), color=ctx.colors["__book__"], lw=1.5,
            label=f"{ctx.model} book", zorder=5)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    sr = ctx.stats["sharpe"]
    if sr is not None:
        ax.axhline(sr, color=ctx.colors["__book__"], lw=0.6, alpha=0.6, zorder=1)
        ax.annotate(f"full sample {sr:.2f}".replace("-", "\u2212"), xy=(1, sr),
                    xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_ylabel(f"Rolling {ROLL_MONTHS}-month Sharpe")
    _date_axis(ax, x.iloc[0], x.iloc[-1], narrow)
    _legend_top(fig, 2)
    return True


def fig_drawdown(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Drawdown of compounded returns: book and EW market scaled to the book's volatility."""
    ts = ctx.ts
    if len(ts) < 2:
        return False
    ax = fig.add_subplot()
    x = [ts.index[0], *list(ts["eom_ret"])]
    dd = -np.concatenate([[0.0], drawdown(ts["ret"].to_numpy())])
    if ts["mkt_ew"].std() > 0:
        mk = ts["mkt_ew"] * ts["ret"].std() / ts["mkt_ew"].std()
        ax.plot(x, -np.concatenate([[0.0], drawdown(mk.to_numpy())]), color=MARKET, lw=0.8,
                label="EW market (vol-matched)")
    book = ctx.colors["__book__"]
    ax.fill_between(x, dd, 0, color=book, alpha=0.12, lw=0)
    ax.plot(x, dd, color=book, lw=1.1, label=f"{ctx.model} book", zorder=5)
    i = int(np.argmin(dd))
    ax.plot([x[i]], [dd[i]], "o", ms=3.5, color=book, mec="white", mew=1.0, zorder=6)
    ax.annotate(f"{dd[i]:.1%} ({pd.Timestamp(x[i]):%b %Y})".replace("-", "\u2212"),
                xy=(x[i], dd[i]), xytext=(5, -1), textcoords="offset points", fontsize=6.5,
                color=INK_2, va="center")
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    _pct_axis(ax)
    ax.set_ylabel("Drawdown")
    _date_axis(ax, x[0], x[-1], narrow)
    _legend_top(fig, 2)
    return True


def fig_meta_weights(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Stacked meta-combination weights of the sleeves over time."""
    mw = wide(ctx.diag, "meta_weights", "sleeve", "theta")
    if mw is None or len(mw) < 2:
        return False
    cols = [s for s in ctx.sleeves if s in mw] or list(mw.columns)
    mw = mw[cols].fillna(0.0)
    ax = fig.add_subplot()
    ax.stackplot(mw.index, mw.T.to_numpy(), colors=[ctx.colors.get(s, OTHER) for s in cols],
                 labels=[SLEEVE_LABELS.get(s, s) for s in cols], edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, max(1.0, float(mw.sum(axis=1).max())))
    ax.grid(False)
    _pct_axis(ax)
    ax.set_ylabel("Meta weight")
    _date_axis(ax, mw.index[0], mw.index[-1], narrow)
    _mark_test_start(ax, ctx, mw.index[0])
    _legend_top(fig, 3 if narrow else min(len(cols), 6))
    return True


def fig_sleeve_corr(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Correlation matrix of sleeve returns and the final book, test period."""
    corr = ctx.stats.get("sleeve_corr")
    if not corr or len(corr) < 3:
        return False
    names = list(corr)
    mat = np.array([[np.nan if corr[r][c] is None else corr[r][c] for c in names] for r in names],
                   dtype=float)
    if np.isfinite(mat[~np.eye(len(names), dtype=bool)]).sum() == 0:
        return False  # too few test months for any correlation
    labels = [SLEEVE_LABELS.get(s, s) if s != "Book" else f"{ctx.model} book" for s in names]
    ax = fig.add_subplot()
    im = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(len(names)), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(names)), labels)
    ax.tick_params(length=0)
    size = 6 if len(names) <= 7 else 5
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(mat[i, j]):
                rgb = DIVERGING((mat[i, j] + 1) / 2)[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(j, i, _no_neg_zero(f"{mat[i, j]:.2f}").replace("-", "\u2212"),
                        ha="center", va="center",
                        fontsize=size, color="white" if lum < 0.45 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-1, 0, 1], aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    return True


def fig_leverage(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Ex-ante volatility forecasts, leverage scale, gross/net exposure, realized vs target vol."""
    bk = ctx.diag.get("book")
    if bk is None or bk.empty:
        return _leverage_from_weights(ctx, fig, narrow)
    bk = bk.set_index("eom").sort_index()
    axs = fig.subplots(2, 2, sharex=True)
    x = bk.index
    a = axs[0, 0]
    a.plot(x, bk["vol_synth"], color=SLOTS[0], lw=0.9, label="forecast: EWMA of daily book returns")
    a.plot(x, bk["vol_fm"], color=SLOTS[1], lw=0.9, label="forecast: factor risk model")
    a.set_title("(a) Forecast vol., unscaled book", fontsize=7, loc="left")
    vols = pd.concat([bk["vol_synth"], bk["vol_fm"]]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vols) > 20:  # a few start-up months must not flatten the panel
        lo, hi = np.nanquantile(vols, [0.005, 0.995])
        a.set_ylim(max(0.0, lo * 0.9), hi * 1.1)
    _pct_axis(a)
    b = axs[0, 1]
    b.plot(x, bk["scale"], color=INK_2, lw=0.9, label="_scale")
    b.set_title("(b) Leverage scale = target / forecast vol.", fontsize=7, loc="left")
    c = axs[1, 0]
    c.plot(x, bk["gross"], color=SLOTS[2], lw=0.9, label="gross exposure $\\Sigma|w|$")
    c.plot(x, bk["net"], color=SLOTS[3], lw=0.9, label="net exposure $\\Sigma w$")
    c.axhline(0, color=AXIS, lw=0.6, zorder=1)
    c.set_title("(c) Final book exposure", fontsize=7, loc="left")
    d = axs[1, 1]
    rv = bk["ret"].rolling(12).std() * math.sqrt(12)
    target = (ctx.stats.get("book_diag") or {}).get("vol_target") or VOL_SCALE
    d.axhline(target, color=INK_MUTED, lw=0.7, zorder=1)
    d.plot(x, rv, color=INK_2, lw=0.9)
    d.set_title(f"(d) Realized 12-month vol. vs. {target:.0%} target", fontsize=7, loc="left")
    _pct_axis(d)
    for ax in axs.flat:
        _date_axis(ax, x[0], x[-1], True)
        _mark_test_start(ax, ctx, x[0])
    _legend_top(fig, 2 if narrow else 4)
    return True


def _leverage_from_weights(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    ts = ctx.ts
    if len(ts) < 2:
        return False
    ax = fig.add_subplot()
    ax.plot(ts.index, ts["gross"], color=SLOTS[0], lw=0.9, label="gross $\\Sigma|w|$")
    ax.plot(ts.index, ts["net"], color=SLOTS[1], lw=0.9, label="net $\\Sigma w$")
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_ylabel("Exposure")
    _date_axis(ax, ts.index[0], ts.index[-1], narrow)
    _legend_top(fig, 2)
    return True


def fig_learner_ic(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Average out-of-sample IC of each return learner by calendar year of eom."""
    icw = wide(ctx.diag, "learner_ic", "learner", "ic")
    if icw is None or icw.empty:
        return False
    yearly = icw.groupby(icw.index.year).mean()
    ax = fig.add_subplot()
    ls = ctx.stats.get("learners") or {}
    for k in [k for k in ctx.learners if k in yearly]:
        m = (ls.get(k) or {}).get("ic_mean_test")
        lab = LEARNER_LABELS.get(k, k) + (f" (test {m:.3f})".replace("-", "\u2212") if m is not None and math.isfinite(m) else "")
        ax.plot(yearly.index, yearly[k], color=ctx.colors.get(f"L_{k}", OTHER), lw=0.9,
                marker="o", ms=2.6, mec="white", mew=0.5, label=lab)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    if yearly.index.min() < ctx.test_start.year:
        _vline_label(ax, ctx.test_start.year + 0.5, "test period")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5 if narrow else 9,
                                                   steps=[1, 2, 5, 10]))
    ax.set_ylabel("Mean monthly IC")
    _legend_top(fig, 2 if narrow else 4)
    return True


def fig_decade_sharpe(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Sharpe ratio by decade with 95% confidence intervals, book vs EW market."""
    dec = ctx.stats["decades"]
    rows = [(d, v) for d, v in dec.items() if v["sharpe"] is not None and math.isfinite(v["sharpe"])]
    if not rows:
        return False
    mk = ctx.stats.get("market_ew") or {}
    labels = [d for d, _ in rows] + ["Full"]
    book = [(v["sharpe"], v["sharpe_ci_low"], v["sharpe_ci_high"]) for _, v in rows]
    book.append((ctx.stats["sharpe"], ctx.stats["sharpe_ci_low"], ctx.stats["sharpe_ci_high"]))
    mkt = [(v["mkt_ew_sharpe"], v["mkt_ew_ci_low"], v["mkt_ew_ci_high"]) for _, v in rows]
    mkt.append((mk.get("sharpe", np.nan), mk.get("sharpe_ci_low", np.nan), mk.get("sharpe_ci_high", np.nan)))
    ax = fig.add_subplot()
    pos = np.arange(len(labels), dtype=float)
    pos[-1] += 0.35  # separate the full sample from the decades
    for vals, off, color, lab, filled in ((mkt, -0.14, MARKET, "EW market", False),
                                          (book, 0.14, ctx.colors["__book__"], f"{ctx.model} book", True)):
        v = np.array(vals, dtype=float)
        ax.errorbar(pos + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o",
                    color=color, ms=4, mfc=color if filled else "white", mec=color, mew=1.0,
                    elinewidth=1.0, capsize=0, label=lab, zorder=5)
        if filled:
            for p, y in zip(pos + off, v[:, 0]):
                ax.annotate(f"{y:.2f}".replace("-", "\u2212"), xy=(p, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_xticks(pos, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.6)
    ax.set_ylabel("Annualized Sharpe ratio")
    _legend_top(fig, 2)
    return True


def fig_zeta(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Ridge shrinkage chosen by cross-validation for the SDF sleeves, at each refit."""
    z = ctx.diag.get("sdf_zeta")
    if z is None or z.empty:
        return False
    ax = fig.add_subplot()
    for s, g in z.groupby("sleeve", sort=False):
        g = g.sort_values("eom")
        ax.step(g["eom"], g["zeta"], where="post", color=ctx.colors.get(s, OTHER), lw=1.0,
                label=SLEEVE_LABELS.get(s, s))
    ax.set_yscale("log")
    ax.set_ylabel("Chosen shrinkage $\\zeta$")
    start, end = z["eom"].min(), z["eom"].max()
    _date_axis(ax, start, end, narrow)
    _mark_test_start(ax, ctx, start)
    _legend_top(fig, 2)
    return True


def fig_sleeve_sharpe(ctx: Ctx, fig: Figure, narrow: bool) -> bool:
    """Dumbbell: Sharpe ratio of each sleeve and the book, pre-test vs test period."""
    sl = ctx.stats.get("sleeves")
    if not sl:
        return False
    items = [(SLEEVE_LABELS.get(s, s), v["sharpe_pre"], v["sharpe_test"], ctx.colors.get(s, OTHER))
             for s, v in sl.items()]
    pre = (ctx.stats.get("book_pre") or {}).get("sharpe", np.nan)
    items.append((f"{ctx.model} book", pre, ctx.stats["sharpe"], ctx.colors["__book__"]))
    ax = fig.add_subplot()
    y = np.arange(len(items))[::-1]
    for yi, (_, a, b, color) in zip(y, items):
        a = np.nan if a is None else a
        b = np.nan if b is None else b
        ax.plot([a, b], [yi, yi], color=AXIS, lw=1.0, zorder=1)
        ax.plot([a], [yi], "o", ms=4.5, mfc="white", mec=color, mew=1.2, zorder=3)
        ax.plot([b], [yi], "o", ms=4.5, color=color, mec="white", mew=0.6, zorder=4)
    ax.set_yticks(y, [it[0] for it in items])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, lw=0.6, zorder=0)
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.plot([], [], "o", ms=4.5, mfc="white", mec=INK_2, mew=1.2, ls="", label="pre-test")
    ax.plot([], [], "o", ms=4.5, color=INK_2, ls="", label="test period")
    _legend_top(fig, 2)
    return True


@dataclass
class FigureSpec:
    func: Callable[[Ctx, Figure, bool], bool]
    width: float  # default width (inches) of the standalone file
    aspect: float  # height / width at full text width
    aspect_narrow: float  # height / width when set at <= 4 inches
    title: str
    caption: str


FIGURES: dict[str, FigureSpec] = {
    "cumret": FigureSpec(fig_cumret, 6.5, 0.42, 0.80, "Cumulative log excess return",
                         "Cumulative log excess return of the {model} book, its sleeves and the "
                         "equal-weighted market over the test period; sleeves and market are "
                         "scaled ex post to the book's realized volatility."),
    "rolling_sharpe": FigureSpec(fig_rolling_sharpe, 6.5, 0.32, 0.70, "Rolling 36-month Sharpe ratio",
                                 "Rolling 36-month annualized Sharpe ratio of the {model} book "
                                 "and the equal-weighted market."),
    "drawdown": FigureSpec(fig_drawdown, 6.5, 0.32, 0.70, "Drawdown",
                           "Drawdown of compounded excess returns; the equal-weighted market is "
                           "scaled to the book's realized volatility."),
    "meta_weights": FigureSpec(fig_meta_weights, 6.5, 0.36, 0.80, "Meta-combination weights",
                               "Meta-combination weights of the sleeves at each formation date."),
    "sleeve_corr": FigureSpec(fig_sleeve_corr, 3.2, 0.62, 0.92, "Sleeve correlations",
                              "Correlations of monthly sleeve returns and the final book, test period."),
    "leverage": FigureSpec(fig_leverage, 6.5, 0.50, 0.95, "Volatility timing",
                           "Volatility timing: ex-ante volatility forecasts of the unscaled "
                           "book, the resulting leverage scale, exposures of the final book "
                           "and its realized 12-month volatility against the target."),
    "learner_ic": FigureSpec(fig_learner_ic, 6.5, 0.34, 0.75, "Learner information coefficients",
                             "Average monthly out-of-sample cross-sectional IC of each return "
                             "learner by calendar year (legend: test-period mean)."),
    "decade_sharpe": FigureSpec(fig_decade_sharpe, 3.2, 0.40, 0.80, "Sharpe ratio by decade",
                                "Annualized Sharpe ratio by decade of the return month with 95% "
                                "confidence intervals (i.i.d., Lo 2002)."),
    "zeta": FigureSpec(fig_zeta, 6.5, 0.30, 0.70, "SDF shrinkage",
                       "Ridge shrinkage of the random-feature SDF sleeves chosen by blocked "
                       "cross-validation at each refit."),
    "sleeve_sharpe": FigureSpec(fig_sleeve_sharpe, 3.2, 0.45, 0.85, "Sleeve Sharpe ratios",
                                "Annualized Sharpe ratio of each sleeve and the final book before "
                                "(hollow) and during (filled) the test period."),
}


def render_figure(name: str, ctx: Ctx, width: float, height: float | None, title: bool,
                  paths: list[Path], dpi: int) -> float | None:
    """Draw figure `name` at width x height inches, save it to each path; height or None."""
    spec = FIGURES[name]
    narrow = width <= 4.0
    h = height or width * (spec.aspect_narrow if narrow else spec.aspect)
    with matplotlib.rc_context(MPL_RC):
        fig = Figure(figsize=(width, h), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        fig._has_title = title  # read by _legend_top
        try:
            ok = spec.func(ctx, fig, narrow)
        except Exception as exc:  # a broken diagnostic must not kill the whole report
            warn(f"figure '{name}' failed: {exc!r}")
            ok = False
        if not ok:
            return None
        if title:
            fig.suptitle(spec.title, x=0.0, ha="left", fontsize=8.5, fontweight="bold", color=INK)
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            meta = {"CreationDate": None} if p.suffix == ".pdf" else {}
            fig.savefig(p, dpi=dpi, metadata=meta)
    return h


def save_all_figures(ctx: Ctx, fig_dir: Path, dpi: int) -> list[str]:
    done = []
    for name, spec in FIGURES.items():
        paths = [fig_dir / f"{name}.pdf", fig_dir / f"{name}.png"]
        if render_figure(name, ctx, spec.width, None, True, paths, dpi) is not None:
            done.append(name)
        else:
            print(f"  figure {name}: skipped (inputs unavailable)")
    return done


# --------------------------------------------------------------------------------------
# Document source parsing
# --------------------------------------------------------------------------------------

@dataclass
class Heading:
    level: int
    text: str
    numbered: bool
    label: str | None


@dataclass
class Para:
    text: str


@dataclass
class ListBlock:
    items: list[str]
    ordered: bool


@dataclass
class MathBlock:
    tex: str
    label: str | None


@dataclass
class Directive:
    kind: str  # figure, table, keyfacts, pagebreak, vspace
    names: list[str]
    opts: dict[str, str]


@dataclass
class PipeTable:
    rows: list[list[str]]
    align: list[str]
    caption: str
    label: str | None


@dataclass
class Div:
    kind: str
    blocks: list[Any]


Block = Heading | Para | ListBlock | MathBlock | Directive | PipeTable | Div

LABEL_RE = re.compile(r"\s*\{#([\w:.-]+)\}\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
DIRECTIVE_RE = re.compile(r"^\[\[\s*([\w-]+)(?::([^\s\]]+))?(.*?)\]\]$", re.S)
OPT_RE = re.compile(r'([\w-]+)\s*=\s*("(?:[^"\\]|\\.)*"|\S+)')


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """`---` key: value `---` at the top of the file -> (dict, remaining text)."""
    m = re.match(r"^\ufeff?---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"')
    return meta, text[m.end():]


def parse_directive(src: str) -> Directive:
    m = DIRECTIVE_RE.match(src.strip())
    if not m:
        raise ValueError(f"malformed directive: {src[:60]!r}")
    kind, names, rest = m.group(1).lower(), m.group(2), m.group(3)
    opts = {k: (v[1:-1].replace('\\"', '"') if v.startswith('"') else v)
            for k, v in OPT_RE.findall(" ".join(rest.split()))}
    if kind in ("vspace", "condbreak") and not opts:
        opts["size"] = rest.strip() or ("6pt" if kind == "vspace" else "1in")
    return Directive(kind, names.split(",") if names else [], opts)


def _parse_label(text: str) -> tuple[str, str | None]:
    m = LABEL_RE.search(text)
    return (text[:m.start()].rstrip(), m.group(1)) if m else (text, None)


def _parse_pipe_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]


def parse_blocks(lines: list[str]) -> list[Block]:
    """Line-oriented block parser for the document source dialect (see module docstring)."""
    blocks: list[Block] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if para:
            blocks.append(Para("\n".join(para)))
            para.clear()

    while i < n:
        s = lines[i].strip()
        if not s:
            flush()
            i += 1
        elif s.startswith(":::") and s[3:].strip():
            flush()
            kind, depth, j = s[3:].strip().lower(), 1, i + 1
            while j < n:
                t = lines[j].strip()
                if t.startswith(":::"):
                    depth += 1 if t[3:].strip() else -1
                    if depth == 0:
                        break
                j += 1
            blocks.append(Div(kind, parse_blocks(lines[i + 1:j])))
            i = j + 1
        elif s == ":::":
            i += 1
        elif s.startswith("$$"):
            flush()
            buf, j = s[2:], i
            while "$$" not in buf and j + 1 < n:
                j += 1
                buf += "\n" + lines[j].strip()
            tex, _, tail = buf.partition("$$")
            _, label = _parse_label(tail) if tail.strip() else ("", None)
            blocks.append(MathBlock(" ".join(tex.split("\n")).strip(), label))
            i = j + 1
        elif s.startswith("[["):
            flush()
            buf, j = s, i
            while not buf.rstrip().endswith("]]") and j + 1 < n:
                j += 1
                buf += " " + lines[j].strip()
            blocks.append(parse_directive(buf))
            i = j + 1
        elif re.match(r"^#{1,3}\s", s):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip()
            numbered = not text.endswith("{-}") and level < 3
            text, label = _parse_label(text.removesuffix("{-}").rstrip())
            blocks.append(Heading(level, text, numbered, label))
            i += 1
        elif s.startswith("|"):
            flush()
            rows, j = [], i
            while j < n and lines[j].strip().startswith("|"):
                rows.append(_parse_pipe_row(lines[j]))
                j += 1
            align = ["l"] * len(rows[0])
            if len(rows) > 1 and all(re.fullmatch(r":?-+:?", c) for c in rows[1] if c):
                align = ["c" if c.startswith(":") and c.endswith(":") else "r" if c.endswith(":")
                         else "l" for c in rows[1]]
                rows.pop(1)
            caption, label = "", None
            if j < n and lines[j].strip().startswith("Table:"):
                caption, label = _parse_label(lines[j].strip()[6:].strip())
                j += 1
            blocks.append(PipeTable(rows, align, caption, label))
            i = j
        elif LIST_RE.match(lines[i]) and not para:
            items, ordered, i = _parse_list(lines, i)
            blocks.append(ListBlock(items, ordered))
        else:
            para.append(s)
            i += 1
    flush()
    return blocks


def _list_item(line: str, ordered: bool) -> re.Match | None:
    """Match a top-level item of a list of the given kind (numbered or bulleted)."""
    m = LIST_RE.match(line)
    return m if m and len(m.group(1)) < 2 and m.group(2)[0].isdigit() == ordered else None


def _parse_list(lines: list[str], i: int) -> tuple[list[str], bool, int]:
    """Items of a flat list starting at line i; indented lines continue the item."""
    ordered = LIST_RE.match(lines[i]).group(2)[0].isdigit()
    items: list[str] = []
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _list_item(line, ordered)
        if m:
            items.append(m.group(3).strip())
        elif line.strip() and line[:1] in (" ", "\t") and items:
            items[-1] += " " + line.strip()
        elif not line.strip():
            k = i + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _list_item(lines[k], ordered):
                i = k
                continue
            break
        else:
            break
        i += 1
    return items, ordered, i


# --------------------------------------------------------------------------------------
# Equations (matplotlib mathtext -> high-resolution PNG)
# --------------------------------------------------------------------------------------

class MathRenderer:
    """Renders TeX snippets with mathtext (STIX font set) to cached transparent PNGs."""

    def __init__(self, build_dir: Path, dpi: int = 600) -> None:
        self.dir = build_dir / "math"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.parser = MathTextParser("path")

    def render(self, tex: str, size: float) -> tuple[Path, float, float, float] | None:
        """(png path, width, height, depth) in points, or None if mathtext rejects `tex`."""
        src = f"${tex}$"
        key = hashlib.sha1(f"{src}|{size}|{self.dpi}".encode()).hexdigest()[:16]
        path = self.dir / f"{key}.png"
        prop = FontProperties(family="STIXGeneral", size=size, math_fontfamily="stix")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w, h, d, _, _ = self.parser.parse(src, dpi=72, prop=prop)
                if not path.exists():
                    fig = Figure(figsize=(w / 72.0, h / 72.0))
                    fig.text(0, d / h, src, fontproperties=prop, color=INK)
                    fig.savefig(path, dpi=self.dpi, transparent=True)
        except Exception as exc:
            why = next((x.strip() for x in str(exc).splitlines() if "Exception" in x), repr(exc))
            warn(f"TeX not supported by mathtext: {tex!r} ({why})")
            return None
        return path, float(w), float(h), float(d)


# --------------------------------------------------------------------------------------
# Inline markup -> reportlab paragraph markup
# --------------------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{\{\s*(stat|config|fig|tab|eq|sec)\s*:\s*([^}|]+?)\s*(?:\|\s*([^}]+?)\s*)?\}\}")
TOKEN_RE = re.compile(
    r"(?P<esc>\\[\\$*`_])"
    r"|(?P<code>`[^`]+`)"
    r"|(?P<link>\[(?P<ltext>[^\]]+)\]\((?P<url>[^)\s]+)\))"
    r"|(?P<math>\$(?=[^\s$])(?:[^$\\]|\\.)*?[^\s\\]\$(?!\d))")


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _emphasis(s: str) -> str:
    """Bold, italic, dashes and true minus signs on already-escaped text."""
    s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![*\w])", r"<i>\1</i>", s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    return re.sub(r"(?<![\w.\-])-(?=\.?\d)", "\u2212", s)


class Inline:
    """Converts the inline dialect to reportlab markup, substituting placeholders."""

    def __init__(self, stats: dict[str, Any], config: dict[str, str], math: MathRenderer,
                 refs: dict[str, str], fonts: dict[str, str]) -> None:
        self.stats, self.config, self.math, self.refs, self.fonts = stats, config, math, refs, fonts

    def placeholders(self, text: str, math: bool = False) -> str:
        """Substitute {{...}} placeholders; inside TeX, '%' is escaped for mathtext."""
        def sub(m: re.Match) -> str:
            value = lookup_placeholder(m)
            return value.replace("%", "\\%") if math else value

        def lookup_placeholder(m: re.Match) -> str:
            kind, key, spec = m.group(1), m.group(2).strip(), m.group(3)
            if kind == "stat":
                return stat_text(self.stats, key, spec)
            if kind == "config":
                if key not in self.config:
                    warn(f"unknown config key '{key}'")
                    return "n/a"
                v = self.config[key]
                if spec and spec != "raw" and _to_float(v) is not None:
                    return format_value(float(v), spec)
                return v
            ref = f"{kind}:{key}"
            if ref not in self.refs:
                warn(f"unknown reference '{ref}'")
                return "??"
            return self.refs[ref]

        return PLACEHOLDER_RE.sub(sub, text)

    def __call__(self, text: str, size: float) -> str:
        """Inline source -> paragraph markup. Escapes, code, links and math are tokenized
        first (stashed behind private-use markers) so emphasis rules never touch them."""
        stash: list[str] = []
        ph = self.placeholders

        def keep(markup: str) -> str:
            stash.append(markup)
            return f"\ue000{len(stash) - 1}\ue001"

        out, pos = [], 0
        for m in TOKEN_RE.finditer(text):
            out.append(ph(text[pos:m.start()]))
            pos = m.end()
            if m.group("esc"):
                out.append(keep(xml_escape(m.group("esc")[1])))
            elif m.group("code"):
                out.append(keep(f'<font face="{self.fonts["mono"]}" size="{size * 0.9:.1f}">'
                                f'{xml_escape(ph(m.group("code")[1:-1]))}</font>'))
            elif m.group("link"):
                inner = _emphasis(xml_escape(ph(m.group("ltext"))))
                out.append(keep(f'<a href="{xml_escape(m.group("url"))}" color="#1c5cab">{inner}</a>'))
            else:
                out.append(keep(self.math_img(ph(m.group("math")[1:-1], math=True), size)))
        out.append(ph(text[pos:]))
        s = _emphasis(xml_escape("".join(out)))
        return re.sub("\ue000(\\d+)\ue001", lambda m: stash[int(m.group(1))], s)

    def math_img(self, tex: str, size: float) -> str:
        r = self.math.render(tex, size)
        if r is None:
            return f"<i>{xml_escape(tex)}</i>"
        path, w, h, d = r
        return f'<img src="{path}" width="{w:.2f}" height="{h:.2f}" valign="{-d:.2f}"/>'


# --------------------------------------------------------------------------------------
# PDF building (reportlab platypus)
# --------------------------------------------------------------------------------------

def register_fonts() -> dict[str, str]:
    """STIX General (Times design, broad Unicode) from matplotlib; built-in Times as fallback."""
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        for name, file in (("STIX", "STIXGeneral.ttf"), ("STIX-Bold", "STIXGeneralBol.ttf"),
                           ("STIX-Italic", "STIXGeneralItalic.ttf"),
                           ("STIX-BoldItalic", "STIXGeneralBolIta.ttf"),
                           ("DejaVuMono", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
        pdfmetrics.registerFontFamily("STIX", normal="STIX", bold="STIX-Bold",
                                      italic="STIX-Italic", boldItalic="STIX-BoldItalic")
        return {"body": "STIX", "bold": "STIX-Bold", "italic": "STIX-Italic", "mono": "DejaVuMono"}
    except Exception as exc:
        warn(f"STIX fonts unavailable ({exc}); using Times-Roman")
        return {"body": "Times-Roman", "bold": "Times-Bold", "italic": "Times-Italic", "mono": "Courier"}


def make_styles(f: dict[str, str]) -> dict[str, ParagraphStyle]:
    ink, ink2 = rl_colors.HexColor(INK), rl_colors.HexColor(INK_2)
    body = ParagraphStyle("body", fontName=f["body"], fontSize=10.5, leading=13.2,
                          alignment=TA_JUSTIFY, spaceAfter=5, textColor=ink, autoLeading="max")
    return {
        "body": body,
        "title": ParagraphStyle("title", parent=body, fontName=f["bold"], fontSize=17, leading=20,
                                alignment=TA_LEFT, spaceAfter=3),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontName=f["italic"], fontSize=11.5,
                                   leading=14, alignment=TA_LEFT, spaceAfter=5, textColor=ink2),
        "byline": ParagraphStyle("byline", parent=body, fontSize=10, leading=12.5,
                                 alignment=TA_LEFT, spaceAfter=0, textColor=ink2),
        "h1": ParagraphStyle("h1", parent=body, fontName=f["bold"], fontSize=12, leading=14.5,
                             alignment=TA_LEFT, spaceBefore=9, spaceAfter=4, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=body, fontName=f["bold"], fontSize=10.5, leading=13,
                             alignment=TA_LEFT, spaceBefore=6, spaceAfter=3, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=body, fontName=f["italic"], fontSize=10.5, leading=13,
                             alignment=TA_LEFT, spaceBefore=4, spaceAfter=2, keepWithNext=1),
        "abstract": ParagraphStyle("abstract", parent=body, fontSize=9.5, leading=12,
                                   leftIndent=0.35 * inch, rightIndent=0.35 * inch, spaceAfter=4),
        "small": ParagraphStyle("small", parent=body, fontSize=9, leading=11.2),
        "box": ParagraphStyle("box", parent=body, fontSize=9.5, leading=12, spaceAfter=3),
        "references": ParagraphStyle("references", parent=body, fontSize=9, leading=11,
                                     leftIndent=12, firstLineIndent=-12, alignment=TA_LEFT,
                                     spaceAfter=2),
        "caption": ParagraphStyle("caption", parent=body, fontSize=9, leading=11, spaceAfter=0,
                                  spaceBefore=3, textColor=ink),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9, leading=11, alignment=TA_LEFT,
                               spaceAfter=0),
        "cell_r": ParagraphStyle("cell_r", parent=body, fontSize=9, leading=11, alignment=TA_RIGHT,
                                 spaceAfter=0),
        "cell_c": ParagraphStyle("cell_c", parent=body, fontSize=9, leading=11, alignment=TA_CENTER,
                                 spaceAfter=0),
        "fact_value": ParagraphStyle("fact_value", parent=body, fontName=f["bold"], fontSize=15,
                                     leading=17, alignment=TA_LEFT, spaceAfter=0),
        "fact_label": ParagraphStyle("fact_label", parent=body, fontSize=8, leading=9.5,
                                     alignment=TA_LEFT, spaceAfter=0, textColor=ink2),
        "footer": ParagraphStyle("footer", parent=body, fontSize=8, textColor=rl_colors.HexColor(INK_MUTED)),
        "panel": ParagraphStyle("panel", parent=body, fontSize=8.5, leading=10, alignment=TA_CENTER,
                                spaceAfter=0),
    }


def parse_length(value: str, text_width: float) -> float:
    """'6.5in', '16cm', '80mm', '300pt', '50%' (of the text width) -> points."""
    m = re.fullmatch(r"([\d.]+)\s*(in|cm|mm|pt|%)?", value.strip())
    if not m:
        raise ValueError(f"bad length {value!r}")
    x, unit = float(m.group(1)), m.group(2) or "pt"
    return {"in": x * inch, "cm": x * cm, "mm": x * mm, "pt": x, "%": x / 100 * text_width}[unit]


def make_canvas_class(footer: str, font: str, pages: list[int], watermark: str | None = None) -> type:
    """Canvas that defers page output to stamp 'page x of y' and records the page count."""

    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["initialFontName"] = kwargs.get("initialFontName") or font  # no unembedded Helvetica
            super().__init__(*args, **kwargs)
            self._pages: list[dict] = []

        def showPage(self) -> None:  # noqa: N802 (reportlab API)
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            pages.append(len(self._pages))
            for state in self._pages:
                self.__dict__.update(state)
                self._footer(len(self._pages))
                super().showPage()
            super().save()

        def _footer(self, total: int) -> None:
            w, _ = self._pagesize
            self.saveState()
            self.setFont(font, 8)
            self.setFillColor(rl_colors.HexColor(INK_MUTED))
            self.drawString(inch, 0.6 * inch, footer)
            self.drawRightString(w - inch, 0.6 * inch, f"{self._pageNumber} / {total}")
            self.restoreState()
            if watermark:  # diagonal stamp and a banner, so a preview cannot pass for results
                h = self._pagesize[1]
                self.saveState()
                self.setFillColor(rl_colors.HexColor("#C0392B"))
                self.setFillAlpha(0.13)
                self.setFont(font, 40)
                self.translate(w / 2, h / 2)
                self.rotate(35)
                self.drawCentredString(0, 0, watermark)
                self.restoreState()
                self.saveState()
                self.setFillColor(rl_colors.HexColor("#C0392B"))
                self.setFont(font, 8)
                self.drawCentredString(w / 2, h - 0.55 * inch, watermark)
                self.restoreState()

    return NumberedCanvas


class Rule(Flowable):
    """A hairline horizontal rule across the text width."""

    def __init__(self, width: float, thickness: float = 0.5, space: float = 4) -> None:
        super().__init__()
        self.width, self.thickness, self.space = width, thickness, space

    def wrap(self, aw: float, ah: float) -> tuple[float, float]:
        return self.width, self.thickness + 2 * self.space

    def draw(self) -> None:
        self.canv.setStrokeColor(rl_colors.HexColor(AXIS))
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


def walk(blocks: list[Block]) -> Iterable[Block]:
    """Blocks in document order, descending into fenced divs."""
    for b in blocks:
        if isinstance(b, Div):
            yield from walk(b.blocks)
        else:
            yield b


class DocBuilder:
    """Turns parsed blocks into platypus flowables."""

    def __init__(self, ctx: Ctx, meta: dict[str, str], config: dict[str, str], build_dir: Path,
                 dpi: int) -> None:
        self.ctx, self.meta, self.build_dir, self.dpi = ctx, meta, build_dir, dpi
        self.fonts = register_fonts()
        self.styles = make_styles(self.fonts)
        self.math = MathRenderer(build_dir)
        self.refs: dict[str, str] = {}
        self.inline = Inline(ctx.stats, config, self.math, self.refs, self.fonts)
        self.text_width = letter[0] - 2 * inch
        self.section = [0, 0]
        self.counters = {"fig": 0, "tab": 0, "eq": 0}
        self.fig_renders = 0
        self.source_dir = ROOT / "docs"

    # numbering ------------------------------------------------------------------------
    def number(self, blocks: list[Block]) -> None:
        """First pass: assign figure, table, equation and section numbers to labels.

        Must mirror the rendering order: every figure/table directive and every captioned
        pipe table produces exactly one caption, every labelled equation one number.
        """
        sec = [0, 0]
        for b in walk(blocks):
            if isinstance(b, Heading) and b.numbered:
                if b.level == 1:
                    sec = [sec[0] + 1, 0]
                else:
                    sec[1] += 1
                if b.label:
                    self.refs[f"sec:{b.label.removeprefix('sec:')}"] = (
                        f"{sec[0]}" if b.level == 1 else f"{sec[0]}.{sec[1]}")
            elif isinstance(b, Directive) and b.kind in ("figure", "table", "image"):
                kind = "tab" if b.kind == "table" else "fig"
                self.counters[kind] += 1
                labels = list(b.names) + ([b.opts["label"]] if "label" in b.opts else [])
                for lab in labels:
                    self.refs.setdefault(f"{kind}:{lab}", str(self.counters[kind]))
            elif isinstance(b, PipeTable) and b.caption:
                self.counters["tab"] += 1
                if b.label:
                    self.refs[f"tab:{b.label.removeprefix('tab:')}"] = str(self.counters["tab"])
            elif isinstance(b, MathBlock) and b.label:
                self.counters["eq"] += 1
                self.refs[f"eq:{b.label.removeprefix('eq:')}"] = str(self.counters["eq"])
        self.counters = {"fig": 0, "tab": 0, "eq": 0}

    # top level ------------------------------------------------------------------------
    def title_block(self) -> list[Flowable]:
        m, st = self.meta, self.styles
        out: list[Flowable] = [Paragraph(self.inline(m.get("title", self.ctx.model), 17), st["title"])]
        if m.get("subtitle"):
            out.append(Paragraph(self.inline(m["subtitle"], 11.5), st["subtitle"]))
        today = date.today().strftime("%B %Y")
        by = " \u00b7 ".join(x for x in (m.get("author", DEFAULT_AUTHOR), m.get("course", DEFAULT_COURSE),
                                          m.get("date", today)) if x)
        out += [Paragraph(self.inline(by, 10), st["byline"]), Rule(self.text_width, 0.6, 5)]
        return out

    def flowables(self, blocks: list[Block], style: str = "body") -> list[Flowable]:
        out: list[Flowable] = []
        for b in blocks:
            out.extend(self.block(b, style))
        return out

    def block(self, b: Block, style: str) -> list[Flowable]:
        st = self.styles
        if isinstance(b, Heading):
            return [self.heading(b)]
        if isinstance(b, Para):
            texts = b.text.split("\n") if style == "references" else [b.text.replace("\n", " ")]
            return [Paragraph(self.inline(t, st[style].fontSize), st[style]) for t in texts]
        if isinstance(b, ListBlock):
            return [self.list_block(b, style)]
        if isinstance(b, MathBlock):
            return self.display_math(b)
        if isinstance(b, PipeTable):
            return self.pipe_table(b)
        if isinstance(b, Div):
            return self.div(b)
        if b.kind == "figure":
            return self.figure(b)
        if b.kind == "image":
            return self.image(b)
        if b.kind == "table":
            return self.gen_table(b)
        if b.kind == "keyfacts":
            return self.keyfacts(b)
        if b.kind == "pagebreak":
            return [PageBreak()]
        if b.kind == "vspace":
            return [Spacer(1, parse_length(b.opts.get("size", "6pt"), self.text_width))]
        if b.kind == "condbreak":
            return [CondPageBreak(parse_length(b.opts.get("size", "1in"), self.text_width))]
        warn(f"unknown directive '{b.kind}'")
        return []

    def heading(self, b: Heading) -> Paragraph:
        st = self.styles
        text = self.inline(b.text, st[f"h{b.level}"].fontSize)
        if b.numbered and b.level == 1:
            self.section = [self.section[0] + 1, 0]
            text = f"{self.section[0]}\u2002{text}"
        elif b.numbered and b.level == 2:
            self.section[1] += 1
            text = f"{self.section[0]}.{self.section[1]}\u2002{text}"
        return Paragraph(text, st[f"h{b.level}"])

    def list_block(self, b: ListBlock, style: str) -> ListFlowable:
        st = self.styles[style]
        item_style = ParagraphStyle(f"{style}_li", parent=st, spaceAfter=1.5)
        items = [ListItem(Paragraph(self.inline(t, st.fontSize), item_style)) for t in b.items]
        kw = dict(bulletType="1", bulletFormat="%s.") if b.ordered else dict(
            bulletType="bullet", start="\u2022")
        return ListFlowable(items, leftIndent=14, bulletFontName=self.fonts["body"],
                            bulletFontSize=st.fontSize, spaceAfter=st.spaceAfter, **kw)

    def display_math(self, b: MathBlock) -> list[Flowable]:
        imgs = []
        for line in [t.strip() for t in self.inline.placeholders(b.tex, math=True).split("\\\\")
                     if t.strip()]:
            r = self.math.render(line, 11)
            if r is None:
                imgs.append(Paragraph(f"<i>{xml_escape(line)}</i>", self.styles["body"]))
            else:
                path, w, h, _ = r
                imgs.append(Image(str(path), width=w, height=h))
        num = ""
        if b.label:
            self.counters["eq"] += 1
            num = f"({self.counters['eq']})"
        side = 0.5 * inch
        rows = [["", img, num if k == len(imgs) - 1 else ""] for k, img in enumerate(imgs)]
        t = Table(rows, colWidths=[side, self.text_width - 2 * side, side])
        t.setStyle(self.table_style([("ALIGN", (1, 0), (1, -1), "CENTER"), ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return [KeepTogether([Spacer(1, 2), t, Spacer(1, 4)])]

    def div(self, b: Div) -> list[Flowable]:
        kind = b.kind if b.kind in self.styles else "body"
        if b.kind not in ("abstract", "box", "small", "references"):
            warn(f"unknown div '{b.kind}'; rendered as body text")
        inner = self.flowables(b.blocks, kind)
        if b.kind == "abstract" and inner and isinstance(inner[0], Paragraph):
            first = b.blocks[0]
            if isinstance(first, Para):
                text = self.inline(first.text.replace("\n", " "), 9.5)
                inner[0] = Paragraph("<b>Abstract.</b> " + text, self.styles["abstract"])
        if b.kind == "box":
            t = Table([[inner]], colWidths=[self.text_width])
            t.setStyle(self.table_style([("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor("#f3f2ee")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                   ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            return [Spacer(1, 2), t, Spacer(1, 6)]
        return inner + ([Spacer(1, 4)] if b.kind == "abstract" else [])

    def table_style(self, cmds: list[tuple]) -> TableStyle:
        """TableStyle with the body font (reportlab otherwise pulls in unembedded Helvetica)."""
        return TableStyle([("FONTNAME", (0, 0), (-1, -1), self.fonts["body"]), *cmds])

    def caption(self, kind: str, text: str) -> Paragraph:
        self.counters[kind] += 1
        word = "Figure" if kind == "fig" else "Table"
        return Paragraph(f"<b>{word} {self.counters[kind]}.</b> {self.inline(text, 9)}",
                         self.styles["caption"])

    # figures --------------------------------------------------------------------------
    def figure(self, b: Directive) -> list[Flowable]:
        names = [n for n in b.names if n] or ["?"]
        unknown = [n for n in names if n not in FIGURES]
        if unknown:
            warn(f"unknown figure(s) {unknown}; known: {', '.join(FIGURES)}")
        k, gap = len(names), 0.15 * inch
        default_w = (self.text_width - gap * (k - 1)) / k
        w = parse_length(b.opts["width"], self.text_width) if "width" in b.opts else default_w
        h = parse_length(b.opts["height"], self.text_width) if "height" in b.opts else None
        cells = []
        for name in names:
            self.fig_renders += 1
            path = self.build_dir / "figures" / f"{name}_{self.fig_renders}.png"
            got = None
            if name in FIGURES:
                got = render_figure(name, self.ctx, w / inch, h / inch if h else None, False,
                                    [path], self.dpi)
            if got is not None:
                cells.append(Image(str(path), width=w, height=got * inch))
            else:
                if name in FIGURES:
                    warn(f"figure '{name}' unavailable (missing diagnostics or too few months); placeholder inserted")
                cells.append(self.placeholder(name, w, h or 0.8 * inch))
        if k == 1:
            default_cap = FIGURES[names[0]].caption if names[0] in FIGURES else ""
        else:
            default_cap = " ".join(f"({chr(97 + j)}) {FIGURES[n].caption}" for j, n in enumerate(names)
                                   if n in FIGURES)
        cap = b.opts.get("caption", default_cap).replace("{model}", self.ctx.model)
        if k == 1:
            body: Flowable = cells[0]
        else:
            labels = [Paragraph(f"({chr(97 + j)})", self.styles["panel"]) for j in range(k)]
            body = Table([cells, labels], colWidths=[w + gap * (k - 1) / k] * k)
            body.setStyle(self.table_style([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                      ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return [KeepTogether([Spacer(1, 3), body, self.caption("fig", cap), Spacer(1, 8)])]

    def image(self, b: Directive) -> list[Flowable]:
        """A static image file (PNG/JPG), path relative to the document source; numbered as a figure."""
        rel = ",".join(b.names) if b.names else ""
        path = (self.source_dir / rel) if not Path(rel).is_absolute() else Path(rel)
        w = parse_length(b.opts["width"], self.text_width) if "width" in b.opts else self.text_width
        cap = b.opts.get("caption", "").replace("{model}", self.ctx.model)
        if not path.exists():
            warn(f"image '{rel}' not found; placeholder inserted")
            body: Flowable = self.placeholder(rel, w, 0.8 * inch)
        else:
            iw, ih = ImageReader(str(path)).getSize()
            h = parse_length(b.opts["height"], self.text_width) if "height" in b.opts else w * ih / iw
            body = Image(str(path), width=w, height=h)
        return [KeepTogether([Spacer(1, 3), body, self.caption("fig", cap), Spacer(1, 8)])]

    def placeholder(self, name: str, w: float, h: float) -> Table:
        p = Paragraph(f"[<i>{xml_escape(name)}</i> unavailable]", self.styles["cell_c"])
        t = Table([[p]], colWidths=[w], rowHeights=[h])
        t.setStyle(self.table_style([("BOX", (0, 0), (-1, -1), 0.5, rl_colors.HexColor(AXIS)),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    # tables ---------------------------------------------------------------------------
    def gen_table(self, b: Directive) -> list[Flowable]:
        name = b.names[0] if b.names else ""
        data = None
        if name not in TABLES:
            warn(f"unknown table '{name}'; known: {', '.join(TABLES)}")
        elif name == "perf":
            rows = [r.strip() for r in b.opts["rows"].split(",")] if "rows" in b.opts else None
            wide = b.opts.get("layout") == "wide"
            short = b.opts.get("labels", "short" if wide else "long") == "short"
            data = perf_table(self.ctx.stats, rows, short)
        else:
            data = TABLES[name](self.ctx.stats)
        if data is None or not data.rows:
            if name in TABLES:
                warn(f"table '{name}' unavailable (missing diagnostics or too few months); placeholder inserted")
            cap = self.caption("tab", b.opts.get("caption", f"[table {name} unavailable]"))
            return [KeepTogether([cap, self.placeholder(f"table {name}", self.text_width, 0.5 * inch),
                                  Spacer(1, 8)])]
        wide = name == "perf" and b.opts.get("layout") == "wide"
        if wide:
            half = (len(data.rows) + 1) // 2
            left, right = data.rows[:half], data.rows[half:] + [["", ""]] * (2 * half - len(data.rows))
            data = TableData(data.header * 2, [a + c for a, c in zip(left, right)], ["l", "r"] * 2,
                             data.caption)
        cap = b.opts.get("caption", data.caption).replace("{model}", self.ctx.model)
        width = parse_length(b.opts["width"], self.text_width) if "width" in b.opts else None
        return self.table_flowables([data.header] + data.rows, data.align, cap, wide, width)

    def pipe_table(self, b: PipeTable) -> list[Flowable]:
        return self.table_flowables(b.rows, b.align, b.caption)

    def table_flowables(self, rows: list[list[str]], align: list[str], caption: str,
                        wide: bool = False, width: float | None = None) -> list[Flowable]:
        st = {"l": self.styles["cell"], "r": self.styles["cell_r"], "c": self.styles["cell_c"]}
        ncol = max(len(r) for r in rows)
        align = (align + ["l"] * ncol)[:ncol]
        cells = []
        for i, r in enumerate(rows):
            r = r + [""] * (ncol - len(r))
            line = []
            for j, c in enumerate(r):
                text = self.inline(c, 9)
                line.append(Paragraph(f"<b>{text}</b>" if i == 0 and text else text, st[align[j]]))
            cells.append(line)
        widths = self.col_widths(cells, align, width)
        t = Table(cells, colWidths=widths, hAlign="CENTER", repeatRows=1)
        ink = rl_colors.HexColor(INK)
        style = [("LINEABOVE", (0, 0), (-1, 0), 0.8, ink), ("LINEBELOW", (0, 0), (-1, 0), 0.4, ink),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.8, ink), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if wide:
            style.append(("LINEBEFORE", (2, 0), (2, -1), 0.4, rl_colors.HexColor(AXIS)))
        t.setStyle(self.table_style(style))
        flow: list[Flowable] = [Spacer(1, 2)]
        if caption:
            flow += [self.caption("tab", caption), Spacer(1, 3)]
        return [KeepTogether(flow + [t, Spacer(1, 8)])]

    def col_widths(self, cells: list[list[Paragraph]], align: list[str],
                   width: float | None) -> list[float]:
        """Natural (unwrapped) column widths; text columns absorb any shortfall or surplus.

        Numeric columns keep their natural width so values never wrap; left-aligned text
        columns are squeezed (and wrap) when the table is wider than the page, or widened
        when a target width is given.
        """
        pad = 8 + 1.5  # left + right cell padding, plus slack against rounding
        nat = []
        for j in range(len(align)):
            w = 0.0
            for row in cells:
                row[j].wrap(1e5, 1e5)
                w = max(w, max(row[j].getActualLineWidths0() or [0.0]))
            nat.append(w + pad)
        target = width if width else min(sum(nat), self.text_width)
        flex = [j for j, a in enumerate(align) if a == "l"] or list(range(len(align)))
        fixed = sum(w for j, w in enumerate(nat) if j not in flex)
        room = target - fixed
        flex_nat = sum(nat[j] for j in flex)
        if room < 0.2 * target:  # numeric columns alone overflow: scale everything
            return [w * target / sum(nat) for w in nat]
        return [nat[j] * room / flex_nat if j in flex else nat[j] for j in range(len(align))]

    def keyfacts(self, b: Directive) -> list[Flowable]:
        spec = b.opts.get("stats", "sharpe,sharpe_ci,mean,sd,max_dd_scaled,turnover")
        defaults = {"sharpe": "Sharpe ratio", "sharpe_ci": "Sharpe 95% CI", "mean": "Excess return p.a.",
                    "sd": "Volatility p.a.", "max_dd": "Max drawdown", "max_dd_scaled": "Max DD at 10% vol",
                    "turnover": "Monthly turnover", "t_stat": "t-statistic", "hit_rate": "Hit rate",
                    "capm_ew.alpha": "Alpha vs. EW market", "gross_leverage": "Gross leverage"}
        cells_v, cells_l = [], []
        for item in [x.strip() for x in spec.split(",") if x.strip()]:
            key, _, label = item.partition(":")
            st = self.ctx.stats
            if key == "sharpe_ci":
                value = f"[{stat_text(st, 'sharpe_ci_low')}, {stat_text(st, 'sharpe_ci_high')}]"
            else:
                value = stat_text(st, key)
            cells_v.append(Paragraph(self.inline(value, 15), self.styles["fact_value"]))
            cells_l.append(Paragraph(self.inline(label or defaults.get(key, key), 8),
                                     self.styles["fact_label"]))
        need = []
        for v, lab in zip(cells_v, cells_l):
            for p in (v, lab):
                p.wrap(1e5, 1e5)
            need.append(max(max(v.getActualLineWidths0() or [0]), max(lab.getActualLineWidths0() or [0])) + 12)
        widths = [w * self.text_width / sum(need) for w in need]
        t = Table([cells_v, cells_l], colWidths=widths)
        t.setStyle(self.table_style([("LINEABOVE", (0, 0), (-1, 0), 0.6, rl_colors.HexColor(AXIS)),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.6, rl_colors.HexColor(AXIS)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, 0), 5),
                               ("BOTTOMPADDING", (0, -1), (-1, -1), 5),
                               ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
        return [Spacer(1, 2), t, Spacer(1, 8)]


def build_pdf(ctx: Ctx, source: Path, pdf_path: Path, build_dir: Path, config: dict[str, str],
              dpi: int, max_pages_cli: int | None, watermark: str | None = None) -> int:
    """Render the document source to pdf_path; returns the page count."""
    meta, text = split_front_matter(source.read_text(encoding="utf-8"))
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    blocks = parse_blocks(text.splitlines())
    builder = DocBuilder(ctx, meta, config, build_dir, dpi)
    builder.source_dir = source.parent
    builder.number(blocks)
    story = builder.title_block() + builder.flowables(blocks)
    pages: list[int] = []
    title = re.sub(r"[*$`]", "", meta.get("title", ctx.model))
    footer = meta.get("footer", f"{ctx.model} \u00b7 {meta.get('author', DEFAULT_AUTHOR)} \u00b7 "
                                f"{meta.get('course', DEFAULT_COURSE)}")
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch, title=title,
                            author=meta.get("author", DEFAULT_AUTHOR), subject=meta.get("subtitle", ""),
                            creator="tools/make_report.py (reportlab)", invariant=1)
    doc.build(story, canvasmaker=make_canvas_class(footer, builder.fonts["body"], pages, watermark))
    n = pages[-1] if pages else 0
    limit = max_pages_cli or int(meta.get("max_pages", 5))
    if n > limit:
        warn(f"{pdf_path.name} has {n} pages, more than the limit of {limit}")
    return n


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring for the document source syntax.")
    ap.add_argument("--data", type=Path, required=True, help="folder with ctff_chars.parquet")
    ap.add_argument("--weights", type=Path, required=True, help="weights CSV (id, eom, w)")
    ap.add_argument("--diagnostics", type=Path, help="folder of <key>.parquet diagnostics")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--doc-source", type=Path, help="document source (default docs/documentation.md)")
    ap.add_argument("--pdf", type=Path, help="PDF path (default <out>/documentation.pdf)")
    ap.add_argument("--model-name", default="PRISM")
    ap.add_argument("--max-pages", type=int, help="page limit (default: front matter or 5)")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution of figures")
    ap.add_argument("--no-pdf", action="store_true", help="statistics and figures only")
    ap.add_argument("--strict", action="store_true", help="exit with status 2 on any warning")
    ap.add_argument("--watermark", help="stamp every page with this text (e.g. for previews)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rets = load_returns(args.data)
    pf = load_weights(args.weights)
    diag = load_diagnostics(args.diagnostics)
    print(f"loaded {len(rets):,} chars rows, {len(pf):,} weights, diagnostics: "
          f"{', '.join(sorted(diag)) or 'none'}")

    stats, ts = compute_all_stats(pf, rets, diag, args.model_name)
    stats["inputs"] = {"data": str(args.data), "weights": str(args.weights),
                       "diagnostics": str(args.diagnostics) if args.diagnostics else None}
    stats["versions"] = {"reportlab": reportlab.Version, "matplotlib": matplotlib.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__}
    (out / "stats.json").write_text(json.dumps(json_clean(stats), indent=2))
    write_markdown(stats, out / "performance_stats.md")
    ts.reset_index().to_csv(out / "monthly_returns.csv", index=False, date_format="%Y-%m-%d")
    print(f"Sharpe {stats['sharpe']:.3f} [{stats['sharpe_ci_low']:.2f}, {stats['sharpe_ci_high']:.2f}], "
          f"mean {stats['mean']:.2%}, sd {stats['sd']:.2%}, turnover {stats['turnover']:.2%}, "
          f"max DD {stats['max_dd']:.2%} over {stats['months']} months")

    sleeves, learners = sleeve_order(diag), learner_order(diag)
    ctx = Ctx(args.model_name, ts, json_clean(stats), diag, rets.loc[rets["test"], "eom"].min(),
              sleeves, learners, assign_colors(sleeves, learners))
    figs = save_all_figures(ctx, out / "figures", args.dpi)
    print(f"figures: {', '.join(figs)}")

    source = args.doc_source or (ROOT / "docs" / "documentation.md")
    if args.no_pdf:
        pass
    elif not source.exists():
        (warn if args.doc_source else note)(f"no document source ({source}); PDF skipped")
    else:
        pdf = args.pdf or out / "documentation.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        n = build_pdf(ctx, source, pdf, out / "_build", config_dict(diag), args.dpi, args.max_pages,
                      args.watermark)
        print(f"PDF: {pdf} ({n} pages)")
    if _WARNINGS:
        print(f"{len(_WARNINGS)} warning(s)", file=sys.stderr)
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
