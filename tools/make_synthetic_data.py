"""Synthetic data in exactly the CTF input format, for end-to-end tests without WRDS.

Writes ``ctff_chars.parquet``, ``ctff_features.parquet`` and ``ctff_daily_ret.parquet``
(layout and dtypes as pinned in docs/DEV_SPEC.md section 2) from a small simulated
economy whose structure is known, so a model can be sanity-checked against the truth:

* Universe. Stocks list and delist over time (size-dependent monthly exit hazard, no exit
  in the first year), ~``n_stocks`` names per month in ``chars`` with ~2 % entries and ~2 %
  exits per month. A stock enters ``chars`` once it has >= 200 trading days of history and a
  row is kept only if the stock survives the whole return month. About 20 % extra ids
  have daily returns but never appear in ``chars``. SIC codes cover all FF12 industries,
  with some missing/empty and unparseable ('abc') values.
* Characteristics. Six persistent latent drivers per stock (monthly AR(1), phi
  0.95-0.98, correlated innovations): size, value, profitability, momentum, investment,
  risk. Features come in 8-12 themed clusters (JKP-style names); each feature is a noisy,
  monotone and often nonlinear transform (log-normal, logistic, power, zero-inflated,
  integer-valued, linear with arbitrary scale) of one cluster signal, which itself mixes one
  or two latents. Features have availability start dates (some only from 1965/1975) and
  persistent (Markov) missingness of 0-30 %. Four features (ret_1_0, ret_12_1,
  rvol_21d, beta_60m) are computed from the simulated returns themselves.
* Returns. Daily excess return = mu_it / n_days + beta_i * mkt + industry factor +
  4 style factors (loadings = size, value, investment and risk latents) + fat-tailed idio
  noise with stock-specific vol (larger for small stocks). The market follows a
  GJR-GARCH(1,1) with 1987, 2008 and 2020 crash shocks; factor and idio vols co-move
  with it. The planted monthly expected return is nonlinear in the latents,
  ``mu ~ value + 0.5 (profit^2 - 1) - 0.5 value * momentum`` (cross-sectionally
  de-meaned), scaled so the ex-ante optimal hedged long-short has an annual Sharpe of
  ``signal_sharpe`` (default 2) on an ``n_stocks`` cross-section. ``ret_exc_lead1m`` equals
  the compounded daily returns over (eom, eom_ret] exactly.

Modes:
  full        1951-12..2023-11 by default; ``ctff_test`` = eom_ret >= 1990-01-31.
  validation  mimics the CTF validation run as replicated by the organizers'
              utils/toy_data.R: 123 months ending 2023-11, ~50 stocks with >= 80 % month
              coverage drawn from a larger world, 10 sorted sampled features, ``ctff_test``
              only for the last 3 months, the Utils industry (SIC 4900-4949) dropped, and
              one Enrgy stock (SIC 1311) that appears only in the last test month with
              daily returns only on dates in (its eom, eom_ret].

Usage:
  python tools/make_synthetic_data.py --mode full --out DIR
      [--start 1951-12 --end 2023-11 --n-stocks 300 --n-features 60 --seed 7]
      [--daily-lead-months 12] [--signal-sharpe 2.0] [--truth]

``--truth`` also writes the latent truth (planted mu, betas, latents per chars row, the
market return/vol path and the feature design) as ``truth_*.parquet``; the CTF format
files are unaffected. In-process use: ``generate(mode, ...)`` returns the three pandas
DataFrames exactly as ``pandas.read_parquet`` would load them.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import ndtr

# --------------------------------------------------------------------------------------
# Constants of the simulated economy
# --------------------------------------------------------------------------------------

TEST_FIRST_EOM_RET = np.datetime64("1990-01-31")
DEFAULTS = {
    "full": {"start": "1951-12", "end": "2023-11", "n_stocks": 300, "n_features": 60},
    "validation": {"start": None, "end": "2023-11", "n_stocks": 50, "n_features": 10},
}
BURN_IN_MONTHS = 120  # monthly-only simulation before the data starts (steady-state universe)
SEASONING_DAYS = 200  # chars require >= 200 daily returns in the past 252 trading days
NO_EXIT_MONTHS = 12  # a new listing cannot delist during its first year
MEAN_HAZARD = 0.025  # monthly delisting hazard of an average-size stock
ALIVE_PER_CHARS = 1.14  # alive (incl. unseasoned) main stocks per chars row, steady state
EXTRA_SHARE = 0.17  # daily-only population relative to the main one (~20 % extra ids)
FEEDBACK = 0.15  # listing-rate feedback that keeps the cross-section near its target
MAIN, EXTRA = 0, 1
HUGE = np.iinfo(np.int64).max // 4

VALIDATION_MONTHS = 123
VALIDATION_TEST_MONTHS = 3
VALIDATION_WORLD_MULT = 20  # world size relative to the validation sample
VALIDATION_MIN_COVERAGE = 0.8
VALIDATION_LATE_SIC = "1311"

LATENTS = ("size", "value", "profit", "momentum", "invest", "risk")
SIZE, VALUE, PROFIT, MOMENTUM, INVEST, RISK = range(6)
LATENT_PHI = np.array([0.98, 0.97, 0.97, 0.95, 0.96, 0.97])
LATENT_CORR = {(SIZE, RISK): -0.4, (SIZE, VALUE): -0.25, (VALUE, PROFIT): -0.3,
               (VALUE, INVEST): -0.25, (PROFIT, RISK): -0.2}
STYLE_LATENTS = np.array([SIZE, VALUE, INVEST, RISK])
STYLE_DAILY_VOL = np.array([0.004, 0.004, 0.003, 0.005])
IND_DAILY_VOL = 0.006
IDIO_T_DF = 4.0
DAYS_PER_MONTH = 261 / 12

MKT_PREMIUM = 0.06 / 252
MKT_LONG_RUN_VAR = 0.16**2 / 252
GJR_ALPHA, GJR_GAMMA, GJR_BETA = 0.03, 0.10, 0.90
MKT_T_DF = 6.0
MKT_MAX_VOL = 0.08
MARKET_EVENTS = {  # forced daily market excess returns (crash-style volatility spikes)
    "1987-10-19": -0.205, "1987-10-20": 0.053, "1987-10-26": -0.083,
    "2008-09-29": -0.088, "2008-10-09": -0.076, "2008-10-13": 0.115, "2008-10-15": -0.090,
    "2008-11-20": -0.067, "2008-12-01": -0.089,
    "2020-03-09": -0.076, "2020-03-12": -0.095, "2020-03-16": -0.120, "2020-03-24": 0.094,
}

# Fama-French 12 industries, mirroring ff12_class() in the organizers' factor_model_utils.R.
FF12_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "NoDur": ((100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)),
    "Durbl": ((2500, 2519), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
              (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)),
    "Manuf": ((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
              (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
              (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)),
    "Enrgy": ((1200, 1399), (2900, 2999)),
    "Chems": ((2800, 2829), (2840, 2899)),
    "BusEq": ((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)),
    "Telcm": ((4800, 4899),),
    "Utils": ((4900, 4949),),
    "Shops": ((5000, 5999), (7200, 7299), (7600, 7699)),
    "Hlth": ((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)),
    "Money": ((6000, 6999),),
}
OTHER_SIC_RANGES = ((1000, 1199), (1400, 1999), (4000, 4799), (4950, 4999), (7000, 7199),
                    (7300, 7369), (7380, 7399), (8100, 8999), (9100, 9999))
FF12_NAMES = (*FF12_RANGES, "Other")
FF12_PROBS = np.array([0.07, 0.03, 0.12, 0.05, 0.03, 0.15, 0.03, 0.04, 0.11, 0.10, 0.17, 0.10])
OTHER = FF12_NAMES.index("Other")
BAD_SIC = {None: 0.010, "": 0.010, "abc": 0.002, "n/a": 0.001, "99x9": 0.001}

SIZE_GROUPS = np.array(["small", "large", "mega"], dtype=object)
SIZE_BREAKS = (0.45, 0.78)  # cross-sectional size-latent percentiles of small|large|mega

# (theme, latent weights) of the feature clusters, in order of inclusion.
CLUSTERS: tuple[tuple[str, dict[int, float]], ...] = (
    ("value", {VALUE: 1.0}),
    ("profitability", {PROFIT: 1.0}),
    ("momentum", {MOMENTUM: 1.0}),
    ("investment", {INVEST: 1.0}),
    ("low_risk", {RISK: 1.0}),
    ("size", {SIZE: 1.0}),
    ("quality", {PROFIT: 0.8, SIZE: 0.4}),
    ("profit_growth", {MOMENTUM: 0.7, PROFIT: 0.5}),
    ("accruals", {INVEST: 0.7, VALUE: -0.4}),
    ("low_leverage", {RISK: -0.6, SIZE: 0.5}),
    ("debt_issuance", {INVEST: 0.6, VALUE: 0.5}),
    ("seasonality", {}),  # pure noise cluster, unrelated to returns
)
CLUSTER_OWN_WEIGHT = 0.4  # weight of the cluster-specific persistent component
CLUSTER_OWN_PHI = 0.9
THEME_NAMES: dict[str, tuple[str, ...]] = {
    "value": ("be_me", "at_me", "bev_mev", "debt_me", "div12m_me", "ebitda_mev", "eq_dur",
              "eqnetis_at", "eqnpo_12m", "eqnpo_me", "eqpo_me", "fcf_me", "ival_me",
              "netis_at", "ni_me", "ocf_me", "sale_me", "chcsho_12m"),
    "profitability": ("ebit_bev", "ebit_sale", "f_score", "ni_be", "niq_be", "o_score", "ocf_at",
                      "ope_be", "ope_bel1", "dolvol_var_126d", "turnover_var_126d"),
    "momentum": ("prc_highprc_252d", "resff3_6_1", "resff3_12_1", "ret_3_1", "ret_6_1",
                 "ret_9_1", "seas_1_1na"),
    "investment": ("aliq_at", "at_gr1", "be_gr1a", "coa_gr1a", "col_gr1a", "emp_gr1", "inv_gr1",
                   "inv_gr1a", "lnoa_gr1a", "mispricing_mgmt", "ncoa_gr1a", "nncoa_gr1a",
                   "noa_gr1a", "ppeinv_gr1a", "ret_60_12", "sale_gr1", "sale_gr3", "saleq_gr1",
                   "seas_2_5na"),
    "low_risk": ("beta_dimson_21d", "betabab_1260d", "betadown_252d", "earnings_variability",
                 "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d", "ivol_hxz4_21d",
                 "ocfq_saleq_std", "rmax1_21d", "rmax5_21d", "seas_6_10na", "turnover_126d",
                 "zero_trades_21d", "zero_trades_126d", "zero_trades_252d"),
    "size": ("ami_126d", "dolvol_126d", "market_equity", "prc", "rd_me"),
    "quality": ("at_turnover", "cop_at", "cop_atl1", "dgp_dsale", "gp_at", "gp_atl1",
                "mispricing_perf", "ni_inc8q", "niq_at", "op_at", "op_atl1", "opex_at", "qmj",
                "qmj_growth", "qmj_prof", "qmj_safety", "sale_bev"),
    "profit_growth": ("dsale_dinv", "dsale_drec", "dsale_dsga", "niq_at_chg1", "niq_be_chg1",
                      "niq_su", "ocf_at_chg1", "ret_12_7", "sale_emp_gr1", "saleq_su",
                      "seas_1_1an", "tax_gr1a"),
    "accruals": ("cowc_gr1a", "oaccruals_at", "oaccruals_ni", "seas_16_20na", "taccruals_at",
                 "taccruals_ni"),
    "low_leverage": ("age", "aliq_mat", "at_be", "bidaskhl_21d", "cash_at", "netdebt_me",
                     "ni_ivol", "rd_sale", "rd5_at", "tangibility", "z_score"),
    "debt_issuance": ("capex_abn", "debt_gr3", "fnl_gr1a", "ncol_gr1a", "nfna_gr1a", "ni_ar1",
                      "noa_at"),
    "seasonality": ("corr_1260d", "coskew_21d", "dbnetis_at", "kz_index", "lti_gr1a", "pi_nix",
                    "seas_2_5an", "seas_6_10an", "seas_11_15an", "seas_11_15na", "seas_16_20an",
                    "sti_gr1a"),
}
# Features computed from the simulated returns: name -> theme.
RETURN_FEATURES = {"ret_1_0": "short_term_reversal", "ret_12_1": "momentum",
                   "rvol_21d": "low_risk", "beta_60m": "low_risk"}
MIN_FEATURES_FOR_RETURN_FEATURES = 20
RING = 60  # months of return history kept for the return-based features

TRANSFORMS = ("linear", "lognormal", "logistic", "power", "zero_inflated", "integer")
TRANSFORM_PROBS = np.array([0.33, 0.22, 0.10, 0.12, 0.15, 0.08])
AVAIL_STARTS = (None, "1963-07", "1965-01", "1975-01")
AVAIL_PROBS = np.array([0.60, 0.10, 0.15, 0.15])
MAX_FEATURE_MISS = 0.25  # own Markov missing rate ~ U(0, 0.25); accounting gaps add ~5 %
MISS_EXIT_PROB = 0.12  # mean missing spell ~8 months
GAP_ENTER_PROB, GAP_EXIT_PROB = 0.01, 0.19  # stock-level accounting gaps (stationary 5 %)
ACCOUNTING_SHARE = 0.6


# --------------------------------------------------------------------------------------
# Configuration, calendar and feature design
# --------------------------------------------------------------------------------------


def _months(k: int) -> np.timedelta64:
    return np.timedelta64(int(k), "M")


@dataclass(frozen=True)
class Config:
    """Resolved generator settings."""

    mode: str
    start: np.datetime64  # first chars eom (datetime64[M])
    end: np.datetime64  # last chars eom (datetime64[M])
    n_stocks: int  # chars cross-section of the output
    n_features: int  # features in the output
    seed: int
    daily_lead_months: int
    signal_sharpe: float
    world_stocks: int  # chars cross-section of the simulated world
    world_features: int


def resolve_config(mode: str, start: str | None = None, end: str | None = None,
                   n_stocks: int | None = None, n_features: int | None = None, seed: int = 7,
                   daily_lead_months: int = 12, signal_sharpe: float = 2.0) -> Config:
    """Fill mode-specific defaults and validate the settings."""
    if mode not in DEFAULTS:
        raise ValueError(f"mode must be one of {sorted(DEFAULTS)}, got {mode!r}")
    d = DEFAULTS[mode]
    end_m = np.datetime64(end or d["end"], "M")
    if mode == "validation" and start is None:
        start_m = end_m - _months(VALIDATION_MONTHS - 1)
    else:
        start_m = np.datetime64(start or d["start"], "M")
    n_stocks = int(n_stocks or d["n_stocks"])
    n_features = int(n_features or d["n_features"])
    if start_m > end_m or n_stocks < 1 or n_features < 1 or daily_lead_months < 0:
        raise ValueError("need start <= end, n_stocks >= 1, n_features >= 1, daily_lead_months >= 0")
    validation = mode == "validation"
    return Config(
        mode=mode, start=start_m, end=end_m, n_stocks=n_stocks, n_features=n_features,
        seed=int(seed), daily_lead_months=int(daily_lead_months),
        signal_sharpe=float(signal_sharpe),
        world_stocks=n_stocks * VALIDATION_WORLD_MULT if validation else n_stocks,
        world_features=max(60, n_features) if validation else n_features,
    )


@dataclass(frozen=True)
class Calendar:
    """Monthly grid and Mon-Fri trading days of the whole simulation."""

    months: np.ndarray  # datetime64[M]
    days: np.ndarray  # datetime64[D], business days
    month_start: np.ndarray  # index of the first trading day of each month
    month_end: np.ndarray  # one past the last trading day of each month
    eom: np.ndarray  # datetime64[D], calendar month ends

    def index(self, month: np.datetime64) -> int:
        return int((month - self.months[0]).astype(np.int64))


def build_calendar(first: np.datetime64, last: np.datetime64) -> Calendar:
    """Business days (no holidays) and calendar month ends for months first..last."""
    months = np.arange(first, last + _months(1))
    next_month_day1 = (months + _months(1)).astype("datetime64[D]")
    all_days = np.arange(months[0].astype("datetime64[D]"), next_month_day1[-1])
    days = all_days[np.is_busday(all_days)]
    start = np.searchsorted(days, months.astype("datetime64[D]"))
    end = np.append(start[1:], days.size)
    eom = next_month_day1 - np.timedelta64(1, "D")
    return Calendar(months, days, start, end, eom)


@dataclass(frozen=True)
class FeatureDesign:
    """How every feature is produced. Latent-driven features come first, then return-based."""

    names: tuple[str, ...]
    themes: tuple[str, ...]
    n_latent: int
    cluster: np.ndarray  # (n_latent,) cluster of each latent-driven feature
    cluster_load: np.ndarray  # (6, K) unit-norm latent weights per cluster
    cluster_own: np.ndarray  # (K,) weight of the cluster-specific component
    rho: np.ndarray  # loading on the cluster signal
    sign: np.ndarray
    noise_phi: np.ndarray  # persistence of the feature-specific noise
    transform: np.ndarray  # index into TRANSFORMS
    p1: np.ndarray
    p2: np.ndarray
    miss_rate: np.ndarray
    accounting: np.ndarray  # subject to stock-level accounting gaps
    avail: np.ndarray  # datetime64[M] first available month (NaT = always)

    @property
    def n_clusters(self) -> int:
        return self.cluster_load.shape[1]


def _cluster_sizes(rng: np.random.Generator, n_latent: int, themes: list[str]) -> np.ndarray:
    k = len(themes)
    base = 2 if n_latent >= 2 * k else 1
    pool = np.array([len(THEME_NAMES[t]) for t in themes], dtype=float)
    return base + rng.multinomial(n_latent - base * k, pool / pool.sum())


def _feature_names(rng: np.random.Generator, themes: list[str], sizes: np.ndarray) -> list[str]:
    names = []
    for theme, size in zip(themes, sizes):
        pool = [str(name) for name in rng.permutation(THEME_NAMES[theme])]
        names += pool[:size] + [f"{theme}_{j:02d}" for j in range(len(pool) + 1, size + 1)]
    return names


def design_features(rng: np.random.Generator, n_features: int) -> FeatureDesign:
    """Draw clusters, names, transforms, missingness and availability for every feature."""
    with_returns = n_features >= MIN_FEATURES_FOR_RETURN_FEATURES
    n_latent = n_features - (len(RETURN_FEATURES) if with_returns else 0)
    k = int(np.clip(round(n_features / 5), 8, 12))
    k = max(1, min(k, n_latent // 2 if n_latent >= 2 else 1))
    themes = [t for t, _ in CLUSTERS[:k]]
    sizes = _cluster_sizes(rng, n_latent, themes)
    names = _feature_names(rng, themes, sizes)
    feat_themes = [t for t, s in zip(themes, sizes) for _ in range(s)]
    if with_returns:
        names += list(RETURN_FEATURES)
        feat_themes += list(RETURN_FEATURES.values())
    assert len(set(names)) == len(names) == n_features

    load = np.zeros((len(LATENTS), k))
    for c, (_, weights) in enumerate(CLUSTERS[:k]):
        for latent, w in weights.items():
            load[latent, c] = w
    norm = np.linalg.norm(load, axis=0)
    load[:, norm > 0] /= norm[norm > 0]
    own = np.where(norm > 0, CLUSTER_OWN_WEIGHT, 1.0)

    n = n_latent
    transform = rng.choice(len(TRANSFORMS), size=n, p=TRANSFORM_PROBS)
    p1, p2 = _transform_params(rng, transform)
    avail_pick = rng.choice(len(AVAIL_STARTS), size=n, p=AVAIL_PROBS)
    avail = np.array([np.datetime64(AVAIL_STARTS[i] or "NaT", "M") for i in avail_pick])
    return FeatureDesign(
        names=tuple(names), themes=tuple(feat_themes), n_latent=n_latent,
        cluster=np.repeat(np.arange(k), sizes), cluster_load=load, cluster_own=own,
        rho=rng.uniform(0.55, 0.95, n), sign=np.where(rng.random(n) < 0.3, -1.0, 1.0),
        noise_phi=rng.uniform(0.2, 0.95, n), transform=transform, p1=p1, p2=p2,
        miss_rate=rng.uniform(0.0, MAX_FEATURE_MISS, n),
        accounting=rng.random(n) < ACCOUNTING_SHARE, avail=avail,
    )


def _transform_params(rng: np.random.Generator, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape (p1) and scale (p2) parameters of each feature's transform."""
    n = transform.size
    scale = 10.0 ** rng.uniform(-2.0, 2.0, n)
    p1 = np.zeros(n)
    p2 = scale.copy()
    t = np.asarray(TRANSFORMS)[transform]
    p1[t == "linear"] = rng.normal(0.0, 2.0, (t == "linear").sum()) * scale[t == "linear"]
    p1[t == "lognormal"] = rng.uniform(0.3, 1.2, (t == "lognormal").sum())
    p1[t == "logistic"] = rng.uniform(0.8, 2.5, (t == "logistic").sum())
    p1[t == "power"] = rng.uniform(1.3, 2.5, (t == "power").sum())
    p1[t == "zero_inflated"] = rng.uniform(-0.3, 0.8, (t == "zero_inflated").sum())
    return p1, p2


def apply_transforms(z: np.ndarray, d: FeatureDesign) -> np.ndarray:
    """Map standardized feature signals (rows x n_latent) to observed feature values."""
    x = np.empty_like(z)
    for k, name in enumerate(TRANSFORMS):
        cols = np.flatnonzero(d.transform == k)
        if cols.size == 0:
            continue
        zc, p1, p2 = z[:, cols], d.p1[cols], d.p2[cols]
        if name == "linear":
            x[:, cols] = p1 + p2 * zc
        elif name == "lognormal":
            x[:, cols] = p2 * np.exp(p1 * zc)
        elif name == "logistic":
            x[:, cols] = 1.0 / (1.0 + np.exp(-p1 * zc))
        elif name == "power":
            x[:, cols] = p2 * np.sign(zc) * np.abs(zc) ** p1
        elif name == "zero_inflated":
            x[:, cols] = p2 * np.maximum(zc - p1, 0.0)
        else:  # integer score 0..9, like f_score
            x[:, cols] = np.round(9.0 * ndtr(zc))
    return x


# --------------------------------------------------------------------------------------
# Stock-level economics
# --------------------------------------------------------------------------------------


def latent_cholesky() -> np.ndarray:
    corr = np.eye(len(LATENTS))
    for (a, b), c in LATENT_CORR.items():
        corr[a, b] = corr[b, a] = c
    return np.linalg.cholesky(corr)


def mu_shape(lat: np.ndarray) -> np.ndarray:
    """Un-scaled planted expected return: nonlinear in value, profitability and momentum."""
    v, p, m = lat[:, VALUE], lat[:, PROFIT], lat[:, MOMENTUM]
    return v + 0.5 * (p * p - 1.0) - 0.5 * v * m


def stock_beta(lat: np.ndarray, fe: np.ndarray) -> np.ndarray:
    return np.clip(1.0 + 0.3 * lat[:, RISK] - 0.1 * lat[:, SIZE] + 0.15 * fe, 0.1, 2.5)


def idio_vol(lat: np.ndarray, fe: np.ndarray) -> np.ndarray:
    """Daily idiosyncratic vol at average market volatility (small stocks are riskier)."""
    return 0.02 * np.exp(-0.3 * lat[:, SIZE] + 0.15 * lat[:, RISK] + 0.15 * fe)


def ex_ante_sharpe2(mu: np.ndarray, lat: np.ndarray, beta: np.ndarray, ind: np.ndarray,
                    gamma: np.ndarray, sigma_m: np.ndarray) -> float:
    """Squared monthly Sharpe mu' Sigma^-1 mu of the mean-variance optimal portfolio.

    Sigma = B F B' + D with B = [market beta, industry dummies * gamma, style loadings], F the
    monthly factor variances at average market volatility and D the idio variances; solved
    with the Woodbury identity. The optimal portfolio is the fully factor-hedged long-short.
    """
    dummies = (ind[:, None] == np.arange(len(FF12_NAMES))) * gamma[:, None]
    b_mat = np.column_stack([beta, dummies, lat[:, STYLE_LATENTS]])
    f_var = DAYS_PER_MONTH * np.concatenate([[MKT_LONG_RUN_VAR], np.full(len(FF12_NAMES), IND_DAILY_VOL**2),
                                             STYLE_DAILY_VOL**2])
    d_inv = 1.0 / sigma_m**2
    btd = b_mat.T * d_inv
    core = np.diag(1.0 / f_var) + btd @ b_mat
    proj = btd @ mu
    return float(mu @ (d_inv * mu) - proj @ np.linalg.solve(core, proj))


def calibrate_alpha_scale(rng: np.random.Generator, n_stocks: int, signal_sharpe: float,
                          draws: int = 40) -> float:
    """Scale of mu_shape giving the optimal hedged portfolio of n_stocks the target Sharpe.

    Averages the ex-ante squared Sharpe over random stationary cross-sections. Hedging matters:
    the linear value part of mu is spanned by the value-factor loading, so it earns only a
    factor-like Sharpe and most of the diversifiable alpha comes from the nonlinear terms.
    """
    if signal_sharpe <= 0:
        return 0.0
    chol = latent_cholesky()
    sr2 = []
    for _ in range(draws):
        lat = rng.standard_normal((n_stocks, len(LATENTS))) @ chol.T
        mu0 = mu_shape(lat)
        mu0 -= mu0.mean()
        beta = stock_beta(lat, rng.standard_normal(n_stocks))
        sigma_m = idio_vol(lat, rng.standard_normal(n_stocks)) * np.sqrt(DAYS_PER_MONTH)
        ind = rng.choice(len(FF12_NAMES), size=n_stocks, p=FF12_PROBS)
        sr2.append(ex_ante_sharpe2(mu0, lat, beta, ind, rng.uniform(0.7, 1.3, n_stocks), sigma_m))
    return signal_sharpe / np.sqrt(12.0) / np.sqrt(np.mean(sr2))


def draw_industries(rng: np.random.Generator, n: int) -> tuple[np.ndarray, list[str | None]]:
    """FF12 industry index and SIC string per stock (some missing or unparseable)."""
    ind = rng.choice(len(FF12_NAMES), size=n, p=FF12_PROBS)
    bad_vals = list(BAD_SIC)
    bad_draw = rng.random(n)
    bad_pick = rng.choice(len(bad_vals), size=n, p=np.array(list(BAD_SIC.values())) / sum(BAD_SIC.values()))
    u = rng.random(n)
    sic: list[str | None] = []
    for i in range(n):
        if bad_draw[i] < sum(BAD_SIC.values()):
            ind[i] = OTHER
            sic.append(bad_vals[bad_pick[i]])
            continue
        ranges = OTHER_SIC_RANGES if ind[i] == OTHER else FF12_RANGES[FF12_NAMES[ind[i]]]
        widths = np.array([b - a + 1 for a, b in ranges])
        j = min(int(np.searchsorted(np.cumsum(widths) / widths.sum(), u[i], side="right")), len(ranges) - 1)
        a, b = ranges[j]
        sic.append(str(int(rng.integers(a, b + 1))))
    return ind, sic


def ff12_class(sic: list[str | None] | np.ndarray) -> np.ndarray:
    """FF12 industry names of SIC strings; missing/unparseable codes map to 'Other' as in R."""
    out = np.full(len(sic), "Other", dtype=object)
    for i, s in enumerate(sic):
        if s is None or not str(s).isdigit():
            continue
        code = int(s)
        for name, ranges in FF12_RANGES.items():
            if any(a <= code <= b for a, b in ranges):
                out[i] = name
                break
    return out


# --------------------------------------------------------------------------------------
# Market and factor returns
# --------------------------------------------------------------------------------------


def simulate_market(rng: np.random.Generator, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily market excess return and conditional variance: GJR-GARCH(1,1) plus crash shocks."""
    n = days.size
    z = rng.standard_t(MKT_T_DF, size=n) * np.sqrt((MKT_T_DF - 2.0) / MKT_T_DF)
    forced = np.full(n, np.nan)
    for date, ret in MARKET_EVENTS.items():
        k = int(np.searchsorted(days, np.datetime64(date)))
        if k < n and days[k] == np.datetime64(date):
            forced[k] = ret
    omega = MKT_LONG_RUN_VAR * (1.0 - GJR_ALPHA - GJR_GAMMA / 2.0 - GJR_BETA)
    ret, var = np.empty(n), np.empty(n)
    h = MKT_LONG_RUN_VAR
    for t in range(n):
        var[t] = h
        r = MKT_PREMIUM + np.sqrt(h) * z[t] if np.isnan(forced[t]) else forced[t]
        e = r - MKT_PREMIUM
        h = min(omega + (GJR_ALPHA + GJR_GAMMA * (e < 0.0)) * e * e + GJR_BETA * h, MKT_MAX_VOL**2)
        ret[t] = r
    return ret, var


# --------------------------------------------------------------------------------------
# The simulated world
# --------------------------------------------------------------------------------------


@dataclass
class WorldResult:
    """Everything simulated: chars-eligible rows, daily returns, stock attributes, truth."""

    cal: Calendar
    design: FeatureDesign
    row_stock: np.ndarray
    row_month: np.ndarray
    row_x: np.ndarray  # (rows, n_features) in design.names order
    row_size: np.ndarray  # index into SIZE_GROUPS
    row_ret: np.ndarray  # ret_exc_lead1m
    row_mu: np.ndarray
    row_beta: np.ndarray
    row_sigma: np.ndarray  # monthly idio vol at average market vol
    row_latent: np.ndarray  # (rows, 6)
    daily_day: np.ndarray  # index into cal.days
    daily_stock: np.ndarray
    daily_ret: np.ndarray
    stock_kind: np.ndarray
    stock_ind: np.ndarray
    stock_sic: list[str | None]
    stock_gamma: np.ndarray
    mkt_first_day: int
    mkt_ret: np.ndarray
    mkt_var: np.ndarray
    alpha_scale: float


class _World:
    """Mutable state of the stock universe during the monthly simulation loop."""

    def __init__(self, design: FeatureDesign, capacity: int) -> None:
        self.d = design
        self.n = 0
        self.cap = 0
        self.chol = latent_cholesky()
        self.alive = np.zeros(0, dtype=np.int64)
        self.sic: list[str | None] = []
        self._specs = {  # name: (trailing shape, dtype, fill)
            "kind": ((), np.int8, 0), "ind": ((), np.int8, 0),
            "list_day": ((), np.int64, 0), "list_month": ((), np.int64, 0),
            "delist_day": ((), np.int64, HUGE), "delist_month": ((), np.int64, HUGE),
            "fe_beta": ((), float, 0.0), "fe_vol": ((), float, 0.0), "gamma": ((), float, 1.0),
            "lat": ((len(LATENTS),), float, 0.0), "own": ((design.n_clusters,), float, 0.0),
            "noise": ((design.n_latent,), float, 0.0), "miss": ((design.n_latent,), bool, False),
            "gap": ((), bool, False), "ret_hist": ((RING,), float, np.nan),
            "rvol": ((), float, np.nan),
        }
        self._grow(capacity)

    def _grow(self, need: int) -> None:
        """Ensure room for `need` stocks; per-stock arrays are attributes named as in _specs."""
        if need <= self.cap:
            return
        cap = max(need, 2 * self.cap)
        for name, (shape, dtype, fill) in self._specs.items():
            new = np.full((cap, *shape), fill, dtype=dtype)
            if self.cap:
                new[: self.n] = getattr(self, name)[: self.n]
            setattr(self, name, new)
        self.cap = cap

    def add(self, rng: np.random.Generator, k: int, kind: int, list_day: np.ndarray, list_month: int) -> None:
        if k == 0:
            return
        self._grow(self.n + k)
        idx = np.arange(self.n, self.n + k)
        d = self.d
        ind, sic = draw_industries(rng, k)
        self.sic += sic
        self.kind[idx] = kind
        self.ind[idx] = ind
        self.list_day[idx] = list_day
        self.list_month[idx] = list_month
        self.fe_beta[idx] = rng.standard_normal(k)
        self.fe_vol[idx] = rng.standard_normal(k)
        self.gamma[idx] = rng.uniform(0.7, 1.3, k)
        self.lat[idx] = rng.standard_normal((k, len(LATENTS))) @ self.chol.T
        self.own[idx] = rng.standard_normal((k, d.n_clusters))
        self.noise[idx] = rng.standard_normal((k, d.n_latent))
        self.miss[idx] = rng.random((k, d.n_latent)) < np.minimum(0.95, 1.5 * d.miss_rate)
        self.gap[idx] = rng.random(k) < GAP_ENTER_PROB / (GAP_ENTER_PROB + GAP_EXIT_PROB)
        self.n += k
        self.alive = np.concatenate([self.alive, idx])

    def count_alive(self, kind: int, month: int) -> int:
        a = self.alive
        return int(np.sum((self.kind[a] == kind) & (self.delist_month[a] != month)))

    def draw_exits(self, rng: np.random.Generator, month: int, d0: int, d1: int) -> None:
        a = self.alive
        hazard = MEAN_HAZARD * np.exp(-0.6 * self.lat[a, SIZE] - 0.18)
        can_exit = self.list_month[a] <= month - NO_EXIT_MONTHS
        out = a[can_exit & (rng.random(a.size) < hazard)]
        self.delist_day[out] = rng.integers(d0, d1, size=out.size)
        self.delist_month[out] = month

    def evolve(self, rng: np.random.Generator) -> None:
        """Advance all persistent monthly states of the alive stocks by one month."""
        a, d = self.alive, self.d
        eps = rng.standard_normal((a.size, len(LATENTS))) @ self.chol.T
        self.lat[a] = LATENT_PHI * self.lat[a] + np.sqrt(1.0 - LATENT_PHI**2) * eps
        self.own[a] = CLUSTER_OWN_PHI * self.own[a] + np.sqrt(1.0 - CLUSTER_OWN_PHI**2) * rng.standard_normal(
            (a.size, d.n_clusters))
        self.noise[a] = d.noise_phi * self.noise[a] + np.sqrt(1.0 - d.noise_phi**2) * rng.standard_normal(
            (a.size, d.n_latent))
        enter = MISS_EXIT_PROB * d.miss_rate / (1.0 - d.miss_rate)
        u = rng.random((a.size, d.n_latent))
        miss = self.miss[a]
        self.miss[a] = np.where(miss, u >= MISS_EXIT_PROB, u < enter)
        u = rng.random(a.size)
        gap = self.gap[a]
        self.gap[a] = np.where(gap, u >= GAP_EXIT_PROB, u < GAP_ENTER_PROB)


def simulate_world(cfg: Config) -> WorldResult:
    """Run the monthly simulation loop and collect chars rows and daily returns."""
    streams = np.random.SeedSequence(cfg.seed).spawn(5)
    rng_design, rng_market, rng_factor, rng_world, rng_calib = (np.random.default_rng(s) for s in streams)
    design = design_features(rng_design, cfg.world_features)
    alpha_scale = calibrate_alpha_scale(rng_calib, cfg.n_stocks, cfg.signal_sharpe)

    daily_first_month = cfg.start - _months(cfg.daily_lead_months - 1)
    first = min(daily_first_month, cfg.start) - _months(BURN_IN_MONTHS)
    cal = build_calendar(first, cfg.end + _months(1))
    m_daily, m_first, m_last = cal.index(daily_first_month), cal.index(cfg.start), cal.index(cfg.end)
    day0 = int(cal.month_start[m_daily])
    mkt, mkt_var = simulate_market(rng_market, cal.days[day0:])
    vol_scale = (mkt_var / MKT_LONG_RUN_VAR) ** 0.25  # factor and idio vol co-move with market vol
    ind_f = rng_factor.standard_normal((mkt.size, len(FF12_NAMES))) * IND_DAILY_VOL * vol_scale[:, None]
    sty_f = rng_factor.standard_normal((mkt.size, STYLE_LATENTS.size)) * STYLE_DAILY_VOL * vol_scale[:, None]

    target_main = cfg.world_stocks * ALIVE_PER_CHARS
    target = {MAIN: target_main, EXTRA: EXTRA_SHARE * target_main}
    est = int(sum(target.values()) * (1.0 + 1.5 * MEAN_HAZARD * cal.months.size)) + 100
    w = _World(design, est)
    mkt_hist = np.full(RING, np.nan)
    daily_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    month_store: dict[int, tuple[np.ndarray, ...]] = {}
    snaps: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for m in range(cal.months.size):
        d0, d1 = int(cal.month_start[m]), int(cal.month_end[m])
        if m == 0:
            for kind, tgt in target.items():
                k = int(round(tgt))
                w.add(rng_world, k, kind, np.full(k, d0 - 2 * SEASONING_DAYS), -NO_EXIT_MONTHS)
        else:
            w.draw_exits(rng_world, m, d0, d1)
            for kind, tgt in target.items():
                lam = max(0.0, MEAN_HAZARD * tgt + FEEDBACK * (tgt - w.count_alive(kind, m)))
                k = int(rng_world.poisson(lam))
                w.add(rng_world, k, kind, rng_world.integers(d0, d1, size=k), m)

        w.ret_hist[: w.n, m % RING] = np.nan
        mkt_hist[m % RING] = np.nan
        if m >= m_daily:
            t = slice(d0 - day0, d1 - day0)
            month_store[m] = _daily_block(w, rng_world, d0, d1, mkt[t], vol_scale[t], ind_f[t], sty_f[t],
                                          alpha_scale, daily_parts)
            w.ret_hist[w.alive, m % RING] = month_store[m][1]
            mkt_hist[m % RING] = np.prod(1.0 + mkt[t]) - 1.0

        w.alive = w.alive[w.delist_month[w.alive] != m]
        w.evolve(rng_world)
        if m_first <= m <= m_last:
            snaps.append(_snapshot(w, cal, m, mkt_hist))

    return _collect(cal, design, w, snaps, month_store, daily_parts, day0, mkt, mkt_var, alpha_scale)


def _daily_block(w: _World, rng: np.random.Generator, d0: int, d1: int, mkt: np.ndarray,
                 vol_scale: np.ndarray, ind_f: np.ndarray, sty_f: np.ndarray, alpha_scale: float,
                 daily_parts: list) -> tuple[np.ndarray, ...]:
    """Daily returns of all stocks alive in one month; returns month-level summaries."""
    a = w.alive
    lat = w.lat[a]
    n_days = d1 - d0
    mu = alpha_scale * mu_shape(lat)
    main = w.kind[a] == MAIN
    mu -= mu[main].mean() if main.any() else mu.mean()
    beta = stock_beta(lat, w.fe_beta[a])
    sig = idio_vol(lat, w.fe_vol[a])
    eps = rng.standard_t(IDIO_T_DF, size=(n_days, a.size)) * np.sqrt((IDIO_T_DF - 2.0) / IDIO_T_DF)
    r = (mu / n_days + np.outer(mkt, beta) + ind_f[:, w.ind[a]] * w.gamma[a]
         + sty_f @ lat[:, STYLE_LATENTS].T + eps * sig * vol_scale[:, None])
    np.maximum(r, -0.95, out=r)

    day = np.arange(d0, d1)[:, None]
    valid = (day >= w.list_day[a]) & (day <= w.delist_day[a])
    n_valid = valid.sum(axis=0)
    month_ret = np.prod(np.where(valid, 1.0 + r, 1.0), axis=0) - 1.0
    s1 = np.where(valid, r, 0.0).sum(axis=0)
    s2 = np.where(valid, r * r, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rvol = np.sqrt(np.maximum(s2 - s1 * s1 / n_valid, 0.0) / (n_valid - 1))
    w.rvol[a] = np.where(n_valid >= 15, rvol, np.nan)

    di, si = np.nonzero(valid)
    daily_parts.append(((d0 + di).astype(np.int32), a[si].astype(np.int32), r[di, si]))
    return a, month_ret, mu, beta, sig * np.sqrt(n_days)


def _snapshot(w: _World, cal: Calendar, m: int, mkt_hist: np.ndarray) -> tuple:
    """Features at eom m of the seasoned main stocks (chars candidates)."""
    a, d = w.alive, w.d
    eom_day = int(cal.month_end[m]) - 1
    cand = a[(w.kind[a] == MAIN) & (eom_day - w.list_day[a] + 1 >= SEASONING_DAYS)]
    lat = w.lat[cand]

    signal = (lat @ d.cluster_load) * np.sqrt(1.0 - d.cluster_own**2) + w.own[cand] * d.cluster_own
    z = d.sign * (d.rho * signal[:, d.cluster] + np.sqrt(1.0 - d.rho**2) * w.noise[cand])
    x = apply_transforms(z, d)
    unavailable = ~np.isnat(d.avail) & (d.avail > cal.months[m])
    missing = w.miss[cand] | (w.gap[cand][:, None] & d.accounting) | unavailable
    x[missing] = np.nan

    if len(d.names) > d.n_latent:
        x = np.hstack([x, _return_features(w, cand, m, mkt_hist)])

    pct = (np.argsort(np.argsort(lat[:, SIZE])) + 0.5) / max(cand.size, 1)
    size = np.searchsorted(SIZE_BREAKS, pct, side="right").astype(np.int8)
    return m, cand, x, size, lat


def _return_features(w: _World, cand: np.ndarray, m: int, mkt_hist: np.ndarray) -> np.ndarray:
    """ret_1_0, ret_12_1, rvol_21d, beta_60m from the simulated return history."""
    hist = w.ret_hist[cand]
    ret_1_0 = hist[:, m % RING]
    past = [(m - k) % RING for k in range(1, 12)]
    ret_12_1 = np.prod(1.0 + hist[:, past], axis=1) - 1.0
    ok = ~np.isnan(hist) & ~np.isnan(mkt_hist)
    n = ok.sum(axis=1)
    xm = np.where(ok, mkt_hist, 0.0)
    ym = np.where(ok, hist, 0.0)
    sx, sy = xm.sum(axis=1), ym.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = ((xm * ym).sum(axis=1) - sx * sy / n) / ((xm * xm).sum(axis=1) - sx * sx / n)
    beta[n < 36] = np.nan
    return np.column_stack([ret_1_0, ret_12_1, w.rvol[cand], beta])


def _collect(cal: Calendar, design: FeatureDesign, w: _World, snaps: list,
             month_store: dict, daily_parts: list, day0: int, mkt: np.ndarray, mkt_var: np.ndarray,
             alpha_scale: float) -> WorldResult:
    """Keep chars rows whose stock survives the return month and attach the realized return."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("stock", "month", "x", "size", "ret", "mu",
                                                          "beta", "sigma", "lat")}
    for m, cand, x, size, lat in snaps:
        keep = w.delist_month[cand] > m + 1
        stocks = cand[keep]
        a_next, ret_next, mu_next, beta_next, sig_next = month_store[m + 1]
        pos = np.searchsorted(a_next, stocks)
        assert np.array_equal(a_next[pos], stocks)
        assert np.all(w.list_day[stocks] <= cal.month_start[m + 1])
        for key, val in (("stock", stocks), ("month", np.full(stocks.size, m)), ("x", x[keep]),
                         ("size", size[keep]), ("ret", ret_next[pos]), ("mu", mu_next[pos]),
                         ("beta", beta_next[pos]), ("sigma", sig_next[pos]), ("lat", lat[keep])):
            parts[key].append(val)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return WorldResult(
        cal=cal, design=design, row_stock=cat["stock"], row_month=cat["month"], row_x=cat["x"],
        row_size=cat["size"], row_ret=cat["ret"], row_mu=cat["mu"], row_beta=cat["beta"],
        row_sigma=cat["sigma"], row_latent=cat["lat"],
        daily_day=np.concatenate([p[0] for p in daily_parts]),
        daily_stock=np.concatenate([p[1] for p in daily_parts]),
        daily_ret=np.concatenate([p[2] for p in daily_parts]),
        stock_kind=w.kind[: w.n].copy(), stock_ind=w.ind[: w.n].copy(), stock_sic=list(w.sic),
        stock_gamma=w.gamma[: w.n].copy(), mkt_first_day=day0, mkt_ret=mkt, mkt_var=mkt_var,
        alpha_scale=alpha_scale,
    )


# --------------------------------------------------------------------------------------
# Output datasets
# --------------------------------------------------------------------------------------


@dataclass
class SyntheticData:
    """Arrow tables of the CTF files plus the (optional) truth tables."""

    chars: pa.Table
    features: pa.Table
    daily_ret: pa.Table
    truth_chars: pa.Table
    truth_market: pa.Table
    truth_features: pa.Table


@dataclass(frozen=True)
class Selection:
    """Which world rows, features, stocks and daily rows make up the output dataset."""

    rows: np.ndarray
    test: np.ndarray  # ctff_test per selected row
    features: np.ndarray  # indices into design.names
    daily: np.ndarray  # indices into the daily arrays
    sic: list[str | None]


def select_full(world: WorldResult) -> Selection:
    rows = np.arange(world.row_stock.size)
    eom_ret = world.cal.eom[world.row_month + 1]
    return Selection(rows=rows, test=eom_ret >= TEST_FIRST_EOM_RET,
                     features=np.arange(len(world.design.names)),
                     daily=np.arange(world.daily_ret.size), sic=world.stock_sic)


def select_validation(world: WorldResult, cfg: Config, ids: np.ndarray, rng: np.random.Generator) -> Selection:
    """Subsample the world the way utils/toy_data.R builds the organizers' validation replica."""
    month = world.row_month
    last = int(month.max())
    n_months = last - int(month.min()) + 1
    ind_name = ff12_class(world.stock_sic)[world.row_stock]

    late_rows = np.flatnonzero((ind_name == "Enrgy") & (month == last))
    if late_rows.size == 0:
        raise RuntimeError("no Enrgy stock in the last month; increase --n-stocks or change --seed")
    late_row = late_rows[np.argmin(ids[world.row_stock[late_rows]])]
    late_stock = int(world.row_stock[late_row])

    base = np.flatnonzero(~np.isin(ind_name, ["Utils", "Enrgy"]))
    stocks, counts = np.unique(world.row_stock[base], return_counts=True)
    coverage = counts / n_months
    good = stocks[coverage >= VALIDATION_MIN_COVERAGE]
    if good.size > cfg.n_stocks:
        chosen = np.sort(rng.choice(good, size=cfg.n_stocks, replace=False))
    elif good.size >= 30:
        chosen = good
    else:
        chosen = np.sort(stocks[np.argsort(-coverage, kind="stable")[: cfg.n_stocks]])
    rows = np.sort(np.append(base[np.isin(world.row_stock[base], chosen)], late_row))

    first_ret_day = world.cal.month_start[last + 1]
    others = np.setdiff1d(np.unique(world.daily_stock), np.append(chosen, late_stock))
    extra_pool = others[world.stock_kind[others] == EXTRA]
    n_extra = min(extra_pool.size, int(round(0.2 * (chosen.size + 1))))
    extras = rng.choice(extra_pool, size=n_extra, replace=False)
    daily = np.flatnonzero(np.isin(world.daily_stock, np.concatenate([chosen, extras]))
                           | ((world.daily_stock == late_stock) & (world.daily_day >= first_ret_day)))

    sic = list(world.stock_sic)
    sic[late_stock] = VALIDATION_LATE_SIC
    names = np.array(world.design.names)
    picked = np.sort(rng.choice(names.size, size=min(cfg.n_features, names.size), replace=False))
    picked = picked[np.argsort(names[picked])]
    test = month[rows] > last - VALIDATION_TEST_MONTHS
    return Selection(rows=rows, test=test, features=picked, daily=daily, sic=sic)


def _string_array(values: np.ndarray | list) -> pa.Array:
    return pa.array(list(values), type=pa.string())


def build_tables(world: WorldResult, sel: Selection, ids: np.ndarray) -> SyntheticData:
    """Arrow tables with the CTF schema: date32 dates, string text, bool flag, int64 ids."""
    cal, design = world.cal, world.design
    stock = world.row_stock[sel.rows]
    month = world.row_month[sel.rows]
    row_id = ids[stock]
    order = np.lexsort((month, row_id))
    rows, stock, month, row_id, test = sel.rows[order], stock[order], month[order], row_id[order], sel.test[order]

    names = np.array(design.names)
    feat_order = sel.features[np.argsort(names[sel.features], kind="stable")]
    sic = np.array(sel.sic, dtype=object)
    cols: dict[str, pa.Array] = {
        "id": pa.array(row_id, type=pa.int64()),
        "eom": pa.array(cal.eom[month]),
        "eom_ret": pa.array(cal.eom[month + 1]),
        "excntry": _string_array(np.full(rows.size, "USA", dtype=object)),
        "sic": _string_array(sic[stock]),
        "size_grp": _string_array(SIZE_GROUPS[world.row_size[rows]]),
        "ret_exc_lead1m": pa.array(world.row_ret[rows], type=pa.float64()),
        "ctff_test": pa.array(test, type=pa.bool_()),
    }
    for j in feat_order:
        cols[str(names[j])] = pa.array(world.row_x[rows, j], type=pa.float64(), from_pandas=True)
    chars = pa.table(cols)
    features = pa.table({"features": _string_array(names[feat_order])})

    d_id = ids[world.daily_stock[sel.daily]]
    d_day = world.daily_day[sel.daily]
    d_order = np.lexsort((d_day, d_id))
    daily = pa.table({
        "id": pa.array(d_id[d_order], type=pa.int64()),
        "date": pa.array(cal.days[d_day[d_order]]),
        "ret_exc": pa.array(world.daily_ret[sel.daily][d_order], type=pa.float64()),
    })

    lat = world.row_latent[rows]
    truth = {
        "id": pa.array(row_id, type=pa.int64()), "eom": pa.array(cal.eom[month]),
        "mu": pa.array(world.row_mu[rows]), "beta": pa.array(world.row_beta[rows]),
        "gamma": pa.array(world.stock_gamma[stock]), "sigma_idio": pa.array(world.row_sigma[rows]),
        "ff12": _string_array(np.array(FF12_NAMES, dtype=object)[world.stock_ind[stock]]),
        **{f"l_{name}": pa.array(lat[:, k]) for k, name in enumerate(LATENTS)},
    }
    mkt_days = cal.days[world.mkt_first_day:]
    truth_market = pa.table({"date": pa.array(mkt_days), "mkt_ret": pa.array(world.mkt_ret),
                             "mkt_vol": pa.array(np.sqrt(world.mkt_var))})
    return SyntheticData(chars, features, daily, pa.table(truth), truth_market, _design_table(design))


def _latent_mix(weights: np.ndarray) -> str:
    mix = [f"{w:+.2f}*{LATENTS[k]}" for k, w in enumerate(weights) if w != 0]
    return " ".join(mix) or "noise"


def _design_table(d: FeatureDesign) -> pa.Table:
    n = len(d.names)
    pad = n - d.n_latent

    def padded(values: np.ndarray, fill: object) -> list:
        return list(values) + [fill] * pad

    return pa.table({
        "feature": _string_array(d.names), "theme": _string_array(d.themes),
        "cluster": pa.array(padded(d.cluster, -1), type=pa.int64()),
        "transform": _string_array(padded(np.asarray(TRANSFORMS)[d.transform], "returns")),
        "latents": _string_array(padded([_latent_mix(d.cluster_load[:, c]) for c in d.cluster], "returns")),
        "sign": pa.array(padded(d.sign, 1.0), type=pa.float64()),
        "rho": pa.array(padded(d.rho, np.nan), type=pa.float64(), from_pandas=True),
        "miss_rate": pa.array(padded(d.miss_rate, 0.0), type=pa.float64()),
        "accounting": pa.array(padded(d.accounting, False), type=pa.bool_()),
        "avail_start": _string_array(padded([None if np.isnat(a) else str(a) for a in d.avail], None)),
    })


def simulate(cfg: Config) -> SyntheticData:
    """Simulate the world and cut the dataset for the configured mode."""
    world = simulate_world(cfg)
    rng_ids, rng_select = (np.random.default_rng(s) for s in np.random.SeedSequence([cfg.seed, 1]).spawn(2))
    n_total = world.stock_kind.size
    if n_total > 90_000:
        raise ValueError(f"{n_total} simulated stocks exceed the 5-digit id space")
    ids = rng_ids.choice(np.arange(10_000, 100_000), size=n_total, replace=False).astype(np.int64)
    sel = select_full(world) if cfg.mode == "full" else select_validation(world, cfg, ids, rng_select)
    return build_tables(world, sel, ids)


def generate(mode: str, start: str | None = None, end: str | None = None, n_stocks: int | None = None,
             n_features: int | None = None, seed: int = 7, *, daily_lead_months: int = 12,
             signal_sharpe: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(chars, features, daily_ret) DataFrames with the dtypes pandas.read_parquet would give."""
    cfg = resolve_config(mode, start, end, n_stocks, n_features, seed, daily_lead_months, signal_sharpe)
    data = simulate(cfg)
    return data.chars.to_pandas(), data.features.to_pandas(), data.daily_ret.to_pandas()


def write_dataset(data: SyntheticData, out_dir: Path, truth: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(data.chars, out_dir / "ctff_chars.parquet")
    pq.write_table(data.features, out_dir / "ctff_features.parquet")
    pq.write_table(data.daily_ret, out_dir / "ctff_daily_ret.parquet")
    if truth:
        pq.write_table(data.truth_chars, out_dir / "truth_chars.parquet")
        pq.write_table(data.truth_market, out_dir / "truth_market.parquet")
        pq.write_table(data.truth_features, out_dir / "truth_features.parquet")


def _summary(data: SyntheticData) -> str:
    chars = data.chars.select(["id", "eom", "ctff_test"]).to_pandas()
    per_month = chars.groupby("eom").size()
    n_daily_ids = len(pc.unique(data.daily_ret["id"]))
    return (f"chars: {data.chars.num_rows:,} rows x {data.chars.num_columns} cols, "
            f"{per_month.size} months ({chars.loc[chars.ctff_test, 'eom'].nunique()} test), "
            f"{chars.id.nunique():,} ids, {per_month.mean():.0f} stocks/month "
            f"[{per_month.min()}-{per_month.max()}]; features: {data.features.num_rows}; "
            f"daily_ret: {data.daily_ret.num_rows:,} rows, {n_daily_ids:,} ids")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=sorted(DEFAULTS), required=True)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--start", help="first chars month YYYY-MM (full: 1951-12; validation: end - 122 months)")
    p.add_argument("--end", help="last chars month YYYY-MM (default 2023-11)")
    p.add_argument("--n-stocks", type=int, help="stocks per month (full: 300; validation: 50)")
    p.add_argument("--n-features", type=int, help="number of features (full: 60; validation: 10)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--daily-lead-months", type=int, default=12,
                   help="months of daily returns before the first chars eom (default 12)")
    p.add_argument("--signal-sharpe", type=float, default=2.0,
                   help="annual Sharpe of the optimal hedged portfolio on the planted mu (0 = no signal)")
    p.add_argument("--truth", action="store_true", help="also write truth_*.parquet")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    cfg = resolve_config(args.mode, args.start, args.end, args.n_stocks, args.n_features, args.seed,
                         args.daily_lead_months, args.signal_sharpe)
    data = simulate(cfg)
    write_dataset(data, args.out, truth=args.truth)
    print(f"[{cfg.mode}] {cfg.start}..{cfg.end} seed={cfg.seed} -> {args.out} "
          f"({time.perf_counter() - t0:.1f}s)\n  {_summary(data)}")


if __name__ == "__main__":
    main()
