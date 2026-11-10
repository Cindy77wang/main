---
title: "Problem Set: Volatility"
subtitle: "MGT 924 — Statistical Foundations"
---

**Data.** CRSP daily data from WRDS, 2 Jan 1990 – 31 Dec 2020 ($T = 7{,}812$ trading days).
The market series is the CRSP value-weighted index return (`vwretd`; using the Fama–French
`mktrf` gives a 0.998 correlation and identical conclusions). MSFT is its CRSP holding-period
return. All returns are in percent per day. Every likelihood below is coded by hand and
maximized numerically (SLSQP, with the stationarity restriction imposed as a constraint);
standard errors are the robust Bollerslev–Wooldridge sandwich $\hat H^{-1}\hat S\hat H^{-1}$
with a numerical Hessian and outer product of numerical scores. As an independent check, the
`arch` Python package reproduces every point estimate to the third decimal. Code:
`volatility_ps.py`; all numbers: `results.json`.

| | Mean | S.d. | Ann. vol | Skew | Excess kurt. | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Market | 0.044 | 1.13 | 17.9% | −0.32 | 11.0 | −11.8 | 11.5 |
| MSFT | 0.102 | 2.02 | 32.1% | 0.23 | 6.5 | −15.6 | 19.6 |

# Question 1

## (a) Autocorrelations of returns and squared returns

**Prior (stated before plotting).** I expect the *market* to show stronger volatility
clustering in the sense of higher and more slowly decaying autocorrelations of $r_t^2$. Write
MSFT's return as $r_{i,t} = \beta r_{m,t} + e_{i,t}$. The persistent part of volatility is
largely economy-wide (macro uncertainty, financial-sector stress, risk appetite), and the
market carries all of it. A single stock adds an idiosyncratic component $e_{i,t}$ that is
dominated by one-off news (earnings releases, product launches, antitrust rulings). These
are big, jump-like and mostly not followed by more of the same, so they add a lot of
variance to $r_{i,t}^2$ without adding much autocovariance, which mechanically lowers the
autocorrelation of squared returns. Diversification removes exactly that noise from the
index. For the levels of returns I expect both ACFs to be close to zero (weak-form
efficiency), with perhaps a small negative first-order coefficient.

![Autocorrelations, lags 1–20](figures/acf.png){width=100%}

**Result: the prior is confirmed.**

* *Returns.* Autocorrelations are tiny for both series (all $|\rho_k| < 0.06$). The market's
  lag-1 coefficient is $-0.056$ and MSFT's is $-0.047$, both slightly outside the $\pm 0.022$
  band. With $T = 7{,}812$ such small values are statistically detectable (Ljung–Box(20)
  $p<0.001$ for both), but they are economically negligible: $\rho_1^2 < 0.4\%$ of the variance is predictable.
* *Squared returns.* Both series cluster strongly (Ljung–Box(20) statistics of 11,017 for the
  market and 1,800 for MSFT, $p \approx 0$), but the market's ACF is two to three times higher
  at every lag: 0.31 / 0.45 at lags 1–2 and still 0.18 at lag 20, versus 0.18 / 0.15 and
  0.05 for MSFT. The same ranking holds for $|r_t|$, which is less sensitive to outliers
  (market 0.29 → 0.24, MSFT 0.24 → 0.12 from lag 1 to lag 20), so the result is not driven
  by a few crash days.
* *Mechanism check.* As Q1(c) shows, MSFT's GARCH persistence $\alpha+\beta$ is the same as the
  market's (0.985 for both). What differs is the ARCH loading $\alpha$ relative to the
  noise: MSFT's squared shocks carry more one-off, idiosyncratic noise, so each squared
  return is a noisier signal of the underlying volatility. That is exactly the "noise
  dilutes the autocorrelation" story behind the prior.

## (b) Gaussian GARCH(1,1) by maximum likelihood

**Model.**
$$
r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \quad z_t \overset{iid}{\sim} N(0,1), \qquad
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,
$$
with $\omega>0$, $\alpha,\beta\ge 0$ and $\alpha+\beta<1$ (covariance stationarity).

**Log likelihood.** Conditional on $\mathcal F_{t-1}$, $r_t \sim N(\mu,\sigma_t^2)$, and
$\sigma_t^2$ is a known function of the parameters and past data. By the prediction-error
decomposition, the log likelihood is the sum of the one-step conditional densities. With
$\theta = (\mu,\omega,\alpha,\beta)$:
$$
\mathcal L(\theta) = \sum_{t=1}^{T} \ell_t(\theta) = -\frac{1}{2}\sum_{t=1}^{T}\left[\log(2\pi) + \log\sigma_t^2(\theta) + \frac{(r_t-\mu)^2}{\sigma_t^2(\theta)}\right],
\qquad \hat\theta = \arg\max_{\theta} \mathcal L(\theta),
$$
with the recursion started at $\sigma_1^2 = T^{-1}\sum_t (r_t-\mu)^2$ (backcast). Because
returns are fat-tailed, the Gaussian likelihood is treated as a quasi-likelihood. The
estimator stays consistent under correct mean and variance equations, and inference uses
sandwich standard errors.

| | $\mu$ | $\omega$ | $\alpha$ | $\beta$ | $\alpha+\beta$ | Half-life (days) | Long-run ann. vol | log L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Market | 0.071 | 0.0175 | 0.112 | 0.873 | 0.985 | 47 | 17.4% | −10,110.2 |
| | (0.009) | (0.0035) | (0.012) | (0.012) | | | | |
| MSFT | 0.116 | 0.066 | 0.079 | 0.906 | 0.985 | 47 | 33.9% | −15,649.6 |
| | (0.018) | (0.036) | (0.025) | (0.030) | | | | |

*Robust standard errors in parentheses. Half-life $= \log 0.5/\log(\alpha+\beta)$. Long-run
volatility $=\sqrt{252\,\omega/(1-\alpha-\beta)}$.*

Both series have the canonical daily-GARCH shape: small $\alpha$, large $\beta$, and
persistence near (but significantly below) one. The market reacts more strongly to news
($\alpha = 0.112$ vs 0.079), while MSFT relies more on the slowly moving component
($\beta = 0.906$). The implied long-run volatilities, 17.4% and 33.9%, match the sample
volatilities (17.9% and 32.1%).

![GARCH(1,1) conditional volatility](figures/garch_vol.png){width=100%}

## (c) "MSFT's volatility is essentially just the market moving"

If the claim were true, $\sigma^2_{MSFT,t} \approx \beta^2\sigma^2_{m,t}$: the systematic
share of MSFT's conditional variance would be close to one, and the market-model residual
would have no volatility dynamics of its own. The evidence rejects both.

1. **Unconditional decomposition.** In the market model $r_{MSFT,t} = a + b\,r_{m,t} + e_t$,
   $\hat b = 1.15$ and $R^2 = 0.41$. So **59% of MSFT's daily variance is idiosyncratic**.
2. **Conditional decomposition from the GARCH estimates.** Using the two fitted GARCH
   variances and a rolling one-year beta (estimated with past data only), the systematic
   share $\hat\beta_t^2\hat\sigma^2_{m,t}/\hat\sigma^2_{MSFT,t}$ averages **0.41** (median
   0.35; 10th–90th percentile 0.17–0.73). With the full-sample beta the average is 0.39.
   The share approaches one only in market-wide crises (2002, late 2008, March 2020). In
   calm periods, and in MSFT-specific episodes (the 1990s antitrust case, the 2000 breakup
   ruling, 2006 earnings shocks), it falls to 0.1–0.3.
3. **Idiosyncratic volatility has its own GARCH.** A GARCH(1,1) on the residual $e_t$ gives
   $\alpha = 0.024\,(0.006)$ and $\beta = 0.971\,(0.007)$, with persistence 0.995 (half-life
   about 150 days) and a long-run volatility of **24.4% a year**. That is larger than the
   market's own 17.4%. The residual's squared ACF is still highly significant (Ljung–Box(20) = 704).
4. **The two volatility paths are related but far from identical.** The fitted
   annualized volatilities have correlation 0.69. Regressing $\log\hat\sigma^2_{MSFT,t}$ on
   $\log\hat\sigma^2_{m,t}$ gives slope 0.50 and $R^2 = 0.39$, so 61% of the time-variation
   in MSFT's log variance is *not* explained by market variance. Adding lagged market variance
   to MSFT's own GARCH ($\sigma^2_t = \omega+\alpha\varepsilon^2_{t-1}+\beta\sigma^2_{t-1}+\delta\sigma^2_{m,t}$)
   gives $\hat\delta = 0.035$ with robust s.e. 0.034. Once MSFT's own shocks are in the
   equation, market variance adds little.
5. **The claim has become more true over time, but it is still not true.**

| Period | $\hat b$ | $R^2$ | MSFT vol | Market vol | Idiosyncratic vol |
|---|---:|---:|---:|---:|---:|
| 1990–1999 | 1.49 | 0.32 | 34.3% | 13.0% | 28.3% |
| 2000–2009 | 1.08 | 0.45 | 36.0% | 22.2% | 26.8% |
| 2010–2020 | 1.07 | 0.54 | 25.4% | 17.4% | 17.3% |

**Verdict.** The claim is wrong as a description of 1990–2020. On average only about 40% of
MSFT's variance is market variance, and the idiosyncratic 60% is persistent and predictable
in its own right. The claim is closest to right in crises, when correlations spike and the
systematic share approaches one, and in the most recent decade, when MSFT became a mega-cap
that is a large part of the index itself (about 54% systematic). A risk model that scaled
market volatility by beta would understate MSFT's risk in normal times and miss
firm-specific volatility episodes entirely.

![MSFT variance decomposition](figures/msft_decomp.png){width=100%}

## (d) What the model says about market risk over the next month (as of 31 Dec 2020)

Under GARCH(1,1), the $h$-step variance forecast is
$E_t\sigma^2_{t+h} = \bar\sigma^2 + (\alpha+\beta)^{h-1}(\sigma^2_{t+1}-\bar\sigma^2)$. The
one-month (21-day) variance is the sum of these forecasts.

* **Markets are calm right now, and that calm is informative.** At the end of 2020 the model
  estimates the market's volatility at about **9% a year**, roughly half its 30-year normal
  of 17%. Because turbulent and quiet periods both tend to persist, the model expects
  January 2021 to be quieter than a typical month. One standard deviation of the
  month's return is about **±3.0%**, versus ±5.0% in an average month. In practice, a month
  of −1.5% to +4.5% would be ordinary (about two months in three land in that range).
* **But calm fades, so the forecast drifts back toward normal.** About half of any deviation
  from normal volatility disappears every ~47 trading days (about two months). Over the month,
  the model's daily volatility forecast rises from about 9% to about 12% (annualized). The
  further ahead you look, the less today's calm matters and the more the forecast looks like
  an average month.
* **Low volatility does not mean no risk, and risk can jump quickly.** The model puts a
  roughly 1-in-20 chance on a monthly loss worse than **about 3.5%**, and a 1-in-100 chance
  on a loss worse than about 6%. These numbers are only as good as the next few weeks'
  news. A single large shock raises the model's risk estimate immediately. For comparison,
  on 17 March 2020 the same model put annualized volatility near 100% and a one-month
  standard deviation of about 27%. (The Gaussian model also tends to understate how severe
  the worst days can be; see Q3.)

![Market volatility term structure](figures/forecast.png){width=80%}

# Question 2 (PhD) — Gaussian GARCH-in-mean

**Model and MLE objective.**
$$
r_t = \mu + \lambda\,g(\sigma_t^2) + \varepsilon_t,\quad \varepsilon_t=\sigma_t z_t,\ z_t\sim N(0,1),\qquad
\sigma_t^2 = \omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2,
$$
where $g(\sigma^2) = \sigma^2$ (the risk–return trade-off) or $g = \sigma$ or $\log\sigma^2$.
With $\theta = (\mu,\lambda,\omega,\alpha,\beta)$,
$$
\hat\theta = \arg\max_\theta\; \mathcal L(\theta) = -\frac12\sum_{t=1}^T\left[\log 2\pi + \log\sigma_t^2(\theta) + \frac{\big(r_t-\mu-\lambda g(\sigma^2_t(\theta))\big)^2}{\sigma^2_t(\theta)}\right]
$$
subject to $\omega>0$, $\alpha,\beta\ge0$, $\alpha+\beta<1$. The key difference from plain
GARCH is that the residual $\varepsilon_t(\theta) = r_t-\mu-\lambda g(\sigma_t^2)$ now
depends on the variance parameters. The recursion must be run jointly: compute
$\sigma_t^2$ from $\varepsilon_{t-1}$, then $\varepsilon_t$ from $\sigma_t^2$. The
information matrix is no longer block-diagonal between mean and variance parameters, so
misspecifying the variance equation makes the mean estimates inconsistent, and they must be
estimated jointly.

*Illustration (not required).* With $g=\sigma^2$, the market gives
$\hat\lambda = 0.041$ (robust s.e. 0.013) and MSFT gives $\hat\lambda = 0.012$ (s.e. 0.009).
The market shows a significant positive risk–return trade-off; MSFT does not, which is
consistent with only systematic risk being priced.

**Adding return predictors $X_t$ to the mean.** Replace the mean with
$m_t(\theta) = \mu + \lambda g(\sigma_t^2) + \gamma' X_{t-1}$, where $X_{t-1}$ is known at
$t-1$ (dividend yield, term spread, lagged returns, ...), and maximize
$$
\mathcal L(\theta,\gamma) = -\tfrac12\sum_t\left[\log2\pi+\log\sigma_t^2+\frac{(r_t-m_t)^2}{\sigma_t^2}\right],\qquad \varepsilon_t = r_t - m_t.
$$
The only changes are that the extra parameters $\gamma$ enter $\varepsilon_t$, and that the
variance recursion is fed the new $\varepsilon_{t-1}$. $X$ must be predetermined, and $X$
and $g(\sigma^2_t)$ must not be collinear. If a predictor is itself a volatility proxy,
$\lambda$ and $\gamma$ become weakly identified.

**Adding variance predictors $Q_t$ to the GARCH equation (GARCH-X).**
$$
\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2 + \delta' Q_{t-1}
$$
(e.g., lagged VIX², realized variance, market variance as in Q1(c)), with the same
objective. The additional regularity conditions are:

1. **Positivity for every $t$**: $\sigma_t^2>0$ is no longer guaranteed by $\omega,\alpha,\beta>0$.
   It suffices to have $Q_t \ge 0$ (e.g., variances, squared spreads) and $\delta\ge0$.
   For sign-indefinite predictors you must either restrict $\omega + \delta'Q_t > 0$ over the
   support of $Q$, or reparameterize: $(\delta' Q)^2$, $\exp(\delta'Q)$, or an EGARCH-X
   form in $\log\sigma^2_t$.
2. **Measurability**: $Q_{t-1}\in\mathcal F_{t-1}$, otherwise $\sigma_t^2$ is not a
   conditional variance and the likelihood factorization fails.
3. **Stationarity and moments**: $\alpha+\beta<1$ is no longer enough by itself. $Q_t$ must
   be strictly stationary and ergodic with $E|Q_t|<\infty$ (and $E\|Q_t\|^2<\infty$ for
   asymptotic normality). Then
   $E\sigma^2 = (\omega+\delta'E[Q])/(1-\alpha-\beta)$. A trending or unit-root $Q$ would
   make $\sigma_t^2$ non-stationary even if $\alpha+\beta<1$.
4. **Identification**: $\omega$ and $\delta'E[Q]$ are separated only if $Q$ varies over
   time. $Q$ must not be (asymptotically) collinear with $\varepsilon_{t-1}^2$ or
   $\sigma^2_{t-1}$. A large $Q$ coefficient can push $\alpha$ to the boundary
   $\alpha = 0$, where standard asymptotics fail (LR tests of $\delta=0$ follow a
   $\bar\chi^2$ mixture).

# Question 3 (PhD) — GJR-GARCH(1,1,1) with Student-$t$ innovations

**Model.**
$$
r_t = \mu+\varepsilon_t,\quad \varepsilon_t=\sigma_t z_t,\quad z_t \overset{iid}{\sim} t_\nu^{std}\ (\text{mean }0,\ \text{variance }1,\ \nu>2),
$$
$$
\sigma_t^2 = \omega + \big(\alpha + \gamma\,\mathbb 1\{\varepsilon_{t-1}<0\}\big)\varepsilon_{t-1}^2 + \beta\sigma^2_{t-1}.
$$
The log-likelihood with the standardized-$t$ density is
$$
\mathcal L = \sum_t\left[\log\Gamma\!\left(\tfrac{\nu+1}{2}\right)-\log\Gamma\!\left(\tfrac{\nu}{2}\right)-\tfrac12\log\big(\pi(\nu-2)\big)-\tfrac12\log\sigma_t^2-\tfrac{\nu+1}{2}\log\!\left(1+\frac{\varepsilon_t^2}{(\nu-2)\sigma_t^2}\right)\right].
$$

**Non-integration (covariance-stationarity) condition.** Write
$\varepsilon_{t-1}^2 = \sigma^2_{t-1}z_{t-1}^2$ and
$\mathbb 1\{\varepsilon_{t-1}<0\} = \mathbb 1\{z_{t-1}<0\}$ (since $\sigma_{t-1}>0$). Then
$$
\sigma_t^2 = \omega + \sigma^2_{t-1}\,\big[\alpha z_{t-1}^2 + \gamma z_{t-1}^2\mathbb 1\{z_{t-1}<0\} + \beta\big].
$$
Because $z_{t-1}$ is independent of $\sigma^2_{t-1}\in\mathcal F_{t-2}$,
$$
E_{t-2}\,\sigma_t^2 = \omega + \big(\alpha\,E z^2 + \gamma\,E[z^2\mathbb 1\{z<0\}] + \beta\big)\sigma_{t-1}^2 = \omega + \phi\,\sigma^2_{t-1}.
$$
Here $E z^2 = 1$. Because the standardized $t$ is symmetric about zero,
$E[z^2\mathbb 1\{z<0\}] = \tfrac12 E z^2 = \tfrac12$, for any $\nu>2$. Hence
$$
\phi = \alpha + \tfrac{\gamma}{2} + \beta.
$$
Iterating, $E_{t-k}\sigma^2_{t} = \omega\sum_{j=0}^{k-2}\phi^j + \phi^{k-1}\sigma^2_{t-k+1}$.
Shocks to variance die out, and the unconditional variance
$E\sigma_t^2 = E\varepsilon^2_t = \omega/(1-\phi)$ is finite and positive, if and only if
$$
\boxed{\ \alpha + \tfrac12\gamma + \beta < 1\ }\qquad(\text{with } \omega>0,\ \alpha\ge0,\ \alpha+\gamma\ge0,\ \beta\ge0,\ \nu>2).
$$
At $\phi=1$ the model is *integrated* (IGARCH-type): variance shocks are permanent, and the
unconditional variance does not exist. (Strict stationarity only needs the weaker
$E\log(\alpha z^2+\gamma z^2\mathbb 1\{z<0\}+\beta)<0$. The condition above is the one for a
finite unconditional variance and mean-reverting forecasts. With asymmetric innovations,
$\tfrac12$ would be replaced by $E[z^2\mathbb 1\{z<0\}]$.) The forecast half-life is
$\log 0.5/\log\phi$.

**Estimates** (robust s.e. in parentheses; $\phi = \alpha+\gamma/2+\beta$):

| | $\mu$ | $\omega$ | $\alpha$ | $\gamma$ | $\beta$ | $\nu$ | $\phi$ | log L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Market | 0.065 | 0.0137 | 0.009 | 0.163 | 0.895 | 6.9 | 0.985 | −9,832.4 |
| | (0.008) | (0.0024) | (0.006) | (0.018) | (0.010) | (0.57) | | |
| MSFT | 0.073 | 0.032 | 0.056 | 0.068 | 0.910 | 5.3 | 1.000† | −15,220.2 |
| | (0.016) | (0.010) | (0.010) | (0.015) | (0.014) | (0.32) | | |

† The constraint $\phi<1$ binds. Without it the MLE is $\phi = 1.0017$, with a
log-likelihood gain of only 0.12, so the data cannot distinguish MSFT's GJR-$t$ variance from
an integrated process.

Log-likelihood ladder (which feature matters):

| | GARCH-N | GARCH-$t$ | GJR-N | GJR-$t$ |
|---|---:|---:|---:|---:|
| Market | −10,110.2 | −9,927.5 | −9,999.7 | −9,832.4 |
| MSFT | −15,649.6 | −15,238.1 | −15,624.0 | −15,220.2 |

**GJR-$t$ vs GARCH(1,1)-N.**

* *Fit.* GJR-$t$ improves log L by 278 (market) and 429 (MSFT) with two extra parameters.
  Both improvements are enormous. For MSFT almost all of the gain comes from fat tails
  ($t$: +412, leverage: +25). For the market both features matter (fat tails: +183,
  leverage: +111).
* *Tails.* $\hat\nu = 6.9$ (market) and 5.3 (MSFT). Even after GARCH scaling, standardized
  shocks are far from Gaussian, and more so for the single stock (jumps from firm-specific
  news). This is why the Gaussian GARCH has to be read as a QMLE.
* *Dynamics.* In GARCH-N, symmetric news enters through $\alpha$. In GJR the market's
  symmetric coefficient collapses to $\alpha = 0.009$ (not significant): essentially *all*
  of the market's ARCH effect comes from negative returns. Persistence is unchanged for the
  market (0.985, half-life about 46 days). For MSFT it rises to the unit-root boundary: both the average news response $\alpha+\gamma/2$ (0.079 → 0.090) and $\beta$ (0.906 → 0.910) go up. Under a fat-tailed density the largest MSFT outliers are down-weighted as $t$-draws instead of forcing a lower $\beta$, which leaves a smoother, more persistent variance path. MSFT's long-run variance is therefore poorly pinned down.
* *Unconditional level.* The market's implied long-run volatility falls from 17.4% to 15.2%,
  because the asymmetric model attributes high-variance periods to sequences of negative
  shocks rather than to a higher baseline.

**Leverage: market vs MSFT.**

| | $\alpha$ (good news) | $\alpha+\gamma$ (bad news) | Ratio bad/good |
|---|---:|---:|---:|
| Market | 0.009 | 0.171 | ≈ 19 |
| MSFT | 0.056 | 0.124 | ≈ 2.2 |

The asymmetry is far stronger for the market. Its $\gamma = 0.163$ is 2.4 times MSFT's 0.068,
and for the index good news barely raises volatility at all, while for MSFT both signs
matter. This is the standard finding, and it argues against a pure *financial leverage*
explanation. MSFT had essentially no debt over this period, yet the asymmetry in its
volatility is mild, while the index's is extreme. The asymmetry instead looks systematic:
**volatility feedback** (a rise in market risk raises required returns and lowers prices,
so negative returns and rising volatility arrive together) and correlation spikes in
selloffs act on the aggregate. Firm-specific shocks, good or bad (blowout earnings, product
launches), raise MSFT's volatility symmetrically. Consistent with Q1(c), MSFT's $\gamma$ is 0.42 times the market's, which is close to the
0.41 average systematic share of its variance. MSFT's leverage effect is about what its
market component alone would deliver.

![News impact curves](figures/news_impact.png){width=100%}
