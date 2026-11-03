# Problem Set: MLE, GMM, and ARMA — Solutions

All estimators are coded from scratch in numpy: OLS normal equations, Poisson and Gaussian
likelihoods, scores and Hessians, Kalman filter, Hannan–Rissanen, ACF/PACF (Durbin–Levinson),
Ljung–Box, GMM, Newey–West, and the Wald/LR/LM, J, Diebold–Mariano and Clark–West tests.
The only library calls are `scipy.optimize` (numerical optimizer) and `scipy.stats`
(χ²/normal tail probabilities). The problem set allows both.

| File | Content |
|---|---|
| `common.py` | OLS, Newey–West, plotting style, table writer |
| `q1_arma.py` | Question 1 (JKP factors: ACF/PACF, AR/ARMA by regression and exact MLE, OOS, forecast) |
| `q2_poisson_ar.py` | Question 2 (OLS vs. exact Poisson MLE, Wald / LR / LM tests) |
| `q3_gmm.py` | Question 3 (GMM for the Poisson λ, J-test) |
| `results/` | Every table (`.md`/`.csv`) and figure produced by the scripts |

```bash
pip install -r requirements.txt
python q2_poisson_ar.py
python q3_gmm.py
python q1_arma.py
```

`q1_arma.py` reads the JKP file in `inputs/jkp/` and takes about 1.5 minutes.

---

## Question 1 — ARMA models for JKP factors

**Data.** `inputs/jkp/usa_all_themes_monthly_vw_cap.csv`: JKP USA *theme* portfolios,
monthly, capped value-weighted. Each theme is the average of the JKP factors in that
cluster. The sample is 1975-01 to 2025-12 (T = 612); returns are in decimals per month.
Sample statistics: `results/q1_summary_stats.md`.

### Factor choice

| Theme | Mean / month | Std | Ann. Sharpe | Why it is interesting |
|---|---|---|---|---|
| **Short-term reversal** | 0.10% | 1.30% | 0.25 | Built from last month's return, so it is a compensation for liquidity provision. Microstructure and liquidity shocks are the most likely source of return autocorrelation. |
| **Momentum** | 0.32% | 3.29% | 0.33 | A large premium with violent crashes. The "factor momentum" literature (Gupta & Kelly 2019; Ehsani & Linnainmaa 2022) claims factor returns are positively autocorrelated, which is exactly what an ARMA model tests. |
| **Value** | 0.36% | 3.43% | 0.36 | Driven by a slow-moving valuation spread, with a long 2017–2020 drawdown. If the expected return μₜ is a persistent AR(1) and returns are μₜ + noise, returns are ARMA(1,1) with φ ≈ 1 and θ ≈ −φ. |
| **Low risk** | 0.12% | 3.91% | 0.10 | Betting against beta and low volatility. The premium is linked to leverage and funding constraints (Frazzini & Pedersen 2014), which are persistent, so predictability is plausible. |

### ACF / PACF and tests (`results/figures/q1_acf_pacf.png`, `results/q1_correlograms.md`)

* Sample ACF: $\hat\rho_k=\sum_{t>k}(x_t-\bar x)(x_{t-k}-\bar x)/\sum_t(x_t-\bar x)^2$.
  The PACF comes from the Durbin–Levinson recursion; it matches statsmodels.
* Individual tests use $z=\hat\rho_k\sqrt T$, N(0,1) under white noise (band ±1.96/√T = ±0.079).
* The joint test is Ljung–Box $Q(m)=T(T+2)\sum_{k\le m}\hat\rho_k^2/(T-k)\sim\chi^2_m$.

| lag | STR ACF | STR PACF | Mom ACF | Mom PACF | Value ACF | Value PACF | Low-risk ACF | Low-risk PACF |
|---|---|---|---|---|---|---|---|---|
| 1 | **−0.114** | **−0.114** | −0.016 | −0.016 | **0.132** | **0.132** | 0.061 | 0.061 |
| 2 | 0.020 | 0.007 | −0.066 | −0.066 | 0.058 | 0.042 | −0.004 | −0.008 |
| 3 | −0.015 | −0.012 | 0.025 | 0.023 | −0.039 | −0.053 | −0.045 | −0.045 |
| 4 | 0.022 | 0.019 | −0.048 | −0.052 | **0.105** | **0.117** | 0.036 | 0.042 |
| 5 | −0.004 | 0.001 | −0.010 | −0.008 | −0.058 | **−0.085** | −0.039 | −0.044 |
| 6 | 0.026 | 0.025 | 0.044 | 0.036 | **0.113** | **0.124** | 0.075 | **0.079** |
| 10 | 0.006 | 0.013 | 0.005 | 0.010 | 0.056 | 0.038 | **0.084** | **0.085** |
| 11 | **0.119** | **0.121** | −0.066 | −0.067 | **0.080** | 0.075 | 0.074 | 0.065 |
| 12 | **−0.124** | **−0.099** | **0.109** | **0.105** | **−0.110** | **−0.148** | **−0.098** | **−0.111** |
| Ljung–Box Q(12), p | 28.8, 0.004 | | 19.0, 0.088 | | 47.9, <0.001 | | 25.6, 0.012 | |

Bold = significant at 5%. Lags 7–9 are omitted here and are in the full tables.

Every theme shows a lag-12 spike, i.e. annual seasonality in factor returns (Heston &
Sadka 2008; Keloharju, Linnainmaa & Nyberg 2016). A low-order ARMA cannot capture it.
The lag choice below therefore uses lags 1–4, applying the rule in `choose_order`.

**Expected model, one sentence per factor:**

* **Short-term reversal → AR(1).** The PACF has a single significant spike at lag 1 (−0.11)
  and nothing at lags 2–10, while the ACF is also only lag 1. This is the signature of an
  AR(1) with negative φ (consistent with reversal).
* **Momentum → ARMA(0,0), white noise.** No ACF or PACF lag from 1 to 11 is significant and
  Ljung–Box(12) does not reject at 5% (p = 0.09). So, perhaps surprisingly given "factor
  momentum", the momentum theme's own past returns should not predict it.
* **Value → AR(4).** The PACF is significant at lags 1 and 4 (and 6), and the ACF does not
  cut off cleanly. That points to an autoregression of order 4.
* **Low risk → ARMA(1,1).** No single low-order lag is significant but Ljung–Box(12)
  rejects (p = 0.012). The dependence is weak and spread out, which a parsimonious ARMA(1,1)
  can approximate (the "persistent expected return + noise" representation).

### Estimation (`results/q1_estimates.md`)

**(a) Regression.**

* AR(p): OLS of $x_t$ on $(1,x_{t-1..t-p})$.
* ARMA(p,q): Hannan–Rissanen. First fit a long AR(m) by OLS, with
  $m=\max(12,\lceil(\ln T)^2\rceil)=41$, to get residuals $\hat\varepsilon_t$; then regress
  $x_t$ on $(1,x_{t-1..t-p},\hat\varepsilon_{t-1..t-q})$.
* $\mu=c/(1-\sum\phi_i)$.

**(b) Exact Gaussian MLE.** For
$x_t-\mu=\sum_{i=1}^p\phi_i(x_{t-i}-\mu)+\varepsilon_t+\sum_{j=1}^q\theta_j\varepsilon_{t-j}$
with $\varepsilon_t\sim N(0,\sigma^2)$, the vector $x$ is
$N(\mu\iota,\sigma^2\Omega(\phi,\theta))$, so the exact log-likelihood is

$$\ell(\mu,\phi,\theta,\sigma^2)=-\tfrac T2\ln(2\pi\sigma^2)-\tfrac12\ln|\Omega|-\tfrac1{2\sigma^2}(x-\mu\iota)'\Omega^{-1}(x-\mu\iota).$$

It is evaluated through the prediction-error decomposition
$\ell=-\tfrac12\sum_t[\ln 2\pi+\ln(\sigma^2F_t)+v_t^2/(\sigma^2F_t)]$. The innovations
$v_t=x_t-E[x_t|x_{1:t-1}]$ and their variances $F_t$ come from the Kalman filter on Harvey's
state-space form, started from the stationary covariance $P_0=TP_0T'+RR'$. That
initialization is what makes the likelihood exact rather than conditional. Details:

* σ² is concentrated out.
* Stationarity and invertibility are imposed through the partial-autocorrelation (tanh)
  transform.
* Standard errors come from the inverse numerical Hessian.
* For AR(1) the closed form
  $\ell=-\tfrac12\ln\frac{2\pi\sigma^2}{1-\phi^2}-\frac{(1-\phi^2)(x_1-\mu)^2}{2\sigma^2}-\frac{T-1}{2}\ln 2\pi\sigma^2-\sum_{t\ge2}\frac{(x_t-\mu-\phi(x_{t-1}-\mu))^2}{2\sigma^2}$
  is checked against the filter on every run.

| Factor | Model | Regression | MLE (se) | log L | BIC |
|---|---|---|---|---|---|
| STR | AR(1) | φ = −0.114 (0.040) | φ = −0.114 (0.040) | 1792.34 | **−3565.4** |
| STR | ARMA(1,1) | φ = −0.329, θ = 0.232 | φ = −0.197 (0.41), θ = 0.085 (0.42) | 1792.36 | −3559.1 |
| Momentum | ARMA(0,0) | μ = 0.32% | μ = 0.32% (0.13%) | 1220.92 | **−2429.0** |
| Momentum | AR(1) | φ = −0.016 (0.040) | φ = −0.016 (0.041) | 1221.00 | −2422.8 |
| Momentum | ARMA(1,1) | φ = 0.067, θ = −0.100 | φ = 0.782 (0.28), θ = −0.816 (0.26) | 1221.82 | −2418.0 |
| Value | AR(1) | φ = 0.132 (0.040) | φ = 0.133 (0.040) | 1201.13 | **−2383.0** |
| Value | ARMA(1,1) | φ = 0.003, θ = 0.151 | φ = 0.353 (0.28), θ = −0.224 (0.30) | 1201.48 | −2377.3 |
| Value | AR(4) | φ = (0.141, 0.041, −0.068, 0.118) | φ = (0.136, 0.043, −0.070, 0.119), se ≈ 0.04 | **1206.82** | −2375.1 |
| Low risk | AR(1) | φ = 0.061 (0.040) | φ = 0.061 (0.040) | 1116.76 | **−2214.3** |
| Low risk | ARMA(1,1) | φ = 0.004, θ = 0.060 | φ = 0.013 (0.40), θ = 0.048 (0.40) | 1116.76 | −2207.9 |

**Regression vs. MLE.**

* For pure AR models the two methods agree to the third decimal. Conditional least squares
  and exact MLE differ only through the first p observations.
* For ARMA(1,1) they differ a lot (momentum: HR gives φ = 0.07, θ = −0.10; MLE gives
  φ = 0.78, θ = −0.82). This is the classic near-cancellation of AR and MA roots.
  When φ ≈ −θ, the ARMA(1,1) is close to white noise and the likelihood is flat along the
  ridge φ = −θ, so the parameters are weakly identified. The huge MLE standard errors
  (≈ 0.3–0.4) show this, and the log-likelihood gain over the simpler model is below 1
  point in every case.
* The ARMA(1,1) MLE values for momentum and value (φ ≈ 0.35–0.8, θ ≈ −φ) are what the
  "small persistent expected-return component" story predicts. The data cannot distinguish
  them from white noise.

### Out-of-sample, 2016–2025 (`results/q1_oos.md`, `results/figures/q1_oos_cumulative.png`)

Every model is re-estimated on 1975–2015. Parameters are held fixed, and one-step-ahead
forecasts are made for 2016-01 to 2025-12 (120 months):

* Kalman filter for the MLE fits;
* the ARMA recursion for the regression fits.

The benchmark is the 1975–2015 mean. $R^2_{OOS}=1-\text{MSE}_{model}/\text{MSE}_{mean}$.
The tests are Diebold–Mariano (DM, HAC) and Clark–West (CW, for nested models, one-sided).

| Factor | Model | R²_OOS (regression) | R²_OOS (MLE) | CW p (MLE) |
|---|---|---|---|---|
| STR | **AR(1)** (chosen) | −1.70% | −1.71% | 0.48 |
| STR | ARMA(1,1) | −1.86% | −2.80% | 0.64 |
| Momentum | **ARMA(0,0)** (chosen) | 0 | 0 | — |
| Momentum | AR(1) | −0.10% | −0.02% | 0.93 |
| Momentum | ARMA(1,1) | +0.19% | −0.63% | 0.68 |
| Value | AR(1) | +2.82% | +2.75% | 0.078 |
| Value | ARMA(1,1) | +2.32% | +3.09% | 0.071 |
| Value | **AR(4)** (chosen) | **+3.75%** | **+3.66%** | 0.080 |
| Low risk | AR(1) | +0.74% | +0.75% | 0.15 |
| Low risk | **ARMA(1,1)** (chosen) | +0.62% | **+0.75%** | 0.15 |

No DM test against the mean or against the AR(1) is significant at 5% (all p > 0.13).

**Does the ACF/PACF choice win out of sample?** Mostly yes, but the margins are tiny and
not statistically significant.

* **Value:** the chosen AR(4) is the best model, with R²_OOS ≈ 3.7%. That is economically
  large for monthly returns, and Clark–West is marginally significant (p ≈ 0.08). Most of
  the gain comes from 2020–2022, the value crash and rebound, when its persistence paid off.
* **Low risk:** the chosen ARMA(1,1) MLE is best, but only by 0.002 pp over the AR(1).
* **Momentum:** the chosen white-noise model is effectively tied for best. Only the HR
  ARMA(1,1) beats it, by 0.19%, and the MLE version of the same model loses.
* **Short-term reversal: the choice fails.** The in-sample lag-1 coefficient
  (φ = −0.11, t = −2.8) does not survive. AR(1) loses 1.7% to the historical mean. Reversal
  profits have shrunk in the recent, more liquid, high-frequency-trading era, and its
  in-sample predictability does not carry over.

The general lesson: monthly factor-return autocorrelations of 0.1 are statistically
detectable in 50 years of data, but they buy at most a few percent of R² and are fragile
out of sample. Parsimony (the mean or an AR(1)) is hard to beat.

### Next month's expected return — Value, AR(4) (`results/q1_next_month.md`)

The preferred model for value is the AR(4): it was chosen from the correlogram, it was the
best out of sample, and it has the highest likelihood. Estimated on 1975–2025 by exact MLE,
it implies

$$E_T[x_{T+1}]=\mu+\sum_{i=1}^4\phi_i(x_{T+1-i}-\mu)=0.36\%+0.136(1.39)+0.043(4.52)-0.070(-4.35)+0.119(-4.22)=0.55\%,$$

where the deviations from the mean (in %) are the value returns of Dec, Nov, Oct and Sep 2025.

So the model expects value to earn **about 0.55% in January 2026**, roughly 0.19 pp above
its long-run mean of 0.36% per month. The deviation is not driven only by last month. The
strong November (+4.9%) and December (+1.8%) returns, plus the positive lag-3 effect of a
weak October, push the forecast up. The weak September works through the lag-4
coefficient (0.12) and partly offsets this.

The revision is small relative to the risk. The one-step predictive standard deviation is
3.4%, so the 95% interval is about [−6.1%, +7.1%], and the predictable component explains
only a few percent of the variance (consistent with R²_OOS ≈ 3.7%). The long-run mean is
itself uncertain: its standard error is 0.18%, and value averaged only 0.17% per month over
2016–2025. In practice the AR(4) supports a modest tilt toward value next month, not a
confident bet. The AR(1) and ARMA(1,1) give 0.55% and 0.64%, so the conclusion does not
depend on the exact specification.

---

## Question 2 — Autoregressive model for WSJ phrase counts

Data: monthly counts for "bubble", "climate change" and "china", 1984-01 to 2017-06
(T = 402, so 401 usable observations). Model: $E_{t-1}[c_{i,t}]=\lambda_{i,t}=\delta_0+\delta_1c_{i,t-1}$.

### (a) OLS

OLS of $c_t$ on $(1,c_{t-1})$: $\hat\delta=(X'X)^{-1}X'c$. Classical and White (HC1)
standard errors are reported. White matters because under the Poisson model
$\text{Var}_{t-1}(c_t)=\lambda_t$, so the errors are heteroskedastic by construction.

### (b) Exact Poisson MLE

The conditional mean is used as the Poisson intensity: $c_t\mid c_{t-1}\sim\text{Poisson}(\lambda_t)$,
$\lambda_t=\delta_0+\delta_1c_{t-1}>0$. The exact likelihood is the product of the marginal
of the first observation and the conditionals:

$$L(\delta)=f(c_1;\delta)\prod_{t=2}^{T}\frac{e^{-\lambda_t}\lambda_t^{c_t}}{c_t!},\qquad
\ell(\delta)=\ln f(c_1;\delta)+\sum_{t=2}^{T}\left[c_t\ln(\delta_0+\delta_1c_{t-1})-(\delta_0+\delta_1c_{t-1})-\ln c_t!\right].$$

The marginal $f(c_1)$ has no closed form for this Poisson autoregression, so the main
estimates condition on $c_1$, which is standard. As a robustness check the script also
includes $c_1\sim\text{Poisson}(\delta_0/(1-\delta_1))$ at the stationary mean
(`results/q2_mle_variants.md`). This changes δ₁ by at most 0.02.

Derivatives used:

* Score: $s_t=(c_t/\lambda_t-1)x_t$.
* Hessian: $H=-\sum_t c_tx_tx_t'/\lambda_t^2$.
* Fisher information: $I=\sum_t x_tx_t'/\lambda_t$.

The likelihood is maximized under λ_t > 0 with L-BFGS-B and polished with Newton steps on
the analytic Hessian; the score at the optimum is about 1e-10. Standard errors come from
$(-H)^{-1}$, plus the QML sandwich $H^{-1}(\sum s_ts_t')H^{-1}$.

| phrase | OLS δ₀ | (se) | (White) | OLS δ₁ | (se) | (White) | MLE δ₀ | (ML se) | (robust) | MLE δ₁ | (ML se) | (robust) | log L | Pearson φ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bubble | 6.797 | 1.158 | 1.027 | 0.749 | 0.033 | 0.047 | 3.739 | 0.274 | 0.559 | 0.863 | 0.013 | 0.037 | −2193.7 | 6.76 |
| climate change | 2.360 | 0.550 | 0.522 | 0.762 | 0.033 | 0.073 | 0.819 | 0.079 | 0.161 | 0.926 | 0.017 | 0.052 | −1610.2 | 6.49 |
| china | 69.04 | 16.26 | 12.70 | 0.888 | 0.023 | 0.025 | 39.04 | 1.64 | 8.41 | 0.938 | 0.003 | 0.021 | −9767.4 | 42.82 |

Pearson φ is the dispersion $\frac1{T-2}\sum(c_t-\hat\lambda_t)^2/\hat\lambda_t$; it equals 1
under the Poisson model. Figures: `results/figures/q2_fitted_means.png` and
`q2_overdispersion.png`.

### (c) Regression vs. MLE

1. **Same long-run mean, different persistence.** The implied unconditional means
   δ₀/(1−δ₁) nearly coincide: bubble 27.1 (OLS) vs. 27.2 (MLE), climate change 9.9 vs. 11.0,
   china 616 vs. 626, against sample means of 26.9, 9.5 and 605. MLE, however, finds much
   more persistence (δ₁ = 0.86 / 0.93 / 0.94 vs. 0.75 / 0.76 / 0.89) with a smaller
   intercept.
2. **Why.** The Poisson first-order condition $\sum_t (c_t-\lambda_t)x_t/\lambda_t=0$ is
   weighted least squares with weights $1/\lambda_t$. The script checks that WLS with the
   MLE weights reproduces the MLE exactly. OLS weights every month equally, so it is
   dominated by high-count months: post-2000 "bubble", post-2007 "climate change", the
   2004–2016 "china" plateau. Those months have the largest variance and contain one-off
   news spikes (2001, 2002, 2009, Copenhagen in Dec. 2009, Aug. 2015) that revert
   immediately, which pulls δ₁ down. MLE down-weights them in proportion to their expected
   variance. It gives more weight to the long low-count stretches, where counts drift slowly,
   so the estimated persistence is higher. If the variance really is proportional to the
   mean, MLE is the efficient weighting. Both estimators are consistent for δ when the
   conditional mean is correctly specified, so the large gap is itself a sign the linear
   AR(1) mean is not exactly right (e.g. regime shifts in topic attention).
3. **Standard errors.** Poisson implies Var = mean, but the Pearson dispersion is 6.5 for
   bubble, 6.5 for climate change and 43 for china, so the counts are strongly overdispersed
   (`q2_overdispersion.png`). The Hessian-based ML standard errors are therefore far too
   small, by roughly √φ. The robust sandwich standard errors are 2× to 6.4× larger. With
   robust standard errors the MLE is still more precise than OLS with White standard
   errors (bubble δ₁: 0.037 vs. 0.047; china: 0.021 vs. 0.025), as the WLS interpretation
   predicts. OLS classical standard errors are also wrong, because they ignore the
   heteroskedasticity.

### Tests of the estimated autoregressive parameter

The restriction $R\delta=r$ with $R=[0\;1]$ is tested two ways, each χ²(1) (5% critical
value 3.84):

* **H₀: δ₁ = 0**, no dynamics. This is the main test.
* **H₀: δ₁ = 1**, a unit-root, random-walk-type mean.

How each test is set up:

* **OLS:** $t=(\hat\delta_1-r)/\text{se}(\hat\delta_1)$ and Wald $=t^2$, with classical or
  White standard errors.
* **ML Wald:** $W=(R\hat\delta-r)'[R\hat V R']^{-1}(R\hat\delta-r)$. It uses only the
  unrestricted MLE, with $\hat V=(-H)^{-1}$ or the sandwich.
* **LR:** $2[\ell(\hat\delta)-\ell(\tilde\delta)]$, where $\tilde\delta$ is the restricted
  MLE. Under δ₁ = 0 it has a closed form, $\tilde\delta_0=\bar c$ (a constant Poisson mean).
  Under δ₁ = 1 it is a 1-D maximization over δ₀.
* **LM (score):** $LM=S(\tilde\delta)'I(\tilde\delta)^{-1}S(\tilde\delta)$, where
  $S=\sum_t(c_t/\tilde\lambda_t-1)x_t$ and $I=\sum_tx_tx_t'/\tilde\lambda_t$. It needs only
  the restricted model. The robust version (Wooldridge) replaces the information-matrix
  equality with the outer product of scores. "LR/φ" is the quasi-LR scaled by the Pearson
  dispersion.

| phrase, H₀ | OLS t | OLS W | OLS W (White) | ML W | ML W (sandwich) | LR | LM | LM (robust) | LR/φ |
|---|---|---|---|---|---|---|---|---|---|
| bubble, δ₁=0 | 22.65 | 513.2 | 253.2 | 4360.6 | 540.6 | 4609.1 | 4184.5 | 118.1 | 682.3 |
| bubble, δ₁=1 | −7.59 | 57.5 | 28.4 | 110.5 | 13.7 | 109.4 | 106.7 | 11.8 | 16.2 |
| climate change, δ₁=0 | 23.09 | 533.1 | 107.8 | 2932.6 | 318.0 | 4563.2 | 4585.3 | 64.4 | 703.3 |
| climate change, δ₁=1 | −7.19 | 51.7 | 10.5 | 19.0 | **2.06** | 18.3 | 17.2 | **1.88** | **2.82** |
| china, δ₁=0 | 38.61 | 1490.7 | 1238.0 | 80225 | 1978.8 | 76931 | 70534 | 196.1 | 1796.4 |
| china, δ₁=1 | −4.87 | 23.7 | 19.7 | 354.8 | 8.75 | 355.1 | 345.0 | 8.00 | 8.29 |

p-values are in `results/q2_tests_with_pvalues.csv`.

**Interpretation.**

* **δ₁ = 0** is rejected overwhelmingly by every test: the counts are strongly persistent.
  The classical ML trio agree closely (bubble W ≈ LR ≈ LM ≈ 4,200–4,600), as asymptotic
  theory says when the model is correct. But the classical statistics are inflated by about
  φ because of overdispersion. The robust Wald, robust LM and dispersion-scaled LR are roughly 8× to
  360× smaller (still huge).
* **δ₁ = 1.** The OLS t-statistic is exactly the Dickey–Fuller τ_μ statistic. Under H₀ the
  series is non-stationary, so the χ² p-values are only indicative and should be compared
  with the DF critical values (−2.87 at 5%, −3.44 at 1%). All three phrases reject a unit
  root by DF. Among the Poisson tests the robust versions are the credible ones.
  "Bubble" and "china" still reject at 1%. **"Climate change" does not** (sandwich
  Wald p = 0.15, robust LM p = 0.17, LR/φ p = 0.09): once overdispersion is accounted for,
  its MLE persistence (0.93) cannot be distinguished from 1. The non-robust Poisson tests
  would have reached the opposite, wrong conclusion. This is the practical lesson: Wald,
  LR and LM are asymptotically equivalent only under correct specification, and here the
  Poisson variance assumption fails.

---

## Question 3 — GMM for the Poisson parameter

### Formulation

For each phrase, with $c_t$ iid Poisson(λ) under H₀, mean and variance both equal λ:

$$g_t(\lambda)=\begin{pmatrix}c_t-\lambda\\(c_t-\lambda)^2-\lambda\end{pmatrix},\qquad E[g_t(\lambda_0)]=0,\qquad
\hat\lambda=\arg\min_\lambda\;\bar g(\lambda)'\,W\,\bar g(\lambda),\quad \bar g=\tfrac1T\textstyle\sum_t g_t.$$

That is 2 moments for 1 parameter, so there is 1 overidentifying restriction.

* **Step 1:** W = I.
* **Step 2:** $W=\hat S^{-1}$, where $\hat S$ is the long-run covariance of $g_t$ at the
  step-1 estimate.
* **Iterated:** repeat step 2 until λ converges.
* **CUE:** S(λ) is re-estimated inside the objective at every λ.

$\hat S$ is computed two ways, with centered moments in both:

* **iid:** $\frac1T\sum g_tg_t'$;
* **HAC:** Newey–West with 5 lags (the $4(T/100)^{2/9}$ rule), because the counts are
  strongly autocorrelated (Question 2).

Asymptotics: $\sqrt T(\hat\lambda-\lambda)\to N(0,(G'S^{-1}G)^{-1})$ with
$G=E[\partial g_t/\partial\lambda]=(-1,\,-1-2(\bar c-\lambda))'$; the general sandwich is
used for W = I. Hansen's $J=T\bar g'\hat S^{-1}\bar g\to\chi^2_1$.

The comparison is the iid Poisson MLE: $\hat\lambda_{ML}=\bar c$ with $\text{se}=\sqrt{\bar c/T}$.

### Results (`results/q3_estimates.md`, `results/q3_jtests.md`, `results/figures/q3_gmm_objective.png`)

| | bubble | climate change | china |
|---|---|---|---|
| sample mean = **MLE** (se; HAC se) | **26.91** (0.26; 2.40) | **9.54** (0.15; 1.46) | **604.6** (1.23; 42.0) |
| sample variance / mean | 18.6 | 20.0 | 223 |
| GMM step 1 (W = I) | 27.41 (2.45) | 10.04 (1.55) | 605.1 (42.1) |
| GMM two-step, HAC | **21.70** (2.01) | **4.51** (0.83) | **569.4** (41.1) |
| GMM iterated, HAC | 17.90 (1.87) | 4.51 (0.83) | 477.5 (39.9) |
| GMM CUE, HAC | 15.07 (1.87) | 4.06 (0.83) | 328.8 (39.9) |
| GMM two-step, iid S | 21.69 (0.89) | 5.78 (0.44) | 556.9 (17.5) |
| J (two-step HAC), p | 53.7, 2e-13 | 17.4, 3e-5 | 124.9, 5e-29 |
| J (two-step iid S), p | 106.7, 5e-25 | 51.4, 8e-13 | 343.7, 1e-76 |

### GMM vs. MLE

* **The MLE uses only the mean.** The iid-Poisson score is $\sum(c_t/\lambda-1)$, so
  $\hat\lambda_{ML}=\bar c$ regardless of the variance. GMM also forces λ to fit the
  variance.
* **The variance moment cannot be satisfied.** $\bar g_2(\lambda)=s^2+(\bar c-\lambda)^2-\lambda$
  has a root only if $s^2\le\bar c+\tfrac14$. Here $s^2/\bar c$ is 19 to 223, so the
  just-identified "variance-only" estimator does not exist, and $\bar g_2>0$ at every λ.
* **Identity weighting.** The objective is dominated by $\bar g_2^2$, which is minimized
  where $\partial\bar g_2/\partial\lambda=0$, i.e. at $\lambda=\bar c+\tfrac12$. That is
  exactly the step-1 result (27.41 = 26.91 + 0.5, and so on).
* **Efficient weighting pulls λ below the mean.** Optimal GMM combines the moments using
  their covariance. Counts are right-skewed, so $g_1$ and $g_2$ are positively correlated
  (third central moment > 0). Roughly, $\hat\lambda\approx\bar c-(\hat S_{12}/\hat S_{22})\,\bar g_2$.
  The large positive violation of the variance moment is read as "the mean was drawn
  high", and λ is pulled down: 21.7 vs. 26.9, 4.5 vs. 9.5, 569 vs. 605.
* **Iterated and CUE estimates move further.** $\hat S(\lambda)$ grows as λ falls, so these
  estimators can shrink J by inflating $\hat S$. This is the familiar instability of
  efficient GMM under misspecification.
* **Misspecification breaks the equivalence.** Under correct specification every one of
  these estimators is consistent for the same λ and asymptotically equivalent; the MLE is
  efficient. Here they converge to different pseudo-true values. Their disagreement is
  itself a Hausman-type signal that the Poisson model is wrong.
* **Standard errors.** The ML standard error √(c̄/T) assumes iid Poisson draws. It is about
  10× smaller than the HAC standard error of the mean (0.26 vs. 2.40 for bubble), because
  the counts are both overdispersed and autocorrelated. The GMM HAC standard errors are of
  the same order as the HAC standard error of the mean.
* **A side result.** The efficient-GMM standard error is the same at every λ (e.g. iterated
  and CUE agree to all digits). With centered moments,
  $g_t-\bar g=A(\lambda)u_t$ and $G=A(\lambda)G_0$, where $u_t=(c_t-\bar c,\;c_t^2-\overline{c^2})'$
  and $A=\begin{pmatrix}1&0\\-2\lambda&1\end{pmatrix}$. Hence $G'S^{-1}G=G_0'S_u^{-1}G_0$
  does not depend on λ.

### Test of the overidentifying restriction

Under H₀ (both Poisson moments hold), $J\sim\chi^2_1$ (5% critical value 3.84). J ranges
from 17 to 344 across phrases, weighting matrices and estimators; every p-value is below
4e-5. **The Poisson restriction that variance equals mean is rejected decisively for all
three phrases.**

The HAC J-statistics are smaller than the iid ones because they allow for serial
correlation in the moments, but they still reject. This matches Question 2. The conditional
mean of the counts moves over time (δ₁ ≈ 0.9), and an unconditional Poisson ignores that,
so the unconditional variance includes Var(λ_t) and becomes overdispersed. Even conditional
on $c_{t-1}$ the counts are overdispersed (Pearson φ = 6.5 to 43). A negative binomial or a
Poisson autoregression with a quasi-likelihood variance $\phi\lambda_t$ would be more
appropriate, and the conclusions on means should rest on robust or HAC inference.
