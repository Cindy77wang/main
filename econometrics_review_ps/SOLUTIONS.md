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

**Data.** Four individual JKP factors from <https://jkpfactors.com/data>: USA, monthly,
capped value-weighted long-short excess returns, one file per factor in `inputs/jkp/`.
The sample is 1975-01 to 2025-12 (T = 612); returns are in decimals per month. Sample
statistics: `results/q1_summary_stats.md`.

### Factor choice

| JKP factor | Mean / month | Std | Ann. Sharpe | Why it is interesting |
|---|---|---|---|---|
| **`ret_1_0`**: short-term reversal (Jegadeesh 1990) | 0.29% | 3.54% | 0.28 | Sorts stocks on last month's return, so it pays for providing liquidity. Microstructure and liquidity shocks are the most likely source of autocorrelation in a factor's own returns. |
| **`ret_12_1`**: momentum, return from t−12 to t−1 (Jegadeesh & Titman 1993) | 0.52% | 4.35% | 0.41 | The largest premium of the four, with violent crashes. The "factor momentum" literature (Gupta & Kelly 2019; Ehsani & Linnainmaa 2022) claims factor returns are positively autocorrelated, which is exactly what an ARMA model tests. |
| **`be_me`**: book-to-market, i.e. value (Fama & French 1992) | 0.24% | 3.79% | 0.22 | Driven by a slow-moving valuation spread, with a long 2017–2020 drawdown. If the expected return μₜ is a persistent AR(1) and returns are μₜ + noise, returns are ARMA(1,1) with small, long-lived autocorrelations. |
| **`betabab_1260d`**: Frazzini–Pedersen beta, i.e. betting against beta (2014) | 0.03% | 4.81% | 0.02 | Its premium is tied to leverage and funding constraints, which are persistent, so predictability is plausible. Its value-weighted premium since 1975 is close to zero. |

### ACF / PACF and tests (`results/figures/q1_acf_pacf.png`, `results/q1_correlograms.md`)

* Sample ACF: $\hat\rho_k=\sum_{t>k}(x_t-\bar x)(x_{t-k}-\bar x)/\sum_t(x_t-\bar x)^2$.
  The PACF comes from the Durbin–Levinson recursion; it matches statsmodels.
* Individual tests use $z=\hat\rho_k\sqrt T$, N(0,1) under white noise (band ±1.96/√T = ±0.079).
* The joint test is Ljung–Box $Q(m)=T(T+2)\sum_{k\le m}\hat\rho_k^2/(T-k)\sim\chi^2_m$.

| lag | STR ACF | STR PACF | Mom ACF | Mom PACF | Value ACF | Value PACF | BAB ACF | BAB PACF |
|---|---|---|---|---|---|---|---|---|
| 1 | −0.075 | −0.075 | 0.015 | 0.015 | **0.115** | **0.115** | 0.058 | 0.058 |
| 2 | **−0.079** | **−0.086** | −0.012 | −0.012 | **0.083** | 0.071 | −0.040 | −0.044 |
| 3 | 0.003 | −0.010 | −0.033 | −0.032 | −0.022 | −0.040 | 0.012 | 0.017 |
| 4 | −0.017 | −0.025 | 0.020 | 0.021 | 0.078 | **0.081** | −0.010 | −0.014 |
| 5 | −0.010 | −0.014 | −0.033 | −0.034 | **−0.096** | **−0.112** | −0.021 | −0.018 |
| 6 | 0.050 | 0.045 | **0.084** | **0.084** | 0.062 | 0.076 | 0.038 | 0.039 |
| 10 | **0.084** | **0.080** | −0.077 | **−0.087** | 0.053 | 0.025 | 0.079 | **0.080** |
| 11 | 0.063 | 0.078 | −0.058 | −0.048 | 0.056 | 0.067 | **0.081** | 0.073 |
| 12 | **−0.088** | −0.069 | 0.069 | 0.060 | −0.016 | −0.051 | −0.020 | −0.026 |
| Ljung–Box Q(12), p | 22.5, 0.032 | | 17.1, 0.147 | | 31.1, 0.002 | | 13.1, 0.364 | |

Bold = significant at 5%. Lags 7–9 are omitted here and are in the full tables.

A few isolated spikes at long lags (6, 10, 11, 12) are expected by chance across 12 lags
and 4 series, and a low-order ARMA would not model them. The order choice therefore uses
lags 1–4 (rule in `choose_order`).

**Expected model, one sentence per factor:**

* **Short-term reversal → AR(2).** The PACF is negative at lags 1–2 (−0.075, −0.086, with
  lag 2 significant) and then drops to zero, and Ljung–Box rejects (p = 0.03). That
  cut-off after lag 2 is the AR(2) signature, and negative coefficients fit reversal.
* **Momentum → ARMA(0,0), white noise.** No ACF or PACF lag from 1 to 5 is significant and
  Ljung–Box(12) does not reject (p = 0.15). So, despite "factor momentum", the momentum
  factor's own past returns should not predict it.
* **Value → MA(2).** The ACF is significant at lags 1–2 (0.115, 0.083) and insignificant at
  lag 3, while the PACF tails off (significant again at lags 4–5). An ACF that cuts off
  after lag 2 is the MA(2) signature; Ljung–Box rejects strongly (p = 0.002).
* **Betting against beta → ARMA(0,0), white noise.** No lag from 1 to 9 is significant and
  Ljung–Box(12) is far from rejecting (p = 0.36).

### Estimation (`results/q1_estimates.md`)

**(a) Regression.**

* AR(p): OLS of $x_t$ on $(1,x_{t-1..t-p})$.
* ARMA(p,q) and MA(q): Hannan–Rissanen. First fit a long AR(m) by OLS, with
  $m=\max(12,\lceil(\ln T)^2\rceil)=41$, to get residuals $\hat\varepsilon_t$; then regress
  $x_t$ on $(1,x_{t-1..t-p},\hat\varepsilon_{t-1..t-q})$.
* $\mu=c/(1-\sum\phi_i)$. For ARMA(0,0) the regression is on a constant only, so
  $\hat\mu=\bar x$.

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
* For ARMA(0,0) the exact MLE is $\hat\mu=\bar x$ and $\hat\sigma^2=\frac1T\sum(x_t-\bar x)^2$.

| Factor | Model | Regression (OLS se) | MLE (se) | log L | BIC |
|---|---|---|---|---|---|
| STR | AR(1) | φ = −0.075 (0.040) | φ = −0.077 (0.041) | 1177.85 | **−2336.4** |
| STR | ARMA(1,1) | φ = −0.050, θ = −0.033 | φ = 0.488 (0.21), θ = −0.586 (0.19) | 1179.79 | −2333.9 |
| STR | **AR(2)** | φ = (−0.089, −0.086), se 0.040 | φ = (−0.084, −0.088), se 0.041 | **1180.18** | −2334.7 |
| Momentum | **ARMA(0,0)** | μ = 0.52% | μ = 0.52% (0.18%) | 1050.72 | **−2088.6** |
| Momentum | AR(1) | φ = 0.016 (0.040) | φ = 0.016 (0.041) | 1050.80 | −2082.3 |
| Momentum | ARMA(1,1) | φ = 0.074, θ = −0.069 | φ = −0.910 (0.16), θ = 0.895 (0.17) | 1051.12 | −2076.6 |
| Value | AR(1) | φ = 0.115 (0.040) | φ = 0.118 (0.041) | 1139.93 | **−2260.6** |
| Value | ARMA(1,1) | φ = −0.051, θ = 0.185 | φ = 0.484 (0.23), θ = −0.368 (0.24) | 1140.93 | −2256.2 |
| Value | **MA(2)** | θ = (0.134, 0.064), se 0.044 | θ = (0.115, 0.079), se (0.042, 0.037) | **1141.53** | −2257.4 |
| BAB | **ARMA(0,0)** | μ = 0.03% | μ = 0.03% (0.19%) | 989.11 | **−1965.4** |
| BAB | AR(1) | φ = 0.058 (0.041) | φ = 0.058 (0.040) | 990.14 | −1961.0 |
| BAB | ARMA(1,1) | φ = −0.067, θ = 0.130 | φ = −0.405 (0.31), θ = 0.472 (0.30) | 990.71 | −1955.8 |

The chosen model is in bold in the Model column.

**Regression vs. MLE.**

* For the pure AR models and ARMA(0,0) the two methods agree almost exactly. Conditional
  least squares and exact MLE differ only through the first p observations.
* The two MA(2) fits for value are close: (0.134, 0.064) by regression vs. (0.115, 0.079)
  by MLE.
* For ARMA(1,1) the methods diverge sharply (momentum: HR gives φ = 0.07, θ = −0.07; MLE
  gives φ = −0.91, θ = 0.90). This is near-cancellation of the AR and MA roots. When
  φ ≈ −θ the ARMA(1,1) is almost white noise and the likelihood is flat along the ridge
  φ = −θ, so the parameters are weakly identified. The large standard errors and the
  log-likelihood gains of about 1 point over simpler models show this. BIC prefers the
  most parsimonious model for every factor.

### Out-of-sample, 2016–2025 (`results/q1_oos.md`, `results/figures/q1_oos_cumulative.png`)

Every model is re-estimated on 1975–2015. Parameters are held fixed, and one-step-ahead
forecasts are made for 2016-01 to 2025-12 (120 months):

* Kalman filter for the MLE fits;
* the ARMA recursion for the regression fits.

The benchmark is the 1975–2015 mean. $R^2_{OOS}=1-\text{MSE}_{model}/\text{MSE}_{mean}$.
The tests are Diebold–Mariano (DM, HAC) and Clark–West (CW, for nested models, one-sided).

| Factor | Model | R²_OOS (regression) | R²_OOS (MLE) | CW p (MLE) |
|---|---|---|---|---|
| STR | AR(1) | +1.48% | +1.27% | 0.19 |
| STR | ARMA(1,1) | +1.88% | −0.40% | 0.37 |
| STR | **AR(2)** (chosen) | **+1.96%** | **+1.58%** | 0.19 |
| Momentum | **ARMA(0,0)** (chosen) | **0** | **0** | — |
| Momentum | AR(1) | −0.76% | −0.57% | 0.77 |
| Momentum | ARMA(1,1) | −0.63% | −0.13% | 0.47 |
| Value | AR(1) | **+0.45%** | +0.27% | 0.18 |
| Value | ARMA(1,1) | −0.52% | **+0.96%** | 0.12 |
| Value | **MA(2)** (chosen) | +0.40% | +0.63% | 0.16 |
| BAB | **ARMA(0,0)** (chosen) | **0** | **0** | — |
| BAB | AR(1) | −0.04% | −0.03% | 0.36 |
| BAB | ARMA(1,1) | −0.19% | −0.09% | 0.35 |

For the white-noise models the forecast is the training mean itself, so R²_OOS = 0 by
construction and there is nothing to test. No DM test against the mean is significant
(all p > 0.19).

**Does the ACF/PACF choice win out of sample?** Yes, for three of the four factors, but
none of the gains is statistically significant.

* **Short-term reversal:** the chosen AR(2) is the best model under both regression
  (R²_OOS = 2.0%) and MLE (1.6%). Most of the gain comes from 2020, the COVID crash and
  rebound, when the factor's negative autocorrelation paid off. The Clark–West p-value is
  0.16–0.19, so it is not significant.
* **Momentum and BAB:** the chosen white-noise model wins. Every AR or ARMA specification
  has R²_OOS < 0. Fitting the in-sample noise in the autocorrelations only adds estimation
  error.
* **Value:** the chosen MA(2) is not best. ARMA(1,1) MLE does slightly better (0.96% vs.
  0.63%), and all value models are within about 1% of the mean. The ranking among them is
  noise.

The general lesson: monthly factor-return autocorrelations of 0.05–0.1 are detectable in
50 years of data, but they buy at most about 2% of out-of-sample R², and the more
parsimonious model usually wins. The ACF/PACF-based choices worked mainly because they were
parsimonious, including white noise where nothing was significant.

### Next month's expected return — Short-term reversal, AR(2) (`results/q1_next_month.md`)

The preferred model for short-term reversal is the AR(2): it was chosen from the
correlogram and it was the best model out of sample. Estimated on 1975–2025 by exact MLE,
it implies

$$E_T[x_{T+1}]=\mu+\phi_1(x_T-\mu)+\phi_2(x_{T-1}-\mu)=0.285\%-0.084(0.073-0.285)-0.088(2.457-0.285)=0.11\%,$$

where $x_T$ and $x_{T-1}$ are the factor's December and November 2025 returns (in %).

So the model expects the short-term reversal factor to earn **about 0.11% in January 2026**,
roughly 0.17 pp below its long-run mean of 0.29% per month. The cut comes almost entirely
from the lag-2 term. November 2025 was a good month (+2.5%), and with φ₂ = −0.09 the model
expects about a tenth of that outperformance to reverse two months later. December's
near-zero return (0.07%) contributes almost nothing.

The revision is tiny relative to the risk. The one-step predictive standard deviation is
3.5%, so the 95% interval is about [−6.8%, +7.0%], and the predictable component explains
roughly 1–2% of the variance (in line with R²_OOS ≈ 1.6–2.0%). The long-run mean is also
uncertain: its standard error is 0.12%, and the factor averaged −0.04% per month over
2016–2025. In practice the forecast supports a slightly smaller-than-usual position in the
reversal factor next month, not a directional bet. The simpler AR(1) gives 0.30%, about
the unconditional mean, so the conclusion depends on including the second lag.

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
