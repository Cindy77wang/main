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

`q1_arma.py` needs the JKP data in `inputs/jkp/` first (see Question 1).

---

## Question 1 — ARMA models for JKP factors

> **Status.** The code is complete and tested, but **the JKP results are not in this file yet.**
> The network policy of the environment where this was written blocks `jkpfactors.com`, so
> the data could not be downloaded. To run it, download the USA monthly factor returns from
> <https://jkpfactors.com/data> (e.g. *all factors, monthly, capped value-weighted*) and save
> the CSV in `inputs/jkp/`. Then run `python q1_arma.py`. The script reads the long JKP layout
> (`location, name, freq, weighting, direction, n_stocks, date, ret`) or a wide file. It
> writes all the tables and figures referenced below.
>
> The pipeline was checked on simulated ARMA data in the JKP layout (not saved). The
> correlogram matches statsmodels' Durbin–Levinson PACF. The exact MLE log-likelihoods and
> parameters match statsmodels' exact state-space ARIMA to about 1e-5. The Kalman-filter
> likelihood equals the closed-form AR(1) exact likelihood, and the script asserts this on
> every run. The correlogram rule recovered the simulated orders (AR(1), MA(1), AR(2)).

### Factor choice (JKP names in brackets)

1. **Short-term reversal** (`ret_1_0`). It is built from last month's return, so it is the
   factor most directly linked to liquidity provision and microstructure. It is a natural
   place to look for return autocorrelation.
2. **Momentum** (`ret_12_1`). It has a large premium and rare but violent crashes. "Factor
   momentum" (Gupta & Kelly 2019; Ehsani & Linnainmaa 2022) is the claim that factor returns
   themselves are positively autocorrelated, which is exactly what an ARMA model measures.
3. **Value** (`be_me`). It has a slow-moving valuation spread and a long drawdown from 2017
   to 2020. If the expected return μ_t is a persistent AR(1) and realized returns are
   μ_t + noise, returns follow an **ARMA(1,1) with φ close to 1 and θ close to −φ**. That
   produces small but long-lived autocorrelations, which an AR(1) misses.
4. **Betting against beta** (`betabab_1260d`). Its premium is tied to funding liquidity and
   leverage constraints (Frazzini & Pedersen 2014). Funding conditions are persistent, so
   predictability is plausible.

The factor list is the `FACTORS` dictionary at the top of `q1_arma.py` and can be swapped
for any four JKP factor names.

### ACF / PACF and tests

* Sample ACF: $\hat\rho_k=\sum_{t>k}(x_t-\bar x)(x_{t-k}-\bar x)/\sum_t(x_t-\bar x)^2$.
  The PACF comes from the Durbin–Levinson recursion.
* Individual tests use $z=\hat\rho_k\sqrt T$ (and the same for the PACF), which is N(0,1)
  under white noise. Bartlett standard errors are also reported. The joint test is
  Ljung–Box $Q(m)=T(T+2)\sum_{k\le m}\hat\rho_k^2/(T-k)\sim\chi^2_m$ for m = 1, …, 12.
* Output: `results/figures/q1_acf_pacf.png` and `results/q1_correlograms.md`.
* Model expected from the correlogram (one sentence per factor). The script writes these to
  the end of `results/q1_correlograms.md` using this rule: a PACF that cuts off at lag p
  while the ACF decays suggests AR(p); an ACF that cuts off at lag q suggests MA(q); joint
  significance without a clean cut-off suggests ARMA(1,1). To override the rule after
  looking at the plots, edit `CHOICE`.

### Estimation

**(a) Regression.** AR(p): OLS of $x_t$ on $(1,x_{t-1},\dots,x_{t-p})$. Models with an MA
part use Hannan–Rissanen:

1. Fit a long AR(m) by OLS, with $m=\max(12,\lceil(\ln T)^2\rceil)$, to get residuals
   $\hat\varepsilon_t$.
2. Regress $x_t$ on $(1,x_{t-1..t-p},\hat\varepsilon_{t-1..t-q})$.

$\mu=c/(1-\sum\phi_i)$.

**(b) Exact Gaussian MLE.** For
$x_t-\mu=\sum_{i=1}^p\phi_i(x_{t-i}-\mu)+\varepsilon_t+\sum_{j=1}^q\theta_j\varepsilon_{t-j}$
with $\varepsilon_t\sim N(0,\sigma^2)$, the vector $x=(x_1,\dots,x_T)'$ is
$N(\mu\iota,\sigma^2\Omega(\phi,\theta))$, so the exact log-likelihood is

$$\ell(\mu,\phi,\theta,\sigma^2)=-\tfrac T2\ln(2\pi\sigma^2)-\tfrac12\ln|\Omega|-\tfrac1{2\sigma^2}(x-\mu\iota)'\Omega^{-1}(x-\mu\iota).$$

It is evaluated without inverting Ω through the prediction-error decomposition
$\ell=-\tfrac12\sum_t[\ln 2\pi+\ln(\sigma^2F_t)+v_t^2/(\sigma^2F_t)]$. Here the innovations
$v_t=x_t-E[x_t|x_{1:t-1}]$ and their variances $F_t$ come from the Kalman filter on Harvey's
state-space form. The filter starts from the stationary state covariance
$P_0=TP_0T'+RR'$, which is what makes the likelihood exact rather than conditional. Details:

* σ² is concentrated out.
* Stationarity and invertibility are imposed by mapping unconstrained parameters to partial
  autocorrelations, tanh(u) ∈ (−1, 1), and then to AR/MA coefficients.
* Standard errors come from the inverse numerical Hessian of the exact log-likelihood in
  natural parameters.
* For AR(1) the closed form
  $\ell=-\tfrac12\ln\frac{2\pi\sigma^2}{1-\phi^2}-\frac{(1-\phi^2)(x_1-\mu)^2}{2\sigma^2}-\frac{T-1}{2}\ln 2\pi\sigma^2-\sum_{t\ge2}\frac{(x_t-\mu-\phi(x_{t-1}-\mu))^2}{2\sigma^2}$
  is used as a check.

Output: `results/q1_estimates.md` (estimates, standard errors, log L and BIC for AR(1),
ARMA(1,1) and the chosen model, by both methods).

### Out-of-sample, 2016–2025

All models are re-estimated on 1975–2015. Parameters are then held fixed and one-step-ahead
forecasts are made for every month from 2016 to 2025:

* Kalman filter for the MLE fits.
* The ARMA recursion for the regression fits.

The benchmark is the training-sample mean. The script reports MSE, RMSE, MAE and
$R^2_{OOS}=1-\text{MSE}_{model}/\text{MSE}_{mean}$, plus three tests:

* Diebold–Mariano (HAC) against the mean;
* Clark–West against the mean, since the mean model is nested;
* Diebold–Mariano against the AR(1).

Output: `results/q1_oos.md` (column `chosen` flags the correlogram-based model) and
`results/figures/q1_oos_cumulative.png` (cumulative SSE gain over the mean).

Whether the ACF/PACF-selected model wins out of sample depends on the downloaded data. Read
it off `q1_oos.md`. Typical outcome for monthly factor returns: autocorrelations are small,
so R²_OOS is within ±2%. Extra lags tend to over-fit, and the parsimonious AR(1) or the mean
is hard to beat. In-sample significance of a lag, such as one Ljung–Box rejection, is
therefore weak evidence of out-of-sample forecastability.

### Next month's expected return

`results/q1_next_month.md` reports, for every model and factor:

* the unconditional mean μ;
* the last observed return;
* the forecast $E[x_{T+1}|x_{1:T}]$ with a 95% predictive interval
  ($\pm1.96\sqrt{\sigma^2F_{T+1}}$).

**Paragraph template** (fill in the numbers from that file for the factor you prefer). For
an AR(1), $E_T[x_{T+1}]=\mu+\phi(x_T-\mu)$. Next month's expected return is the long-run mean
μ plus a fraction φ of this month's deviation from it. With φ around 0.1, a 5% return this
month raises next month's expectation by only about 0.5 percentage points. The predictive
standard deviation is roughly σ, so the 95% interval is many times wider than the forecast
revision. The forecast is statistically meaningful but economically modest: it tilts
exposure at the margin rather than timing the factor. For an ARMA(1,1),
$E_T[x_{T+1}]=\mu+\phi(x_T-\mu)+\theta\hat\varepsilon_T$. With φ ≈ 1 and θ ≈ −φ this is a
slowly updated weighted average of past returns, i.e. a filtered estimate of the persistent
expected return μ_T.

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
