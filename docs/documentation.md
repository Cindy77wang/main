---
title: PRISM: A Precision-Rotated, Integrated SDF Machine
subtitle: A machine-learning tangency portfolio of US stocks for the JKP Common Task Framework
author: Cindy Wang
course: MGT 924 Statistical Foundations, Yale SOM
max_pages: 5
---

::: abstract
PRISM is a monthly long-short portfolio of liquid US stocks. It is built to maximize the out-of-sample Sharpe ratio on the JKP Common Task Framework (CTF) data. It estimates the conditional tangency portfolio $w_t \propto \Sigma_t^{-1}\mu_t$ in two complementary ways. The first predicts and then optimizes: four machine-learning forecasts of $\mu_t$ are combined with a Barra-style risk model $\Sigma_t$. The second optimizes directly: a random-feature stochastic discount factor (SDF) learns the Sharpe-maximizing combination of thousands of characteristic-managed portfolios. A non-negative meta-portfolio weights the resulting six sleeves by their past out-of-sample returns, and a volatility-timing overlay sets the leverage. Over the {{stat:months}}-month test period, PRISM earned a Sharpe ratio of **{{stat:sharpe}}** (95% CI {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}). Every estimate uses only information available at the portfolio date.
:::

[[keyfacts stats="sharpe:Sharpe ratio,mean:Return p.a.,sd:Volatility p.a.,max_dd_scaled:Max DD at 10% vol,capm_ew.alpha_t:CAPM alpha t-stat"]]

[[figure:cumret width=6.5in height=2.3in caption="Cumulative excess return (log scale) of PRISM and its sleeves, and the equal-weighted market scaled to the same volatility, over the test period."]]

# Problem and design

The CTF scores the annualized Sharpe ratio of monthly returns $\sum_i w_{i,t}\,r_{i,t+1}$ from December 1989 to November 2023. Leverage and short sales are free and trading costs are ignored. The ex-ante optimal answer is therefore the conditional tangency portfolio, and the statistical problem is to estimate $\mu_t$ and $\Sigma_t$ well enough that $\Sigma_t^{-1}\mu_t$ survives out of sample. PRISM rests on three principles:

- **Rotate by risk.** Signals are turned into portfolios through an explicit covariance model rather than rank-sorted. Two models with identical forecasts can differ greatly in Sharpe ratio, depending on how the forecasts are mapped into weights.
- **Use complexity with shrinkage.** Rich nonlinear models are controlled by cross-validated ridge penalties [Kelly, Malamud and Zhou 2024; Didisheim, Ke, Kelly and Malamud 2024].
- **Diversify across model families and time the leverage.** The sleeves are only moderately correlated (average pairwise correlation {{stat:sleeve_corr_avg}}). Volatility is predictable while expected returns barely move with it, so constant-volatility leverage raises the Sharpe ratio [Moreira and Muir 2017].

**Data and validation protocol.** The model uses only the CTF files: {{stat:n_features}} characteristics and next-month excess returns of about 2,000 stocks per month, plus daily returns. Each month, every characteristic is ranked across stocks onto $[-0.5, 0.5]$, and missing values are set to the median, zero. Nothing is selected or referenced by name. Every component is refit each December on a fixed calendar, using only labels whose return month has ended and daily returns dated on or before the portfolio date. Refitting starts in the 1950s as soon as five years of data exist, so **1957–1989 is a genuine pseudo-out-of-sample validation period**. All data-driven tuning happens inside each training window: validation blocks, cross-validation and past out-of-sample returns. The remaining constants were fixed in advance from the cited literature. The whole design was developed and debugged on simulated data in the CTF format, and none of it was tuned on the test period. The code passes replicas of the CTF's tests: determinism, invariance to input order and dtypes, and truncation of later data, where earlier weights stay bit-identical. It also passes a stricter test that replaces the last month's future returns with noise.

# Model

**Risk model.** As in MSCI Barra USE4 [Menchero, Orr and Wang 2011], returns follow a factor model with observable exposures:
$$\Sigma_t = X_t\,\Omega_t\,X_t' + D_t .$$
The exposures $X_t$ are the market, Fama-French 12 industry dummies and up to 13 characteristic *themes* [Jensen, Kelly and Pedersen 2023]. The themes are rebuilt every December by average-linkage hierarchical clustering of the characteristics, using the distance $1-|\bar\rho|$ on their five-year average cross-sectional correlation. Only characteristics available in most of the window are clustered. Daily ridge cross-sectional regressions give factor returns and residuals. $\Omega_t$ is an exponentially weighted covariance (half-lives: 504 days for correlations, 84 for variances), and $D_t$ is an EWMA of squared residuals (half-life 84 days). Using a few clustered themes instead of all raw characteristics keeps $\Omega_t$ well conditioned.

**Return learners.** Four learners map the ranked characteristics into the next-month return, standardized and winsorized within each month. Each uses a rolling 20-year window whose last 3 years form a time-ordered validation block. *Ridge* picks its penalty by validation information coefficient (IC). *XGBoost* picks its tree depth (3 or 6) and number of trees by early stopping. The *MLP* is an ensemble of three early-stopped NN3 networks [Gu, Kelly and Xiu 2020]. The *LSTM* reads each stock's 12-month trajectory of 12 characteristic principal components, so it can learn from *changes* in characteristics.

**Sleeves.** Each learner $k$ gives a Markowitz sleeve $w^{B_k}_t \propto \Sigma_t^{-1}\hat\mu_{k,t}$. Two further sleeves follow the large-factor-model SDF of Didisheim et al. (2024). Each stock's signal vector stacks its ranked characteristics $x_{i,t}$, $P$ = 3,000 random Fourier feature pairs and a constant:
$$s_{i,t} = \left[x_{i,t},\ \sin(\gamma_p\,\omega_p' x_{i,t}),\ \cos(\gamma_p\,\omega_p' x_{i,t}),\ 1\right]_{p=1}^{P}, \qquad \omega_p \sim N(0, I).$$
The bandwidths $\gamma_p$ cycle through 0.5, 1, 2 and 3. Every signal defines a managed portfolio scaled to unit ex-ante volatility. In sleeve A0 the portfolio is the de-meaned signal itself. In A1 it is rotated by $\Sigma_t^{-1}$: if expected returns are spanned by the signals, the efficient portfolio $\Sigma_t^{-1}S_t\lambda$ lies exactly in the span of the rotated signals. Given past managed-portfolio returns $F$ (360 months), the SDF loadings solve a ridge-regularized Markowitz problem, computed cheaply in its dual $T\times T$ form:
$$\hat b_t = \arg\min_b \sum_{\tau<t}\left(1-b'F_{\tau}\right)^2 + \zeta \sum_p b_p^2 = F'\left(FF'+\zeta I\right)^{-1}\mathbf{1}.$$
The shrinkage $\zeta$ maximizes the pooled Sharpe ratio across five blocked cross-validation folds and is re-chosen every December.

**Meta-combination and volatility timing.** The sleeves, each at unit ex-ante volatility, get non-negative weights $\theta_t = \arg\min_{\theta \geq 0} \sum_{\tau<t}(1-\theta' R_\tau)^2$, where $R_\tau$ holds the sleeves' realized out-of-sample returns over the past 20 years. This is the long-only, Sharpe-maximizing combination, and it is shrunk 30% toward equal weights to limit estimation error [DeMiguel, Garlappi and Uppal 2009]. Finally, the book is scaled to a 10% annual volatility target. The volatility forecast applies today's weights to the past 252 days of returns and takes an EWMA ($\lambda$ = 0.97, the RiskMetrics monthly decay) of the resulting synthetic book returns.

# Results

[[table:perf layout=wide caption="Out-of-sample performance over the test period. Statistics follow the CTF organizers' definitions; the 95% interval for the Sharpe ratio uses the i.i.d. standard error."]]

[[table:sleeves caption="Annualized Sharpe ratios of each sleeve and of the combinations, in the pre-test validation period (1957–1989) and the test period, with the average meta weight and correlation with the final book in the test period."]]

[[figure:meta_weights,learner_ic width=3.15in height=1.75in caption="(a) Meta weights of the six sleeves, estimated from past out-of-sample sleeve returns. (b) Out-of-sample information coefficient of each return learner, by year."]]

The sleeve table isolates each layer of the model. The rows for the individual sleeves show what each forecasting approach earns on its own. The equal-weight row shows the pure diversification gain from combining them. The gap between that row and the meta book without volatility timing measures what the learned weights add. The last two rows measure the volatility overlay: it moves the test Sharpe ratio from {{stat:book_unscaled.sharpe_test}} to {{stat:sharpe}}. The book is close to market-neutral (CAPM beta {{stat:capm_ew.beta}}, alpha t-statistic {{stat:capm_ew.alpha_t}}), so its returns are not a disguised market bet.

[[figure:rolling_sharpe,drawdown width=3.15in height=1.7in caption="(a) Rolling 36-month Sharpe ratio. (b) Drawdown of PRISM scaled to 10% volatility."]]

[[figure:leverage width=6.5in height=2.3in caption="Volatility timing. The forecast volatility of the meta book before scaling, the leverage scale applied, the resulting gross and net exposure, and the realized 12-month volatility of the final book against its 10% target."]]

# What an investor should expect

The 95% interval for the test Sharpe ratio, {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}, is wide even over 34 years, and we would plan on the lower half of it for three reasons:

- **Performance decay.** Anomaly returns decay after publication [McLean and Pontiff 2016]. PRISM's Sharpe ratio was {{stat:decades.1990s.sharpe}} in the 1990s, {{stat:decades.2000s.sharpe}} in the 2000s, {{stat:decades.2010s.sharpe}} in the 2010s and {{stat:decades.2020s.sharpe}} in 2020–2023, which shows how much of the edge persists.
- **Trading costs.** The CTF scores gross returns, but PRISM trades heavily, with average monthly turnover of {{stat:turnover}} at gross leverage {{stat:gross_leverage}}. Net returns would be materially lower and capacity is limited. Trading costs could be built into the objective, as in Jensen, Kelly, Malamud and Pedersen (2024).
- **Crash risk.** The volatility forecast reacts within weeks, so a sudden shock can still cause a large one-month loss before leverage adjusts. The worst month was {{stat:worst_month}} and the maximum drawdown at 10% volatility was {{stat:max_dd_scaled}}.

# Reproducibility and disclosures

PRISM is a single deterministic Python 3.13 file with pinned dependencies (numpy, pandas, scipy, threadpoolctl, xgboost). Seeds are fixed and the inputs are put in canonical order first. The run took {{stat:runtime_hours|.1f}} hours. The risk model re-implements, in Python, the Barra-style specification of the CTF organizers' benchmark models, but uses clustered themes as factors. The idea of volatility-targeting the whole book came from studying public leaderboard submissions; no code was copied. The code was developed with the assistance of an AI tool (Claude), and the author takes responsibility for the submission.

::: references
DeMiguel, V., L. Garlappi and R. Uppal (2009). Optimal versus naive diversification. *Review of Financial Studies* 22, 1915–1953.
Didisheim, A., S. Ke, B. Kelly and S. Malamud (2024). APT or "AIPT"? The surprising dominance of large factor models. Working paper.
Gu, S., B. Kelly and D. Xiu (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33, 2223–2273.
Jensen, T. I., B. Kelly, S. Malamud and L. H. Pedersen (2024). Machine learning and the implementable efficient frontier. Working paper.
Jensen, T. I., B. Kelly and L. H. Pedersen (2023). Is there a replication crisis in finance? *Journal of Finance* 78, 2465–2518.
Kelly, B., S. Malamud and K. Zhou (2024). The virtue of complexity in return prediction. *Journal of Finance* 79, 459–503.
McLean, R. D. and J. Pontiff (2016). Does academic research destroy stock return predictability? *Journal of Finance* 71, 5–32.
Menchero, J., D. J. Orr and J. Wang (2011). The Barra US equity model (USE4): methodology notes. MSCI.
Moreira, A. and T. Muir (2017). Volatility-managed portfolios. *Journal of Finance* 72, 1611–1644.
:::
