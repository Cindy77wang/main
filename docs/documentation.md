---
title: PRISM: A Precision-Rotated, Integrated SDF Machine
subtitle: A machine-learning tangency portfolio of US stocks for the JKP Common Task Framework, designed in a digital twin
author: Cindy Wang
course: MGT 924 Statistical Foundations, Yale SOM
max_pages: 5
---

::: abstract
PRISM is a monthly long-short portfolio of US stocks that maximizes the out-of-sample Sharpe ratio on the JKP Common Task Framework (CTF) data. It estimates the conditional tangency portfolio $w_t \propto \Sigma_t^{-1}\mu_t$ in two complementary ways. The first predicts and then optimizes: an ensemble of four machine-learning forecasts of $\mu_t$ is combined with a Barra-style risk model $\Sigma_t$. The second optimizes directly: a random-feature stochastic discount factor (SDF) learns the Sharpe-maximizing combination of thousands of characteristic-managed portfolios. A non-negative meta-portfolio weights the three sleeves, and a volatility-timing overlay sets the leverage. Every design decision was made in a *digital twin*, simulated markets in the exact CTF format whose true tangency portfolio is known, never on the CTF test period. Over the {{stat:months}}-month test period, PRISM earned a Sharpe ratio of **{{stat:sharpe}}** (95% CI {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}).
:::

[[keyfacts stats="sharpe:Sharpe ratio,mean:Return p.a.,sd:Volatility p.a.,max_dd_scaled:Max DD at 10% vol,capm_ew.alpha_t:CAPM alpha t-stat"]]

::: box
**What is new in PRISM.**

- **An SDF learned in risk-rotated space.** Sleeve A1 multiplies every random-feature managed portfolio by $\Sigma_t^{-1}$ before the ridge SDF of Didisheim et al. (2024) combines them. If the signals span expected returns, the tangency portfolio is an exact combination of the candidates, so the SDF need only learn which signals are priced.
- **Trees that learn the alpha the optimizer keeps.** $\Sigma_t^{-1}$ hedges away any forecast that a risk factor prices, so XGBoost is trained on residual returns per unit of specific volatility, a generalized-least-squares target.
- **Stocks read as trajectories.** An LSTM reads each stock's 12-month path of characteristic principal components: the *direction* in which a firm's characteristics move is a predictor.
- **Designed in a digital twin.** Simulated CTF markets with a known truth show how much Sharpe ratio each layer loses. A change was adopted only if it raised the pre-1990 Sharpe ratio in two of them (Section {{sec:twin}}).
:::

[[figure:cumret width=6.5in height=1.8in caption="Cumulative excess return (log scale) of PRISM and its sleeves, and the equal-weighted market scaled to the same volatility, over the test period."]]

# Problem and design

The CTF scores the annualized Sharpe ratio of monthly returns from December 1989 to November 2023, ignoring trading costs; the ex-ante optimal portfolio is the conditional tangency portfolio.

The statistical problem is to estimate $\mu_t$ and $\Sigma_t$ well enough that $\Sigma_t^{-1}\mu_t$ survives out of sample. PRISM follows three principles. *Rotate by risk*: signals become portfolios through an explicit covariance model, not through rank sorts. *Use complexity, with shrinkage*: rich nonlinear models are controlled by cross-validated ridge penalties [Kelly, Malamud and Zhou 2024; Didisheim, Ke, Kelly and Malamud 2024]. *Diversify across model families and time the leverage*: the sleeves' average pairwise correlation is {{stat:sleeve_corr_avg}}, and volatility is predictable while expected returns barely move with it [Moreira and Muir 2017].

**Data and validation protocol.** The model uses only the CTF files: {{stat:n_features}} characteristics and next-month excess returns of about 2,000 stocks per month, plus daily returns. Each month, every characteristic is ranked across stocks onto $[-0.5, 0.5]$, and missing values are set to the median, zero. Nothing is selected or referenced by name. Every component is refit each December on a fixed calendar, using only labels whose return month has ended and daily returns dated on or before the portfolio date. Refitting starts in the 1950s, as soon as five years of data exist, so **1957–1989 is a genuine pseudo-out-of-sample validation period**. All tuning happens inside each training window; the remaining constants were fixed in advance from the cited literature. The code passes replicas of the CTF's tests (determinism, invariance to input order and dtypes, and truncation of later data, with earlier weights bit-identical) and a stricter test that replaces the last month's future returns with noise.

# Model

**Risk model.** As in MSCI Barra USE4 [Menchero, Orr and Wang 2011], returns follow a factor model with observable exposures:
$$\Sigma_t = X_t\,\Omega_t\,X_t' + D_t .$$
The exposures $X_t$ are the market, the Fama-French 12 industry dummies and every ranked characteristic, standardized within the month. This is the specification of the organizers' Markowitz-ML benchmark. In the digital twin it hedged clearly better than 13 clustered characteristic *themes* [Jensen, Kelly and Pedersen 2023]; the themes are still built every December and reported as a diagnostic. Daily ridge cross-sectional regressions give factor returns and residuals. $\Omega_t$ is an exponentially weighted covariance, with half-lives of 504 days for correlations and 84 days for variances. $D_t$ is an EWMA of squared residuals (half-life 84 days). With about 415 factors, the factor correlation matrix is shrunk toward the identity in proportion to the number of factors relative to the effective sample size.

**Return learners.** Four learners map the ranked characteristics into next-month returns, standardized and winsorized within each month. Each uses a rolling 20-year window whose last 3 years form a time-ordered validation block. *Ridge* picks its penalty by the validation information coefficient (IC). *XGBoost* learns residual returns: each month's return net of the risk model's factor exposures, divided by the stock's specific volatility [Grinold and Kahn 2000]. Raw returns are dominated by common factor shocks, and $\Sigma_t^{-1}$ hedges away the factor-priced part of any forecast anyway. XGBoost picks its tree depth (3 or 6) and its number of trees by early stopping. The *MLP* is an ensemble of three early-stopped NN3 networks [Gu, Kelly and Xiu 2020]. The *LSTM* reads each stock's 12-month trajectory of its first 12 characteristic principal components.

**Sleeves.** The learners' standardized forecasts are averaged into one forecast [Timmermann 2006]; the residual forecast is first scaled back to returns by the specific volatility. That forecast gives one Markowitz sleeve, $w^{B}_t \propto \Sigma_t^{-1}\hat\mu_t$. Two further sleeves follow the large-factor-model SDF of Didisheim et al. (2024). Each stock's signal vector stacks its ranked characteristics $x_{i,t}$, $P$ = 3,000 random Fourier feature pairs and a constant:
$$s_{i,t} = \left[x_{i,t},\ \sin(\gamma_p\,\omega_p' x_{i,t}),\ \cos(\gamma_p\,\omega_p' x_{i,t}),\ 1\right]_{p=1}^{P}, \qquad \omega_p \sim N(0, I).$$
The bandwidths $\gamma_p$ cycle through 0.5, 1, 2 and 3. Every signal defines a managed portfolio, scaled to unit ex-ante volatility. In sleeve A0 the portfolio is the de-meaned signal itself; in A1 it is the rotated signal $\Sigma_t^{-1}s$. Given past managed-portfolio returns $F$ (360 months), the SDF loadings solve a ridge-regularized Markowitz problem, which is cheap to compute in its dual $T\times T$ form:
$$\hat b_t = \arg\min_b \sum_{\tau<t}\left(1-b'F_{\tau}\right)^2 + \zeta \sum_p b_p^2 = F'\left(FF'+\zeta I\right)^{-1}\mathbf{1}.$$
The shrinkage $\zeta$ maximizes the pooled Sharpe ratio across five blocked cross-validation folds and is re-chosen every December.

**Meta-combination and volatility timing.** The sleeves, each at unit ex-ante volatility, get non-negative weights $\theta_t = \arg\min_{\theta \geq 0} \sum_{\tau<t}(1-\theta' R_\tau)^2$. Here $R_\tau$ holds the sleeves' realized out-of-sample returns over the past 20 years. This is the long-only, Sharpe-maximizing combination, and it is shrunk 30% toward equal weights to limit estimation error [DeMiguel, Garlappi and Uppal 2009]. Finally, the book is scaled to a 10% annual volatility target. The volatility forecast applies today's weights to the past 252 days of returns and takes an EWMA ($\lambda$ = 0.97, the RiskMetrics monthly decay) of the resulting synthetic book returns.

# Designed in a digital twin {#sec:twin}

The real data cannot show how far a model is from the best achievable portfolio, because the true $\mu_t$ and $\Sigma_t$ are unknown. We therefore built a digital twin of the CTF: a simulator that writes the three CTF files in the exact format. Its stocks list and delist, belong to FF12 industries and have missing and late-starting characteristics. Their fat-tailed daily returns load on a GJR-GARCH market with crashes in 1987, 2008 and 2020 and on industry and style factors. The 60–80 characteristics are noisy, often nonlinear transforms of six persistent latent firm traits. Expected returns are a planted nonlinear function of these traits (value, squared profitability and a value-momentum interaction), calibrated so that the true tangency portfolio earns a Sharpe ratio of about 2. Because the truth is known, every layer of PRISM can be measured against it, and so can replicas of the organizers' benchmarks run on the same data (Figure {{fig:twin}}).

[[image:figures/digital_twin.png label=twin width=6.5in caption="The digital twin. Annualized Sharpe ratios in two independently simulated CTF markets, every portfolio at unit ex-ante volatility, in the test period (bars) and the pre-test period (circles). The oracle holds the true tangency portfolio; the benchmark replicas apply the organizers' recipes to the same data."]]

Three lessons shaped PRISM. *The risk model is half the battle*: even with the true expected returns, PRISM's estimated risk model reaches Sharpe ratios of 1.56 and 1.63, against the oracle's 2.11 and 2.46; this gap led to the all-characteristic risk model (Table {{tab:twin}}). *Forecasts must survive hedging*: the original learners mostly predicted factor-priced returns, and hedged with the true covariance their ensemble earned only 0.14 before 1990, hence the residual target. *Learning the SDF directly works best*: the rotated SDF sleeve is the strongest sleeve in both markets, and the meta-combination comes close to it without knowing in advance which sleeve will be best. Replicas of the organizers' benchmarks earn 0.24–0.38, below the 1/N market. The simulated markets have only 300–500 stocks, so the comparisons matter, not the levels.

| Idea tested | Motivation | Pre-1990 change (1 / 2) | Decision |
|:--|:--|:-:|:-:|
| All characteristics as risk factors | Hedging loss vs. the oracle | +0.16 / +0.25 | Adopted |
| Residual-return XGBoost, one ensemble sleeve | Forecasts were hedged away | +0.10 / +0.12 | Adopted |
| Neural SDF trained on the realized Sharpe ratio | Top leaderboard entries | −0.01 / +0.00 | Rejected |
| Learner forecasts as extra SDF columns | Let the SDF weight the learners | −0.02 to −0.16 | Rejected |
| 6,000 instead of 3,000 random features | Virtue of complexity | +0.02 (under 0.5 s.e.) | Rejected |
| Statistical factors added to the risk model | Residual factor structure | Oracle hedge 1.48 → 1.51 | Rejected |
| Shrink the ensemble where learners disagree | Trust agreement | +0.00 / +0.00 at best | Rejected |
| Leverage timed by the opportunity set | Time-varying Sharpe ratio | ≤ +0.04 even with the truth | Rejected |
| XGBoost tree leaves as extra SDF signals | Trees as managed portfolios | −0.02 / −0.02 | Rejected |
| About 250 meta and overlay variants | Windows, shrinkage, decays | Within noise | Rejected |
Table: Design decisions in the digital twin. Each idea was implemented, re-run independently and judged only by the book's pre-1990 Sharpe ratio in both markets. The neural SDF was rejected although it raised the simulated test-period Sharpe ratio (+0.12 / +0.04). {#tab:twin}

# Results

[[table:perf layout=wide caption="Out-of-sample performance over the test period (CTF organizers' definitions; i.i.d. 95% interval)."]]

[[table:sleeves caption="Annualized Sharpe ratios of the sleeves and combinations before 1990 and in the test period, with the average test-period meta weight and correlation with the book."]]

[[figure:meta_weights,rolling_sharpe width=3.15in height=1.6in caption="(a) Meta weights of the three sleeves, estimated from past out-of-sample sleeve returns. (b) Rolling 36-month Sharpe ratio of PRISM and of the equal-weighted market."]]

The volatility overlay moves the test Sharpe ratio from {{stat:book_unscaled.sharpe_test}} to {{stat:sharpe}}, and the book is close to market-neutral (CAPM beta {{stat:capm_ew.beta}}, alpha t-statistic {{stat:capm_ew.alpha_t}}), so its returns are not a disguised market bet.

# What an investor should expect

The 95% interval for the test Sharpe ratio, {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}, is wide; we would plan on its lower half, for three reasons:

- **Performance decay.** Anomaly returns decay after publication [McLean and Pontiff 2016]. PRISM's Sharpe ratio was {{stat:decades.1990s.sharpe}} in the 1990s, {{stat:decades.2000s.sharpe}} in the 2000s, {{stat:decades.2010s.sharpe}} in the 2010s and {{stat:decades.2020s.sharpe}} in 2020–2023.
- **Trading costs.** The CTF scores gross returns, but PRISM trades heavily: average monthly turnover is {{stat:turnover}} at gross leverage {{stat:gross_leverage}}. Net returns would be materially lower and capacity is limited. Trading costs could be built into the objective [Jensen, Kelly, Malamud and Pedersen 2024].
- **Crash risk.** The volatility forecast reacts within weeks, so a sudden shock can still cause a large one-month loss before leverage adjusts. The worst month was {{stat:worst_month}}, and the maximum drawdown at 10% volatility was {{stat:max_dd_scaled}}.

# Reproducibility and disclosures

PRISM is a single deterministic Python 3.13 file with pinned dependencies; seeds are fixed and the inputs are put in canonical order. The run took {{stat:runtime_hours|.1f}} hours. The simulator and the digital twin are in the repository's `tools/` folder. The risk model re-implements the Barra-style specification of the organizers' benchmarks. Volatility-targeting the whole book is an idea from public leaderboard submissions; no code was copied. The code was developed with the help of an AI tool (Claude); the author is responsible for the submission.

::: references
DeMiguel, V., L. Garlappi and R. Uppal (2009). Optimal versus naive diversification. *Review of Financial Studies* 22, 1915–1953.
Didisheim, A., S. Ke, B. Kelly and S. Malamud (2024). APT or "AIPT"? The surprising dominance of large factor models. Working paper.
Grinold, R. C. and R. N. Kahn (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
Gu, S., B. Kelly and D. Xiu (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33, 2223–2273.
Jensen, T. I., B. Kelly, S. Malamud and L. H. Pedersen (2024). Machine learning and the implementable efficient frontier. Working paper.
Jensen, T. I., B. Kelly and L. H. Pedersen (2023). Is there a replication crisis in finance? *Journal of Finance* 78, 2465–2518.
Kelly, B., S. Malamud and K. Zhou (2024). The virtue of complexity in return prediction. *Journal of Finance* 79, 459–503.
McLean, R. D. and J. Pontiff (2016). Does academic research destroy stock return predictability? *Journal of Finance* 71, 5–32.
Menchero, J., D. J. Orr and J. Wang (2011). The Barra US equity model (USE4): methodology notes. MSCI.
Moreira, A. and T. Muir (2017). Volatility-managed portfolios. *Journal of Finance* 72, 1611–1644.
Timmermann, A. (2006). Forecast combinations. *Handbook of Economic Forecasting* 1, 135–196.
:::
