# Summary

PRISM (*Precision-Rotated, Integrated SDF Machine*) is a systematic long-short portfolio of US stocks. It is rebalanced monthly and built to maximize the out-of-sample Sharpe ratio on the JKP Common Task Framework (CTF) data. Over the {{stat:months}}-month test period ({{stat:start}} to {{stat:end}}), the strategy earned an annualized excess return of {{stat:mean_pct}} with a volatility of {{stat:sd_pct}}. That is a **Sharpe ratio of {{stat:sharpe}}** (95% CI {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}). The maximum drawdown, scaled to 10% volatility, was {{stat:max_dd_scaled_pct}}. Every number in this paper is strictly out of sample: at each month the model uses only data that was available at that date, and no choice was tuned on the test period.

The design rests on one idea from asset pricing. The portfolio with the highest conditional Sharpe ratio is the tangency portfolio, $w_t^* \propto \Sigma_t^{-1}\mu_t$. PRISM estimates it in two complementary ways and lets past out-of-sample performance decide how to combine them:

* **Predict, then optimize.** Machine-learning forecasts of $\mu_t$ (ridge, XGBoost, a neural network, an LSTM) are combined with a factor risk model $\Sigma_t$.
* **Optimize directly.** A random-feature stochastic discount factor (SDF) learns the Sharpe-maximizing combination of thousands of characteristic-managed portfolios.

[[figure:cumret width=6.5in caption="Cumulative excess return (log scale) of PRISM, its sleeves and the equal-weighted market scaled to the same volatility. The shaded region is the pre-test period, used only for validation."]]

# Data, protocol and validation

**Data.** The model uses only the CTF files: monthly characteristics and next-month excess returns for US stocks in the small, large and mega size groups (about 2,000 stocks per month from 1952 to 2023), the list of {{stat:n_features}} characteristics, and daily excess returns. Each month, every characteristic is ranked across stocks and mapped to $[-0.5, 0.5]$; missing values are set to the median (zero). No characteristic is selected or referenced by hand. Industries are the Fama-French 12 groups, mapped from SIC codes.

**Protocol.** Each component is refit every December on a fixed calendar, using only information available at that date. Labels are next-month returns whose return month has ended; daily returns are dated on or before the formation date. The first refits happen in the 1950s, as soon as five years of data exist. The **1957–1989 pre-test period therefore serves as a genuine pseudo-out-of-sample validation sample**, and the test period (portfolios formed from December 1989) is never used for any choice. All data-dependent tuning is done inside each training window:
* time-ordered validation blocks for the return learners;
* blocked cross-validation for the SDF shrinkage;
* past out-of-sample returns for the sleeve weights.

The remaining constants (half-lives, network sizes, window lengths) were fixed a priori from the cited literature. The code passes the CTF's checks for determinism, invariance to input order and look-ahead (truncated data), plus a stricter test in which the last month's future returns are replaced by noise.

# Model

**Risk model.** Following MSCI Barra USE4 [Menchero, Orr and Wang 2011], returns follow a linear factor model with observable exposures:
$$\Sigma_t = X_t\,\Omega_t\,X_t^\prime + D_t .$$
The exposures $X_t$ are the market, FF12 industry dummies and characteristic *themes*. Themes are rebuilt every December by average-linkage hierarchical clustering of the characteristics on the distance $1-|\bar\rho_{jk}|$, where $\bar\rho$ is the average cross-sectional rank correlation over the past five years. The number of clusters is 13, following the JKP themes [Jensen, Kelly and Pedersen 2023]. Daily ridge cross-sectional regressions give factor returns and residuals. $\Omega_t$ is an exponentially weighted covariance with half-lives of 504 days for correlations and 84 days for variances, and $D_t$ is an EWMA of squared residuals with an 84-day half-life. Using a few clustered themes instead of all raw characteristics makes $\Omega_t$ better conditioned and the factor model more parsimonious.

**Return learners.** Each learner predicts the cross-sectionally standardized, winsorized next-month return from the ranked characteristics. It is trained on a rolling 20-year window, with the final 3 years as a time-ordered validation block:
* *Ridge*: the penalty is chosen on the validation block by the information coefficient (IC).
* *XGBoost*: the tree depth (3 or 6) and number of trees are chosen by early stopping on the validation block.
* *MLP*: the NN3 architecture of Gu, Kelly and Xiu (2020), an ensemble of 3 early-stopped networks.
* *LSTM*: reads each stock's 12-month trajectory of 12 characteristic principal components, so that it can learn from *changes* in characteristics that a single snapshot cannot show.

[[figure:learner_ic width=6.5in caption="Out-of-sample cross-sectional information coefficient of each return learner, by year."]]

**Sleeves.** For each learner $k$, the *Markowitz sleeve* is $w^{B_k}_t \propto \Sigma_t^{-1}\hat\mu_{k,t}$. The *SDF sleeves* follow the large-factor-model approach of Didisheim, Ke, Kelly and Malamud (2024). Let $x_{i,t}$ be stock $i$'s ranked characteristics. The signal matrix stacks the characteristics, $P = 3{,}000$ pairs of random Fourier features $\sin(\gamma_p\,\omega_p^\prime x_{i,t})$ and $\cos(\gamma_p\,\omega_p^\prime x_{i,t})$ with $\omega_p \sim N(0, I)$ and bandwidths $\gamma_p$, and a constant. Each column $s_{p,t}$ defines a managed portfolio, scaled to unit ex-ante volatility. In sleeve A0 the portfolio is the de-meaned signal itself. In sleeve A1 it is rotated by $\Sigma_t^{-1}$: if expected returns are spanned by the signals, $\mu_t = S_t\lambda$, then the efficient portfolio $\Sigma_t^{-1}S_t\lambda$ lies exactly in the span of the rotated signals. The SDF loadings solve the ridge-regularized Markowitz problem over the past 30 years of managed-portfolio returns $F_\tau$:
$$\hat b_t = \arg\min_b \sum_{\tau<t}\left(1-b^\prime F_{\tau}\right)^2 + \zeta\,\lVert b\rVert^2 = F^\prime\left(FF^\prime+\zeta I\right)^{-1}\mathbf{1} .$$
The dual form costs a $T\times T$ solve, so thousands of factors are cheap. The shrinkage $\zeta$ maximizes the Sharpe ratio pooled across 5 blocked cross-validation folds, and is re-chosen every December.

**Meta-combination and volatility timing.** The sleeves, each at unit ex-ante volatility, are combined with non-negative weights $\theta_t = \mathrm{NNLS}(\mathbf{1} \sim R)$. Here $R$ holds the sleeves' realized out-of-sample returns over the past 20 years, and the weights are shrunk 30% toward equal weights to limit estimation error [DeMiguel, Garlappi and Uppal 2009]. Finally, the book is scaled to a 10% annual volatility target. The forecast applies today's weights to the past 252 days of daily returns and takes an EWMA ($\lambda = 0.97$, RiskMetrics) of the resulting synthetic book returns. This exploits volatility clustering, as in volatility-managed portfolios [Moreira and Muir 2017].

# Results

[[table:perf]]

[[table:sleeves]]

[[figure:meta width=6.5in caption="Meta-combination weights of the six sleeves over time, estimated from past out-of-sample sleeve returns."]]

The sleeves are imperfectly correlated, so the combination beats each of them. The ablations in the sleeve table isolate the gain from the learned combination (versus equal weights) and from volatility timing (versus the unscaled book).

[[figure:rolling width=6.5in caption="Rolling 36-month Sharpe ratio (top) and drawdown of PRISM scaled to 10% volatility (bottom)."]]

# What an investor should expect

Statistically, the 95% confidence interval for the test Sharpe ratio is {{stat:sharpe_ci_low}} to {{stat:sharpe_ci_high}}. We would plan on the lower half of this range, for three reasons:
* **Performance decay.** The anomaly literature documents decay after publication [McLean and Pontiff 2016], visible in the Sharpe-by-decade breakdown.
* **Transaction costs.** The CTF scores gross returns, but the book trades heavily (average turnover {{stat:turnover_pct}} per month at gross leverage {{stat:gross}}). Net returns would be materially lower, and capacity is limited.
* **Model risk in crises.** The volatility forecast reacts within weeks rather than days, so sudden volatility spikes such as 1987 or March 2020 can produce large single-month losses before leverage adjusts.

The worst month was {{stat:worst_month_pct}}.

# Reproducibility and disclosures

The model is one deterministic Python 3.13 file, `prism.py`, with pinned dependencies. Random seeds are fixed and the inputs are put in canonical order first. A run takes about {{stat:runtime_hours}} hours on 32 cores. The risk model re-implements, in Python, the Barra-style specification of the CTF organizers' benchmark models; it differs by using clustered themes. The idea of volatility-targeting the book came from studying public leaderboard submissions; no code was copied. The code was written with the assistance of an AI tool (Claude), and the author takes responsibility for it.
