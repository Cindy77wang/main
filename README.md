# PRISM — Precision-Rotated, Integrated SDF Machine

A submission to the [JKP Common Task Framework](https://jkpfactors.com/ctf/rules) (Problem
Set 3, MGT 924 Statistical Foundations, Yale SOM). The model maps the CTF data (monthly
characteristics and returns of liquid US stocks, plus daily returns) into monthly portfolio
weights that aim to maximize the out-of-sample Sharpe ratio.

## The model in one paragraph

All characteristics are ranked within each month. A Barra-style risk model, re-estimated
every December, uses the market, FF12 industries and every characteristic as factors. That
is the organizers' Markowitz-ML specification; clustered characteristic themes are kept as
a diagnostic. Four return learners (ridge, XGBoost, a feed-forward network and an LSTM over
each stock's 12-month characteristic trajectory) are refit every December on a rolling
20-year window, strictly out of sample. XGBoost learns residual returns net of the risk
factors, per unit of specific volatility, so it targets the alpha that hedging keeps. The
learners' forecasts are averaged into one Markowitz sleeve `inv(Sigma) mu`. Two more sleeves
are random-feature SDFs in the style of Didisheim, Ke, Kelly and Malamud: thousands of random
Fourier features of the characteristics define managed portfolios, raw or rotated by
`inv(Sigma)`, which are combined by ridge-regularized Markowitz with cross-validated
shrinkage. A non-negative Sharpe-maximizing meta-portfolio weights the three sleeves by their
past out-of-sample returns. Finally, the book is scaled to a 10% volatility target forecast
from its synthetic daily returns. The full description is in the module docstring of
[`submission/prism.py`](submission/prism.py) and in the documentation PDF.

## How the design was chosen

The CTF data was never used to make design choices. Everything was developed on simulated
markets in the exact CTF format whose true expected returns and covariances are known
(`tools/make_synthetic_data.py`). An oracle decomposition measured how much Sharpe ratio each
layer loses relative to the true tangency portfolio. Candidate changes were then tested in
parallel and kept only if they raised the **pre-1990** Sharpe ratio in **two independently
simulated markets** and survived an independent re-run. In those simulations the adopted
changes (all-characteristic risk model, residual-return XGBoost, ensemble sleeve) lifted the
pre-1990 Sharpe ratio from 0.57 to 0.83 and from 0.24 to 0.61, and the test-period Sharpe
ratio from 0.84 to 0.94 and from 1.00 to 1.14. On the same simulated data, replicas of the
organizers' Markowitz-ML benchmark (2.52 on the real leaderboard) earn only 0.16-0.39.

`tools/digital_twin.py` reproduces the decomposition and draws
[`docs/figures/digital_twin.png`](docs/figures/digital_twin.png): each layer of PRISM, the
oracle and the benchmark replicas, in two simulated markets. The documentation includes it
together with the table of ideas that were tested and rejected.

## Repository layout

| Path | Purpose |
|---|---|
| `submission/prism.py` | **The model** — single self-contained file with `main(chars, features, daily_ret)` |
| `submission/requirements.txt` | Exact dependency pins (Python 3.13) |
| `tools/run_local.py` | Runs the model on the real data; writes the weights CSV and diagnostics |
| `tools/test_submission.py` | Pre-submission checks: rules scan, output contract, determinism, order invariance, look-ahead truncation test |
| `tools/make_synthetic_data.py` | Synthetic data in the exact CTF format (full-period and validation-like) |
| `tools/make_report.py` | Performance statistics, figures and the documentation PDF |
| `tools/digital_twin.py` | Oracle decomposition in simulated markets with known truth (the design lab) |
| `docs/documentation.md` | Source text of the documentation (rendered by `make_report.py`) |
| `docs/figures/` | Digital-twin figure and table included in the documentation |
| `docs/preview/` | **Preview** of the documentation PDF built from simulated data (watermarked; not results) |
| `docs/DEV_SPEC.md` | Data formats and the model/tool interfaces |
| `slurm/run_prism.slurm` | Job script for a 32-core SLURM node (e.g. Yale McCleary/Grace) |

## How to reproduce

1. **Environment** (Python 3.13, the CTF runtime). Install [uv](https://docs.astral.sh/uv/)
   first if needed (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:
   ```bash
   uv venv --python 3.13 .venv && source .venv/bin/activate
   uv pip install -r submission/requirements.txt pyarrow       # the model
   uv pip install matplotlib==3.11.2 reportlab==5.0.1           # the report tools only
   ```
2. **Data.** Download `ctff_chars.parquet`, `ctff_features.parquet` and
   `ctff_daily_ret.parquet` (see [dataset access](https://jkpfactors.com/ctf/dataset-access))
   into `data/raw/`. The data is licensed through WRDS and is git-ignored.
3. **Check the code first** on synthetic data in the exact CTF format (minutes):
   ```bash
   python tools/make_synthetic_data.py --mode validation --out data/synthetic/validation --daily-lead-months 0
   python tools/test_submission.py --model submission/prism.py --data data/synthetic/validation
   ```
   This runs the rule scan, the output contract, determinism, input-order/dtype invariance,
   the organizers' truncation (look-ahead) test and a stricter label-leakage test.
4. **Run the model** on the real data:
   ```bash
   python tools/run_local.py --data data/raw --out output/prism --threads 32
   # or on a cluster, from the repository root: sbatch slurm/run_prism.slurm
   # (follow progress with: tail -f prism_<jobid>.out)
   ```
   Estimated from scaled benchmarks: about 3–4 hours on 32 cores (at most ~8 hours in the
   worst case; longer on a laptop) and a peak of about 30–35 GB of RAM (the 415-factor risk
   model is stored for every month). Ask for 128 GB to be safe. The log reports every December
   refit with per-learner timings.
5. **Build the documentation** from the run's output (statistics, figures, 5-page PDF):
   ```bash
   python tools/make_report.py --data data/raw --weights output/prism/prism_weights.csv \
       --diagnostics output/prism/diagnostics --out docs/report --doc-source docs/documentation.md --strict
   ```
   Every number in the PDF is filled in from the run; the digital-twin figure comes from
   `docs/figures/`. Before submitting, re-read the narrative in `docs/documentation.md`
   against the actual results and adjust the interpretation where needed. Until then,
   `docs/preview/documentation_PREVIEW_synthetic.pdf` shows the layout, filled in with
   simulated data (every page is stamped as a preview).
6. **Optional check of the saved weights file** against the CTF format and coverage:
   ```bash
   python tools/test_submission.py --model submission/prism.py --data data/raw --static-only \
       --weights output/prism/prism_weights.csv
   ```

## Submission checklist (jkpfactors.com/ctf/submit)

| Form field | File |
|---|---|
| Model script | `submission/prism.py` |
| Portfolio weights | `output/prism/prism_weights.csv` |
| Dependencies | `submission/requirements.txt` |
| Documentation | `docs/report/documentation.pdf` |

## Compliance with the CTF rules

* **No look-ahead (Rule 1).** Every estimate used at month t uses only characteristics at
  t, monthly returns realized by t and daily returns dated at or before t. Refits happen on
  a fixed calendar (each December), so truncating the data never changes earlier weights.
  `tools/test_submission.py` replicates the organizers' truncation test.
* **Algorithmic feature use (Rule 2).** All provided features are used; nothing is
  selected by hand, and no feature is referenced by name.
* **Competition data only (Rule 3), reproducibility (Rules 4, 18).** Fixed seeds, canonical
  input ordering (rows, columns and feature list), and pinned dependencies.
* **Security (Rule 15).** No network, shell, dynamic-code or file-system access inside the
  model.

## Disclosures

* The risk model follows the Barra USE4S-style specification used in the organizers'
  benchmark models (Minimum Variance and Markowitz-ML in
  [theisij/common-task-framework-SDF](https://github.com/theisij/common-task-framework-SDF)),
  re-implemented in Python, with all characteristics and the FF12 industries as factors as
  in their specification.
* Public leaderboard submissions
  ([Hemasrikar/jkp-ctf-portfolio-models](https://github.com/Hemasrikar/jkp-ctf-portfolio-models))
  were studied for ideas. The volatility-targeting overlay is the idea we took from them;
  our implementation and forecast are different, and no code was copied.
* The code was developed with the help of an AI assistant (Claude); the author is
  responsible for the submission.
