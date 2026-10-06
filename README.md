# PRISM — Precision-Rotated, Integrated SDF Machine

A submission to the [JKP Common Task Framework](https://jkpfactors.com/ctf/rules) (Problem
Set 3, MGT 924 Statistical Foundations, Yale SOM). The model maps the CTF data (monthly
characteristics and returns of liquid US stocks, plus daily returns) into monthly portfolio
weights that aim to maximize the out-of-sample Sharpe ratio.

## The model in one paragraph

All 402 characteristics are ranked within each month. A Barra-style risk model, rebuilt
every December, combines the market, FF12 industries and characteristic *themes* found by
hierarchical clustering. Four return learners (ridge, XGBoost, a feed-forward network and
an LSTM over each stock's 12-month characteristic trajectory) are refit every December on a
rolling 20-year window, strictly out of sample. Their forecasts become Markowitz sleeves
`inv(Sigma) mu`. Two more sleeves are random-feature SDFs à la Didisheim, Ke, Kelly and
Malamud: thousands of random Fourier features of the characteristics define managed
portfolios, raw or rotated by `inv(Sigma)`, which are combined by ridge-regularized
Markowitz with cross-validated shrinkage. A non-negative Sharpe-maximizing meta-portfolio
weights the six sleeves using their past out-of-sample returns. Finally, the book is scaled
to a 10% volatility target forecast from its synthetic daily returns. The full description
is in the module docstring of [`submission/prism.py`](submission/prism.py) and in the
documentation PDF.

## Repository layout

| Path | Purpose |
|---|---|
| `submission/prism.py` | **The model** — single self-contained file with `main(chars, features, daily_ret)` |
| `submission/requirements.txt` | Exact dependency pins (Python 3.13) |
| `tools/run_local.py` | Runs the model on the real data; writes the weights CSV and diagnostics |
| `tools/test_submission.py` | Pre-submission checks: rules scan, output contract, determinism, order invariance, look-ahead truncation test |
| `tools/make_synthetic_data.py` | Synthetic data in the exact CTF format (full-period and validation-like) |
| `tools/make_report.py` | Performance statistics, figures and the documentation PDF |
| `docs/documentation.md` | Source text of the documentation |
| `slurm/run_prism.slurm` | Job script for a 32-core SLURM node (e.g. Yale McCleary/Grace) |

## How to reproduce

1. **Environment** (Python 3.13, the CTF runtime):
   ```bash
   uv venv --python 3.13 .venv && source .venv/bin/activate
   uv pip install -r submission/requirements.txt pyarrow matplotlib reportlab
   ```
2. **Data.** Download `ctff_chars.parquet`, `ctff_features.parquet` and
   `ctff_daily_ret.parquet` (see [dataset access](https://jkpfactors.com/ctf/dataset-access))
   into `data/raw/`. The data is licensed through WRDS and is git-ignored.
3. **Run the model** (about 3–6 hours on 32 cores; peak memory well under 64 GB):
   ```bash
   python tools/run_local.py --data data/raw --out output/prism --threads 32
   # or: sbatch slurm/run_prism.slurm
   ```
4. **Check the submission** against the CTF rules (fast checks on synthetic data first):
   ```bash
   python tools/make_synthetic_data.py --mode validation --out data/synthetic/validation
   python tools/test_submission.py --model submission/prism.py \
       --requirements submission/requirements.txt --data data/synthetic/validation
   ```
5. **Build the documentation** from the run's output:
   ```bash
   python tools/make_report.py --data data/raw --weights output/prism/prism_weights.csv \
       --diagnostics output/prism/diagnostics --out docs/report --doc-source docs/documentation.md
   ```

## Submission checklist (jkpfactors.com/ctf/submit)

| Form field | File |
|---|---|
| Model script | `submission/prism.py` |
| Portfolio weights | `output/prism/prism_weights.csv` |
| Dependencies | `submission/requirements.txt` |
| Documentation | `docs/report/prism_documentation.pdf` |

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
  re-implemented in Python. It differs in using clustered characteristic themes instead of
  all 402 raw characteristics as factors.
* Public leaderboard submissions
  ([Hemasrikar/jkp-ctf-portfolio-models](https://github.com/Hemasrikar/jkp-ctf-portfolio-models))
  were studied for ideas. The volatility-targeting overlay is the idea we took from them;
  our implementation and forecast are different, and no code was copied.
* The code was developed with the help of an AI assistant (Claude); the author is
  responsible for the submission.
