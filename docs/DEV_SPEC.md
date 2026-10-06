# PRISM — developer specification (internal)

This file pins the interfaces between the model (`submission/prism.py`) and the local
tooling (`tools/`). It is the single source of truth for column names, dtypes and file
layout. Competition rules: <https://jkpfactors.com/ctf/rules> (mirrored in the organizer
repo `theisij/common-task-framework-SDF`, file `docs/ctf_rules.md`, updated 2026-09-29).

## 1. Repository layout

```
submission/prism.py          # THE submission: single self-contained file, defines main()
submission/requirements.txt  # exact pins (==) for every non-stdlib import in prism.py
tools/make_synthetic_data.py # synthetic CTF-format data (full-period and validation-like)
tools/test_submission.py     # static rule checks + contract/determinism/shuffle/lookahead tests
tools/run_local.py           # run main() on real data, save weights CSV + diagnostics
tools/make_report.py         # performance stats, figures, documentation PDF
docs/                        # this spec, documentation sources
slurm/                       # HPC job script
```

Python 3.13 (the CTF runtime). Dev venv: `/workspace/main/.venv` (`.venv/bin/python`).
Installed: numpy 2.5.3, pandas 3.0.6, polars 1.44.2, pyarrow 25.0.1, scipy 1.18.1,
scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0, matplotlib (tools only).

## 2. CTF input data (what `main()` receives)

All three arguments are pandas DataFrames loaded by the CTF from parquet files written by
R `arrow::write_parquet`. **pandas 3 reads `date32` columns as `object` dtype holding
`datetime.date`, and strings as `StringDtype`** — code must accept that, and also
`datetime64[ns]`, and bool-or-int flags.

### `chars` (`ctff_chars.parquet`) — one row per (id, eom)

| column           | type (as read by pandas)        | meaning |
|------------------|---------------------------------|---------|
| `id`             | int (int32/int64; maybe float)  | CRSP permno (< 100000) |
| `eom`            | date (object datetime.date)     | portfolio formation month-end (calendar month end) |
| `eom_ret`        | date                            | month-end of the return month = eom + 1 month (month-end) |
| `excntry`        | string                          | "USA" |
| `sic`            | string, may be missing/""       | 4-digit SIC code as text |
| `size_grp`       | string                          | one of "small", "large", "mega" |
| `ret_exc_lead1m` | float64                         | excess return from eom to eom_ret (realized at eom_ret) |
| `ctff_test`      | bool                            | True iff eom_ret >= 1990-01-31 (test sample) |
| <features>       | float64 with NaN                | ~402 characteristics (JKP names, e.g. `be_me`, `ret_12_1`) |

Column order in the real file: `id, eom, eom_ret, excntry, sic, size_grp, ret_exc_lead1m,
ctff_test, <features...>`. Sample: eom from 1951-12-31 to 2023-11-30 (eom_ret 1952-01-31 ..
2023-12-31); test = eom 1989-12-31 .. 2023-11-30 (408 months); ~1,800-2,500 stocks/month;
`ret_exc_lead1m` is never missing; every stock has >= 200 daily returns in the past 252
trading days at eom.

### `features` (`ctff_features.parquet`)
One column, `features` (string): the feature column names in `chars`.

### `daily_ret` (`ctff_daily_ret.parquet`)
| column    | type   |
|-----------|--------|
| `id`      | int    |
| `date`    | date (object datetime.date) |
| `ret_exc` | float64 (non-missing) |

Covers all CRSP ids (also ids never in `chars`), trading days, typically starting before the
first `chars` month.

### CTF validation run (Rule 14)
A ~4 MB subset with **123 monthly observations**: 10 years of training + 3 test months
(`ctff_test` True only for the last 3 months), ~50 stocks, ~10 features. The organizers'
toy replica (utils/toy_data.R) also: drops one FF12 industry entirely (Utils, SIC 4900-4949),
and adds ONE late-industry stock (Enrgy, e.g. SIC 1311) that appears ONLY in the last test
month and whose daily returns exist only AFTER its eom (date in (eom, eom_ret]).

## 3. `main()` output contract (Rules 11, 12)
`main(chars, features, daily_ret) -> pd.DataFrame` with exactly the columns `id`, `eom`, `w`:
`id` integer (int64), `eom` date (same values as the input `eom` of that row), `w` float64;
no NaN; no duplicated (id, eom); **exactly one row per `ctff_test` row of `chars`, and no
other rows**; non-zero gross exposure each test month; < 50 MB as CSV.

## 4. Tests the CTF runs (and we replicate)
* **Determinism**: same input -> identical output (rtol 1e-5, atol 1e-8: |a-b| <= atol + rtol*|b|).
* **Order invariance**: shuffled rows of every input, shuffled feature list, shuffled column order -> same weights.
* **Lookahead (truncation)**: rerun on `chars[eom <= c]`, `daily_ret[date <= c]` for a test
  month c -> weights for eom <= c must match the full run.
* **Static/security**: file < 1 MB, UTF-8, no NUL bytes; no `subprocess`, `os.system`,
  `os.popen`, `socket`, `urllib`, `requests`, `http.client`, `eval(`, `exec(`, `compile(`;
  no file I/O inside `main()`; exact `main` signature; pinned requirements.

## 5. Diagnostics interface (tools <-> model)
`prism.run_model(chars, features, daily_ret, config=None, diagnostics=True)` returns
`(weights_df, diag)`; `main()` returns only `weights_df`. `diag` is a dict of DataFrames:

| key              | columns                                   | meaning |
|------------------|-------------------------------------------|---------|
| `sleeve_returns` | `eom, sleeve, ret`                        | realized return over (eom, eom_ret] of each vol-targeted sleeve formed at eom (pre-test months included) |
| `meta_weights`   | `eom, sleeve, theta`                      | meta-combination weight of each sleeve at eom |
| `book`           | `eom, scale, vol_synth, vol_fm, gross, net, n_stocks, ret` | final-book overlay diagnostics; `ret` = realized final-book return |
| `learner_ic`     | `eom, learner, ic`                        | OOS cross-sectional Pearson IC of each return learner |
| `sdf_zeta`       | `eom, sleeve, zeta`                       | ridge shrinkage chosen by CV for the SDF sleeves |
| `themes`         | `refit_eom, feature, cluster`             | hierarchical-clustering theme assignment at each risk-model refit |
| `timing`         | `step, seconds`                           | wall-clock per pipeline step |
| `config`         | `key, value`                              | configuration used |
