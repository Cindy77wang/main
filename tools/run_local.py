"""Run PRISM on the CTF data and save the submission weights plus diagnostics.

Usage:
    python tools/run_local.py --data data/raw --out output/prism [--threads 32]

Writes:
    <out>/prism_weights.csv          the "Portfolio Weights" file for the CTF submission form
    <out>/diagnostics/<key>.parquet  sleeve returns, meta weights, book overlay, learner ICs, ...
    <out>/run_info.json              runtime, row counts, package versions

The model itself (submission/prism.py) never touches the file system; this wrapper does the
I/O, exactly like the CTF pipeline, which loads the three parquet files, calls main(), and
stores its return value.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_model(path):
    spec = importlib.util.spec_from_file_location("prism", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with ctff_chars/features/daily_ret.parquet")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--model", default=str(ROOT / "submission" / "prism.py"))
    ap.add_argument("--threads", type=int, default=0, help="XGBoost threads (0 = min(32, cpu count))")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chars = pd.read_parquet(data / "ctff_chars.parquet")
    features = pd.read_parquet(data / "ctff_features.parquet")
    daily_ret = pd.read_parquet(data / "ctff_daily_ret.parquet")
    print(f"loaded chars {chars.shape}, features {len(features)}, daily {daily_ret.shape} "
          f"in {time.time() - t0:.0f}s", flush=True)

    prism = load_model(args.model)
    config = {"n_threads": args.threads} if args.threads else None
    weights, diag = prism.run_model(chars, features, daily_ret, config=config, diagnostics=True)

    weights.to_csv(out / "prism_weights.csv", index=False)
    for key, frame in diag.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_parquet(out / "diagnostics" / f"{key}.parquet", index=False)

    test = chars.loc[chars["ctff_test"].astype(bool), ["id", "eom", "ret_exc_lead1m"]]
    merged = test.merge(weights, on=["id", "eom"], how="left")
    port = (merged["w"] * merged["ret_exc_lead1m"]).groupby(merged["eom"]).sum()
    info = {
        "runtime_minutes": round((time.time() - t0) / 60, 1),
        "rows": int(len(weights)),
        "months": int(weights["eom"].nunique()),
        "missing_test_rows": int(merged["w"].isna().sum()),
        "test_sharpe_annualized": float(port.mean() / port.std() * np.sqrt(12)),
        "python": platform.python_version(),
        "packages": {m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "xgboost")
                     if m in sys.modules},
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
