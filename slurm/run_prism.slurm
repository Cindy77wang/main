#!/bin/bash
# Run PRISM on the full CTF data on a SLURM cluster (e.g. Yale's McCleary or Grace).
# Mirrors the CTF execution environment: 32 cores, Python 3.13, pinned requirements.
# Submit from the repository root: sbatch slurm/run_prism.slurm  (log: prism_<jobid>.out)
#SBATCH --job-name=prism
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=prism_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"  # where the uv installer puts uv

# One-time environment setup (uv: https://docs.astral.sh/uv/)
if [ ! -d .venv-ctf ]; then
  uv venv --python 3.13 .venv-ctf
  VIRTUAL_ENV=.venv-ctf uv pip install -r submission/requirements.txt pyarrow
fi

export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
.venv-ctf/bin/python tools/run_local.py --data data/raw --out output/prism --threads 32
