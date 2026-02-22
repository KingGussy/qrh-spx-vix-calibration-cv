# qrh-spx-vix-calibration-cv

CUDA-accelerated simulator + synthetic data generator for the Quadratic Rough Heston model, producing SPX and VIX implied-vol smiles on fixed log-moneyness grids. Includes a deterministic forward-variance routine for VIX targets. Control-variate and NN calibration components will be added/expanded on later.

## What’s in this repo
- `src/qrh_sim/` — reusable library code (kernel build, params, parsing, pricing helpers)
- `scripts/generate_synth_qrh_spx_vix.py` — synthetic dataset generator (writes `.npz` shards)
- `scripts/sim_paths_cuda.cu` + CMake/pyproject — CUDA extension sources and build config
- `tests/` — quick sanity tests (WIP)

## Quickstart
Create a venv and install editable:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -U pip
pip install -e .