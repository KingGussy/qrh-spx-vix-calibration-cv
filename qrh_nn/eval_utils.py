from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from qrh_nn.model import ResMLP, ResMLPConfig 
from qrh_nn.model_k import ContinuousKModel, ContinuousKConfig # build_ctsk_model

# PATHS
REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
CKPT_FIXED = REPO / "models" / "full_mtp" / "checkpoints_resmlp_03" / "best.pt"
CKPT_FIXED_ARB_LAM01 = REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam01" / "best.pt"
CKPT_FIXED_ARB_LAM10 = REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam10" / "best.pt"
CKPT_FIXED_ARB_LAM50 = REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam50" / "best.pt"
NORM_FIXED = REPO / "models" / "norm" / "norm.npz"

CKPT_CTSK = REPO / "models" / "full_mtp_spxk" / "modelA_run01" / "best.pt"
NORM_CTSK = REPO / "models" / "norm" / "norm_spxk.npz"

# GRIDS
K_SPX_FIXED = np.array([-0.15, -0.12, -0.10, -0.08, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00,  0.01,  0.02,  0.03,  0.04,  0.05], dtype=np.float32)
K_VIX_FIXED = np.array([-0.10, -0.05, -0.03, -0.01,  0.01,  0.03,  0.05,  0.07,  0.09, 0.11,  0.13,  0.15,  0.17,  0.19,  0.21], dtype=np.float32)
K_SPX_DENSE = np.linspace(K_SPX_FIXED.min(), K_SPX_FIXED.max(), 151, dtype=np.float32)
K_VIX_DENSE = np.linspace(K_VIX_FIXED.min(), K_VIX_FIXED.max(), 151, dtype=np.float32)


# ============== general utils =================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=(device.type == "cuda"))

def _device_or_default(device: Optional[torch.device | str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _as_t(x, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    return obj



def infer_T_unique(X: np.ndarray, T_col: int = -1) -> np.ndarray:
    T = X[:, T_col]
    return np.unique(T)

def infer_grid_dims(X: np.ndarray, Y: np.ndarray, T_col: int = -1) -> Tuple[int, int]:
    nT = int(infer_T_unique(X, T_col=T_col).size)
    nK = int(Y.shape[1]) // 2 # assumes Y = [spx( nK )), vix( nK )], i.e. nK for vix = nK for spx
    return nT, nK

def split_spx_vix(Y: np.ndarray, nK: int) -> Tuple[np.ndarray, np.ndarray]:
    return Y[:, :nK], Y[:, nK:]



def unnormalise(Yn: np.ndarray, Y_mu: np.ndarray, Y_sd: np.ndarray) -> np.ndarray:
    return Yn * Y_sd[None, :] + Y_mu[None, :]

def normalise(X: np.ndarray, X_mu: np.ndarray, X_sd: np.ndarray) -> np.ndarray:
    return (X - X_mu[None, :]) / X_sd[None, :]


# ==== predictors ===== #
@torch.no_grad()
def predict_norm(
    model: nn.Module,
    Xn: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    takes in NORMALISED Xn,  N x d_in
    returns Yn_pred: NORMALISED predictions
    """
    model.eval()
    N = Xn.shape[0]
    preds = []
    #with torch.no_grad():
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(Xn[i:i+batch_size]).to(device)
        yb = model(xb).detach().cpu().numpy()
        preds.append(yb)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_raw(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    X_mu: np.ndarray,
    X_sd: np.ndarray,
    Y_mu: np.ndarray,
    Y_sd: np.ndarray,
    device: torch.device,
    batch_size: int = 4096
) -> np.ndarray:
    '''
    Just predict_norm as above but doesn't assume normalisation
    '''
    Xn = (X_raw - X_mu[None, :]) / X_sd[None, :]
    preds = []
    for i in range(0, len(Xn), batch_size):
        xb = torch.from_numpy(Xn[i:i+batch_size]).to(device)
        yb = model(xb).detach().cpu().numpy()
        preds.append(yb)
    Yn = np.concatenate(preds, axis=0)
    Y = Yn * Y_sd[None, :] + Y_mu[None, :]
    return Y


# ===== Universal model loading ==== #

def _build_fixed_from_cfg(cfg):
    return ResMLP(cfg)

def _build_ctsk_from_cfg(cfg):
    return ContinuousKModel(cfg)

MODEL_SPECS = {
    "fixed": {
        "ckpt": CKPT_FIXED,
        "norm": NORM_FIXED,
        "cfg_cls": ResMLPConfig,
        "builder": _build_fixed_from_cfg,
        "trim_last_input": False,
    },
    "ctsk": {
        "ckpt": CKPT_CTSK,
        "norm": NORM_CTSK,
        "cfg_cls": ContinuousKConfig,
        "builder": _build_ctsk_from_cfg,
        "trim_last_input": True,   # trim unused positional feat
    },
}

def load_model_and_norm(
        model_type: str, 
        device: torch.device,
        *,
        ckpt_override: Optional[Path] = None,
        norm_override: Optional[Path] = None
):
    model_type = model_type.lower()
    if model_type not in MODEL_SPECS:
        raise ValueError(f"Unknown model_type={model_type!r}. Available: {list(MODEL_SPECS)}")

    spec = MODEL_SPECS[model_type]

    ckpt_path = Path(ckpt_override) if ckpt_override is not None else Path(spec["ckpt"])
    norm_path = Path(norm_override) if norm_override is not None else Path(spec["norm"])


    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("something's missing boy")

    cfg_dict = ckpt.get("cfg", None)
    cfg_cls = spec["cfg_cls"]
    cfg = cfg_cls(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_cls()

    model = spec["builder"](cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    with np.load(norm_path, allow_pickle=False) as z:
        X_mu = z["X_mu"].astype(np.float32, copy=False)
        X_sd = z["X_sd"].astype(np.float32, copy=False)
        Y_mu = z["Y_mu"].astype(np.float32, copy=False)
        Y_sd = z["Y_sd"].astype(np.float32, copy=False)

    if spec["trim_last_input"] and X_mu.shape[0] == 18:
        X_mu = X_mu[:17]
        X_sd = X_sd[:17]

    return model, cfg, X_mu, X_sd, Y_mu, Y_sd



# ====== Eval metrics ======
def error_stats(abs_err: np.ndarray) -> Dict[str, float]:
    """
    abs_err: array of abs errors in raw IV units, shape (...,)
    """
    flat = abs_err.reshape(-1)
    return {
        "mae": float(flat.mean()),
        "rmse": float(np.sqrt(np.mean(flat * flat))),
        "p50": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
    }


def per_grid_mean_abs_error(
    abs_err: np.ndarray,
    X: np.ndarray,
    T_unique: np.ndarray,
    nK: int,
    T_col: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean abs error per T x K grid point
    abs_err: shape (N, 2*nK) raw abs errors for all outputs
    returns:
      spx_grid: (nT, nK)
      vix_grid: (nT, nK)
    """
    N = abs_err.shape[0]
    nT = T_unique.size

    spx_err, vix_err = split_spx_vix(abs_err, nK)

    spx_grid = np.zeros((nT, nK), dtype=np.float64)
    vix_grid = np.zeros((nT, nK), dtype=np.float64)
    counts = np.zeros((nT,), dtype=np.int64)

    # Map each row to its T index
    T = X[:, T_col]
    # T_unique is sorted
    t_idx = np.searchsorted(T_unique, T)

    for ti in range(nT):
        mask = (t_idx == ti)
        counts[ti] = int(mask.sum())
        if counts[ti] == 0:
            continue
        spx_grid[ti] = spx_err[mask].mean(axis=0)
        vix_grid[ti] = vix_err[mask].mean(axis=0)

    return spx_grid, vix_grid


# ===== Plotting helpers ======

def plot_heatmap(grid: np.ndarray, title: str, out_path: Path, x_label: str = "strike index", y_label: str = "T index"):
    plt.figure(figsize=(8, 4))
    plt.imshow(grid, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

