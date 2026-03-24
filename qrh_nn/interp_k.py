from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from qrh_nn.model import ResMLP, ResMLPConfig
from qrh_nn.eval_utils import (
    K_SPX_FIXED as K_SPX,K_VIX_FIXED as K_VIX, K_SPX_DENSE, K_VIX_DENSE,
    CKPT_FIXED, NORM_FIXED,
    predict_raw, 
)

# ==== CFG ====

REPO = Path(".")

PACKED_PATH = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
TEST_IDX_PATH = REPO / "models" / "split" / "test_idx.npy"

OUT_DIR = REPO / "models" / "full_mtp" / "interpC"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# the grids on which our MLP trained
T_GRID = np.array([0.03, 0.05, 0.07, 0.09], dtype=np.float32)

try:
    from scipy.interpolate import PchipInterpolator, UnivariateSpline
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ============================================================
# RAW LATTICE METRICS
# ============================================================

def second_diff_roughness(smiles: np.ndarray) -> np.ndarray:
    """
    Crude roughness approximation
    smiles: (N, nK)
    returns per-row mean abs second difference, shape (N,)
    """
    d2 = smiles[:, 2:] - 2.0 * smiles[:, 1:-1] + smiles[:, :-2]
    return np.mean(np.abs(d2), axis=1)


def lattice_derivative(smiles: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """
    Finite-difference derivative d sigma / d k on the lattice.
    smiles: (N, nK)
    returns derivs: (N, nK)
    """
    N, nK = smiles.shape
    out = np.zeros_like(smiles)

    # endpoints
    out[:, 0] = (smiles[:, 1] - smiles[:, 0]) / (k_grid[1] - k_grid[0])
    out[:, -1] = (smiles[:, -1] - smiles[:, -2]) / (k_grid[-1] - k_grid[-2])

    # central differences
    for j in range(1, nK - 1):
        out[:, j] = (smiles[:, j + 1] - smiles[:, j - 1]) / (k_grid[j + 1] - k_grid[j - 1]) # ./\. --> __

    return out


# ============================================================
# INTERPOLATION / SMOOTHING
# ============================================================

class SmileInterpolator:
    """
    Wraps one smile on one maturity slice.
    Supports: linear, pchip or spline interpolation schemes

    """
    def __init__(
        self,
        k_grid: np.ndarray,
        sigma_grid: np.ndarray,
        method: str = "pchip",
        smooth: bool = False,
        smooth_s: Optional[float] = None
    ):
        self.k_grid = np.asarray(k_grid, dtype=np.float64)
        self.sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
        self.method = method
        self.smooth = smooth and HAVE_SCIPY

        if method == "linear" or (not HAVE_SCIPY):
            self._kind = "linear"
            self._interp = None
        elif method == "pchip" and not self.smooth:
            self._kind = "pchip"
            self._interp = PchipInterpolator(self.k_grid, self.sigma_grid)
        elif method == "spline" or self.smooth:
            # smoothing spline: choose a small smoothing if none supplied
            s = smooth_s if smooth_s is not None else 1e-6 * len(self.k_grid)
            self._kind = "spline"
            self._interp = UnivariateSpline(self.k_grid, self.sigma_grid, s=s, k=3)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def eval(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        if self._kind == "linear":
            return np.interp(k, self.k_grid, self.sigma_grid)
        return self._interp(k)

    def deriv(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        if self._kind == "linear":
            # derivative of piecewise linear interpolant:
            vals = self.eval(k)
            eps = 1e-5
            vals_p = self.eval(k + eps)
            vals_m = self.eval(k - eps)
            return (vals_p - vals_m) / (2 * eps)
        if self._kind == "pchip":
            return self._interp.derivative()(k)
        if self._kind == "spline":
            return self._interp.derivative()(k)
        raise RuntimeError("Unexpected interpolation kind")


def smoothed_value_and_derivative_metrics(
    true_smiles: np.ndarray,
    pred_smiles: np.ndarray,
    k_grid: np.ndarray,
    method: str = "spline",
    smooth: bool = True
) -> Dict[str, float]:
    """
    Compare smoothed predicted smile against finite-difference on the original lattice.
    - values: compare smoothed pred evaluated at lattice points vs "true" lattice values
    - derivatives: compare smoothed pred derivative at lattice points vs "true" lattice FD derivative

    true_smiles, pred_smiles: (N, nK)
    """
    assert true_smiles.shape == pred_smiles.shape
    N, nK = true_smiles.shape

    # "True" derivative baseline = finite difference on the true lattice
    d_true = lattice_derivative(true_smiles, k_grid)   # (N, nK)

    val_errs = []
    deriv_errs = []
    pred_rough_smoothed = []
    true_rough = second_diff_roughness(true_smiles)

    for i in range(N):
        interp = SmileInterpolator(
            k_grid,
            pred_smiles[i],
            method=method,
            smooth=smooth
        )

        # Smoothed predicted smile evaluated back at the lattice points
        pred_smoothed_vals = interp.eval(k_grid)
        pred_smoothed_deriv = interp.deriv(k_grid)

        val_errs.append(np.abs(pred_smoothed_vals - true_smiles[i]))
        deriv_errs.append(np.abs(pred_smoothed_deriv - d_true[i]))

        # Roughness of the smoothed smile sampled on the lattice
        pred_rough_smoothed.append(
            second_diff_roughness(pred_smoothed_vals[None, :])[0]
        )

    val_errs = np.concatenate(val_errs, axis=0)
    deriv_errs = np.concatenate(deriv_errs, axis=0)
    pred_rough_smoothed = np.asarray(pred_rough_smoothed)

    return {
        "value_mae_smoothed": float(val_errs.mean()),
        "value_p95_smoothed": float(np.quantile(val_errs, 0.95)),
        "deriv_mae_smoothed": float(deriv_errs.mean()),
        "deriv_p95_smoothed": float(np.quantile(deriv_errs, 0.95)),
        "rough_true_mean": float(true_rough.mean()),
        "rough_pred_smoothed_mean": float(pred_rough_smoothed.mean()),
        "rough_ratio_smoothed_over_true": float(
            pred_rough_smoothed.mean() / (true_rough.mean() + 1e-12)
        ),
    }