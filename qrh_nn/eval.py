from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as qrh_nn

import matplotlib.pyplot as plt

from qrh_nn.model import ResMLP, ResMLPConfig
from qrh_nn.eval_utils import (
    ensure_dir, load_npz, 
    load_model_and_norm,
    predict_norm, 
    normalise, unnormalise,
    error_stats, 
    infer_grid_dims, infer_T_unique, per_grid_mean_abs_error,
    split_spx_vix, 
    plot_heatmap
)

# ===== Main Eval ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=str, default=r".\data\synthetic_qrh_spx_vix\full_data.npz")
    ap.add_argument("--split_dir", type=str, default=r".\models\split")
    ap.add_argument("--norm", type=str, default=r".\models\norm\norm.npz")
    ap.add_argument("--ckpt", type=str, default=r".\models\full_mtp\checkpoints_resmlp_03\best.pt")
    ap.add_argument("--out_dir", type=str, default=r".\models\full_mtp\eval_final")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--n_smile_examples", type=int, default=6)
    ap.add_argument("--do_calib_sanity", action="store_true")
    ap.add_argument("--calib_trials", type=int, default=10)
    ap.add_argument("--calib_steps", type=int, default=200)
    ap.add_argument("--calib_lr", type=float, default=5e-2)
    args = ap.parse_args()

    packed_path = Path(args.packed)
    split_dir = Path(args.split_dir)
    norm_path = Path(args.norm)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)
    ensure_dir(out_dir / "plots")

    device = torch.device(args.device)

    # ---- Load packed data ----
    packed = load_npz(packed_path)
    X = packed["X"].astype(np.float32, copy=False)
    Y = packed["Y"].astype(np.float32, copy=False)

    # ---- Load splits ----
    test_idx = np.load(split_dir / "test_idx.npy").astype(np.int64)
    val_idx_path = split_dir / "val_idx.npy"
    val_idx = np.load(val_idx_path).astype(np.int64) if val_idx_path.exists() else None

    # We'll evaluate primarily on test
    def subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx]

    X_test = subset(X, test_idx)
    Y_test = subset(Y, test_idx)

    # # ---- Infer grid ----
    T_unique = infer_T_unique(X, T_col=-1)
    nT, nK   = infer_grid_dims(X, Y, T_col=-1)

    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("fixed", device=device)

    Xn_test = normalise(X_test, X_mu, X_sd)
    Yn_pred_test = predict_norm(model, Xn_test, device=device, batch_size=args.batch_size)
    Y_pred_test = unnormalise(Yn_pred_test, Y_mu, Y_sd)


    # ---- Errors ----
    abs_err_test = np.abs(Y_pred_test - Y_test)  # raw IV units

    # Overall and split (SPX vs VIX)
    spx_err, vix_err = split_spx_vix(abs_err_test, nK)
    metrics = {
        "overall": error_stats(abs_err_test),
        "spx": error_stats(spx_err),
        "vix": error_stats(vix_err),
        "meta": {
            "n_test_rows": int(test_idx.size),
            "nT": int(nT),
            "nK": int(nK),
            "T_unique": [float(x) for x in T_unique.tolist()],
            "ckpt": str(ckpt_path),
            "packed": str(packed_path),
            "device": str(device),
        },
    }

    # per point mean abs err
    spx_grid, vix_grid = per_grid_mean_abs_error(abs_err_test, X_test, T_unique, nK, T_col=-1)
    # metrics["_"] = {
    #     "spx_mean_abs_error_grid_shape": [int(nT), int(nK)],
    #     "vix_mean_abs_error_grid_shape": [int(nT), int(nK)],
    # }

    # Relative error grid (abs error / mean IV) 
    spx_true, vix_true = split_spx_vix(Y_test, nK)
    spx_rel = spx_grid / (np.maximum(1e-6, np.mean(spx_true.reshape(-1, nK), axis=0))[None, :])
    vix_rel = vix_grid / (np.maximum(1e-6, np.mean(vix_true.reshape(-1, nK), axis=0))[None, :])


    # Basic sanity: negative IVs?
    metrics["sanity"] = {
        "frac_pred_iv_negative": float((Y_pred_test < 0).mean()),
        "pred_iv_min": float(Y_pred_test.min()),
        "pred_iv_max": float(Y_pred_test.max()),
    }

    # ---- Save metrics ----
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plots ----
    plot_dir = out_dir / "plots"

    # Heatmaps: abs error
    plot_heatmap(spx_grid, "SPX mean abs IV error", plot_dir / "heat_spx_abs.png")
    plot_heatmap(vix_grid, "VIX mean abs IV error", plot_dir / "heat_vix_abs.png")

    # Heatmaps: relative error
    plot_heatmap(spx_rel, "SPX mean abs IV error / mean IV (relative)", plot_dir / "heat_spx_rel.png")
    plot_heatmap(vix_rel, "VIX mean abs IV error / mean IV (relative)", plot_dir / "heat_vix_rel.png")

    print("Saved eval outputs to:", out_dir)
    print("Key metrics (test):")
    print("  overall:", metrics["overall"])
    print("  spx    :", metrics["spx"])
    print("  vix    :", metrics["vix"])
    print("See plots in:", plot_dir)

if __name__ == "__main__":
    main()