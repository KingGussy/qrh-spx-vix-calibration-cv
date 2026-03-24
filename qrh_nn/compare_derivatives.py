from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from qrh_nn.model import ResMLP, ResMLPConfig
from qrh_nn.model_k import ContinuousKConfig #build_ctsk_model

from qrh_nn.eval_utils import (
    K_SPX_FIXED as K_SPX, K_SPX_DENSE,
    CKPT_FIXED, CKPT_CTSK, NORM_FIXED, NORM_CTSK,
    load_model_and_norm,
)

from qrh_nn.interp_k import (
    HAVE_SCIPY,
    SmileInterpolator,
    predict_raw,
    second_diff_roughness,
    lattice_derivative,
    smoothed_value_and_derivative_metrics,
)


# ============================================================
# PATHS
# ============================================================

REPO = Path(__file__).resolve().parents[1]

PACKED_FIXED = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
FIXED_TEST_IDX = REPO / "models" / "split" / "test_idx.npy"

PACKED_CTSK = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data_spxk.npz"
CTSK_TEST_IDX = REPO / "models" / "split" / "test_idx_spxk.npy"

OUT_DIR = REPO / "models" / "derivatives_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOADERS
# ============================================================

def load_fixed_test():
    idx = np.load(FIXED_TEST_IDX).astype(np.int64)
    with np.load(PACKED_FIXED, allow_pickle=False) as z:
        X = z["X"].astype(np.float32, copy=False)
        Y = z["Y"].astype(np.float32, copy=False)
    return X[idx], Y[idx]

def load_ctsk_test():
    idx = np.load(CTSK_TEST_IDX).astype(np.int64)
    with np.load(PACKED_CTSK, allow_pickle=False) as z:
        X = z["X"].astype(np.float32, copy=False)
        Y = z["Y"].astype(np.float32, copy=False)
    return X[idx], Y[idx]


# ============================================================
# PREDICTION
# ============================================================


def regroup_pointwise_to_smiles(y_pointwise: np.ndarray, nK: int = 15) -> np.ndarray:
    """
    takes a 1-D y and reforms smiles
    y_pointwise: (N*15, 1) or (N*15,)
    returns: (N, 15)
    """
    if y_pointwise.ndim == 2 and y_pointwise.shape[1] == 1:
        y_pointwise = y_pointwise[:, 0] # (* , 1) -> (* ,)
    assert y_pointwise.shape[0] % nK == 0
    return y_pointwise.reshape(-1, nK)

### left wing diag
# def derivative_metrics_interior(
#     d_true: np.ndarray,
#     d_pred: np.ndarray,
#     n_trim_left: int = 1,
#     n_trim_right: int = 1,
# ) -> Dict[str, float]:
#     """
#     d_true, d_pred: (N, nK)
#     Computes derivative error excluding edge points.
#     """
#     if d_true.shape != d_pred.shape:
#         raise ValueError("Shapes must match.")

#     sl = slice(n_trim_left, d_true.shape[1] - n_trim_right)
#     err = np.abs(d_true[:, sl] - d_pred[:, sl]).reshape(-1)

#     return {
#         "mae_interior": float(err.mean()),
#         "p95_interior": float(np.quantile(err, 0.95)),
#         "p99_interior": float(np.quantile(err, 0.99)),
#         "n_points_used": int(err.size),
#     }


# def bucket_value_metrics(
#     true_smiles: np.ndarray,
#     pred_smiles: np.ndarray,
#     k_grid: np.ndarray,
# ) -> Dict[str, Dict[str, float]]:
#     """
#     Split value errors into left wing / near ATM / right wing.
#     """
#     abs_err = np.abs(pred_smiles - true_smiles)

#     left_mask = k_grid <= -0.08
#     atm_mask = (k_grid > -0.08) & (k_grid < 0.02)
#     right_mask = k_grid >= 0.02

#     def summarise(mask: np.ndarray) -> Dict[str, float]:
#         e = abs_err[:, mask].reshape(-1)
#         return {
#             "mae": float(e.mean()),
#             "p95": float(np.quantile(e, 0.95)),
#             "p99": float(np.quantile(e, 0.99)),
#             "n_points": int(e.size),
#         }

#     return {
#         "left_wing": summarise(left_mask),
#         "near_atm": summarise(atm_mask),
#         "right_wing": summarise(right_mask),
#     }


# def count_left_edge_outliers(
#     true_smiles: np.ndarray,
#     pred_smiles: np.ndarray,
#     first_k_only: bool = True,
#     threshold_abs_err: float = 0.05,
# ) -> Dict[str, float]:
#     """
#     Counts large misses at the extreme left edge.
#     threshold_abs_err in raw IV units.
#     """
#     if first_k_only:
#         err = np.abs(pred_smiles[:, 0] - true_smiles[:, 0])
#     else:
#         err = np.abs(pred_smiles[:, :2] - true_smiles[:, :2]).max(axis=1)

#     return {
#         "threshold_abs_err": float(threshold_abs_err),
#         "n_rows_flagged": int((err > threshold_abs_err).sum()),
#         "frac_rows_flagged": float((err > threshold_abs_err).mean()),
#         "max_left_edge_abs_err": float(err.max()),
#         "p95_left_edge_abs_err": float(np.quantile(err, 0.95)),
#     }


# ============================================================
# AUTOGRAD FOR CTS K MODEL DERIVATIVE
# ============================================================

def ctsk_deriv(
    model: torch.nn.Module,
    X_full_raw: np.ndarray,
    X_mu: np.ndarray,
    X_sd: np.ndarray,
    Y_mu: np.ndarray,
    Y_sd: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Computes d sigma_raw / d k_raw for cts model using autograd.
    X_full_raw: (N*15, 18) including u
    Returns shape: (N*15,)
    """
    k_col_full = 16  # [old16, k, ...] => k at index 16

    Y_scale = float(np.squeeze(Y_sd))
    Xk_scale = float(X_sd[k_col_full])

    grads_all = []

    for i in range(0, len(X_full_raw), batch_size):
        xb_full = X_full_raw[i:i+batch_size]
        xb_norm = (xb_full - X_mu[None, :]) / X_sd[None, :]
        # xb_norm = xb_norm_full[:, :-1]  # drop u for cts model input (we now do this on input)

        xt = torch.tensor(xb_norm, dtype=torch.float32, device=device, requires_grad=True)
        y_norm = model(xt).squeeze(-1)  # (B,)

        # gradient wrt normalized k input (which is column 16 after dropping only u)
        grad_norm = torch.autograd.grad(
            outputs=y_norm.sum(),
            inputs=xt,
            create_graph=False,
            retain_graph=False,
        )[0][:, 16]

        # chain rule:
        # y_raw = Y_scale * y_norm + Y_mu
        # x_norm_k = (k_raw - mu_k)/Xk_scale
        # d y_raw / d k_raw = Y_scale * d y_norm / d x_norm_k * (1 / Xk_scale)
        grad_raw = Y_scale * grad_norm / Xk_scale
        grads_all.append(grad_raw.detach().cpu().numpy())

    return np.concatenate(grads_all, axis=0)


# ============================================================
# PLOTTING
# ============================================================

def plot_smile_compare(
    k_dense: np.ndarray,
    k_grid: np.ndarray,
    true_row: np.ndarray,
    fixedk_dense: np.ndarray,
    ctsk_dense: np.ndarray,
    ctsk_lattice: np.ndarray,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(7, 4))
    plt.plot(k_grid, true_row, "x", color="tomato", label="true lattice")
    plt.plot(k_dense, fixedk_dense, "-", color="goldenrod", label="Fixed-k")
    plt.plot(k_dense, ctsk_dense, "-", color="mediumaquamarine", label="Continuous-k")
    #plt.plot(k_grid, ctsk_lattice, "x", label="cts modellattice preds")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("IV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_derivative_compare(
    k_dense: np.ndarray,
    k_grid: np.ndarray,
    d_true_lattice: np.ndarray,
    fixedk_dense_deriv: np.ndarray,
    ctsk_dense_deriv: np.ndarray,
    ctsk_lattice_deriv: np.ndarray,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(7, 4))
    plt.plot(k_grid, d_true_lattice, "x", color="tomato", label="finite-difference derivative")
    plt.plot(k_dense, fixedk_dense_deriv, "-", color="goldenrod", label="Fixed k smooth interpolation")
    plt.plot(k_dense, ctsk_dense_deriv, "-", color="mediumaquamarine", label="Continuous-k AAD")
    #plt.plot(k_grid, ctsk_lattice_deriv, "x", label="cts model deriv @ lattice")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("d sigma / d k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fixed-grid model + data
    fixed_model, fixed_cfg, fixed_X_mu, fixed_X_sd, fixed_Y_mu, fixed_Y_sd = load_model_and_norm("fixed", device) #load_fixed_model(device)
    X_fixed_test, Y_fixed_test = load_fixed_test()
    Y_fixed_pred = predict_raw(fixed_model, X_fixed_test, fixed_X_mu, fixed_X_sd, fixed_Y_mu, fixed_Y_sd, device) # predict_fixed_raw(fixed_model, X_fixed_test, fixed_X_mu, fixed_X_sd, fixed_Y_mu, fixed_Y_sd, device)

    spx_true = Y_fixed_test[:, :15]
    spx_fixed_pred = Y_fixed_pred[:, :15]

    # fixed-k smoothed baseline metrics (same as interp_k baseline)
    fixedk_metrics = smoothed_value_and_derivative_metrics(
        spx_true, spx_fixed_pred, K_SPX, method="spline", smooth=True
    ) if HAVE_SCIPY else {"warning": "scipy not available"}

    # Load cts model  + pointwise test data
    new_model, new_cfg, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd = load_model_and_norm("ctsk", device) #load_new_model(device)
    X_new_test, Y_new_test = load_ctsk_test()

    # Y_new_pred = predict_new_pointwise_raw(
    #     new_model, X_new_test, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device
    # )
    X_new_test = X_new_test[:, :-1]
    Y_new_pred = predict_raw(new_model, X_new_test, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device)

    # Pointwise metrics
    abs_err_pointwise = np.abs(Y_new_pred.reshape(-1) - Y_new_test.reshape(-1))
    ctsk_pointwise_metrics = {
        "value_mae_pointwise": float(abs_err_pointwise.mean()),
        "value_p95_pointwise": float(np.quantile(abs_err_pointwise, 0.95)),
        "value_p99_pointwise": float(np.quantile(abs_err_pointwise, 0.99)),
    }

    # Regroup pointwise into smile rows
    spx_ctsk = regroup_pointwise_to_smiles(Y_new_pred, nK=15)  # (N_test, 15)
    assert spx_ctsk.shape == spx_true.shape

    # Lattice roughness
    rough_true = second_diff_roughness(spx_true)
    rough_ctsk = second_diff_roughness(spx_ctsk)

    # cts model derivative via autograd at lattice points
    d_ctsk_lattice = regroup_pointwise_to_smiles(
        ctsk_deriv(
            new_model, X_new_test, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device
        ),
        nK=15
    )
    d_true_lattice = lattice_derivative(spx_true, K_SPX)

    d_abs_err = np.abs(d_ctsk_lattice - d_true_lattice).reshape(-1)
    ctsk_deriv_metrics = {
        "deriv_mae_autograd_vs_trueFD": float(d_abs_err.mean()),
        "deriv_p95_autograd_vs_trueFD": float(np.quantile(d_abs_err, 0.95)),
        "rough_true_mean": float(rough_true.mean()),
        "rough_ctsk_mean": float(rough_ctsk.mean()),
        "rough_ratio_ctsk_over_true": float(rough_ctsk.mean() / (rough_true.mean() + 1e-12)),
    }

    # Compare against fixed-k baseline numbers
    compare = {
        "fixedk_smoothed_baseline_spx": fixedk_metrics,
        "ctsk_pointwise": ctsk_pointwise_metrics,
        "ctsk_derivative": ctsk_deriv_metrics,
    }

    with open(OUT_DIR / "metrics_compare.json", "w") as f:
        json.dump(compare, f, indent=2)

    # --------------------------------------------------------
    # Plots on same axes
    # --------------------------------------------------------
    example_rows = [0, min(10, len(spx_true)-1), min(100, len(spx_true)-1), min(400, len(spx_true)-1), min(1000, len(spx_true)-1)] # [15, min(30, len(spx_true)-1), min(115, len(spx_true)-1), min(415, len(spx_true)-1), min(900, len(spx_true)-1)]

    for idx in example_rows:
        T_val = float(X_fixed_test[idx, -1])

        # fixed-k smoothed curve
        interp_fixed = SmileInterpolator(K_SPX, spx_fixed_pred[idx], method="spline", smooth=True)
        fixedk_dense = interp_fixed.eval(K_SPX_DENSE)
        fixedk_dense_deriv = interp_fixed.deriv(K_SPX_DENSE)

        # cts model dense curve by querying dense k for this same (omega,z0,T)
        base16 = X_fixed_test[idx]  # old-style input: [omega,z0,T]
        X_dense = np.zeros((len(K_SPX_DENSE), 18), dtype=np.float32)
        X_dense[:, :16] = base16[None, :]
        X_dense[:, 16] = K_SPX_DENSE
        X_dense[:, 17] = np.interp(K_SPX_DENSE, K_SPX, np.arange(1, 16, dtype=np.float32))  # dummy u for alignment, immmediately dropped
        X_dense = X_dense[:, :-1] # immediately drop unused positional encoding (left from diagnostics)

        # ctsk_vals = predict_new_pointwise_raw(
        #     new_model, X_dense, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device
        # ).reshape(-1)
        ctsk_vals = predict_raw(new_model, X_dense, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device).reshape(-1)

        ctsk_dense_deriv = ctsk_deriv(
            new_model, X_dense, new_X_mu, new_X_sd, new_Y_mu, new_Y_sd, device, batch_size=2048
        )

        plot_smile_compare(
            k_dense=K_SPX_DENSE,
            k_grid=K_SPX,
            true_row=spx_true[idx],
            fixedk_dense=fixedk_dense,
            ctsk_dense=ctsk_vals,
            ctsk_lattice=spx_ctsk[idx],
            title=f"SPX smile compare | row {idx} | T={T_val:g}",
            out_path=OUT_DIR / f"spx_smile_compare_row{idx}.png",
        )

        plot_derivative_compare(
            k_dense=K_SPX_DENSE,
            k_grid=K_SPX,
            d_true_lattice=d_true_lattice[idx],
            fixedk_dense_deriv=fixedk_dense_deriv,
            ctsk_dense_deriv=ctsk_dense_deriv,
            ctsk_lattice_deriv=d_ctsk_lattice[idx],
            title=f"SPX derivative compare | row {idx} | T={T_val:g}",
            out_path=OUT_DIR / f"spx_deriv_compare_row{idx}.png",
        )

    # --------------------------------------------------------
    # Extra diagnostics: interior derivative + wing buckets
    # --------------------------------------------------------
    # fixed-k derivative at lattice points from smoothed spline
    # d_fixedk_lattice = []
    # for i in range(len(spx_fixed_pred)):
    #     interp_fixed = SmileInterpolator(K_SPX, spx_fixed_pred[i], method="spline", smooth=True)
    #     d_fixedk_lattice.append(interp_fixed.deriv(K_SPX))
    # d_fixedk_lattice = np.asarray(d_fixedk_lattice)

    # extra_diagnostics = {
    #     "fixedk_derivative_interior": derivative_metrics_interior(
    #         d_true_lattice, d_fixedk_lattice, n_trim_left=1, n_trim_right=1
    #     ),
    #     "ctsk_derivative_interior": derivative_metrics_interior(
    #         d_true_lattice, d_ctsk_lattice, n_trim_left=1, n_trim_right=1
    #     ),
    #     "fixedk_value_buckets": bucket_value_metrics(
    #         spx_true, spx_fixed_pred, K_SPX
    #     ),
    #     "ctsk_value_buckets": bucket_value_metrics(
    #         spx_true, spx_ctsk, K_SPX
    #     ),
    #     "fixedk_left_edge_outliers": count_left_edge_outliers(
    #         spx_true, spx_fixed_pred, first_k_only=True, threshold_abs_err=0.05
    #     ),
    #     "ctsk_left_edge_outliers": count_left_edge_outliers(
    #         spx_true, spx_ctsk, first_k_only=True, threshold_abs_err=0.05
    #     ),
    # }

    # compare["extra_diagnostics"] = extra_diagnostics

    print("Saved comparison outputs to:", OUT_DIR)
    print(json.dumps(compare, indent=2))


if __name__ == "__main__":
    main()