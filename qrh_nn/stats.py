from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def compute_mean_std(A: np.ndarray, eps: float = 1e-8):   
    # small helper for mean and std that avoids division by zero
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = sd + eps
    return mu, sd

def main(
    packed_path: str | Path,
    train_idx_path: str | Path,
    out_dir: str | Path,
):
    packed_path = Path(packed_path)
    train_idx_path = Path(train_idx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx = np.load(train_idx_path).astype(np.int64)

    with np.load(packed_path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32, copy=False)  # (N,16)
        Y = z["Y"].astype(np.float32, copy=False)  # (N,30)
        x_cols = z["x_cols"] if "x_cols" in z.files else None
        y_cols = z["y_cols"] if "y_cols" in z.files else None

    # grab train data and stats
    X_tr = X[train_idx]  # (N_train,16)
    Y_tr = Y[train_idx]  # (N_train,30)

    X_mu, X_sd = compute_mean_std(X_tr)
    Y_mu, Y_sd = compute_mean_std(Y_tr)

    # save
    np.savez(
        out_dir / "norm_spxz.npz",
        X_mu=X_mu.astype(np.float32),
        X_sd=X_sd.astype(np.float32),
        Y_mu=Y_mu.astype(np.float32),
        Y_sd=Y_sd.astype(np.float32),
        x_cols=x_cols,
        y_cols=y_cols,
    )

    meta = {
        "packed_path": str(packed_path),
        "train_idx_path": str(train_idx_path),
        "N_total_rows": int(X.shape[0]),
        "N_train_rows": int(train_idx.size),
        "X_dim": int(X.shape[1]),
        "Y_dim": int(Y.shape[1]),
        "X_sd_min": float(X_sd.min()),
        "X_sd_max": float(X_sd.max()),
        "Y_sd_min": float(Y_sd.min()),
        "Y_sd_max": float(Y_sd.max()),
    }
    with open(out_dir / "norm_spxk_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_dir / "norm.npz")
    print("X_sd min/max:", float(X_sd.min()), float(X_sd.max()))
    print("Y_sd min/max:", float(Y_sd.min()), float(Y_sd.max()))


if __name__ == "__main__":
    main(
        packed_path=r"C:\Users\angus\QuantProjects\QRH\data\synthetic_qrh_spx_vix\full_data_spxk.npz",
        train_idx_path=r"C:\Users\angus\QuantProjects\QRH\models\split\train_idx_spxk.npy",
        out_dir=r"C:\Users\angus\QuantProjects\QRH\models\norm",
    )