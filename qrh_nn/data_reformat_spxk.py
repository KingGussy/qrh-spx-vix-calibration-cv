from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# SPX only
# we reformat the data to now include k as an input feature rather than a fixed k grid of targets
# we add k to X and expand within the splits (so no leakage) (15x more rows)
# we include a positional coordinate for k in the original grid as an optional feature

REPO = Path(".")

PACKED_IN = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
OUT_NPZ   = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data_spxk.npz"

# old split indices
SPLIT_DIR = REPO / "models" / "split"
TRAIN_IDX_IN = SPLIT_DIR / "train_idx.npy"
VAL_IDX_IN   = SPLIT_DIR / "val_idx.npy"
TEST_IDX_IN  = SPLIT_DIR / "test_idx.npy"

# new split indices
TRAIN_IDX_OUT = SPLIT_DIR / "train_idx_spxk.npy"
VAL_IDX_OUT   = SPLIT_DIR / "val_idx_spxk.npy"
TEST_IDX_OUT  = SPLIT_DIR / "test_idx_spxk.npy"

# SPX grid
K_SPX = np.array(
    [-0.15, -0.12, -0.10, -0.08, -0.05, -0.04, -0.03, -0.02, -0.01,
      0.00,  0.01,  0.02,  0.03,  0.04,  0.05],
    dtype=np.float32
)

# Positional coordinate u in [1, 15]
U_SPX = np.arange(1, len(K_SPX) + 1, dtype=np.float32)


# ============================================================
# HELPERS
# ============================================================

def expand_rows_spxk(X_old: np.ndarray, Y_old: np.ndarray):
    """
    X_old: (N, 16) = [params(5), z0(10), T]
    Y_old: (N, 30) = [spx_iv(15), vix_iv(15)]

    Returns:
      X_new: (N*15, 18) = old 16 cols + raw k + positional u
      Y_new: (N*15, 1)  = scalar SPX IV target
      old_to_new_start: (N,) where old row i expands to new rows
                        [old_to_new_start[i], ..., old_to_new_start[i]+14]
    """
    N = X_old.shape[0]
    nK = len(K_SPX)

    spx_old = Y_old[:, :nK]  # (N, 15)

    # Repeat old X 15 times
    X_rep = np.repeat(X_old, nK, axis=0)  # (N*15, 16)

    # Tile raw k and positional u
    k_col = np.tile(K_SPX, N).reshape(-1, 1)   # (N*15, 1)
    u_col = np.tile(U_SPX, N).reshape(-1, 1)   # (N*15, 1)

    # New X = old X + raw k + positional u (u last)
    X_new = np.concatenate([X_rep, k_col, u_col], axis=1).astype(np.float32, copy=False)

    # Targets: flatten SPX smile row-wise
    Y_new = spx_old.reshape(-1, 1).astype(np.float32, copy=False)

    old_to_new_start = np.arange(N, dtype=np.int64) * nK
    return X_new, Y_new, old_to_new_start


def expand_split_idx(old_idx: np.ndarray, nK: int = 15) -> np.ndarray:
    """
    Expand old row indices so that each old row i maps to
    [i*nK, i*nK+1, ..., i*nK+(nK-1)] in the new dataset.
    """
    old_idx = np.asarray(old_idx, dtype=np.int64)
    expanded = (old_idx[:, None] * nK + np.arange(nK, dtype=np.int64)[None, :]).reshape(-1)
    return expanded


# ============================================================
# MAIN
# ============================================================

def main():
    with np.load(PACKED_IN, allow_pickle=False) as z:
        X_old = z["X"].astype(np.float32, copy=False)
        Y_old = z["Y"].astype(np.float32, copy=False)

    X_new, Y_new, old_to_new_start = expand_rows_spxk(X_old, Y_old)

    # Load old split indices
    train_idx_old = np.load(TRAIN_IDX_IN).astype(np.int64)
    val_idx_old   = np.load(VAL_IDX_IN).astype(np.int64)
    test_idx_old  = np.load(TEST_IDX_IN).astype(np.int64)

    # Expand within each split (prevents leakage)
    train_idx_new = expand_split_idx(train_idx_old, nK=len(K_SPX))
    val_idx_new   = expand_split_idx(val_idx_old,   nK=len(K_SPX))
    test_idx_new  = expand_split_idx(test_idx_old,  nK=len(K_SPX))

    # Save new packed dataset
    np.savez_compressed(
        OUT_NPZ,
        X=X_new,
        Y=Y_new,
        k_spx=K_SPX,
        u_spx=U_SPX,
        old_to_new_start=old_to_new_start,
    )

    # Save expanded split indices
    np.save(TRAIN_IDX_OUT, train_idx_new)
    np.save(VAL_IDX_OUT, val_idx_new)
    np.save(TEST_IDX_OUT, test_idx_new)

    meta = {
        "source_packed": str(PACKED_IN),
        "output_packed": str(OUT_NPZ),
        "X_old_shape": list(X_old.shape),
        "Y_old_shape": list(Y_old.shape),
        "X_new_shape": list(X_new.shape),
        "Y_new_shape": list(Y_new.shape),
        "spx_k_grid": K_SPX.tolist(),
        "u_spx": U_SPX.tolist(),
        "input_columns": [
            "a", "b", "c0", "lam", "eta",
            "z0_0", "z0_1", "z0_2", "z0_3", "z0_4",
            "z0_5", "z0_6", "z0_7", "z0_8", "z0_9",
            "T", "k", "u"
        ],
        "target_columns": ["spx_iv"],
        "split_note": "Expanded within existing train/val/test splits to avoid leakage.",
    }

    meta_path = OUT_NPZ.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved new packed dataset:", OUT_NPZ)
    print("Saved expanded split indices:")
    print("  train:", TRAIN_IDX_OUT)
    print("  val  :", VAL_IDX_OUT)
    print("  test :", TEST_IDX_OUT)
    print("X_new shape:", X_new.shape)
    print("Y_new shape:", Y_new.shape)
    print("Meta saved to:", meta_path)


if __name__ == "__main__":
    main()