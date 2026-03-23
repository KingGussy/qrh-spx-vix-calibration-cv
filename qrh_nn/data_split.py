from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# Computes the indices for the data split train/val/test


def expand_sample_ids(sample_ids: np.ndarray, T_LEN: int) -> np.ndarray:
    # For sample i, rows are [i*T_LEN, ..., i*T_LEN + (T_LEN-1)]
    return (sample_ids[:, None] * T_LEN + np.arange(T_LEN)[None, :]).reshape(-1).astype(np.int64)


def main(
    packed_path: str | Path,
    out_dir: str | Path,
    seed: int = 123,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    T_col: int = -1,      # last columns of the X cols
):
    packed_path = Path(packed_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(packed_path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32, copy=False)

    N = X.shape[0]
    T = X[:, T_col]
    T_unique = np.unique(T)
    T_LEN = int(T_unique.size) # will always be 4

    if N % T_LEN != 0:
        raise ValueError(f"N not divisible by {T_LEN}.")

    n_samples = N // T_LEN
    sample_ids = np.arange(n_samples, dtype=np.int64)

    # should already be randomised by extra layer here
    rng = np.random.default_rng(seed)
    rng.shuffle(sample_ids)

    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)
    n_test = n_samples - n_train - n_val
    if n_test <= 0:
        raise ValueError("Bad split fractions; test set is empty.")

    train_sid = sample_ids[:n_train]
    val_sid = sample_ids[n_train:n_train + n_val]
    test_sid = sample_ids[n_train + n_val:]

    train_idx = expand_sample_ids(train_sid, T_LEN)
    val_idx = expand_sample_ids(val_sid, T_LEN)
    test_idx = expand_sample_ids(test_sid, T_LEN)

    # save
    np.save(out_dir / "train_sid.npy", train_sid)
    np.save(out_dir / "val_sid.npy", val_sid)
    np.save(out_dir / "test_sid.npy", test_sid)

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    meta = {
        "packed_path": str(packed_path),
        "N_rows": N,
        "T_unique": [float(x) for x in T_unique.tolist()],
        "T_LEN": T_LEN,
        "n_samples": int(n_samples),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_train_samples": int(n_train),
        "n_val_samples": int(n_val),
        "n_test_samples": int(n_test),
        "n_train_rows": int(train_idx.size),
        "n_val_rows": int(val_idx.size),
        "n_test_rows": int(test_idx.size),
    }
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved split to:", out_dir)
    print("T_LEN =", T_LEN, "| n_samples =", n_samples)
    print("train rows =", train_idx.size, "val rows =", val_idx.size, "test rows =", test_idx.size)


if __name__ == "__main__":
    main(
        packed_path=r"C:\Users\angus\QuantProjects\QRH\data\synthetic_qrh_spx_vix\full_data.npz",
        out_dir=r".\models\split",
        seed=123,
        train_frac=0.80,
        val_frac=0.10,
    )