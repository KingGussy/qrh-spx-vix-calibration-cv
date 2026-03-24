from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PackedNPZDatasetK(Dataset):
    """
    As in dataset but 
    If drop_last_x_col=True, drops the final column (u) once up front.
    """
    def __init__(
        self,
        packed_npz: str | Path,
        idx_npy: str | Path,
        norm_npz: str | Path,
        normalize: bool = True,
        drop_last_x_col: bool = False,
    ):
        self.packed_npz = Path(packed_npz)
        self.idx_npy = Path(idx_npy)
        self.norm_npz = Path(norm_npz)

        idx = np.load(self.idx_npy).astype(np.int64)

        with np.load(self.packed_npz, allow_pickle=False) as z:
            X = z["X"].astype(np.float32, copy=False)
            Y = z["Y"].astype(np.float32, copy=False)

        with np.load(self.norm_npz, allow_pickle=False) as z:
            X_mu = z["X_mu"].astype(np.float32, copy=False)
            X_sd = z["X_sd"].astype(np.float32, copy=False)
            Y_mu = z["Y_mu"].astype(np.float32, copy=False)
            Y_sd = z["Y_sd"].astype(np.float32, copy=False)

        X = X[idx]
        Y = Y[idx]

        if Y.ndim == 1:
            Y = Y[:, None]

        if normalize:
            X = (X - X_mu) / X_sd
            Y = (Y - Y_mu) / Y_sd

        if drop_last_x_col:
            X = X[:, :-1]

        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.Y = torch.from_numpy(Y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]