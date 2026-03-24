from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class PackedPaths:
    packed_npz: Path
    split_dir: Path
    norm_npz: Path


def load_norm(norm_path: str | Path):
    # grab the mean and sd stats
    norm_path = Path(norm_path)
    with np.load(norm_path, allow_pickle=False) as z:
        X_mu = z["X_mu"].astype(np.float32, copy=False)
        X_sd = z["X_sd"].astype(np.float32, copy=False)
        Y_mu = z["Y_mu"].astype(np.float32, copy=False)
        Y_sd = z["Y_sd"].astype(np.float32, copy=False)
    return X_mu, X_sd, Y_mu, Y_sd

class PackedNPZDataset(Dataset):
    """
    dataset view over the X, Y
    returns (per item):
      x: (d_in,)  torch.float32
      y: (d_out,) torch.float32
    """
    def __init__(
        self,
        packed_npz: str | Path,
        idx_npy: str | Path,
        norm_npz: Optional[str | Path] = None,
        normalise: bool = True,
    ):
        self.packed_npz = Path(packed_npz)
        self.idx = np.load(Path(idx_npy)).astype(np.int64)

        # load full arrays
        with np.load(self.packed_npz, allow_pickle=False) as z:
            self.X = z["X"].astype(np.float32, copy=False)
            self.Y = z["Y"].astype(np.float32, copy=False)

        self.normalise = normalise and (norm_npz is not None)
        if self.normalise:
            X_mu, X_sd, Y_mu, Y_sd = load_norm(norm_npz)
            self.X_mu = X_mu
            self.X_sd = X_sd
            self.Y_mu = Y_mu
            self.Y_sd = Y_sd
        else:
            self.X_mu = self.X_sd = self.Y_mu = self.Y_sd = None

        # quick sanity
        if self.idx.min() < 0 or self.idx.max() >= self.X.shape[0]:
            raise IndexError("split indices out of range")

    def __len__(self) -> int:
        return int(self.idx.size)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = int(self.idx[i])  # row id into packed arrays

        x = self.X[j]
        y = self.Y[j]

        if self.normalise:
            x = (x - self.X_mu) / self.X_sd
            y = (y - self.Y_mu) / self.Y_sd

        # Convert to torch tensors
        return torch.from_numpy(x), torch.from_numpy(y)


def make_datasets(
    packed_npz: str | Path,
    split_dir: str | Path,
    norm_npz: str | Path,
):
    split_dir = Path(split_dir)
    train = PackedNPZDataset(
        packed_npz=packed_npz,
        idx_npy=split_dir / "train_idx.npy",
        norm_npz=norm_npz,
        normalise=True,
    )
    val = PackedNPZDataset(
        packed_npz=packed_npz,
        idx_npy=split_dir / "val_idx.npy",
        norm_npz=norm_npz,
        normalise=True,
    )
    test = PackedNPZDataset(
        packed_npz=packed_npz,
        idx_npy=split_dir / "test_idx.npy",
        norm_npz=norm_npz,
        normalise=True,
    )
    return train, val, test


if __name__ == "__main__":
    # quick smoke test
    packed = r"C:\Users\angus\QuantProjects\QRH\data\synthetic_qrh_spx_vix\full_data.npz"
    split = r".\models\split"
    norm = r".\models\norm\norm.npz"

    train_ds, val_ds, test_ds = make_datasets(packed, split, norm)
    x0, y0 = train_ds[0]
    print("train len:", len(train_ds))
    print("x0 shape:", tuple(x0.shape), "y0 shape:", tuple(y0.shape))
    print("x0 first 5:", x0[:5])
    print("y0 first 5:", y0[:5])