from __future__ import annotations

from pathlib import Path
import numpy as np


def pack_run(
    in_dir: str | Path,
    out_path: str | Path,
    pattern: str = "*.npz",
    param_keys=("a", "b", "c0", "lam", "eta"),
    z0_key="z0",
    T_key="T",
    spx_iv_key="spx_iv",
    vix_iv_key="vix_iv",
):
    """
    Input: path to shards, param cols (X) and target IV (y)
    Output: concatenated shards
    """
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    shards = sorted(in_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found in {in_dir} with pattern {pattern}")

    # validate row counts and sum
    total_rows = 0
    z0_dim = None
    iv_dim = None

    required_keys = set(param_keys) | {z0_key, T_key, spx_iv_key, vix_iv_key}

    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            missing = required_keys - set(z.files)
            if missing:
                raise KeyError(f"{p.name} missing keys: {sorted(missing)}; has: {z.files}")

            n = int(z[T_key].shape[0])
            total_rows += n

            # quick shape checking for each col
            for k in param_keys:
                if z[k].shape != (n,):
                    raise ValueError(f"{p.name}: key {k} has shape {z[k].shape}, expected {(n,)}")
            if z[z0_key].ndim != 2 or z[z0_key].shape[0] != n:
                raise ValueError(f"{p.name}: key {z0_key} bad shape {z[z0_key].shape}, expected (n, dz)")
            dz = int(z[z0_key].shape[1])
            if z0_dim is None:
                z0_dim = dz
            elif z0_dim != dz:
                raise ValueError(f"{p.name}: z0 dim {dz} differs from earlier {z0_dim}")
            if z[spx_iv_key].ndim != 2 or z[spx_iv_key].shape[0] != n:
                raise ValueError(f"{p.name}: spx_iv bad shape {z[spx_iv_key].shape}, expected (n, d_iv)")
            if z[vix_iv_key].ndim != 2 or z[vix_iv_key].shape[0] != n:
                raise ValueError(f"{p.name}: vix_iv bad shape {z[vix_iv_key].shape}, expected (n, d_iv)")

            d_iv = int(z[spx_iv_key].shape[1])
            if iv_dim is None:
                iv_dim = d_iv
            elif iv_dim != d_iv:
                raise ValueError(f"{p.name}: iv dim {d_iv} differs from earlier {iv_dim}")

            if z[vix_iv_key].shape[1] != d_iv:
                raise ValueError(f"{p.name}: vix_iv dim {z[vix_iv_key].shape[1]} != spx_iv dim {d_iv}")

    assert z0_dim is not None and iv_dim is not None
    d_in = len(param_keys) + z0_dim + 1
    d_out = 2 * iv_dim

    print(f"Found {len(shards)} shards")
    print(f"Total rows: {total_rows}")
    print(f"Input dim d_in: {d_in} (params {len(param_keys)} + z0 {z0_dim} + T 1)")
    print(f"Output dim d_out: {d_out} (spx_iv {iv_dim} + vix_iv {iv_dim})")

    # preallocate 
    X = np.empty((total_rows, d_in), dtype=np.float32)
    Y = np.empty((total_rows, d_out), dtype=np.float32)

    # Column name metadata
    x_cols = list(param_keys) + [f"z0_{j}" for j in range(z0_dim)] + ["T"]
    y_cols = [f"spx_iv_{j}" for j in range(iv_dim)] + [f"vix_iv_{j}" for j in range(iv_dim)]

    # fill shard by shard
    offset = 0
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            n = int(z[T_key].shape[0])

            params = np.stack([z[k] for k in param_keys], axis=1).astype(np.float32, copy=False)  # (n,5)
            z0 = z[z0_key].astype(np.float32, copy=False)                                        # (n,10)
            T = z[T_key].reshape(n, 1).astype(np.float32, copy=False)                            # (n,1)

            X_chunk = np.concatenate([params, z0, T], axis=1)                                    # (n,d_in)

            spx_iv = z[spx_iv_key].astype(np.float32, copy=False)                                # (n,15)
            vix_iv = z[vix_iv_key].astype(np.float32, copy=False)                                # (n,15)
            Y_chunk = np.concatenate([spx_iv, vix_iv], axis=1)                                   # (n,d_out)

            X[offset:offset+n] = X_chunk
            Y[offset:offset+n] = Y_chunk

            offset += n

    assert offset == total_rows

    # save the combined shard
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        x_cols=np.array(x_cols),
        y_cols=np.array(y_cols),
    )
    print(f"Saved packed dataset to: {out_path}")


if __name__ == "__main__":
    pack_run(
        in_dir=r"C:\Users\angus\data\RUN200K",
        out_path=r"C:\Users\angus\QuantProjects\QRH\training_data",
    )