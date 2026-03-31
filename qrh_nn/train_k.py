from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qrh_nn.train import set_seed
from qrh_nn.dataset_k import PackedNPZDatasetK
from qrh_nn.model_k import build_ctsk_model, ContinuousKConfig


# ============================================================
# Small helpers
# ============================================================

# def set_seed(seed: int) -> None:
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    norm_path: Path,
) -> dict:
    model.eval()

    with np.load(norm_path, allow_pickle=False) as z:
        Y_mu = torch.from_numpy(z["Y_mu"].astype(np.float32)).to(device)  # (1,) or (d_out,)
        Y_sd = torch.from_numpy(z["Y_sd"].astype(np.float32)).to(device)

    total_loss = 0.0
    n = 0
    abs_errors = []

    for Xb, Yb in dl:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        pred = model(Xb)
        loss = loss_fn(pred, Yb)

        bs = Xb.size(0)
        total_loss += float(loss) * bs
        n += bs

        pred_raw = pred * Y_sd + Y_mu
        y_raw = Yb * Y_sd + Y_mu
        abs_errors.append((pred_raw - y_raw).abs().detach().cpu())

    val_loss_norm = total_loss / max(1, n)

    abs_err = torch.cat(abs_errors, dim=0).reshape(-1)
    mae_iv = float(abs_err.mean())
    p95_iv = float(torch.quantile(abs_err, 0.95))
    p99_iv = float(torch.quantile(abs_err, 0.99))

    return {
        "val_loss_norm": val_loss_norm,
        "mae_iv": mae_iv,
        "p95_iv": p95_iv,
        "p99_iv": p99_iv,
    }


# ============================================================
# Main
# ============================================================

def main():
    # ------------------------
    # Paths
    # ------------------------
    repo_root = Path(__file__).resolve().parents[1]

    packed_path = repo_root / "data" / "synthetic_qrh_spx_vix" / "full_data_spxk.npz"
    split_dir = repo_root / "models" / "split"
    train_idx = split_dir / "train_idx_spxk.npy"
    val_idx   = split_dir / "val_idx_spxk.npy"
    test_idx  = split_dir / "test_idx_spxk.npy"

    norm_path = repo_root / "models" / "norm"/ "norm_spxk.npz"
    out_dir = repo_root / "models" / "full_mtp_spxk" / "modelA_run01"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Hyperparams
    # ------------------------
    seed = 123
    epochs = 40
    batch_size = 4096
    val_batch_size = 8192
    lr = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0
    huber_delta = 1.0
    num_workers = 0

    # Model A ignores positional coordinate u
    drop_last_x_col = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # Input dim after dropping u:
    d_in_model = 17

    # Slightly smaller/faster on CPU if needed
    cpu_only = (device.type == "cpu")

    cfg = ContinuousKConfig(
        d_in=d_in_model,
        d_out=1,
        d_model=128 if cpu_only else 256,
        d_hidden=256 if cpu_only else 512,
        n_blocks=4 if cpu_only else 6,
        dropout=0.0,
        use_layernorm=True,
        act="silu",
        out_act=None,
    )

    # ------------------------
    # Setup
    # ------------------------
    set_seed(seed)

    train_ds = PackedNPZDatasetK(
        packed_npz=packed_path,
        idx_npy=train_idx,
        norm_npz=norm_path,
        normalize=True,
        drop_last_x_col=drop_last_x_col,
    )
    val_ds = PackedNPZDatasetK(
        packed_npz=packed_path,
        idx_npy=val_idx,
        norm_npz=norm_path,
        normalize=True,
        drop_last_x_col=drop_last_x_col,
    )
    test_ds = PackedNPZDatasetK(
        packed_npz=packed_path,
        idx_npy=test_idx,
        norm_npz=norm_path,
        normalize=True,
        drop_last_x_col=drop_last_x_col,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_ctsk_model(
        d_in=d_in_model,
        d_model=cfg.d_model,
        d_hidden=cfg.d_hidden,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
        act=cfg.act,
    ).to(device)

    print("Using device:", device)
    print("Model param device:", next(model.parameters()).device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "val_batch_size": val_batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "huber_delta": huber_delta,
                "device": str(device),
                "packed_path": str(packed_path),
                "norm_path": str(norm_path),
                "train_idx": str(train_idx),
                "val_idx": str(val_idx),
                "test_idx": str(test_idx),
                "drop_last_x_col": drop_last_x_col,
                "model_cfg": asdict(cfg),
            },
            f,
            indent=2,
        )

    # ------------------------
    # Training loop
    # ------------------------
    best = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total = 0.0
        n = 0

        for Xb, Yb in train_dl:
            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(Xb)
                loss = loss_fn(pred, Yb)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = Xb.size(0)
            total += float(loss) * bs
            n += bs

        train_loss_norm = total / max(1, n)

        val_metrics = evaluate(
            model=model,
            dl=val_dl,
            device=device,
            loss_fn=loss_fn,
            norm_path=norm_path,
            drop_last_x_col=drop_last_x_col,
        )

        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss_norm": train_loss_norm,
            **val_metrics,
            "seconds": dt,
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_norm={train_loss_norm:.6f} | "
            f"val_norm={val_metrics['val_loss_norm']:.6f} | "
            f"val_MAE_IV={val_metrics['mae_iv']:.6g} | "
            f"val_P95_IV={val_metrics['p95_iv']:.6g} | "
            f"{dt:.1f}s"
        )

        if val_metrics["val_loss_norm"] < best:
            best = val_metrics["val_loss_norm"]
            best_epoch = epoch
            torch.save(
                {
                    "model_class": "ContinuousKModelA",
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "best_val_norm": best,
                },
                out_dir / "best.pt",
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_norm": best,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Done. Best epoch={best_epoch}, best val_norm={best:.6f}")
    print("Saved run to:", out_dir)


if __name__ == "__main__":
    main()