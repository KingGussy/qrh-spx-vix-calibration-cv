from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qrh_nn.dataset import make_datasets
from qrh_nn.model import ResMLP, ResMLPConfig


def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    norm_path: Path,
) -> dict:
    """
    Returns:
      - val_loss_norm: loss on normalised data
      - mae_iv: mean abs error in raw IV units (ie not normalised)
      - q95_abs_iv: 95% quantile abs error in raw IV units
    """
    model.eval()

    # Un-normalise Y
    with np.load(norm_path, allow_pickle=False) as z:
        Y_mu = torch.from_numpy(z["Y_mu"].astype(np.float32)).to(device)
        Y_sd = torch.from_numpy(z["Y_sd"].astype(np.float32)).to(device)

    total_loss = 0.0
    n = 0

    abs_errors = []  # collect per-batch abs errors in raw IV units

    for Xb, Yb in dl:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        pred = model(Xb)
        loss = loss_fn(pred, Yb)

        bs = Xb.size(0)
        total_loss += float(loss) * bs
        n += bs

        # Un-normalise to raw IV units
        pred_raw = pred * Y_sd + Y_mu
        y_raw = Yb * Y_sd + Y_mu
        abs_errors.append((pred_raw - y_raw).abs().detach().cpu())

    val_loss_norm = total_loss / n

    abs_err = torch.cat(abs_errors, dim=0).reshape(-1)  # flatten over samples and dims
    mae_iv = float(abs_err.mean())
    q95_abs_iv = float(torch.quantile(abs_err, 0.95))

    return {
        "val_loss_norm": val_loss_norm,
        "mae_iv": mae_iv,
        "q95_abs_iv": q95_abs_iv,
    }


def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    packed_path = repo_root / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
    split_dir = repo_root / "models" / "split"
    norm_path = repo_root / "models" / "norm" / "norm.npz"

    out_dir = repo_root / "models" / "full_mtp" / "checkpoints_resmlp_03"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparams
    seed = 123
    epochs = 100
    batch_size = 2048
    val_batch_size = 4096
    lr = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0
    huber_delta = 1.0  # in normalised units
    num_workers = 0

    # (smaller cpu cfg) - never needed 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cpu_only = (device.type == "cpu")
    if cpu_only:
        print("cuda not found- running on cpu")

    cfg = ResMLPConfig(
        d_in=16,
        d_out=30,
        d_model=128 if cpu_only else 256,
        d_hidden=256 if cpu_only else 512,
        n_blocks=4 if cpu_only else 6,
        dropout=0.0,
        use_layernorm=True,
        act="silu",
        out_act=None,
    )

    # Setup
    set_seed(seed)

    train_ds, val_ds, _test_ds = make_datasets(
        packed_npz=packed_path,
        split_dir=split_dir,
        norm_npz=norm_path,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_dl = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    model = ResMLP(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    use_amp = (device.type == "cuda")
    scaler = torch.GradScaler("cuda", enabled=use_amp)

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
                "split_dir": str(split_dir),
                "norm_path": str(norm_path),
                "model_cfg": asdict(cfg),
            },
            f,
            indent=2,
        )

    # TRAINING LOOP
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
            if epoch == 1: 
                print("Batch device:", Xb.device, Yb.device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(Xb)
                loss = loss_fn(pred, Yb)

            # scale grads to prevent underflow
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

        # Validate
        val_metrics = evaluate(model, val_dl, device, loss_fn, norm_path)
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
            f"val_Q95_IV={val_metrics['q95_abs_iv']:.6g} | "
            f"{dt:.1f}s"
        )

        # Save the best
        if val_metrics["val_loss_norm"] < best:
            best = val_metrics["val_loss_norm"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "best_val_norm": best,
                },
                out_dir / "best.pt",
            )

    # Save training history
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


if __name__ == "__main__":
    main()