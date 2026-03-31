from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from src.qrh_sim.pricing_utils import norm_cdf, black_call_forward

from qrh_nn.dataset import make_datasets
from qrh_nn.model import ResMLP, ResMLPConfig

from qrh_nn.eval_utils import K_SPX_FIXED
from qrh_nn.train import set_seed


def load_norm_stats(norm_path: Path, device: torch.device):
    with np.load(norm_path, allow_pickle=False) as z:
        X_mu = torch.from_numpy(z["X_mu"].astype(np.float32)).to(device)
        X_sd = torch.from_numpy(z["X_sd"].astype(np.float32)).to(device)
        Y_mu = torch.from_numpy(z["Y_mu"].astype(np.float32)).to(device)
        Y_sd = torch.from_numpy(z["Y_sd"].astype(np.float32)).to(device)
    return X_mu, X_sd, Y_mu, Y_sd


# torch.tensor variants of our pricing utils
def norm_cdf_t(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def black_call_from_forward_t(
    F: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-12
    sigma = torch.clamp(sigma, min=eps)
    T = torch.clamp(T, min=eps)

    sqrtT = torch.sqrt(T)
    volT = sigma * sqrtT

    d1 = (torch.log(torch.clamp(F / K, min=eps)) + 0.5 * volT * volT) / torch.clamp(volT, min=eps)
    d2 = d1 - volT

    return F * norm_cdf_t(d1) - K * norm_cdf_t(d2)


# ============================================================
# Convexity penalty on a NONUNIFORM strike grid
# ============================================================

def spx_convexity_penalty_from_pred(
    pred_norm: torch.Tensor,      # (B, 30) normalized outputs
    Xb_norm: torch.Tensor,        # (B, 16) normalized inputs
    X_mu: torch.Tensor,
    X_sd: torch.Tensor,
    Y_mu: torch.Tensor,
    Y_sd: torch.Tensor,
    k_grid_spx: torch.Tensor,     # (15,)
) -> torch.Tensor:
    """
    Applies convexity penalty to SPX smile only (first 15 outputs).
    Uses correct nonuniform-strike discrete convexity test:
      slopes in strike must be nondecreasing.
    """
    # Un-normalise outputs
    pred_raw = pred_norm * Y_sd + Y_mu         # (B, 30)
    spx_iv = pred_raw[:, : len(k_grid_spx)]    # (B, 15)

    # Un-normalise inputs to recover maturity T
    X_raw = Xb_norm * X_sd + X_mu
    T = X_raw[:, -1]                           # assumes last feature is T
    T = torch.clamp(T, min=1e-8)               # (B,)

    # Forward-normalised setting: F = 1
    F = torch.ones_like(spx_iv)
    K = torch.exp(k_grid_spx).unsqueeze(0).expand_as(spx_iv)  # (B, 15)
    T_mat = T.unsqueeze(1).expand_as(spx_iv)                  # (B, 15)

    C = black_call_from_forward_t(F, K, T_mat, spx_iv)        # (B, 15)

    dK = K[:, 1:] - K[:, :-1]                                 # (B, 14)
    slopes = (C[:, 1:] - C[:, :-1]) / torch.clamp(dK, min=1e-12)   # (B, 14)

    # Convexity requires slopes[:, 1:] >= slopes[:, :-1]
    conv_viols = torch.relu(slopes[:, :-1] - slopes[:, 1:])   # (B, 13)

    return torch.mean(conv_viols ** 2)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    fit_loss_fn: nn.Module,
    norm_path: Path,
    k_grid_spx: np.ndarray,
    lambda_conv: float,
) -> dict:
    """
    Returns:
      - val_fit_loss_norm: fit loss on normalised data
      - val_conv_penalty: convexity penalty term
      - val_total_loss: fit + lambda * conv
      - mae_iv: mean abs error in raw IV units
      - q95_abs_iv: 95% quantile abs error in raw IV units
    """
    model.eval()

    X_mu, X_sd, Y_mu, Y_sd = load_norm_stats(norm_path, device)
    k_grid_spx_t = torch.from_numpy(k_grid_spx.astype(np.float32)).to(device)

    total_fit = 0.0
    total_conv = 0.0
    n = 0
    abs_errors = []

    for Xb, Yb in dl:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)

        pred = model(Xb)
        fit_loss = fit_loss_fn(pred, Yb)
        conv_pen = spx_convexity_penalty_from_pred(
            pred, Xb, X_mu, X_sd, Y_mu, Y_sd, k_grid_spx_t
        )

        bs = Xb.size(0)
        total_fit += float(fit_loss) * bs
        total_conv += float(conv_pen) * bs
        n += bs

        pred_raw = pred * Y_sd + Y_mu
        y_raw = Yb * Y_sd + Y_mu
        abs_errors.append((pred_raw - y_raw).abs().detach().cpu())

    val_fit_loss_norm = total_fit / max(1, n)
    val_conv_penalty = total_conv / max(1, n)
    val_total_loss = val_fit_loss_norm + lambda_conv * val_conv_penalty

    abs_err = torch.cat(abs_errors, dim=0).reshape(-1)
    mae_iv = float(abs_err.mean())
    q95_abs_iv = float(torch.quantile(abs_err, 0.95))

    return {
        "val_fit_loss_norm": val_fit_loss_norm,
        "val_conv_penalty": val_conv_penalty,
        "val_total_loss": val_total_loss,
        "mae_iv": mae_iv,
        "q95_abs_iv": q95_abs_iv,
    }


def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    packed_path = repo_root / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
    split_dir = repo_root / "models" / "split"
    norm_path = repo_root / "models" / "norm" / "norm.npz"

    out_dir = repo_root / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam50"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparams
    seed = 123
    epochs = 100
    batch_size = 2048
    val_batch_size = 4096
    lr = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0
    huber_delta = 1.0
    lambda_conv = 5.0
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cpu_only = (device.type == "cpu")
    if cpu_only:
        print("cuda not found - running on cpu")

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

    set_seed(seed)

    train_ds, val_ds, _test_ds = make_datasets(
        packed_npz=packed_path,
        split_dir=split_dir,
        norm_npz=norm_path,
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

    model = ResMLP(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    fit_loss_fn = nn.HuberLoss(delta=huber_delta)

    X_mu, X_sd, Y_mu, Y_sd = load_norm_stats(norm_path, device)
    k_grid_spx_t = torch.from_numpy(K_SPX_FIXED.astype(np.float32)).to(device)

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
                "lambda_conv": lambda_conv,
                "device": str(device),
                "packed_path": str(packed_path),
                "split_dir": str(split_dir),
                "norm_path": str(norm_path),
                "k_grid_spx": K_SPX_FIXED.tolist(),
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

        total_fit = 0.0
        total_conv = 0.0
        n = 0

        for Xb, Yb in train_dl:
            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)

            if epoch == 1:
                print("Batch device:", Xb.device, Yb.device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(Xb)
                fit_loss = fit_loss_fn(pred, Yb)
                conv_pen = spx_convexity_penalty_from_pred(
                    pred, Xb, X_mu, X_sd, Y_mu, Y_sd, k_grid_spx_t
                )
                loss = fit_loss + lambda_conv * conv_pen

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = Xb.size(0)
            total_fit += float(fit_loss) * bs
            total_conv += float(conv_pen) * bs
            n += bs

        train_fit_loss_norm = total_fit / n
        train_conv_penalty = total_conv / n
        train_total_loss = train_fit_loss_norm + lambda_conv * train_conv_penalty

        # Validate
        val_metrics = evaluate(
            model=model,
            dl=val_dl,
            device=device,
            fit_loss_fn=fit_loss_fn,
            norm_path=norm_path,
            k_grid_spx=K_SPX_FIXED,
            lambda_conv=lambda_conv,
        )
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_fit_loss_norm": train_fit_loss_norm,
            "train_conv_penalty": train_conv_penalty,
            "train_total_loss": train_total_loss,
            **val_metrics,
            "seconds": dt,
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_fit={train_fit_loss_norm:.6f} | "
            f"train_conv={train_conv_penalty:.6f} | "
            f"train_total={train_total_loss:.6f} | "
            f"val_fit={val_metrics['val_fit_loss_norm']:.6f} | "
            f"val_conv={val_metrics['val_conv_penalty']:.6f} | "
            f"val_total={val_metrics['val_total_loss']:.6f} | "
            f"val_MAE_IV={val_metrics['mae_iv']:.6g} | "
            f"val_Q95_IV={val_metrics['q95_abs_iv']:.6g} | "
            f"{dt:.1f}s"
        )

        if val_metrics["val_total_loss"] < best:
            best = val_metrics["val_total_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "best_val_total": best,
                    "lambda_conv": lambda_conv,
                    "k_grid_spx": K_SPX_FIXED.tolist(),
                },
                out_dir / "best.pt",
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_total": best,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Done. Best epoch={best_epoch}, best val_total={best:.6f}")


if __name__ == "__main__":
    main()