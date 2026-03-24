from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import torch

# Param names and ranges:
PARAM_NAMES = ["a", "b", "c", "lam", "eta"] + [f"z0_{i}" for i in range(10)]

LOW = np.array([0.1, 0.01, 0.0001, 0.5, 1.0] + [-0.5] * 10, dtype=np.float32)
HIGH = np.array([0.6, 0.5, 0.03, 2.5, 1.5] + [0.5] * 10, dtype=np.float32)

V1_N_STARTS = 6
V1_LBFGS_STEPS = 80

from qrh_nn.model import ResMLP as ResMLP_fixed
from qrh_nn.model import ResMLPConfig as ResMLPConfig_fixed

from qrh_nn.model_k import ContinuousKModel as ResMLP_ctsk
from qrh_nn.model_k import ContinuousKConfig as ResMLPConfig_ctsk
from qrh_nn.eval_utils import (
    CKPT_FIXED, NORM_FIXED, CKPT_CTSK, NORM_CTSK,
    K_SPX_FIXED, K_VIX_FIXED,
    load_model_and_norm,
    _device_or_default, _as_t
)



def theta_from_u_bounded(u: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(LOW, dtype=torch.float32, device=u.device)
    high = torch.tensor(HIGH, dtype=torch.float32, device=u.device)
    return low + (high - low) * torch.sigmoid(u)


def u_from_theta_bounded(theta_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = (theta_np - LOW) / (HIGH - LOW)
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y)).astype(np.float32)


# random starting params
def make_default_starts(n_starts: int = V1_N_STARTS, seed: int = 123) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [((LOW + HIGH) / 2.0).astype(np.float32)]
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(LOW, HIGH).astype(np.float32))
    return starts



def linear_interp(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    w = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)


def _rmse_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


# metrics and ...
def smile_error_metrics(iv_hat: np.ndarray, iv_obs: np.ndarray) -> Dict[str, float]:
    iv_hat = np.asarray(iv_hat, dtype=np.float64)
    iv_obs = np.asarray(iv_obs, dtype=np.float64)

    if iv_hat.size == 0 or iv_obs.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "max_abs_err": float("nan"),
            "mean_signed_err": float("nan"),
            "n_points": 0,
        }

    err = iv_hat - iv_obs
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
        "mean_signed_err": float(np.mean(err)),
        "n_points": int(err.size),
    }

# generic fixed smile predictor
def _predict_fixed_smile_generic(
    model: torch.nn.Module,
    theta_raw: torch.Tensor, T_raw: float, 
    k_obs_spx: torch.Tensor, k_obs_vix: torch.Tensor,
    X_mu: np.ndarray, X_sd: np.ndarray,
    Y_mu: np.ndarray, Y_sd: np.ndarray,
    device: torch.device,
):
    X_mu_t, X_sd_t  = _as_t(X_mu, device), _as_t(X_sd, device)
    Y_mu_t, Y_sd_t  = _as_t(Y_mu, device), _as_t(Y_sd, device)

    x_raw = torch.zeros((1, 16), dtype=torch.float32, device=device)
    x_raw[0, :15] = theta_raw
    x_raw[0, 15] = float(T_raw)

    x_n = (x_raw - X_mu_t[None, :]) / X_sd_t[None, :]
    y_n = model(x_n)
    y_raw = y_n * Y_sd_t[None, :] + Y_mu_t[None, :]

    spx_fixed = y_raw[0, :15]
    vix_fixed = y_raw[0, 15:]

    k_spx_t = _as_t(K_SPX_FIXED, device)
    k_vix_t = _as_t(K_VIX_FIXED, device)

    spx_hat = linear_interp(k_spx_t, spx_fixed, k_obs_spx)
    vix_hat = linear_interp(k_vix_t, vix_fixed, k_obs_vix)
    return spx_hat, vix_hat

def predict_fixedk_joint_smiles_from_theta(
    model: torch.nn.Module,
    theta_raw: torch.Tensor, T_raw: float, 
    k_obs_spx: torch.Tensor, k_obs_vix: torch.Tensor,
    X_mu: np.ndarray, X_sd: np.ndarray,
    Y_mu: np.ndarray, Y_sd: np.ndarray,
    device: torch.device,
):
    return _predict_fixed_smile_generic(model, theta_raw, T_raw, k_obs_spx, k_obs_vix, X_mu, X_sd, Y_mu, Y_sd, device)

def predict_fixedk_smile_from_theta(
    model: torch.nn.Module,
    theta_raw: torch.Tensor, T_raw: float, 
    k_obs_spx: torch.Tensor, 
    X_mu: np.ndarray, X_sd: np.ndarray,
    Y_mu: np.ndarray, Y_sd: np.ndarray,
    device: torch.device,
):
    spx, _ = _predict_fixed_smile_generic(model, theta_raw, T_raw, k_obs_spx, torch.zeros_like(k_obs_spx), X_mu, X_sd, Y_mu, Y_sd, device)
    return spx


# predict cts k

def predict_ctsk_smile_from_theta(
    model: torch.nn.Module,
    theta_raw: torch.Tensor, T_raw: float, k_obs: torch.Tensor,
    X_mu: np.ndarray, X_sd: np.ndarray,
    Y_mu: np.ndarray, Y_sd: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    X_mu_t = _as_t(X_mu, device)
    X_sd_t = _as_t(X_sd, device)

    y_mu_scalar = float(np.asarray(Y_mu).reshape(-1)[0])
    y_sd_scalar = float(np.asarray(Y_sd).reshape(-1)[0])
    y_mu_t = torch.tensor(y_mu_scalar, dtype=torch.float32, device=device)
    y_sd_t = torch.tensor(y_sd_scalar, dtype=torch.float32, device=device)

    n = k_obs.shape[0]
    x_raw = torch.zeros((n, 17), dtype=torch.float32, device=device)
    x_raw[:, :15] = theta_raw[None, :]
    x_raw[:, 15] = float(T_raw)
    x_raw[:, 16] = k_obs

    x_n = (x_raw - X_mu_t[None, :]) / X_sd_t[None, :]
    y_n = model(x_n).reshape(-1)
    y_raw = y_n * y_sd_t + y_mu_t
    return y_raw


# --
# fixed k calibrators, SPX-only first, then SPX + VIX

def calibrate_fixedk_from_smile(
    k_obs: np.ndarray,
    iv_obs: np.ndarray,
    T_raw: float,
    *,
    device: Optional[torch.device | str] = None,
    n_starts: int = V1_N_STARTS,
    lbfgs_steps: int = V1_LBFGS_STEPS,
) -> Dict[str, object]:
    device = _device_or_default(device)
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("fixed", device)

    k_obs_t = _as_t(np.asarray(k_obs, dtype=np.float32), device)
    iv_obs_t = _as_t(np.asarray(iv_obs, dtype=np.float32), device)

    starts = make_default_starts(n_starts=n_starts, seed=123)
    best = None

    for theta0_np in starts:
        u = torch.tensor(
            u_from_theta_bounded(theta0_np),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.no_grad():
            theta0 = theta_from_u_bounded(u)
            iv0 = predict_fixedk_smile_from_theta(
                model, theta0, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            init_loss = torch.mean((iv0 - iv_obs_t) ** 2).item()

        opt = torch.optim.LBFGS([u], lr=0.8, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            theta = theta_from_u_bounded(u)
            iv_hat = predict_fixedk_smile_from_theta(
                model, theta, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            loss = torch.mean((iv_hat - iv_obs_t) ** 2)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            theta_hat = theta_from_u_bounded(u)
            iv_hat = predict_fixedk_smile_from_theta(
                model, theta_hat, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            final_loss = torch.mean((iv_hat - iv_obs_t) ** 2).item()
            rmse = _rmse_t(iv_hat, iv_obs_t).item()

        metrics = smile_error_metrics(
            iv_hat.detach().cpu().numpy(),
            np.asarray(iv_obs, dtype=np.float32),
        )

        row = {
            "theta_hat_raw": theta_hat.detach().cpu().numpy(),
            "iv_hat": iv_hat.detach().cpu().numpy(),
            "init_loss": float(init_loss),
            "final_loss": float(final_loss),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_abs_err": metrics["max_abs_err"],
            "mean_signed_err": metrics["mean_signed_err"],
            "method": "fixedk_v1",   # or ctsk_v1
            "k_obs": np.asarray(k_obs, dtype=np.float32),
            "iv_obs": np.asarray(iv_obs, dtype=np.float32),
            "T_raw": float(T_raw),
            "param_names": PARAM_NAMES,
        }

        if best is None or row["final_loss"] < best["final_loss"]:
            best = row

    return best

# SPX + VIX fixed calibration
def calibrate_fixedk_joint_from_smiles(
    k_obs_spx: np.ndarray,
    iv_obs_spx: np.ndarray,
    k_obs_vix: np.ndarray,
    iv_obs_vix: np.ndarray,
    T_raw: float,
    *,
    w_spx: float = 1.0,
    w_vix: float = 1.0,
    device: Optional[torch.device | str] = None,
    n_starts: int = V1_N_STARTS,
    lbfgs_steps: int = V1_LBFGS_STEPS,
) -> Dict[str, object]:
    device = _device_or_default(device)
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("fixed", device)

    k_obs_spx_t = _as_t(np.asarray(k_obs_spx, dtype=np.float32), device)
    iv_obs_spx_t = _as_t(np.asarray(iv_obs_spx, dtype=np.float32), device)

    k_obs_vix_t = _as_t(np.asarray(k_obs_vix, dtype=np.float32), device)
    iv_obs_vix_t = _as_t(np.asarray(iv_obs_vix, dtype=np.float32), device)

    starts = make_default_starts(n_starts=n_starts, seed=321)
    best = None

    for theta0_np in starts:
        u = torch.tensor(
            u_from_theta_bounded(theta0_np),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.no_grad():
            theta0 = theta_from_u_bounded(u)
            spx0, vix0 = predict_fixedk_joint_smiles_from_theta(
                model, theta0, T_raw,
                k_obs_spx_t, k_obs_vix_t,
                X_mu, X_sd, Y_mu, Y_sd, device
            )
            init_loss = (
                w_spx * torch.mean((spx0 - iv_obs_spx_t) ** 2)
                + w_vix * torch.mean((vix0 - iv_obs_vix_t) ** 2)
            ).item()

        opt = torch.optim.LBFGS([u], lr=0.8, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            theta = theta_from_u_bounded(u)
            spx_hat, vix_hat = predict_fixedk_joint_smiles_from_theta(
                model, theta, T_raw,
                k_obs_spx_t, k_obs_vix_t,
                X_mu, X_sd, Y_mu, Y_sd, device
            )
            loss = (
                w_spx * torch.mean((spx_hat - iv_obs_spx_t) ** 2)
                + w_vix * torch.mean((vix_hat - iv_obs_vix_t) ** 2)
            )
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            theta_hat = theta_from_u_bounded(u)
            spx_hat, vix_hat = predict_fixedk_joint_smiles_from_theta(
                model, theta_hat, T_raw,
                k_obs_spx_t, k_obs_vix_t,
                X_mu, X_sd, Y_mu, Y_sd, device
            )
            final_loss = (
                w_spx * torch.mean((spx_hat - iv_obs_spx_t) ** 2)
                + w_vix * torch.mean((vix_hat - iv_obs_vix_t) ** 2)
            ).item()

        spx_hat_np = spx_hat.detach().cpu().numpy()
        vix_hat_np = vix_hat.detach().cpu().numpy()

        row = {
            "theta_hat_raw": theta_hat.detach().cpu().numpy(),
            "spx_iv_hat": spx_hat_np,
            "vix_iv_hat": vix_hat_np,
            "init_loss": float(init_loss),
            "final_loss": float(final_loss),
            "spx_metrics": smile_error_metrics(spx_hat_np, np.asarray(iv_obs_spx, dtype=np.float32)),
            "vix_metrics": smile_error_metrics(vix_hat_np, np.asarray(iv_obs_vix, dtype=np.float32)),
            "method": "fixedk_joint",
            "k_obs_spx": np.asarray(k_obs_spx, dtype=np.float32),
            "iv_obs_spx": np.asarray(iv_obs_spx, dtype=np.float32),
            "k_obs_vix": np.asarray(k_obs_vix, dtype=np.float32),
            "iv_obs_vix": np.asarray(iv_obs_vix, dtype=np.float32),
            "T_raw": float(T_raw),
            "param_names": PARAM_NAMES,
        }

        if best is None or row["final_loss"] < best["final_loss"]:
            best = row

    return best

def calibrate_ctsk_from_smile(
    k_obs: np.ndarray,
    iv_obs: np.ndarray,
    T_raw: float,
    *,
    device: Optional[torch.device | str] = None,
    n_starts: int = V1_N_STARTS,
    lbfgs_steps: int = V1_LBFGS_STEPS,
) -> Dict[str, object]:
    device = _device_or_default(device)
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("ctsk", device) #load_ctsk_model_and_norm(device)

    k_obs_t = _as_t(np.asarray(k_obs, dtype=np.float32), device)
    iv_obs_t = _as_t(np.asarray(iv_obs, dtype=np.float32), device)

    starts = make_default_starts(n_starts=n_starts, seed=456)
    best = None

    for theta0_np in starts:
        u = torch.tensor(
            u_from_theta_bounded(theta0_np),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        with torch.no_grad():
            theta0 = theta_from_u_bounded(u)
            iv0 = predict_ctsk_smile_from_theta(
                model, theta0, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            init_loss = torch.mean((iv0 - iv_obs_t) ** 2).item()

        opt = torch.optim.LBFGS([u], lr=0.8, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            theta = theta_from_u_bounded(u)
            iv_hat = predict_ctsk_smile_from_theta(
                model, theta, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            loss = torch.mean((iv_hat - iv_obs_t) ** 2)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            theta_hat = theta_from_u_bounded(u)
            iv_hat = predict_ctsk_smile_from_theta(
                model, theta_hat, T_raw, k_obs_t, X_mu, X_sd, Y_mu, Y_sd, device
            )
            final_loss = torch.mean((iv_hat - iv_obs_t) ** 2).item()
            rmse = _rmse_t(iv_hat, iv_obs_t).item()

        metrics = smile_error_metrics(
            iv_hat.detach().cpu().numpy(),
            np.asarray(iv_obs, dtype=np.float32),
        )

        row = {
            "theta_hat_raw": theta_hat.detach().cpu().numpy(),
            "iv_hat": iv_hat.detach().cpu().numpy(),
            "init_loss": float(init_loss),
            "final_loss": float(final_loss),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "max_abs_err": metrics["max_abs_err"],
            "mean_signed_err": metrics["mean_signed_err"],
            "method": "fixedk_v1",   # or ctsk_v1
            "k_obs": np.asarray(k_obs, dtype=np.float32),
            "iv_obs": np.asarray(iv_obs, dtype=np.float32),
            "T_raw": float(T_raw),
            "param_names": PARAM_NAMES,
        }

        if best is None or row["final_loss"] < best["final_loss"]:
            best = row

    return best


# save fns
def save_fixedk_joint_summary(res: Dict[str, object], out_json: Path) -> None:
    payload = {
        "method": res["method"],
        "T_raw": float(res["T_raw"]),
        "init_loss": float(res["init_loss"]),
        "final_loss": float(res["final_loss"]),
        "spx_metrics": res["spx_metrics"],
        "vix_metrics": res["vix_metrics"],
        "theta_hat_raw": np.asarray(res["theta_hat_raw"]).tolist(),
        "param_names": list(res["param_names"]),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to: {out_json}")

def save_calibration_result(res: Dict[str, object], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": res["method"],
        "T_raw": float(res["T_raw"]),
        "init_loss": float(res["init_loss"]),
        "final_loss": float(res["final_loss"]),
        "rmse": float(res["rmse"]),
        "mae": float(res["mae"]),
        "max_abs_err": float(res["max_abs_err"]),
        "mean_signed_err": float(res["mean_signed_err"]),
        "theta_hat_raw": np.asarray(res["theta_hat_raw"]).tolist(),
        "k_obs": np.asarray(res["k_obs"]).tolist(),
        "iv_obs": np.asarray(res["iv_obs"]).tolist(),
        "iv_hat": np.asarray(res["iv_hat"]).tolist(),
        "param_names": list(res["param_names"]),
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics to: {out_json}")

def print_calibration_summary(res: Dict[str, object]) -> None:
    print(f"method          : {res['method']}")
    print(f"T_raw           : {res['T_raw']}")
    print(f"init_loss       : {res['init_loss']:.8f}")
    print(f"final_loss      : {res['final_loss']:.8f}")
    print(f"rmse            : {res['rmse']:.8f}")
    print(f"mae             : {res['mae']:.8f}")
    print(f"max_abs_err     : {res['max_abs_err']:.8f}")
    print(f"mean_signed_err : {res['mean_signed_err']:.8f}")
    print("theta_hat:")
    theta = np.asarray(res["theta_hat_raw"])
    for n, v in zip(PARAM_NAMES, theta):
        print(f"  {n:>6s} = {v: .6f}")
