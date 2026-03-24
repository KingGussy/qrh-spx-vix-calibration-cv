from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.qrh_sim.kernel import fit_kernel_weights
from src.qrh_sim.pricing_utils import (
    norm_cdf, norm_pdf,
    bs_call_price,
    bs_implied_vol_call,
    bs_call_delta, bs_vega
)

from qrh_nn.demo_joint_calibration import build_spx_smile
from qrh_nn.interp_k import HAVE_SCIPY, SmileInterpolator
from qrh_nn.calibration import (
    calibrate_fixedk_from_smile,
    calibrate_ctsk_from_smile,
    # load_fixed_model_and_norm,
    # load_ctsk_model_and_norm,
)
from qrh_nn.eval_utils import load_model_and_norm, K_SPX_FIXED as K_SPX

# def implied_vol_call_from_price(
#     price: float,
#     S: float,
#     K: float,
#     T: float,
#     r: float = 0.0,
#     q: float = 0.0,
#     tol: float = 1e-8,
#     max_iter: int = 100,
# ) -> float:
#     if T <= 0 or price <= 0 or S <= 0 or K <= 0:
#         return float("nan")

#     intrinsic = max(math.exp(-q * T) * S - math.exp(-r * T) * K, 0.0)
#     if price < intrinsic - 1e-10:
#         return float("nan")

#     low = 1e-6
#     high = 5.0
#     f_low = bs_call_price(S, K, T, low, r, q) - price
#     f_high = bs_call_price(S, K, T, high, r, q) - price

#     if f_low * f_high > 0:
#         return float("nan")

#     for _ in range(max_iter):
#         mid = 0.5 * (low + high)
#         f_mid = bs_call_price(S, K, T, mid, r, q) - price
#         if abs(f_mid) < tol:
#             return mid
#         if f_low * f_mid <= 0:
#             high = mid
#             f_high = f_mid
#         else:
#             low = mid
#             f_low = f_mid
#     return 0.5 * (low + high)

# ============================================================
# DATA PREP
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data SPX hedging demo: v1 or v2, BS vs fixed-k vs cts-k")
    p.add_argument("--kernel-npz", type=str, default=None,
                help="npz file containing kernel arrays c and gamma")
    p.add_argument("--data-dir", type=str, required=True, help="Directory containing held_option_quotes.csv, day0_chain.csv and optionally daily_chains/")
    p.add_argument("--output-dir", type=str, default="outputs/hedge_real_spx")
    p.add_argument("--k-min", type=float, default=-0.15)
    p.add_argument("--k-max", type=float, default=0.07)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

def business_dt_years(date_prev: str, date_next: str) -> float:
    d0 = np.datetime64(date_prev)
    d1 = np.datetime64(date_next)
    n_bus = np.busday_count(d0, d1)
    n_bus = max(int(n_bus), 1)
    return n_bus / 252.0


# def build_spx_smile_(
#     df: pd.DataFrame,
#     *,
#     k_min: float = -0.15,
#     k_max: float = 0.07,
#     r: float = 0.0,
#     q: float = 0.0,
# ) -> Tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
#     out = df.copy()
#     for c in ["strike", "bid", "ask", "mid", "underlyingPrice", "dte"]:
#         if c in out.columns:
#             out[c] = pd.to_numeric(out[c], errors="coerce")

#     out = out[out["mid"].notna() & (out["mid"] > 0)].copy()
#     out = out[out["underlyingPrice"].notna()].copy()
#     out = out[out["strike"].notna()].copy()
#     out = out[out["dte"].notna()].copy()
#     if out.empty:
#         raise RuntimeError("No usable SPX rows after cleaning.")

#     spot = float(out["underlyingPrice"].iloc[0])
#     T_raw = float(out["dte"].iloc[0]) / 365.0

#     out["iv_mid_filled"] = [
#         implied_vol_call_from_price(float(row["mid"]), spot, float(row["strike"]), T_raw, r=r, q=q)
#         for _, row in out.iterrows()
#     ]
#     out["iv_bid_filled"] = [
#         implied_vol_call_from_price(float(row["bid"]), spot, float(row["strike"]), T_raw, r=r, q=q)
#         if pd.notna(row["bid"]) and row["bid"] > 0 else np.nan
#         for _, row in out.iterrows()
#     ]
#     out["iv_ask_filled"] = [
#         implied_vol_call_from_price(float(row["ask"]), spot, float(row["strike"]), T_raw, r=r, q=q)
#         if pd.notna(row["ask"]) and row["ask"] > 0 else np.nan
#         for _, row in out.iterrows()
#     ]
#     out["k_log_money"] = np.log(out["strike"] / spot)

#     out = out[out["iv_mid_filled"].notna()].copy()
#     out = out[(out["k_log_money"] >= k_min) & (out["k_log_money"] <= k_max)].copy()
#     out = out.sort_values("k_log_money").reset_index(drop=True)

#     return (
#         out["k_log_money"].to_numpy(np.float32),
#         out["iv_mid_filled"].to_numpy(np.float32),
#         T_raw,
#         out,
#     )



def load_held_quotes(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "held_option_quotes.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    q = pd.read_csv(fp)
    for c in ["strike", "bid", "ask", "mid", "underlyingPrice", "dte"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")
    if "updated" in q.columns:
        try:
            q["updated_dt"] = pd.to_datetime(q["updated"], unit="s", utc=True)
        except Exception:
            q["updated_dt"] = pd.to_datetime(q["updated"])
        q["quote_date"] = q["updated_dt"].dt.strftime("%Y-%m-%d")
    elif "snapshot_date" in q.columns:
        q["quote_date"] = pd.to_datetime(q["snapshot_date"]).dt.strftime("%Y-%m-%d")
    else:
        raise RuntimeError("Need updated or snapshot_date in held quotes.")
    q = q.sort_values("quote_date").reset_index(drop=True)
    return q

# ============================================================
# Factor update helpers
# ============================================================
def qrh_variance_from_state(theta_raw: np.ndarray, z_state: np.ndarray, c_fac: np.ndarray) -> float:
    a, b, c0 = float(theta_raw[0]), float(theta_raw[1]), float(theta_raw[2])
    z_sum = float(np.dot(c_fac, z_state))
    V = a * (z_sum - b) ** 2 + c0
    return max(V, 1e-12)

def infer_dW_from_spot_move(s_prev: float, s_next: float, V_prev: float, dt: float) -> float:
    dlogS = math.log(s_next / s_prev)
    return (dlogS + 0.5 * V_prev * dt) / math.sqrt(max(V_prev, 1e-12))


def update_z_state_one_step_kfac(
    theta_raw: np.ndarray,
    z_prev: np.ndarray,
    s_prev: float,
    s_next: float,
    c_fac: np.ndarray,
    gamma_fac: np.ndarray,
    dt: float,
) -> np.ndarray:
    lam, eta = float(theta_raw[3]), float(theta_raw[4])

    z_prev = np.asarray(z_prev, dtype=np.float64)
    c_fac = np.asarray(c_fac, dtype=np.float64)
    gamma_fac = np.asarray(gamma_fac, dtype=np.float64)

    z_sum = float(np.dot(c_fac, z_prev))
    V_prev = qrh_variance_from_state(theta_raw, z_prev, c_fac)
    dW = infer_dW_from_spot_move(s_prev, s_next, V_prev, dt)

    decay = np.exp(-gamma_fac * dt)
    kfac = np.where(gamma_fac > 1e-12, (1.0 - decay) / gamma_fac, dt)

    driftQ = -lam * z_sum
    noiseQ = eta * math.sqrt(V_prev) * dW

    z_next = decay * z_prev + driftQ * kfac + noiseQ
    return z_next.astype(np.float32)

# ============================================================
# MODEL EVALUATION HELPERS
# ============================================================

def eval_fixedk_lattice(theta_raw: np.ndarray, T_raw: float, device: torch.device) -> np.ndarray:
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("fixed", device) #load_fixed_model_and_norm(device)
    X_mu_t = torch.tensor(X_mu, dtype=torch.float32, device=device)
    X_sd_t = torch.tensor(X_sd, dtype=torch.float32, device=device)
    Y_mu_t = torch.tensor(Y_mu, dtype=torch.float32, device=device)
    Y_sd_t = torch.tensor(Y_sd, dtype=torch.float32, device=device)

    x_raw = torch.zeros((1, 16), dtype=torch.float32, device=device)
    x_raw[0, :15] = torch.tensor(theta_raw, dtype=torch.float32, device=device)
    x_raw[0, 15] = float(T_raw)

    with torch.no_grad():
        x_n = (x_raw - X_mu_t[None, :]) / X_sd_t[None, :]
        y_n = model(x_n)
        y_raw = y_n * Y_sd_t[None, :] + Y_mu_t[None, :]
    return y_raw[0, :15].detach().cpu().numpy()



def fixedk_sigma_skew_and_zsens(
    theta_raw: np.ndarray,
    T_raw: float,
    k: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    '''
    Computes sig, dsig/dk and factor state (z)-sensitivites dsig/dZi
    '''
    # grid sig
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("fixed", device) #load_fixed_model_and_norm(device)

    x_raw = np.zeros((1, 16), dtype=np.float32)
    x_raw[0, :15] = theta_raw.astype(np.float32)
    x_raw[0, 15] = np.float32(T_raw)

    x_n = (x_raw - X_mu[None, :]) / X_sd[None, :]
    xt = torch.tensor(x_n, dtype=torch.float32, device=device, requires_grad=True)

    Y_mu_t = torch.tensor(Y_mu, dtype=torch.float32, device=device)
    Y_sd_t = torch.tensor(Y_sd, dtype=torch.float32, device=device)

    y_n = model(xt)
    y_raw = y_n * Y_sd_t[None, :] + Y_mu_t[None, :]
    smile = y_raw[0, :15]

    # interp and skew
    smile_np = smile.detach().cpu().numpy()
    method = "spline" if HAVE_SCIPY else "linear"
    interp = SmileInterpolator(K_SPX, smile_np, method=method, smooth=True if HAVE_SCIPY else False)

    sigma = float(interp.eval(np.array([k], dtype=np.float32))[0])
    dsigmadk = float(interp.deriv(np.array([k], dtype=np.float32))[0])

    # grid zsens
    z_cols = list(range(5, 15))
    dsig_dz_lattice = np.zeros((15, 10), dtype=np.float32)

    for j in range(15):
        grad = torch.autograd.grad(
            outputs=smile[j],
            inputs=xt,
            retain_graph=True,
            create_graph=False,
        )[0][0].detach().cpu().numpy()

        for zi, col in enumerate(z_cols):
            dsig_dz_lattice[j, zi] = grad[col] / float(X_sd[col])

    # interp zsens
    dsig_dz = np.zeros(10, dtype=np.float32)
    for zi in range(10):
        dsig_dz[zi] = np.interp(k, K_SPX, dsig_dz_lattice[:, zi])

    return sigma, dsigmadk, dsig_dz


def ctsk_sigma_skew_and_zsens_(
    theta_raw: np.ndarray,
    T_raw: float,
    k: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("ctsk", device) #load_ctsk_model_and_norm(device)

    if X_mu.shape[0] == 18:
        X_mu = X_mu[:17]
        X_sd = X_sd[:17]

    x_raw = np.zeros((1, 17), dtype=np.float32)
    x_raw[0, :15] = theta_raw.astype(np.float32)
    x_raw[0, 15] = np.float32(T_raw)
    x_raw[0, 16] = np.float32(k)

    x_n = (x_raw - X_mu[None, :]) / X_sd[None, :]
    xt = torch.tensor(x_n, dtype=torch.float32, device=device, requires_grad=True)

    y_n = model(xt).reshape(-1)
    y_mu = float(np.asarray(Y_mu).reshape(-1)[0])
    y_sd = float(np.asarray(Y_sd).reshape(-1)[0])
    sigma = y_n * y_sd + y_mu

    grad = torch.autograd.grad(
        outputs=sigma.sum(),
        inputs=xt,
        create_graph=False,
        retain_graph=False,
    )[0][0].detach().cpu().numpy()

    dsigmadk = grad[16] / float(X_sd[16])
    dsig_dz = np.array([grad[col] / float(X_sd[col]) for col in range(5, 15)], dtype=np.float32)

    return float(sigma.detach().cpu().numpy()[0]), float(dsigmadk), dsig_dz


# richer delta with sum term
def ctsk_sigma_skew_and_zsens(
    theta_raw: np.ndarray,
    T_raw: float,
    k: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm("ctsk", device) # load_ctsk_model_and_norm(device)

    if X_mu.shape[0] == 18:
        X_mu = X_mu[:17]
        X_sd = X_sd[:17]

    x_raw = np.zeros((1, 17), dtype=np.float32)
    x_raw[0, :15] = theta_raw.astype(np.float32)
    x_raw[0, 15] = np.float32(T_raw)
    x_raw[0, 16] = np.float32(k)

    x_n = (x_raw - X_mu[None, :]) / X_sd[None, :]
    xt = torch.tensor(x_n, dtype=torch.float32, device=device, requires_grad=True)

    y_n = model(xt).reshape(-1)
    y_mu = float(np.asarray(Y_mu).reshape(-1)[0])
    y_sd = float(np.asarray(Y_sd).reshape(-1)[0])
    sigma = y_n * y_sd + y_mu

    grad = torch.autograd.grad(
        outputs=sigma.sum(),
        inputs=xt,
        create_graph=False,
        retain_graph=False,
    )[0][0].detach().cpu().numpy()

    dsigmadk = grad[16] / float(X_sd[16])
    dsig_dz = np.array([grad[col] / float(X_sd[col]) for col in range(5, 15)], dtype=np.float32)

    return float(sigma.detach().cpu().numpy()[0]), float(dsigmadk), dsig_dz



def model_delta(
    S: float,
    K: float,
    T: float,
    sigma_hat: float,
    dsigmadk: float,
    dsig_dz: np.ndarray,
    eta: float,
    r: float = 0.0,
    q: float = 0.0,
) -> tuple[float, float, float]:
    # direct spot term
    delta_bs_local = bs_call_delta(S, K, T, sigma_hat, r=r, q=q)
    vega_local = bs_vega(S, K, T, sigma_hat, r=r, q=q)

    direct_term = delta_bs_local - (vega_local / max(S, 1e-12)) * dsigmadk
    factor_term = (eta / max(S, 1e-12)) * vega_local * float(np.sum(dsig_dz))
    total_delta = direct_term + factor_term

    return float(total_delta), float(direct_term), float(factor_term)


# ============================================================
# HEDGE ENGINE
# ============================================================

def summarise_hedge(df: pd.DataFrame, col: str) -> Dict[str, float]:
    x = df[col].to_numpy(dtype=float)
    return {
        "final": float(x[-1]),
        "mean_abs": float(np.mean(np.abs(x))),
        "max_abs": float(np.max(np.abs(x))),
        "std": float(np.std(x)),
    }


def run_hedge(data_dir: Path, out_dir: Path, k_min: float, k_max: float, r: float, q: float, device: torch.device):
    out_dir.mkdir(parents=True, exist_ok=True)

    quotes = load_held_quotes(data_dir)
    if quotes.empty:
        raise RuntimeError("No held quotes loaded.")

    K = float(quotes["strike"].dropna().iloc[0])
    if K is None:
        raise RuntimeError("Need strike column in held quotes for this script.")
    
    # grab expiration date from unix
    if "expiration" in quotes.columns:
        exp_raw = quotes["expiration"].iloc[0]
        try:
            expiry_str = pd.to_datetime(exp_raw, unit="s").strftime("%d-%m-%Y")
        except Exception:
            expiry_str = str(exp_raw)
    else:
        expiry_str = "unknown_expiry"

    # --------------------------------------------------------
    # Day-0 calibration
    # --------------------------------------------------------

    chain0 = pd.read_csv(data_dir / "day0_chain.csv")
    k0, iv0, T0, chain0_used = build_spx_smile(chain0, k_min=k_min, k_max=k_max) #, r=r, q=q)
    res_fixed_0 = calibrate_fixedk_from_smile(k0, iv0, T0, device=device)
    res_cts_0 = calibrate_ctsk_from_smile(k0, iv0, T0, device=device)
    theta_fixed_0 = np.asarray(res_fixed_0["theta_hat_raw"], dtype=np.float32)
    theta_cts_0 = np.asarray(res_cts_0["theta_hat_raw"], dtype=np.float32)
    

    # --------------------------------------------------------
    # Day-0 BS IV (fixed throughout)
    # --------------------------------------------------------
    S0 = float(quotes.loc[0, "underlyingPrice"])
    V0 = float(quotes.loc[0, "mid"])
    T0_held = float(quotes.loc[0, "dte"]) / 365.0
    # sigma_bs0 = implied_vol_call_from_price(V0, S0, K, T0_held, r=r, q=q)
    sigma_bs0 = bs_implied_vol_call(V0, S0, K, T0_held, r=r, q=q)
    print(f"Using fixed day-0 BS IV throughout hedge: sigma_bs0 = {sigma_bs0:.6f}")

    # --------------------------------------------------------
    # state setup
    # --------------------------------------------------------

    c_fac, gamma_fac = fit_kernel_weights(n=10, x_n=3.92, alpha=0.51)
    c_fac = np.asarray(c_fac, dtype=np.float64)
    gamma_fac = np.asarray(gamma_fac, dtype=np.float64)

    z_fixed_state = theta_fixed_0[5:15].copy()
    z_cts_state = theta_cts_0[5:15].copy()

    # --------------------------------------------------------
    # Per-date hedge inputs
    # --------------------------------------------------------
    rows = []

    for idx_row, row in quotes.iterrows():
        qdate = row["quote_date"]
        S = float(row["underlyingPrice"])
        V_mkt = float(row["mid"])
        T = float(row["dte"]) / 365.0
        k_hold = math.log(K / S)

        if idx_row == 0:
            theta_fixed = theta_fixed_0.copy()
            theta_cts = theta_cts_0.copy()
        else:
            prev_S = float(rows[-1]["S"])
            prev_date = rows[-1]["quote_date"]
            dt_years = business_dt_years(prev_date, qdate)

            z_fixed_state = update_z_state_one_step_kfac(
                theta_fixed_0, z_fixed_state, prev_S, S, c_fac, gamma_fac, dt_years
            )
            z_cts_state = update_z_state_one_step_kfac(
                theta_cts_0, z_cts_state, prev_S, S, c_fac, gamma_fac, dt_years
            )

            theta_fixed = theta_fixed_0.copy()
            theta_fixed[5:15] = z_fixed_state

            theta_cts = theta_cts_0.copy()
            theta_cts[5:15] = z_cts_state

        # 
        sigma_bs = sigma_bs0
        delta_bs = bs_call_delta(S, K, T, sigma_bs0, r=r, q=q) if np.isfinite(sigma_bs0) else np.nan

        sigma_fixed, dsigmadk_fixed, dsig_dz_fixed = fixedk_sigma_skew_and_zsens(theta_fixed, T, k_hold, device)
        delta_fixed, direct_fixed, factor_fixed = model_delta(
            S, K, T, sigma_fixed, dsigmadk_fixed, dsig_dz_fixed,
            eta=float(theta_fixed[4]), r=r, q=q
        )

        sigma_cts, dsigmadk_cts, dsig_dz_cts = ctsk_sigma_skew_and_zsens(theta_cts, T, k_hold, device)
        delta_cts, direct_cts, factor_cts = model_delta(
            S, K, T, sigma_cts, dsigmadk_cts, dsig_dz_cts,
            eta=float(theta_cts[4]), r=r, q=q
        )


        rows.append({
            "quote_date": qdate,
            "S": S,
            "K": K,
            "T": T,
            "k_hold": k_hold,
            "V_mkt": V_mkt,
            "sigma_bs": sigma_bs,
            "delta_bs": delta_bs,

            "sigma_fixed": sigma_fixed,
            "dsigmadk_fixed": dsigmadk_fixed,
            "delta_fixed": delta_fixed,


            "sigma_cts": sigma_cts,
            "dsigmadk_cts": dsigmadk_cts,
            "delta_cts": delta_cts,

            "theta_fixed_used": theta_fixed.copy(),
            "theta_cts_used": theta_cts.copy(),
            "z_fixed_norm": float(np.linalg.norm(theta_fixed[5:15])),
            "z_cts_norm": float(np.linalg.norm(theta_cts[5:15])),
        })

    hd = pd.DataFrame(rows).sort_values("quote_date").reset_index(drop=True)

    # --------------------------------------------------------
    # Self-financing hedged portfolios
    # --------------------------------------------------------
    for label, delta_col in [("bs", "delta_bs"), ("fixed", "delta_fixed"), ("cts", "delta_cts")]:
        stock_pos = np.zeros(len(hd))
        cash = np.zeros(len(hd))
        total = np.zeros(len(hd))

        d0 = float(hd.loc[0, delta_col])
        stock_pos[0] = -d0
        cash[0] = -float(hd.loc[0, "V_mkt"]) + d0 * float(hd.loc[0, "S"])
        total[0] = float(hd.loc[0, "V_mkt"]) + stock_pos[0] * float(hd.loc[0, "S"]) + cash[0]

        for i in range(1, len(hd)):
            S_i = float(hd.loc[i, "S"])
            V_i = float(hd.loc[i, "V_mkt"])

            stock_pos[i] = stock_pos[i - 1]
            cash[i] = cash[i - 1] * math.exp(r / 252.0)
            total[i] = V_i + stock_pos[i] * S_i + cash[i]

            d_new = float(hd.loc[i, delta_col])
            new_stock = -d_new
            trade = new_stock - stock_pos[i]
            cash[i] -= trade * S_i
            stock_pos[i] = new_stock

        hd[f"stock_{label}"] = stock_pos
        hd[f"cash_{label}"] = cash
        hd[f"hedge_error_{label}"] = total

    # --------------------------------------------------------
    # Normalise by V0
    # --------------------------------------------------------
    V0 = float(hd.loc[0, "V_mkt"])
    hd["V0"] = V0
    hd["hedge_error_bs_norm"] = hd["hedge_error_bs"] / V0
    hd["hedge_error_fixed_norm"] = hd["hedge_error_fixed"] / V0
    hd["hedge_error_cts_norm"] = hd["hedge_error_cts"] / V0

    metrics = {
        #"mode": mode,
        "n_dates": int(len(hd)),
        "V0": float(V0),
        "bs": summarise_hedge(hd, "hedge_error_bs"),
        "fixed": summarise_hedge(hd, "hedge_error_fixed"),
        "cts": summarise_hedge(hd, "hedge_error_cts"),
        "bs_norm": summarise_hedge(hd, "hedge_error_bs_norm"),
        "fixed_norm": summarise_hedge(hd, "hedge_error_fixed_norm"),
        "cts_norm": summarise_hedge(hd, "hedge_error_cts_norm"),
        # "delta_diag": delta_diag,
    }

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    x = pd.to_datetime(hd["quote_date"])

    plt.figure(figsize=(9, 5))
    plt.plot(x, hd["hedge_error_bs_norm"], color="cornflowerblue", linewidth=1.5, label="BS hedge / V0")
    plt.plot(x, hd["hedge_error_fixed_norm"], linewidth=1.5, color="goldenrod", label="Fixed-k hedge / V0")
    plt.plot(x, hd["hedge_error_cts_norm"], linewidth=1.5, color="seagreen", label="Cts-k hedge / V0")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title(f"Normalised SPX hedge error | Expiry = {expiry_str}")
    plt.xlabel("date")
    plt.ylabel("hedged portfolio value / V0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hedge_error_norm.png", dpi=150)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(x, hd["delta_bs"], linewidth=1.2, color="cornflowerblue",label="BS delta")
    plt.plot(x, hd["delta_fixed"], linewidth=1.2, color="goldenrod", label="Fixed-k delta")
    plt.plot(x, hd["delta_cts"], linewidth=1.2, color="seagreen",label="Cts-k delta")
    plt.title(f"Delta paths | Expiry = {expiry_str}")
    plt.xlabel("date")
    plt.ylabel("delta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_paths.png", dpi=150)
    plt.show()


    hd.to_csv(out_dir / f"hedge_timeseries.csv", index=False)
    with open(out_dir / f"hedge_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if res_fixed_0 is not None and res_cts_0 is not None and chain0_used is not None:
        plt.figure(figsize=(8, 4))
        plt.scatter(chain0_used["k_log_money"], chain0_used["iv_bid_filled"], marker="x", s=18, color="tomato", label="bid IV")
        plt.scatter(chain0_used["k_log_money"], chain0_used["iv_ask_filled"], marker="x", s=18, color="mediumslateblue", label="ask IV")
        plt.plot(res_fixed_0["k_obs"], res_fixed_0["iv_hat"], linewidth=1.5, color="goldenrod", label="fixed-k fit")
        plt.plot(res_cts_0["k_obs"], res_cts_0["iv_hat"], linewidth=1.5, color="seagreen", label="cts-k fit")
        plt.xlabel("log-moneyness k")
        plt.ylabel("implied vol")
        plt.title("Day-0 SPX calibration | ")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "day0_calibration_overlay.png", dpi=150)
        plt.show()

        # zoomed in plot of the same
        atm_mask = (chain0_used["k_log_money"] >= -0.00) & (chain0_used["k_log_money"] <= 0.06)
        plt.figure(figsize=(8, 4))
        plt.scatter(chain0_used.loc[atm_mask, "k_log_money"],chain0_used.loc[atm_mask, "iv_bid_filled"],marker="x",s=18,color="tomato",label="bid IV",)
        plt.scatter(chain0_used.loc[atm_mask, "k_log_money"],chain0_used.loc[atm_mask, "iv_ask_filled"],marker="x",s=18,color="mediumslateblue",label="ask IV",)
        fixed_mask = (res_fixed_0["k_obs"] >= -0.00) & (res_fixed_0["k_obs"] <= 0.06)
        cts_mask = (res_cts_0["k_obs"] >= -0.00) & (res_cts_0["k_obs"] <= 0.06)
        plt.plot(np.asarray(res_fixed_0["k_obs"])[fixed_mask],np.asarray(res_fixed_0["iv_hat"])[fixed_mask],marker="o",markersize=3,linewidth=1.5,color="goldenrod",label="fixed-k fit",)
        plt.plot(np.asarray(res_cts_0["k_obs"])[cts_mask],np.asarray(res_cts_0["iv_hat"])[cts_mask],marker="o",markersize=3,linewidth=1.5, color="seagreen",label="cts-k fit",)
        plt.xlim(-0.00, 0.06)
        plt.xlabel("log-moneyness k")
        plt.ylabel("implied vol")
        plt.title(f"Day-0 SPX calibration overlays (ATM zoom) | expiry={expiry_str}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "day0_calibration_overlay_atm_zoom.png", dpi=150)
        plt.show()

    return hd, metrics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    hd, metrics = run_hedge(
        #mode=args.mode,
        data_dir=data_dir,
        out_dir=out_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        r=args.r,
        q=args.q,
        device=device,
    )

    print("Saved outputs to:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
