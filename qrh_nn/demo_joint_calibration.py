from __future__ import annotations
"""
Run the fixed k model calibration jointly to spx and vix smiles simulatenously
"""
from pathlib import Path
from typing import Dict, Optional

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.qrh_sim.pricing_utils import norm_cdf, bs_call_price, bs_implied_vol_call

from qrh_nn.calibration import (
    calibrate_fixedk_joint_from_smiles,
    save_fixedk_joint_summary,
)


REPO = Path(__file__).resolve().parents[1]

SPX_PATH = REPO / "data" / "hedging_data" / "raw" / "corrected" / "0105" / "SPX_0105_calls_20" / "day0_chain.csv"
VIX_CALL_PATH = REPO / "data" / "hedging_data" / "raw" / "corrected" / "0105" / "VIX_0105_calls" / "day0_chain.csv"
VIX_PUT_PATH = REPO / "data" / "hedging_data" / "raw" / "corrected" / "0105" / "VIX_0105_puts" / "day0_chain.csv"

OUT_DIR = REPO / "models" / "calibration_joint_real_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_spx_smile(
    df: pd.DataFrame,
    *,
    k_min: float = -0.15,
    k_max: float = 0.07,
):
    out = df.copy()
    for c in ["strike", "bid", "ask", "mid", "underlyingPrice", "dte"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[out["mid"].notna() & (out["mid"] > 0)].copy()
    out = out[out["underlyingPrice"].notna()].copy()
    out = out[out["strike"].notna()].copy()
    out = out[out["dte"].notna()].copy()

    if out.empty:
        raise RuntimeError("No usable SPX rows after cleaning.")

    spot = float(out["underlyingPrice"].iloc[0])
    T_raw = float(out["dte"].iloc[0]) / 365.0

    out["iv_mid_filled"] = [
        bs_implied_vol_call(price=float(row["mid"]), S0=spot, K=float(row["strike"]), T=T_raw)
        for _, row in out.iterrows()
    ]

    out["k_log_money"] = np.log(out["strike"] / spot)
    out = out[out["iv_mid_filled"].notna()].copy()
    out = out[(out["k_log_money"] >= k_min) & (out["k_log_money"] <= k_max)].copy()
    out = out.sort_values("k_log_money").reset_index(drop=True)

    out["iv_bid_filled"] = [
        bs_implied_vol_call(price=float(row["bid"]), S0=spot, K=float(row["strike"]), T=T_raw
        ) if pd.notna(row["bid"]) and row["bid"] > 0 else np.nan
        for _, row in out.iterrows()
    ]

    out["iv_ask_filled"] = [
        bs_implied_vol_call(price=float(row["ask"]), S0=spot, K=float(row["strike"]), T=T_raw
        ) if pd.notna(row["ask"]) and row["ask"] > 0 else np.nan
        for _, row in out.iterrows()
    ]

    return (
        out["k_log_money"].to_numpy(np.float32),
        out["iv_mid_filled"].to_numpy(np.float32),
        T_raw,
        out,
    )


# separately for futures approach to VIX
def infer_vix_future_from_parity(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    *,
    r: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """
    Infers the VIX futures using put-call parity from our data for our IV computation (per Romer)
    C - P = F - K ==> F = C - P + K
    We take rates as 0 throughout (assumed negligable over ~30 days)
    """
    c, p = calls_df.copy(), puts_df.copy()

    for df in (c, p):
        for col in ["strike", "mid", "dte"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    c = c[["strike", "mid", "dte"]].rename(columns={"mid": "call_mid", "dte": "dte_call"})
    p = p[["strike", "mid", "dte"]].rename(columns={"mid": "put_mid", "dte": "dte_put"})

    m = pd.merge(c, p, on="strike", how="inner")
    m = m.dropna(subset=["strike", "call_mid", "put_mid", "dte_call", "dte_put"]).copy()

    if m.empty:
        raise RuntimeError("No call/put strikes found.")

    #m["dte"] = 0.5 * (m["dte_call"] + m["dte_put"])

    T_raw = float(m["dte_call"].iloc[0]) / 365.0
    m["F_from_parity"] = m["strike"] + np.exp(r * T_raw) * (m["call_mid"] - m["put_mid"])

    # take median of these for robustness
    F = float(np.median(m["F_from_parity"].to_numpy()))
    print(f"Inferred VIX future F(T) from parity: {F:.6f}")
    return F, m


def build_vix_smile_from_calls_puts(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    *,
    k_min: float = -0.10,
    k_max: float = 0.21,
):
    F, parity_df = infer_vix_future_from_parity(calls_df, puts_df, r=0.0)

    out = calls_df.copy()
    for c in ["strike", "bid", "ask", "mid", "dte"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[out["mid"].notna() & (out["mid"] > 0)].copy()
    out = out[out["strike"].notna()].copy()
    out = out[out["dte"].notna()].copy()

    if out.empty:
        raise RuntimeError("No usable VIX call rows after cleaning.")

    T_raw = float(out["dte"].iloc[0]) / 365.0

    out["iv_mid_filled"] = [
        bs_implied_vol_call(price=float(row["mid"]), S0=F, K=float(row["strike"]), T=T_raw, r=0.0, q=0.0,)
        for _, row in out.iterrows()
    ]

    out["k_log_money"] = np.log(out["strike"] / F)
    out = out[out["iv_mid_filled"].notna()].copy()
    out = out[(out["k_log_money"] >= k_min) & (out["k_log_money"] <= k_max)].copy()
    out = out.sort_values("k_log_money").reset_index(drop=True)

    out["iv_bid_filled"] = [
        bs_implied_vol_call(
            price=float(row["bid"]),
            S0=F,
            K=float(row["strike"]),
            T=T_raw,
            r=0.0,
            q=0.0,
        ) if pd.notna(row["bid"]) and row["bid"] > 0 else np.nan
        for _, row in out.iterrows()
    ]

    out["iv_ask_filled"] = [
        bs_implied_vol_call(
            price=float(row["ask"]),
            S0=F,
            K=float(row["strike"]),
            T=T_raw,
            r=0.0,
            q=0.0,
        ) if pd.notna(row["ask"]) and row["ask"] > 0 else np.nan
        for _, row in out.iterrows()
    ]

    return (
        out["k_log_money"].to_numpy(np.float32),
        out["iv_mid_filled"].to_numpy(np.float32),
        T_raw,
        out,
        F,
        parity_df,
    )


# ==========================================



# Plotter:
def plot_fixedk_joint_realdata_result(
    res: Dict[str, object],
    spx_used: pd.DataFrame,
    vix_used: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> None:
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # SPX
    plt.figure(figsize=(7, 4))
    plt.scatter(spx_used["k_log_money"], spx_used["iv_bid_filled"], marker="x", s=18, color="tomato", label="SPX bid IV")
    plt.scatter(spx_used["k_log_money"], spx_used["iv_ask_filled"], marker="x", s=18, color="mediumslateblue", label="SPX ask IV")
    plt.plot(res["k_obs_spx"], res["spx_iv_hat"], linewidth=1.8, color="forestgreen", label="SPX fitted")
    plt.xlabel("log-moneyness k")
    plt.ylabel("implied vol")
    plt.title("Fixed-k joint calibration | SPX | 30 DTE")
    plt.legend()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir / "fixedk_joint_spx.png", dpi=150)
    plt.show()

    # VIX
    plt.figure(figsize=(7, 4))
    plt.scatter(vix_used["k_log_money"], vix_used["iv_bid_filled"], marker="x", s=18, color="tomato", label="VIX bid IV")
    plt.scatter(vix_used["k_log_money"], vix_used["iv_ask_filled"], marker="x", s=18, color="mediumslateblue", label="VIX ask IV")
    plt.plot(res["k_obs_vix"], res["vix_iv_hat"], linewidth=1.5, color="forestgreen", label="VIX fitted")
    plt.xlabel("log-moneyness k")
    plt.ylabel("implied vol")
    plt.title("Fixed-k joint calibration | VIX | 30 DTE")
    plt.legend()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir / "fixedk_joint_vix.png", dpi=150)
    plt.show()

def main():
    spx_df = pd.read_csv(SPX_PATH)
    vix_call_df = pd.read_csv(VIX_CALL_PATH)
    vix_put_df = pd.read_csv(VIX_PUT_PATH)

    k_spx, iv_spx, T_spx, spx_used = build_spx_smile(
        spx_df,
        k_min=-0.15,
        k_max=0.07,
    )

    k_vix, iv_vix, T_vix, vix_used, F_vix, parity_used = build_vix_smile_from_calls_puts(
        vix_call_df,
        vix_put_df,
        k_min=-0.10,
        k_max=0.21,
    )

    print(f"SPX usable points: {len(k_spx)} | k-range [{k_spx.min():.4f}, {k_spx.max():.4f}]")
    print(f"VIX usable points: {len(k_vix)} | k-range [{k_vix.min():.4f}, {k_vix.max():.4f}]")
    print(f"Inferred VIX future: {F_vix:.6f}")

    spx_used.to_csv(OUT_DIR / "spx_smile_used.csv", index=False)
    vix_used.to_csv(OUT_DIR / "vix_smile_used.csv", index=False)
    parity_used.to_csv(OUT_DIR / "vix_parity_pairs_used.csv", index=False)

    T_raw = 0.5 * (T_spx + T_vix)

    res = calibrate_fixedk_joint_from_smiles(
        k_obs_spx=k_spx,
        iv_obs_spx=iv_spx,
        k_obs_vix=k_vix,
        iv_obs_vix=iv_vix,
        T_raw=T_raw,
    )

    print("Final joint loss:", res["final_loss"])
    print("SPX metrics:", res["spx_metrics"])
    print("VIX metrics:", res["vix_metrics"])

    plot_fixedk_joint_realdata_result(
        res,
        spx_used=spx_used,
        vix_used=vix_used,
        out_dir=OUT_DIR,
    )
    save_fixedk_joint_summary(res, OUT_DIR / "fixedk_joint_summary.json")

if __name__ == "__main__":
    main()