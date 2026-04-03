from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qrh_sim_cuda as sim_mod
import full_path_sim

from src.qrh_sim.kernel import fit_kernel_weights
from src.qrh_sim.sim_utils import KernelSpec, QRHParams, LRHParams, x_cap_from_sigma_cap
from src.qrh_sim.affine_params import pilot_ab_from_qrh_cuda


OUT_DIR = Path("outputs/inspect_b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

B_VALUES = [-0.30, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# Main sim config
S0 = 100.0
T = 0.10
N_STEPS = 400
M_METRICS = 20000
M_PATHS_PLOT = 60
SCHEME = "inv"
QUAD = "left"
SEED = 12345

# Kernel config
N_FACTORS = 10
KERNEL_ALPHA = 0.51
X_STAR = 3.92

# QRH base params
Q_A = 0.4
Q_C0 = 0.01
LAM = 1.0
ETA = 0.6

# Optional cap
V_CAP = None


def bs_call_from_integrated_variance(S0: float, K: float, I: np.ndarray) -> np.ndarray:
    I = np.asarray(I, dtype=np.float64)
    eps = 1e-12
    I = np.maximum(I, eps)
    rootI = np.sqrt(I)

    d1 = (np.log(S0 / K) + 0.5 * I) / rootI
    d2 = d1 - rootI

    from math import erf, sqrt
    norm_cdf = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0))))
    return S0 * norm_cdf(d1) - K * norm_cdf(d2)


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-15 or sy < 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))


def to_extension_obj_kernel(kernel: KernelSpec):
    return SimpleNamespace(c=np.asarray(kernel.c, dtype=np.float64),
                           gamma=np.asarray(kernel.gamma, dtype=np.float64))


def to_extension_obj_q(qpars: QRHParams):
    return SimpleNamespace(a=float(qpars.a),
                           b=float(qpars.b),
                           c0=float(qpars.c0),
                           lam=float(qpars.lam),
                           eta=float(qpars.eta),
                           z0=np.asarray(qpars.z0, dtype=np.float64))


def to_extension_obj_l(lpars: LRHParams):
    return SimpleNamespace(alpha=float(lpars.alpha),
                           beta=float(lpars.beta),
                           lam=float(lpars.lam),
                           eta=float(lpars.eta))


def fit_lrh_for_b(kernel: KernelSpec, qpars: QRHParams, x_cap):
    pilot = pilot_ab_from_qrh_cuda(
        sim_mod=sim_mod,
        kernel=kernel,
        qpars=qpars,
        lrh_zero=LRHParams(0.0, 0.0, qpars.lam, qpars.eta),
        T=float(T),
        n_steps=int(N_STEPS),
        m_pilot=100_000,
        scheme=str(SCHEME),
        quad=str(QUAD),
        S0=float(S0),
        seed=123456,
        x_cap=x_cap,
    )
    return float(pilot["alpha"]), float(pilot["beta"]), pilot


def simulate_joint_paths(kernel: KernelSpec, qpars: QRHParams, lpars: LRHParams, m: int, seed: int):
    return full_path_sim.simulate_qrh_lrh_paths_cuda(
        m=m,
        kernel=to_extension_obj_kernel(kernel),
        q=to_extension_obj_q(qpars),
        l=to_extension_obj_l(lpars),
        T=T,
        n_steps=N_STEPS,
        scheme=SCHEME,
        quad=QUAD,
        S0=S0,
        vcap_obj=x_cap_from_sigma_cap(qpars, float(V_CAP)) if V_CAP is not None else None,
        z_cap_obj=None,
        seed_obj=seed,
        record_full_factors=False,
    )


def compute_metrics(I_Q, I_L, V_Q_paths, V_L_paths, Zsum_Q_paths, Zsum_L_paths):
    K_atm = S0
    price_Q = bs_call_from_integrated_variance(S0, K_atm, I_Q)
    price_L = bs_call_from_integrated_variance(S0, K_atm, I_L)

    rho_price = corr_safe(price_Q, price_L)
    vr = np.inf if (np.isfinite(rho_price) and abs(rho_price) >= 1.0) else (
        float(1.0 / max(1e-12, 1.0 - rho_price**2)) if np.isfinite(rho_price) else np.nan
    )

    return {
        "corr_I": corr_safe(I_Q, I_L),
        "rmse_I": rmse(I_Q, I_L),
        "mae_I": mae(I_Q, I_L),
        "corr_price": rho_price,
        "vr_price": vr,
        "corr_V_terminal": corr_safe(V_Q_paths[:, -1], V_L_paths[:, -1]),
        "corr_Zsum_terminal": corr_safe(Zsum_Q_paths[:, -1], Zsum_L_paths[:, -1]),
        "rmse_V_path": rmse(V_Q_paths, V_L_paths),
        "rmse_Zsum_path": rmse(Zsum_Q_paths, Zsum_L_paths),
        "mean_I_Q": float(np.mean(I_Q)),
        "mean_I_L": float(np.mean(I_L)),
        "std_I_Q": float(np.std(I_Q)),
        "std_I_L": float(np.std(I_L)),
    }


def plot_metric_vs_b(df: pd.DataFrame, col: str, ylabel: str, filename: str):
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["b"], df[col], linewidth=1.4)
    plt.xlabel("b")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs b")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.show()


def plot_I_scatter(b: float, I_Q: np.ndarray, I_L: np.ndarray):
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(I_Q, I_L, s=7, alpha=0.25)
    mn = min(np.min(I_Q), np.min(I_L))
    mx = max(np.max(I_Q), np.max(I_L))
    plt.plot([mn, mx], [mn, mx], linewidth=1.0, color="black")
    plt.xlabel("I_Q")
    plt.ylabel("I_L")
    plt.title(f"Integrated variance scatter | b={b:g}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"I_scatter_b_{b:g}.png", dpi=150)
    plt.show()


def plot_mean_band(times: np.ndarray, Y: np.ndarray, color: str, label: str):
    mean = np.mean(Y, axis=0)
    lo = np.quantile(Y, 0.10, axis=0)
    hi = np.quantile(Y, 0.90, axis=0)
    plt.plot(times, mean, color=color, linewidth=1.3, label=label)
    plt.fill_between(times, lo, hi, color=color, alpha=0.18)


def plot_V_bands(b: float, times: np.ndarray, V_Q: np.ndarray, V_L: np.ndarray):
    plt.figure(figsize=(8, 4.8))
    plot_mean_band(times, V_Q, "tomato", "QRH")
    plot_mean_band(times, V_L, "deepskyblue", "LRH")
    plt.xlabel("t")
    plt.ylabel("V_t")
    plt.title(f"Variance paths: mean and 10-90% band | b={b:g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"V_bands_b_{b:g}.png", dpi=150)
    plt.show()


def main():
    kernel_c, kernel_gamma = fit_kernel_weights(n=N_FACTORS, x_n=X_STAR, alpha=KERNEL_ALPHA)
    kernel = KernelSpec(c=np.array(kernel_c, float), gamma=np.array(kernel_gamma, float))

    rows = []
    selected_b = {-0.30, 0.01, 0.30, 0.50}

    for b in B_VALUES:
        print(f"\n=== Running b={b:g} ===")

        qpars = QRHParams(Q_A, b, Q_C0, LAM, ETA, z0=np.zeros(N_FACTORS))
        x_cap = None if V_CAP is None else x_cap_from_sigma_cap(qpars, sigma_cap=float(V_CAP))

        lrh_alpha, lrh_beta, pilot = fit_lrh_for_b(kernel, qpars, x_cap)
        lpars = LRHParams(alpha=lrh_alpha, beta=lrh_beta, lam=LAM, eta=ETA)

        print(f"Pilot fit: lrh_alpha={lrh_alpha:.6f}, lrh_beta={lrh_beta:.6f}")

        (
            times,
            _S_Q_paths,
            _S_L_paths,
            Zsum_Q_paths,
            Zsum_L_paths,
            V_Q_paths,
            V_L_paths,
            I_Q,
            I_L,
            _ZQ_paths,
            _ZL_paths,
        ) = simulate_joint_paths(kernel, qpars, lpars, m=M_METRICS, seed=SEED)

        I_Q = np.asarray(I_Q)
        I_L = np.asarray(I_L)
        Zsum_Q_paths = np.asarray(Zsum_Q_paths)
        Zsum_L_paths = np.asarray(Zsum_L_paths)
        V_Q_paths = np.asarray(V_Q_paths)
        V_L_paths = np.asarray(V_L_paths)
        times = np.asarray(times)

        metrics = compute_metrics(I_Q, I_L, V_Q_paths, V_L_paths, Zsum_Q_paths, Zsum_L_paths)

        row = {
            "b": float(b),
            "lrh_alpha": lrh_alpha,
            "lrh_beta": lrh_beta,
            **metrics,
        }
        rows.append(row)

        if b in selected_b:
            plot_I_scatter(b, I_Q, I_L)

            # Smaller run for path plots
            (
                times_s,
                _SQ_s,
                _SL_s,
                _ZsumQ_s,
                _ZsumL_s,
                VQ_s,
                VL_s,
                _IQ_s,
                _IL_s,
                _ZQ_s,
                _ZL_s,
            ) = simulate_joint_paths(kernel, qpars, lpars, m=M_PATHS_PLOT, seed=SEED + 999)

            plot_V_bands(b, np.asarray(times_s), np.asarray(VQ_s), np.asarray(VL_s))

    df = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
    df.to_csv(OUT_DIR / "b_sweep_summary.csv", index=False)
    with open(OUT_DIR / "b_sweep_summary.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    plot_metric_vs_b(df, "corr_I", "Corr(I_Q, I_L)", "corr_I_vs_b.png")
    plot_metric_vs_b(df, "rmse_I", "RMSE(I_Q - I_L)", "rmse_I_vs_b.png")
    plot_metric_vs_b(df, "corr_price", "Corr(price_Q, price_L)", "corr_price_vs_b.png")
    plot_metric_vs_b(df, "vr_price", "Variance reduction factor", "vr_price_vs_b.png")
    plot_metric_vs_b(df, "mean_I_Q", "mean(I_Q)", "mean_I_Q_vs_b.png")
    plot_metric_vs_b(df, "mean_I_L", "mean(I_L)", "mean_I_L_vs_b.png")

    print("\nSummary:")
    print(df.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()