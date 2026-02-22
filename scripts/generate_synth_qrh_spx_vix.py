# scripts/generate_synth_qrh_spx_vix.py
import sys, time, json, argparse, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Project bits
from src.qrh_sim.kernel import fit_kernel_weights
from src.qrh_sim.sim_utils import KernelSpec, QRHParams, LRHParams, x_cap_from_sigma_cap
from src.qrh_sim.parse_utils import _parse_float_list, _parse_T_grid, parse_param_sets
from src.qrh_sim.io_utils import _ensure_dir, _load_json, _log, _save_json, save_shard_npz
from src.qrh_sim.pricing_utils import black_call_forward, implied_vol_black_forward

# CUDA module
import importlib, qrh_sim_cuda as sim
importlib.reload(sim)

# =================
# Sampling helpers:
# =================
def sample_params_uniform(rng: np.random.Generator, ranges: dict):
    def U(name):
        lo, hi = ranges[name]
        return float(rng.uniform(lo, hi))
    return dict(
        a=U("a"),
        b=U("b"),
        c0=U("c0"),
        lam=U("lam"),
        eta=U("eta"),
    )

def sample_z0_uniform(rng: np.random.Generator, n_factors: int, z0_lo: float, z0_hi: float) -> np.ndarray:
    return rng.uniform(float(z0_lo), float(z0_hi), size=int(n_factors)).astype(np.float64)


# ==================
# Adapter functions:
# ==================
def simulate_qrh_terminal(
    m_paths: int,
    kernel: KernelSpec,
    q: QRHParams,
    S0: float,
    T: float,
    n_steps_mc: int,
    scheme: str,
    quad: str,
    seed: int,
    sigma_cap: float | None=None,
    dtype=np.float64):
    """
    Runs the CUDA path simulator to maturity T on the QRH only (no CV)
    Returns: S_T - terminal spot price, Z_T - terminal QRH factors
    """
    l_dummy = LRHParams(alpha=0.0, beta=0.0, lam=q.lam, eta=q.eta)
    if sigma_cap is None: 
        vcap_obj = None
    else:vcap_obj = float(x_cap_from_sigma_cap(q, float(sigma_cap)))

    I_Q, I_L, (S_Q, S_L), Z_T, cZT = sim.simulate_paths_cuda(
        m=int(m_paths),
        kernel=kernel,
        q=q,
        l=l_dummy,
        T=float(T),
        n_steps=int(n_steps_mc),
        use_CV=False,
        scheme=str(scheme),
        quad=str(quad),
        S0=float(S0),
        return_ST=True,
        return_ZT=True,
        return_cZT=False,
        return_J=False,
        vcap_obj=vcap_obj,
        v_floor=0.0,
        z_cap_obj=None,
        seed_obj=int(seed),
        dW_shared_obj=None,
    )
    _log(f"[diag] S_T mean={S_Q.mean():.6f} std={S_Q.std():.6f} min={S_Q.min():.6f} max={S_Q.max():.6f}")

    S_Q = np.asarray(S_Q, dtype=dtype)
    Z_T = np.asarray(Z_T, dtype=dtype)

    # === diag 1 ===
    x = np.log(np.maximum(S_Q, 1e-300) / float(S0))
    diag = {
        "spx_ST_mean": float(S_Q.mean()),
        "spx_ST_std": float(S_Q.std(ddof=1)),
        "spx_logret_var": float(x.var(ddof=1)),
    }
    # ===        ===

    if Z_T.ndim != 2 or Z_T.shape[0] != int(m_paths):
        raise RuntimeError(f"Unexpected Z_T shape {Z_T.shape}, expected (m_paths, n_factors).")

    return S_Q, Z_T, diag


def compute_vix_paths_from_ZT(
    Z_T: np.ndarray,           # (m, n_factors)
    kernel: KernelSpec,
    q: QRHParams,
    vix_window_days: float,
    n_steps_vix: int,
):
    """
    Deterministic forward variance computed by CUDA solve_xi_cuda, then:
      VIX_T^2 ~ averaged frward variance
      VIX_T = 100 * sqrt(VIX_T^2)
    Returns:
      vix2_paths (m,), vix_paths (m,), plus means
    """
    Z_T = np.asarray(Z_T, dtype=np.float64)
    if Z_T.ndim != 2:
        raise ValueError("Z_T must be 2-D (m, n_factors).")

    delta = (float(vix_window_days) / 365.0) / float(n_steps_vix)

    xi_paths = sim.solve_xi_cuda(
        np.ascontiguousarray(Z_T, dtype=np.float64),
        float(q.a), float(q.b), float(q.c0), float(q.lam), float(q.eta),
        np.ascontiguousarray(np.asarray(kernel.c, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(kernel.gamma, dtype=np.float64)),
        float(delta),
        int(n_steps_vix),
    )
    xi_paths = np.asarray(xi_paths, dtype=np.float64)  # (m, n_steps_vix)

    vix2_paths = np.maximum(xi_paths.mean(axis=1), 0.0)
    vix_paths = 100.0 * np.sqrt(vix2_paths)

    out = dict(
        vix2_paths=vix2_paths,
        vix_paths=vix_paths,
        vix2_mean=float(vix2_paths.mean()),
        vix_level_mean=float(vix_paths.mean()),
        vix_level_sqrt_Evix2=100.0 * float(np.sqrt(max(float(vix2_paths.mean()), 0.0))),
    )
    return out


def spx_iv_surface_from_ST(S_T: np.ndarray, S0: float, T: float, k_spx: np.ndarray) -> dict:
    """
    fixed log-moneyness k_spx;  K = S0 * exp(k).
    Returns call prices + implied vols (BS with forward F=S0).
    """
    S_T = np.asarray(S_T, np.float64)
    k_spx = np.asarray(k_spx, np.float64).reshape(-1)
    Ks = float(S0) * np.exp(k_spx)

    pay = np.maximum(S_T[:, None] - Ks[None, :], 0.0)
    C = pay.mean(axis=0)  # (nK,)

    # F = float(S0)  # we don't use this as found that rough MC slightly disobeys martingale property
    F = float(S_T.mean())
    iv = np.array([implied_vol_black_forward(float(C[j]), F, float(Ks[j]), float(T))
                   for j in range(Ks.shape[0])], dtype=np.float64)

    return dict(K_spx=Ks, C_spx=C.astype(np.float64), iv_spx=iv.astype(np.float64))


def vix_iv_surface_from_vix_paths(vix_paths: np.ndarray, T: float, k_vix: np.ndarray) -> dict:
    """
    fixed log-moneyness, Fvix = E[VIX_T]
    Returns call prices + implied vols (BS on forward F=Fvix)
    """
    vix_paths = np.asarray(vix_paths, np.float64)
    k_vix = np.asarray(k_vix, np.float64).reshape(-1)

    Fvix = float(vix_paths.mean())
    Kvix = Fvix * np.exp(k_vix)

    pay = np.maximum(vix_paths[:, None] - Kvix[None, :], 0.0)
    C = pay.mean(axis=0)

    iv = np.array([implied_vol_black_forward(float(C[j]), Fvix, float(Kvix[j]), float(T))
                   for j in range(Kvix.shape[0])], dtype=np.float64)

    return dict(F_vix=Fvix, K_vix=Kvix, C_vix=C.astype(np.float64), iv_vix=iv.astype(np.float64))


# ======================
# main data gen run:
# ======================
def main():
    ap = argparse.ArgumentParser(
        "Generate synthetic data: (QRH params + z0 -> SPX IVS + VIX IVS) using fixed log-moneyness grids"
    )

    # output/resume
    ap.add_argument("--out_dir", type=str, default="data/synthetic_qrh_spx_vix")
    ap.add_argument("--run_name", type=str, default="run0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard_size", type=int, default=200)  # counts rows

    # test mode (no saving)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test_verbose", action="store_true", help="Print full SPX + VIX IV smiles (all strikes) per T")
    ap.add_argument("--test_z0_reps", type=int, default=2, help="How many random z0 draws per param set in test")
    ap.add_argument("--param_sets", type=str, default=
        "0.45,0.15,0.007,1.2,1.0;"
        "0.40,0.30,0.001,2.0,1.2;"
        "0.35,0.35,0.012,1.2,1.5"
    )

    # dataset size
    ap.add_argument("--n_samples", type=int, default=50_000)  # counts param draws
    ap.add_argument("--seed", type=int, default=123456)

    # model / kernel
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=0.10)
    ap.add_argument("--T_list", type=str, default="0.03,0.05,0.07,0.09", help="Comma-separated maturities")
    ap.add_argument("--H", type=float, default=0.01) #0.10)
    ap.add_argument("--n_factors", type=int, default=10)
    ap.add_argument("--x_star", type=float, default=3.92)

    # grids: log-moneyness
    ap.add_argument("--k_spx_list", type=str, default="-0.15,-0.12,-0.10,-0.08,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05")
    ap.add_argument("--k_vix_list", type=str, default="-0.10,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21")

    # sim
    ap.add_argument("--m_paths", type=int, default=50_000)
    ap.add_argument("--n_steps_mc", type=int, default=8000)
    ap.add_argument("--scheme", type=str, default="inv")
    ap.add_argument("--quad", type=str, default="left")

    ap.add_argument("--sigma_cap", type=float, default=None, help="apply phi-cap the quadratic term")

    # deterministic VIX
    ap.add_argument("--vix_window_days", type=float, default=30.0)
    ap.add_argument("--n_steps_vix", type=int, default=1024)

    # sampling ranges — ranges borrowed from reference paper
    ap.add_argument("--a_lo", type=float, default=0.1);   ap.add_argument("--a_hi", type=float, default=0.6)
    ap.add_argument("--b_lo", type=float, default=0.01);   ap.add_argument("--b_hi", type=float, default=0.5) 
    ap.add_argument("--c0_lo", type=float, default=1e-4); ap.add_argument("--c0_hi", type=float, default=0.03)
    ap.add_argument("--lam_lo", type=float, default=0.5);ap.add_argument("--lam_hi", type=float, default=2.5)
    ap.add_argument("--eta_lo", type=float, default=1.0);ap.add_argument("--eta_hi", type=float, default=1.5)

    # z0 sampling (10-dim)
    ap.add_argument("--z0_lo", type=float, default=-0.5)
    ap.add_argument("--z0_hi", type=float, default=0.5)

    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.run_name
    _ensure_dir(out_dir)
    meta_path = out_dir / "meta.json"
    state_path = out_dir / "state.json"

    # kernel
    c_w, gamma = fit_kernel_weights(n=args.n_factors, x_n=args.x_star, alpha=args.H + 0.5)
    kernel = KernelSpec(c=np.asarray(c_w, float), gamma=np.asarray(gamma, float))

    # grids
    T_grid = _parse_T_grid(args.T, args.T_list)
    k_spx  = _parse_float_list(args.k_spx_list)
    k_vix  = _parse_float_list(args.k_vix_list)

    cfg = {
        "S0": float(args.S0),
        "T_grid": list(map(float, T_grid)),
        "kernel": {"H": float(args.H), "n_factors": int(args.n_factors), "x_star": float(args.x_star)},
        "MC": {"m_paths": int(args.m_paths), "n_steps_mc": int(args.n_steps_mc), "scheme": str(args.scheme), "quad": str(args.quad), "seed": int(args.seed)},
        "grids": {"k_spx": list(map(float, k_spx)), "k_vix": list(map(float, k_vix))},
        "z0_range": [float(args.z0_lo), float(args.z0_hi)],
        "out_dir": str(out_dir),
        "note": "n_samples counts param draws; rows generated = n_samples * len(T_grid). Each row stores IV smiles on fixed log-moneyness grids.",
    }
    _log("[0] Config-")
    _log(json.dumps(cfg, indent=2))

    # meta --
    if (not args.resume) or (not meta_path.exists()):
        meta = dict(
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            S0=float(args.S0),
            T_grid=list(map(float, T_grid)),
            k_spx=list(map(float, k_spx)),
            k_vix=list(map(float, k_vix)),
            kernel=dict(
                H=float(args.H),
                n_factors=int(args.n_factors),
                x_star=float(args.x_star),
                c=list(map(float, kernel.c)),
                gamma=list(map(float, kernel.gamma)),
            ),
            sim=dict(
                m_paths=int(args.m_paths),
                n_steps_mc=int(args.n_steps_mc),
                scheme=str(args.scheme),
                quad=str(args.quad),
            ),
            vix=dict(window_days=float(args.vix_window_days), n_steps=int(args.n_steps_vix)),
            sampling=dict(
                params=["a","b","c0","lam","eta"],
                z0=f"U[{args.z0_lo},{args.z0_hi}]^{args.n_factors}"
            ),
            outputs=dict(
                per_row=["a","b","c0","lam","eta","z0[n_factors]","T",
                         "spx_iv[nK]","vix_iv[nKvix]",
                         "vix_forward","vix2_mean","vix_level_mean","vix_level_sqrt_Evix2"]
            ),
        )
        _save_json(meta_path, meta)

    # state
    if args.resume and state_path.exists():
        st = _load_json(state_path)
        next_id = int(st["next_id"])
        shard_id = int(st["shard_id"])
    else:
        next_id, shard_id = 0, 0
        _save_json(state_path, dict(next_id=next_id, shard_id=shard_id, seed=int(args.seed)))

    # ------------------------------
    # TEST MODE
    # ------------------------------
    if args.test:
        psets = parse_param_sets(args.param_sets)
        if len(psets) == 0:
            raise ValueError("param_sets is empty after parsing.")

        _log("\n[TEST] Running param_sets with random z0 draws (no saving) :)")
        base_rng = np.random.default_rng(int(args.seed) + 999)

        for i, (a, b, c0, lam, eta) in enumerate(psets, 1):
            _log(f"\nParam set {i}: a={a} b={b} c0={c0} lam={lam} eta={eta} 🩷")

            for r in range(int(args.test_z0_reps)):
                z0 = sample_z0_uniform(base_rng, args.n_factors, args.z0_lo, args.z0_hi)
                _log(f"  z0_rep={r}  z0[:5]={list(map(float, z0[:5]))} ...")

                for T in T_grid:
                    t0 = time.perf_counter()
                    q = QRHParams(a=a, b=b, c0=c0, lam=lam, eta=eta, z0=z0)

                    row_seed = int(args.seed + 10_000 * i + 100 * r + int(round(1e6 * float(T))))
                    S_T, Z_T, diag = simulate_qrh_terminal(
                        m_paths=int(args.m_paths),
                        kernel=kernel,
                        q=q,
                        S0=float(args.S0),
                        T=float(T),
                        n_steps_mc=int(args.n_steps_mc),
                        scheme=str(args.scheme),
                        quad=str(args.quad),
                        seed=row_seed,
                        sigma_cap=args.sigma_cap
                    )
                    

                    spx = spx_iv_surface_from_ST(S_T=S_T, S0=float(args.S0), T=float(T), k_spx=k_spx)

                    vix_paths_out = compute_vix_paths_from_ZT(
                        Z_T=Z_T,
                        kernel=kernel,
                        q=q,
                        vix_window_days=float(args.vix_window_days),
                        n_steps_vix=int(args.n_steps_vix),
                    )
                    vix = vix_iv_surface_from_vix_paths(vix_paths=vix_paths_out["vix_paths"], T=float(T), k_vix=k_vix)

                    dt = time.perf_counter() - t0

                    payload = dict(
                        T=float(T),
                        time_sec=float(dt),
                        vix_forward=float(vix["F_vix"]),
                        vix2_mean=float(vix_paths_out["vix2_mean"]),
                        vix_level_mean=float(vix_paths_out["vix_level_mean"]),
                        vix_level_sqrt_Evix2=float(vix_paths_out["vix_level_sqrt_Evix2"]),
                        spx_iv_atm=float(spx["iv_spx"][int(np.argmin(np.abs(k_spx - 0.0)))]),
                        vix_iv_atm=float(vix["iv_vix"][int(np.argmin(np.abs(k_vix - 0.0)))]),
                    )

                    if args.test_verbose:
                        payload.update(dict(
                            k_spx=list(map(float, k_spx)),
                            K_spx=list(map(float, spx["K_spx"])),
                            C_spx=list(map(float, spx["C_spx"])),
                            iv_spx=list(map(float, spx["iv_spx"])),
                            k_vix=list(map(float, k_vix)),
                            K_vix=list(map(float, vix["K_vix"])),
                            C_vix=list(map(float, vix["C_vix"])),
                            iv_vix=list(map(float, vix["iv_vix"])),   
                        ))
                        payload.update(diag)

                    _log(json.dumps(payload, indent=2))

        _log("\n[TEST] Done.")
        return

    # ----------------------------
    # RUN MODE (save shards)
    # ----------------------------
    ranges = dict(
        a=(args.a_lo, args.a_hi),
        b=(args.b_lo, args.b_hi),
        c0=(args.c0_lo, args.c0_hi),
        lam=(args.lam_lo, args.lam_hi),
        eta=(args.eta_lo, args.eta_hi),
    )

    buffer = {
        "sample_id": [],
        "row_seed": [],
        "a": [], "b": [], "c0": [], "lam": [], "eta": [],
        "z0": [],               # (rows, n_factors)
        "T": [],
        "spx_iv": [],           # (rows, nK)
        "vix_iv": [],           # (rows, nKvix)
        "spx_C": [],
        "vix_C": [],
        "vix_forward": [],      # per row
        "vix2_mean": [],
        "vix_level_mean": [],
        "vix_level_sqrt_Evix2": [],

        "spx_ST_mean": [],
        "spx_ST_std": [],
        "spx_logret_var": [],
    }

    t_run0 = time.perf_counter()

    while next_id < int(args.n_samples):
        sid = int(next_id)

        rng = np.random.default_rng(int(args.seed) + 1_000_003 * sid)
        prm = sample_params_uniform(rng, ranges)
        z0 = sample_z0_uniform(rng, args.n_factors, args.z0_lo, args.z0_hi)


        q = QRHParams(
            a=float(prm["a"]), b=float(prm["b"]), c0=float(prm["c0"]),
            lam=float(prm["lam"]), eta=float(prm["eta"]),
            z0=z0.astype(np.float64),
        )

        for T in T_grid:
            row_seed = int(args.seed + sid + int(round(1e6 * float(T))))

            S_T, Z_T, diag = simulate_qrh_terminal(
                m_paths=int(args.m_paths),
                kernel=kernel,
                q=q,
                S0=float(args.S0),
                T=float(T),
                n_steps_mc=int(args.n_steps_mc),
                scheme=str(args.scheme),
                quad=str(args.quad),
                seed=row_seed,
                sigma_cap=args.sigma_cap
            )
            

            spx = spx_iv_surface_from_ST(S_T=S_T, S0=float(args.S0), T=float(T), k_spx=k_spx)

            vix_paths_out = compute_vix_paths_from_ZT(
                Z_T=Z_T,
                kernel=kernel,
                q=q,
                vix_window_days=float(args.vix_window_days),
                n_steps_vix=int(args.n_steps_vix),
            )
            vix = vix_iv_surface_from_vix_paths(vix_paths=vix_paths_out["vix_paths"], T=float(T), k_vix=k_vix)

            buffer["sample_id"].append(sid)
            buffer["row_seed"].append(row_seed)

            buffer["a"].append(float(prm["a"]))
            buffer["b"].append(float(prm["b"]))
            buffer["c0"].append(float(prm["c0"]))
            buffer["lam"].append(float(prm["lam"]))
            buffer["eta"].append(float(prm["eta"]))

            buffer["z0"].append(z0.astype(np.float32))
            buffer["T"].append(float(T))

            buffer["spx_iv"].append(spx["iv_spx"].astype(np.float32))
            buffer["vix_iv"].append(vix["iv_vix"].astype(np.float32))

            ## new for snaity
            buffer["spx_C"].append(spx["C_spx"].astype(np.float32))
            buffer["vix_C"].append(vix["C_vix"].astype(np.float32))

            buffer["vix_forward"].append(float(vix["F_vix"]))
            buffer["vix2_mean"].append(float(vix_paths_out["vix2_mean"]))
            buffer["vix_level_mean"].append(float(vix_paths_out["vix_level_mean"]))
            buffer["vix_level_sqrt_Evix2"].append(float(vix_paths_out["vix_level_sqrt_Evix2"]))

            # diags
            buffer["spx_ST_mean"].append(diag["spx_ST_mean"])
            buffer["spx_ST_std"].append(diag["spx_ST_std"])
            buffer["spx_logret_var"].append(diag["spx_logret_var"])


        next_id += 1

        if (sid % 10) == 0:
            elapsed = time.perf_counter() - t_run0
            _log(f"[gen] sample_id={sid} elapsed={elapsed:.1f}s :)")

        if len(buffer["sample_id"]) >= int(args.shard_size):
            shard_path = out_dir / f"synthetic_qrh_spx_vix_ivs__{shard_id:06d}.npz"
            rows = {
                "sample_id": np.asarray(buffer["sample_id"], np.int64),
                "row_seed": np.asarray(buffer["row_seed"], np.int64),
                "a": np.asarray(buffer["a"], np.float32),
                "b": np.asarray(buffer["b"], np.float32),
                "c0": np.asarray(buffer["c0"], np.float32),
                "lam": np.asarray(buffer["lam"], np.float32),
                "eta": np.asarray(buffer["eta"], np.float32),
                "z0": np.stack(buffer["z0"], axis=0).astype(np.float32),          # (rows, n_factors)
                "T": np.asarray(buffer["T"], np.float32),
                "spx_iv": np.stack(buffer["spx_iv"], axis=0).astype(np.float32),  # (rows, nK)
                "vix_iv": np.stack(buffer["vix_iv"], axis=0).astype(np.float32),  # (rows, nKvix)
                "spx_C": np.stack(buffer["spx_C"], axis=0).astype(np.float32),
                "vix_C": np.stack(buffer["vix_C"], axis=0).astype(np.float32),
                "vix_forward": np.asarray(buffer["vix_forward"], np.float32),
                "vix2_mean": np.asarray(buffer["vix2_mean"], np.float32),
                "vix_level_mean": np.asarray(buffer["vix_level_mean"], np.float32),
                "vix_level_sqrt_Evix2": np.asarray(buffer["vix_level_sqrt_Evix2"], np.float32),

                "spx_ST_mean": np.asarray(buffer["spx_ST_mean"], np.float32),
                "spx_ST_std": np.asarray(buffer["spx_ST_std"], np.float32),
                "spx_logret_var": np.asarray(buffer["spx_logret_var"], np.float32),
            }
            save_shard_npz(shard_path, rows)

            shard_id += 1
            _save_json(state_path, dict(next_id=int(next_id), shard_id=int(shard_id), seed=int(args.seed)))
            _log(f"[save] shard={shard_id-1} rows={rows['sample_id'].shape[0]} -> {shard_path}")

            for k in buffer:
                buffer[k] = []

    # final flush
    if len(buffer["sample_id"]) > 0:
        shard_path = out_dir / f"synthetic_qrh_spx_vix_ivs__{shard_id:06d}.npz"
        rows = {
            "sample_id": np.asarray(buffer["sample_id"], np.int64),
            "row_seed": np.asarray(buffer["row_seed"], np.int64),
            "a": np.asarray(buffer["a"], np.float32),
            "b": np.asarray(buffer["b"], np.float32),
            "c0": np.asarray(buffer["c0"], np.float32),
            "lam": np.asarray(buffer["lam"], np.float32),
            "eta": np.asarray(buffer["eta"], np.float32),
            "z0": np.stack(buffer["z0"], axis=0).astype(np.float32),
            "T": np.asarray(buffer["T"], np.float32),
            "spx_iv": np.stack(buffer["spx_iv"], axis=0).astype(np.float32),
            "vix_iv": np.stack(buffer["vix_iv"], axis=0).astype(np.float32),
            "spx_C": np.stack(buffer["spx_C"], axis=0).astype(np.float32),
            "vix_C": np.stack(buffer["vix_C"], axis=0).astype(np.float32),
            "vix_forward": np.asarray(buffer["vix_forward"], np.float32),
            "vix2_mean": np.asarray(buffer["vix2_mean"], np.float32),
            "vix_level_mean": np.asarray(buffer["vix_level_mean"], np.float32),
            "vix_level_sqrt_Evix2": np.asarray(buffer["vix_level_sqrt_Evix2"], np.float32),

            "spx_ST_mean": np.asarray(buffer["spx_ST_mean"], np.float32),
            "spx_ST_std": np.asarray(buffer["spx_ST_std"], np.float32),
            "spx_logret_var": np.asarray(buffer["spx_logret_var"], np.float32),
        }
        save_shard_npz(shard_path, rows)

        shard_id += 1
        _save_json(state_path, dict(next_id=int(next_id), shard_id=int(shard_id), seed=int(args.seed)))
        _log(f"[save] final shard={shard_id-1} rows={rows['sample_id'].shape[0]} -> {shard_path} :)")

    _log("Done :)")


if __name__ == "__main__":
    main()
