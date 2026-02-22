import numpy as np
import time

def pilot_beta_given_alpha_from_qrh_cuda(
    sim_mod,
    kernel,
    qpars,
    alpha_fixed: float,
    T: float,
    n_steps: int,
    m_pilot: int,
    scheme: str,
    quad: str,
    S0: float,
    seed: int,
):
    """
    Fit beta in the regression I_Q ≈ alpha*T + beta*J where J = ∫ Z_sum dt,
    holding alpha fixed.

    Requires pilot_ab_from_qrh_cuda to return moments:
      E_IQ, E_J, VarJ, CovIJ
    """
    stats = pilot_ab_from_qrh_cuda(
        sim_mod=sim_mod,
        kernel=kernel,
        qpars=qpars,
        lrh_zero=None,          # or LRHParams(0,0,...) depending on your existing function signature
        T=T,
        n_steps=n_steps,
        m_pilot=m_pilot,
        scheme=scheme,
        quad=quad,
        S0=S0,
        seed=seed,
    )

    # beta = Cov(J, IQ - alpha*T) / Var(J)
    cov = float(stats["covIJ"]) - float(alpha_fixed * T) * 0.0  # alpha*T is constant so doesn't change cov
    varJ = float(stats["varJ"])
    if varJ <= 0.0:
        return 0.0
    beta = cov / varJ
    return beta



def fit_alpha_beta_from_IJ(I, J, T):
    """
    Fit I ≈ alpha*T + beta*J by OLS across paths.
      beta = Cov(I,J)/Var(J)
      alpha = (E[I] - beta*E[J]) / T
    """
    I = np.asarray(I, float).ravel()
    J = np.asarray(J, float).ravel()

    EJ = J.mean()
    EI = I.mean()
    varJ = max((J*J).mean() - EJ*EJ, 0.0)
    varI = max((I*I).mean() - EI*EI, 0.0)
    covIJ = (I*J).mean() - EI*EJ

    # diagnostic snaity checks
    # Correlation (population)
    denom = np.sqrt(varI*varJ) + 1e-30
    corr  = covIJ / denom
    # Cauchy–Schwarz sanity (should never be violated meaningfully)
    cs_gap = abs(covIJ) - np.sqrt(varI*varJ)

    beta = 0.0 if varJ <= 1e-30 else covIJ / varJ
    alpha = (EI - beta*EJ) / float(T)
    return float(alpha), float(beta), float(varJ), float(covIJ), float(varI), float(corr), float(cs_gap)

def pilot_ab_from_qrh_cuda(
    sim_mod,
    kernel, qpars, lrh_zero,   # lrh_zero = LRHParams(0,0,lam,eta) or LRHParams(0,0,0,0)
    T: float, n_steps: int,
    m_pilot: int,
    scheme: str = "inv",
    quad: str = "left",
    S0: float = 100.0,
    seed: int = 123456,
    x_cap=None,
    v_floor: float = 0.0,
    z_cap_obj=None,
):
    t0 = time.perf_counter()

    ret = sim_mod.simulate_paths_cuda(
        m_pilot, kernel, qpars, lrh_zero,
        T, n_steps,
        False,                 # use_CV=False
        scheme, quad,
        S0,
        False,                 # return_ST
        False,                 # return_ZT
        False,                 # return_cZT
        True,                  # return_J
        x_cap,
        v_floor,
        z_cap_obj,
        seed,
        None                   # dW_shared_obj
    )

    # Expected (with return_J=True):
    # (I_Q, I_L, (S_Q,S_L), ZT, cZT, J_Q, J_L)
    if not (isinstance(ret, tuple) and len(ret) >= 7):
        raise RuntimeError(
            f"simulate_paths_cuda returned len={len(ret) if isinstance(ret, tuple) else '??'}, "
            "but return_J=True expects >=7. Check the binding + return tuple."
        )

    I_Q = np.asarray(ret[0], float)
    J_Q = np.asarray(ret[5], float)

    thr = 10
    print("P(I>10) =", float(np.mean(I_Q > thr)))
    print("top  5 I =", np.sort(I_Q)[-5:])

    # temorary diag
    def _stat(x):
        x = np.asarray(x, float).ravel()
        return {
            "mean": float(x.mean()),
            "std":  float(x.std()),
            "min":  float(x.min()),
            "p01":  float(np.quantile(x, 0.01)),
            "p50":  float(np.quantile(x, 0.50)),
            "p99":  float(np.quantile(x, 0.99)),
            "max":  float(x.max()),
        }

    print("DIAG I:", _stat(I_Q))
    print("DIAG J:", _stat(J_Q))
    ####

    alpha, beta, varJ, covIJ, varI, corr, cs_gap = fit_alpha_beta_from_IJ(I_Q, J_Q, T)
    out = {
        "alpha": alpha,
        "beta": beta,
        "varJ": float(varJ),
        "varI": float(varI),
        "covIJ": float(covIJ),
        "cs_gap": float(cs_gap),
        "E_I": float(I_Q.mean()),
        "E_J": float(J_Q.mean()),
        "time_sec": float(time.perf_counter() - t0),
    }
    return out


















# ==========================
# everything below is old (as in, as of today 19/12/25 I dont remember any of it)
# 






# ===============================
# 
# # from pathlib import Path
# import sys
# ROOT = Path(__file__).resolve().parents[1]
# SRC = ROOT / "src"
# sys.path.insert(0, str(SRC)) 

import numpy as np
from src.qrh_sim.pricing_utils import bs_price_from_I

def qrh_V0_from_params(qpars: "QRHParams", kernel: "KernelSpec") -> float:
    """
    QRH initial variance when Z0 is given (often Z0=0): V0 = a (c^T z0 - b)^2 + c0.
    If z0==0, this is simply a*b^2 + c0.
    """
    z0 = np.asarray(qpars.z0, float).reshape(-1)
    c  = np.asarray(kernel.c, float).reshape(-1)
    return float(qpars.a * (float(np.dot(c, z0)) - qpars.b)**2 + qpars.c0)


def calibrate_alpha_beta_quick(*,
    # model / grids / pricing inputs
    kernel: "KernelSpec", simcfg: "SimConfig", qpars: "QRHParams",
    S0: float, K: float, T: float,
    t_grid_fft: np.ndarray, lambdas: np.ndarray,
    etaL: float, lamL: float,
    # anchors (provide two candidates; we’ll pick the better one)
    alpha_ols: float,
    alpha_qrh0: float,
    # search details
    beta_grid: np.ndarray = None,
    m_cal: int = 3000,
    seed: int = 123456
) -> dict:
    """
    Lightweight calibration:
      - Try two alpha anchors: alpha_ols, alpha_qrh0.
      - For each alpha, scan beta_grid and pick beta that minimizes |C_LRH_fft - C_LRH_mc|.
      - Returns dict with chosen alpha,beta and a short report.

    Notes:
      * Uses Convention B′: g0(t) = alpha + (K * (lamL * alpha))(t)   (theta~(t)=lamL*alpha).
      * sigma for FFT Riccati = etaL * beta.
      * Pathwise MC LRH price is computed from IL via BS-on-I.
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.4, 1.2, 17)  # coarse but effective

    m_use = int(min(m_cal, max(1000, m_cal), 100000))  # keep calibration quick

    def build_g0(grid, alpha):
        return g0_grid_(grid, kernel, (lambda s: lamL*alpha), V0=alpha)

    best = None
    report = []

    for alpha in [float(alpha_ols), float(alpha_qrh0)]:
        # FFT-side g0 on FFT grid (alpha only)
        g0_fft = build_g0(t_grid_fft, alpha)

        for beta in beta_grid:
            sig_fft = etaL * float(beta)

            # FFT LRH price at (alpha,beta)
            phi_line = precompute_cf_on_midline(
                T=T, t_grid=t_grid_fft, g0_grid=g0_fft, kernel=kernel,
                S0=S0, lam=lamL, sig=sig_fft, rho=1.0, lambdas=lambdas
            )
            C_fft = price_midline_uncentered(S0, K, lambdas, phi_line)

            # SIM-side LRH MC price at (alpha,beta)
            t_grid_sim = np.linspace(0.0, T, simcfg.n_steps+1)
            g0_sim = build_g0(t_grid_sim, alpha)
            lpars = LRHParams(alpha=float(alpha), beta=float(beta), eta=etaL, lam=lamL)

            # mini-sim for speed
            rng = np.random.default_rng(seed)
            dW_tmp = rng.standard_normal((m_use, simcfg.n_steps)) * np.sqrt(T/simcfg.n_steps)
            _, _, _, _, _, IL, _ = simulate_paths6(
                m=m_use, kernel=kernel, q=qpars, l=lpars, cfg=simcfg,
                _g0_sim=g0_sim, dW_shared=dW_tmp
            )
            C_mc = float(np.mean(bs_price_from_I(IL, S0, K, T)))

            gap = abs(C_fft - C_mc)
            report.append({"alpha": alpha, "beta": float(beta), "C_fft": C_fft, "C_mc": C_mc, "gap": gap})

            if (best is None) or (gap < best["gap"]):
                best = {"alpha": alpha, "beta": float(beta), "C_fft": C_fft, "C_mc": C_mc, "gap": gap}

    return {"best": best, "report": report}



##### above is newest (sept)

def fit_lrh_alpha_beta_IJ(Zsum_paths: np.ndarray, VQ_paths: np.ndarray, dt: float, T: float):
    """
    Closed-form OLS over integrals: minimize E[(I_Q - (alpha*T + beta*J))^2],
    where J = ∫ Z_sum dt and I_Q = ∫ V^Q dt.
    """
    J  = Zsum_paths[:, 1:].sum(axis=1) * dt
    IQ = VQ_paths[:, 1:].sum(axis=1)   * dt
    Jm, IQm = J.mean(), IQ.mean()
    varJ    = np.mean((J - Jm)**2)
    covIJ   = np.mean((J - Jm)*(IQ - IQm))
    beta  = covIJ / (varJ + 1e-16)
    alpha = (IQm - beta * Jm) / T
    return float(alpha), float(beta)

# ---------- 1D helpers ----------
def _bracket_min(fun, x0, step, grow=1.8, maxit=40):
    f0 = fun(x0)
    L, R = x0 - step, x0 + step
    fL, fR = fun(L), fun(R)
    it = 0
    while not (f0 <= fL and f0 <= fR) and it < maxit:
        if fL < fR:
            R += step; fR = fun(R)
        else:
            L -= step; fL = fun(L)
        step *= grow; it += 1
    return L, R

def _golden_min(fun, a, b, iters=60):
    phi = 0.5 * (1.0 + 5.0**0.5)
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc, fd = fun(c), fun(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = fun(d)
    x = 0.5 * (a + b)
    return x, fun(x)

# ====================== Price-space fit (stable) ======================

def fit_lrh_alpha_beta_price(
    Zsum_paths: np.ndarray,
    VQ_paths:   np.ndarray,
    dt: float,
    S0: float, K: float, T: float,
    objective: str = "mse",          # or "corr"
    beta0: float | None = None       # optional slope seed in original units
):
    """
    Borrowed-Z pilot: J := ∫ Z^Q dt. We center/scale J and search over a
    dimensionless slope beta_tilde = beta * sJ / sI in a compact interval.

    For each beta_tilde:
      beta = beta_tilde * sI / sJ,
      I^L  = muI + delta*T + beta*(J - muJ).
    Inner loop: golden-section over delta to minimize price MSE (or maximize Corr).
    """
    # integrals and QRH prices
    J   = Zsum_paths[:, 1:].sum(axis=1) * dt
    IQ  = VQ_paths[:, 1:].sum(axis=1)   * dt
    X   = bs_price_from_I(IQ, S0, K, T)

    muI, muJ = float(IQ.mean()), float(J.mean())
    sI,  sJ  = float(IQ.std(ddof=0)), float(J.std(ddof=0))
    if sJ < 1e-14 or sI < 1e-14:
        # degenerate input -> fall back to integral OLS
        alpha_IJ, beta_IJ = fit_lrh_alpha_beta_IJ(Zsum_paths, VQ_paths, dt, T)
        return alpha_IJ, beta_IJ, {
            "beta0": None, "beta0_tilde": None,
            "alpha_star": float(alpha_IJ), "beta_star": float(beta_IJ),
            "beta_tilde": 0.0, "rho": 0.0,
            "mse": float(np.mean((X - X.mean())**2)), "fallback": True
        }

    # dimensionless seed for beta_tilde
    if beta0 is None:
        rho0 = float(np.cov(IQ, J, bias=True)[0,1] / (sI * sJ))  # Corr(IQ,J)
        bt0  = np.clip(rho0, -0.99, 0.99)
        beta0 = bt0 * sI / sJ
    else:
        bt0 = float(beta0 * sJ / sI)
        bt0 = np.clip(bt0, -1.5, 1.5)

    # inner: minimize price loss over delta for fixed beta_tilde
    def solve_delta(beta_tilde: float):
        beta = beta_tilde * sI / sJ
        Jc   = J - muJ
        def loss_delta(delta: float) -> float:
            IL = muI + delta*T + beta * Jc
            Y  = bs_price_from_I(IL, S0, K, T)
            if objective == "corr":
                return -float(np.corrcoef(X, Y)[0,1])
            else:
                return float(np.mean((Y - X) ** 2))

        step = 0.2 * (sI / max(T, 1.0))  # units: variance per time
        dL, dR = _bracket_min(loss_delta, 0.0, step=step)
        delta_star, loss_star = _golden_min(loss_delta, dL, dR)

        IL = muI + delta_star*T + beta * Jc
        Y  = bs_price_from_I(IL, S0, K, T)
        rho = float(np.corrcoef(X, Y)[0,1])
        mse = float(np.mean((Y - X) ** 2))
        return float(delta_star), float(beta), rho, mse, float(loss_star)

    # outer: loss as a function of beta_tilde
    def loss_bt(bt: float):
        delta_star, beta, rho, mse, loss = solve_delta(bt)
        return loss, delta_star, beta, rho, mse

    # bracket & optimize beta_tilde near bt0
    def f_only(bt): return loss_bt(bt)[0]
    btL, btR = _bracket_min(f_only, bt0, step=0.25)  # compact, dimensionless
    bt_star, _ = _golden_min(f_only, btL, btR)

    # recover optimal params
    delta_star, beta_star, rho_star, mse_star, _ = loss_bt(bt_star)
    alpha_star = (muI - beta_star * muJ) / T + delta_star

    diags = {
        "beta0": float(beta0),
        "beta0_tilde": float(bt0),
        "alpha_star": float(alpha_star),
        "beta_star": float(beta_star),
        "beta_tilde": float(bt_star),
        "rho": float(rho_star),
        "mse": float(mse_star)
    }
    return float(alpha_star), float(beta_star), diags