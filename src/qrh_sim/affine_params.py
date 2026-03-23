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
