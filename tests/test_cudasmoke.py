import numpy as np
import pytest

@pytest.mark.gpu
def test_cuda_simulate_paths_smoke():
    import qrh_sim_cuda as sim
    from src.qrh_sim.kernel import fit_kernel_weights
    from src.qrh_sim.sim_utils import KernelSpec, QRHParams, LRHParams

    # tiny config
    S0 = 100.0
    T = 0.05
    n_steps = 64
    m = 512
    n_factors = 6
    x_star = 3.92
    alpha = 0.51

    c, gamma = fit_kernel_weights(n=n_factors, x_n=x_star, alpha=alpha)
    kernel = KernelSpec(c=np.asarray(c, float), gamma=np.asarray(gamma, float))

    q = QRHParams(a=0.35, b=-0.30, c0=0.01, lam=1.5, eta=0.6, z0=np.zeros(n_factors))
    l_dummy = LRHParams(alpha=0.0, beta=0.0, lam=q.lam, eta=q.eta)

    I_Q, I_L, (S_Q, S_L), Z_T, cZT = sim.simulate_paths_cuda(
        m=m,
        kernel=kernel,
        q=q,
        l=l_dummy,
        T=T,
        n_steps=n_steps,
        use_CV=False,
        scheme="inv",
        quad="left",
        S0=S0,
        return_ST=True,
        return_ZT=True,
        return_cZT=False,
        return_J=False,
        vcap_obj=None,
        v_floor=0.0,
        z_cap_obj=None,
        seed_obj=123,
        dW_shared_obj=None,
    )

    S_Q = np.asarray(S_Q)
    Z_T = np.asarray(Z_T)
    I_Q = np.asarray(I_Q)

    assert S_Q.shape == (m,)
    assert Z_T.shape == (m, n_factors)
    assert I_Q.shape == (m,)
    assert np.isfinite(S_Q).all()
    assert np.isfinite(Z_T).all()
    assert (I_Q >= 0).all()