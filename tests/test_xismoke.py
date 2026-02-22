import numpy as np
import pytest

@pytest.mark.gpu
def test_cuda_solve_xi_smoke():
    import qrh_sim_cuda as sim
    from src.qrh_sim.kernel import fit_kernel_weights
    from src.qrh_sim.sim_utils import KernelSpec

    m = 256
    n_factors = 6
    n_steps_vix = 64
    delta = (30.0/365.0)/n_steps_vix

    # kernel
    c, gamma = fit_kernel_weights(n=n_factors, x_n=3.92, alpha=0.51)
    kernel = KernelSpec(c=np.asarray(c, float), gamma=np.asarray(gamma, float))

    # fake ZT (small random)
    rng = np.random.default_rng(0)
    ZT = rng.normal(size=(m, n_factors)).astype(np.float64)

    # QRH params (representative)
    a, b, c0, lam, eta = 0.35, -0.30, 0.01, 1.5, 0.6

    xi = sim.solve_xi_cuda(
        np.ascontiguousarray(ZT),
        float(a), float(b), float(c0), float(lam), float(eta),
        np.ascontiguousarray(kernel.c),
        np.ascontiguousarray(kernel.gamma),
        float(delta),
        int(n_steps_vix),
    )
    xi = np.asarray(xi, float)
    assert xi.shape == (m, n_steps_vix)
    assert np.isfinite(xi).all()
    assert (xi >= -1e-10).all()