import numpy as np
from math import gamma

def fit_kernel_weights(n: int, x_n: float, alpha: float = 0.51):
    """
    Inputs:
      n     : number of factors (always 10 for us)
      x_n   : geometric ratio (tuned to T) - borrowed value of 3.92 from references
      alpha : kernel param, alpha = H + 1/2 (always 0.51 for us)
    Returns:
      c   : shape (n,) weights
      gam : shape (n,) gammas
    """
    i = np.arange(1, n + 1, dtype=float)

    # weights c_i^n
    c = ( np.power(x_n, (1.0 - alpha) * (i - 0.5*n)) * (1.0 - np.power(x_n, alpha - 1.0))
        ) / ( (1.0 - alpha) * gamma(alpha) * gamma(1.0 - alpha) )
    # rates gamma_i^n
    gam = (1.0 - alpha) * np.power(x_n, (i - 1.0 - 0.5*n)) * ( np.power(x_n, 2.0 - alpha) - 1.0
          ) / ( (np.power(x_n, 1.0 - alpha) - 1.0 ) * (2.0 - alpha) )

    return c, gam


def fractional_kernel(t, H):
    """
    Original fractional kernel: K(t) = t^{alpha-1} / Gamma(alpha)
    """
    alpha = H + 0.5
    t = np.asarray(t)
    K = np.zeros_like(t, dtype=float)
    mask = t > 0
    K[mask] = t[mask]**(alpha-1) / gamma(alpha)
    return K


def _kernel_error_L2(n: int, x_n: float, T: float, n_grid: int = 600, alpha: float = 0.51) -> float:
    """
    Relic from testing
    L2 error of ||K - K^n|| for a given x_n
    """
    c, gam = fit_kernel_weights(n, x_n, alpha)
    t = np.linspace(1e-6, T, n_grid)
    K = fractional_kernel(t, H=alpha - .5)
    Kn = np.exp(-np.outer(t, gam)) @ c

    # approximated the error
    diff = (K - Kn) ** 2
    dt = t[1] - t[0]
    return float((diff[0] + diff[-1] + 2.0 * diff[1:-1].sum()) * dt / 2.0) # trapezium rule
