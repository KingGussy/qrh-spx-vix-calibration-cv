import math
import numpy as np

from src.qrh_sim.pricing_utils import black_call_forward, implied_vol_black_forward

def test_black_forward_iv_roundtrip():
    F = 100.0
    K = 105.0
    T = 0.25
    sigma = 0.6
    C = black_call_forward(F, K, T, sigma)
    sigma_hat = implied_vol_black_forward(C, F, K, T)
    assert math.isfinite(sigma_hat)
    assert abs(sigma_hat - sigma) < 1e-6

def test_black_forward_bounds_handled():
    F, K, T = 100.0, 100.0, 0.5
    intrinsic = max(F - K, 0.0)
    # below intrinsic should clip to near intrinsic and return ~0 vol
    sigma0 = implied_vol_black_forward(intrinsic - 1.0, F, K, T)
    assert sigma0 >= 0.0
    # near upper bound should return large vol (finite)
    sigma_big = implied_vol_black_forward(F - 1e-10, F, K, T)
    assert math.isfinite(sigma_big)