from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from typing import Optional, Tuple

# ====================== Data classes ======================

@dataclass
class KernelSpec:
    c: np.ndarray          # (n,)
    gamma: np.ndarray      # (n,)

@dataclass
class QRHParams:
    a: float
    b: float
    c0: float
    lam: float
    eta: float
    z0: np.ndarray         # (n,)

@dataclass
class LRHParams:
    alpha: float           # intercept in V^L = alpha + beta * (Z @ c)
    beta: float            # slope     in V^L = alpha + beta * (Z @ c)
    eta: float
    lam: float = 0.0

@dataclass
class SimConfig:
    T: float
    n_steps: int
    z_cap: float = 1e6
    v_floor: float = 1e-10
    sigma_cap: float = 10.0
    seed: int | None = None


# Helper to convert from a set vol_cap to an x_star for it 
def x_cap_from_sigma_cap(qpars, sigma_cap: float) -> float:
    # sigma_cap is annualised vol cap
    a = float(qpars.a)
    c0 = float(qpars.c0)
    Vmax = sigma_cap * sigma_cap
    if a <= 0:
        return 0.0
    return math.sqrt(max(0.0, (Vmax - c0) / a))

