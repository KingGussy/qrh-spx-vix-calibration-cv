import numpy as np
import math
from scipy.stats import norm

# in our data gen
def norm_cdf(x: float) -> float:
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def black_call_forward(F: float, K: float, T: float, sigma: float) -> float:
    # Black(76) call on forward, undiscounted (df=1).
    if T <= 0.0:
        return max(F - K, 0.0)
    if sigma <= 0.0:
        return max(F - K, 0.0)
    vol_sqrt = sigma * math.sqrt(T)
    if vol_sqrt < 1e-12:
        return max(F - K, 0.0)
    lnFK = math.log(F / K)
    d1 = (lnFK + 0.5 * vol_sqrt * vol_sqrt) / vol_sqrt
    d2 = d1 - vol_sqrt
    return F * norm_cdf(d1) - K * norm_cdf(d2)

def implied_vol_black_forward(C: float, F: float, K: float, T: float,
                              sigma_lo: float = 1e-8, sigma_hi: float = 5.0) -> float:
    """
    implied vol for undiscounted Black call:
      C = BlackCall(F,K,T,sigma)
    """
    # if T <= 0.0:
    #     return 0.0

    # no-arb bounds (undiscounted):
    intrinsic = max(F - K, 0.0)
    upper = F  # call <= forward
    # Clip slightly to avoid bracketing issues at boundaries
    eps = 1e-12
    Cc = min(max(C, intrinsic + eps), upper - eps)

    # If almost intrinsic, vol ~ 0
    if Cc - intrinsic < 1e-10:
        return 0.0

    # Ensure sigma_hi brackets
    f_lo = black_call_forward(F, K, T, sigma_lo) - Cc
    f_hi = black_call_forward(F, K, T, sigma_hi) - Cc
    if f_lo >= 0.0:
        return sigma_lo
    while f_hi < 0.0 and sigma_hi < 20.0:
        sigma_hi *= 2.0
        f_hi = black_call_forward(F, K, T, sigma_hi) - Cc

    if f_hi < 0.0:
        # extreme price near upper bound -> huge vol
        return float(sigma_hi)

    # Bisection
    lo, hi = sigma_lo, sigma_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = black_call_forward(F, K, T, mid) - Cc
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))



# in the CV demo
def bs_call_price(S0, K, T, sigma, r=0.0, q=0.0):
    # European call price under BS

    # if T <= 0:
    #     return max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    sigma = max(float(sigma), 1e-12)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_implied_vol_call(price, S0, K, T, r=0.0, q=0.0, lo=1e-6, hi=5.0, tol=1e-7, iters=80):
    # Bisection
    if T <= 0:
        return np.nan
    pv = lambda sigma: bs_call_price(S0,K,T,sigma,r,q)
    plo, phi = pv(lo), pv(hi)

    # Expand hi if needed
    while phi < price and hi < 1000: hi *= 2.0; phi = pv(hi)

    if price <= plo: return lo
    if price >= phi: return hi

    for _ in range(iters):
        mid = 0.5*(lo+hi)
        pm  = pv(mid)
        if abs(pm - price) < tol:
            return mid
        if pm > price:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


# greeks for hedging
def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    # if T <= 0:
    #     return 1.0 if S > K else 0.0
    # if sigma <= 0:
    #     return 1.0 if math.exp(-q * T) * S > math.exp(-r * T) * K else 0.0
    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    return math.exp(-q * T) * norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    # if T <= 0 or sigma <= 0:
    #     return 0.0
    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    return math.exp(-q * T) * S * math.sqrt(T) * norm_pdf(d1)





