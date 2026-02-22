import numpy as np
import os

def _parse_float_list(s: str) -> np.ndarray:
    xs = [float(x) for x in s.split(",") if x.strip()]
    if len(xs) == 0:
        raise ValueError("Empty list after parsing.")
    return np.asarray(xs, dtype=np.float64)

def _parse_T_grid(T: float, T_list: str) -> np.ndarray:
    if T_list and T_list.strip():
        Ts = _parse_float_list(T_list)
        if np.any(Ts <= 0.0):
            raise ValueError("All maturities must be > 0.")
        return Ts
    T = float(T)
    if T <= 0.0:
        raise ValueError("--T must be > 0.")
    return np.asarray([T], dtype=np.float64)

def parse_K_list(K_arg):
    """
    Accepts: float (single K), or a comma/space separated string like "90,95,100".
    Returns: list of floats.
    """
    if isinstance(K_arg, (int, float)):
        return [float(K_arg)]
    if isinstance(K_arg, str):
        parts = [p.strip() for p in K_arg.replace(",", " ").split()]
        return [float(p) for p in parts if p]
    raise ValueError(f"Unrecognised K_arg: {K_arg}")


def parse_param_sets(s: str):
    """
    "a,b,c0,lam,eta; a,b,c0,lam,eta; ..."
    """
    out = []
    for tok in [t for t in s.split(";") if t.strip()]:
        a, b, c0, lam, eta = map(float, tok.split(","))
        out.append((a, b, c0, lam, eta))
    return out



def _parse_m_list(s: str) -> list[int]:
    """
    Parse m-list like '10000,20000,50000:100000:25000' (commas and/or slice notation).
    Examples:
      '10000,20000,30000' -> [10000,20000,30000]
      '20000:100000:20000' -> [20000,40000,60000,80000,100000]
      '10000, 25000:50000:12500' -> [10000,25000,37500,50000]
    """
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            parts = [p.strip() for p in tok.split(':')]
            if len(parts) == 2:
                start, stop = map(int, parts)
                step = start if start == stop else (stop - start)
                vals = list(range(start, stop + (1 if step>0 else -1), step))
            elif len(parts) == 3:
                start, stop, step = map(int, parts)
                vals = list(range(start, stop + (1 if step>0 else -1), step))
            else:
                raise ValueError(f"Bad slice token: {tok}")
            out.extend(vals)
        else:
            out.append(int(tok))
    # dedupe preserving order
    seen = set(); cleaned = []
    for v in out:
        if v not in seen:
            cleaned.append(v); seen.add(v)
    return cleaned