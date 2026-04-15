"""Shared pricing math for debasement-LEAPS tooling.

Black-Scholes with continuous dividend yield q, plus a convenience wrapper
that sets q = r - μ to express a debasement-drift assumption. Imported by
the TUI and both scanners.
"""
import math
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes call with continuous dividend yield q."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def debase_fair_value(S: float, K: float, T: float, r: float, sigma: float, mu: float) -> float:
    """BS call where the underlying is assumed to drift at rate μ (not r).

    Implemented by setting q = r - μ. If μ > r, q < 0 and calls get richer.
    Mathematically equivalent to Garman-Kohlhagen with r_f = r - μ, r_d = r.
    """
    return bs_call(S, K, T, r, sigma, q=r - mu)


def breakeven_annualized_growth(S: float, K: float, T: float, premium: float) -> float:
    """Annualized growth rate needed for the long call to break even at expiry."""
    if T <= 0 or S <= 0:
        return float("nan")
    target = K + premium
    if target <= 0:
        return float("nan")
    return (target / S) ** (1 / T) - 1
