import numpy as np
from scipy.stats import norm

def black_price_forward(F, K, T, D, sigma, option_type):
    """Black-76 price with forward F and discount factor D."""
    if any(x is None for x in [F, K, T, D, sigma]):
        return np.nan
    if F <= 0 or K <= 0 or T <= 0 or D <= 0 or sigma <= 0:
        return np.nan

    vsqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vsqrt
    d2 = d1 - vsqrt

    if option_type == "call":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
