import numpy as np
from scipy.stats import norm

def black_price_forward(F, K, T, D, sigma, option_type):
    """Black-76 price with forward F and discount factor D."""
    if any(x is None for x in [F, K, T, D, sigma]):
        return np.nan

    # Vectorise sigma if it is an array, keep scalar path otherwise
    sigma = np.asarray(sigma, dtype=float)
    scalar_output = sigma.ndim == 0
    sigma = np.atleast_1d(sigma)

    if np.any(sigma <= 0) or F <= 0 or K <= 0 or T <= 0 or D <= 0:
        # Return nan with same shape as input sigma
        out = np.full(sigma.shape, np.nan)
        return float(out[0]) if scalar_output else out

    vsqrt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vsqrt
    d2 = d1 - vsqrt

    if option_type == "call":
        out = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        out = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(out[0]) if scalar_output else out

def _d1d2(tau, x, k, sigma):
    sq = sigma * np.sqrt(tau)
    d1 = (x - k) / sq + 0.5 * sq
    return d1, d1 - sq


def bs_vega(tau, x, k, sigma):
    """∂C/∂σ = S · φ(d₁) · √τ"""
    d1, _ = _d1d2(tau, x, k, sigma)
    return np.exp(x) * norm.pdf(d1) * np.sqrt(tau)


def G_func(tau, x, k, sigma):
    """
    G(t, x, σ) := (∂²/∂x² − ∂/∂x) BS(t, x, σ)

    Closed-form simplification (using e^x φ(d₁) = e^k φ(d₂)):
      ∂BS/∂x   = e^x N(d₁)
      ∂²BS/∂x² = e^x N(d₁) + e^x φ(d₁)/(σ√τ)
      ⟹  G = e^x φ(d₁) / (σ√τ)
    """
    d1, _ = _d1d2(tau, x, k, sigma)
    return np.exp(x) * norm.pdf(d1) / (sigma * np.sqrt(tau))


def H_func(tau, x, k, sigma):
    """
    H := ∂G/∂x = G · (1 − d₁/(σ√τ))

    Derivation:
      ∂G/∂x = G + e^x φ(d₁) · (−d₁) / (σ√τ)² · (1/σ√τ) ... simplifies to above.
    """
    d1, _ = _d1d2(tau, x, k, sigma)
    g     = G_func(tau, x, k, sigma)
    return g * (1.0 - d1 / (sigma * np.sqrt(tau)))

def dH_dk(tau, x, k, sigma, tau_min=1e-3):
    """Closed-form ∂H/∂k."""
    tau_r = np.maximum(tau, tau_min)
    m     = sigma * np.sqrt(tau_r)                    # σ√τ
    d1, _ = _d1d2(tau_r, x, k, sigma)
    g     = G_func(tau_r, x, k, sigma)
    return g * (1.0 + d1 * m - d1**2) / m**2