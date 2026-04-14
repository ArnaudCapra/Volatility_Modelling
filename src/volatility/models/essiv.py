from src.volatility.config.config import WEIGHT_EPS
import numpy as np
def rho_of_theta(theta, rho_inf, rho_0, c_rho):
    """ρ(θ) = ρ_∞ + (ρ₀ − ρ_∞)·exp(−c·θ)  — varies from ρ₀ (short) to ρ_∞ (long)."""
    return rho_inf + (rho_0 - rho_inf) * np.exp(-c_rho * np.maximum(theta, WEIGHT_EPS))


def essvi_total_variance(k, theta, rho_inf, rho_0, c_rho, eta, gamma):
    """Natural SSVI surface with θ-dependent skew."""
    theta = np.asarray(theta, dtype=float)
    k     = np.asarray(k,     dtype=float)
    rho   = rho_of_theta(theta, rho_inf, rho_0, c_rho)
    phi   = eta / np.maximum(theta, WEIGHT_EPS) ** gamma
    return 0.5 * theta * (
        1.0 + rho * phi * k
        + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
    )