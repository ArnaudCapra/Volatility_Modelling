import numpy as np
from src.volatility.models.essiv import essvi_total_variance, rho_of_theta
from src.volatility.config.config import WEIGHT_EPS

def _unpack_essvi_params(x, n_t):
    theta_knots = np.cumsum(np.exp(x[:n_t]))
    rho_inf = np.tanh(x[n_t])
    rho_0   = np.tanh(x[n_t + 1])
    c_rho   = np.exp(x[n_t + 2])          # must be positive
    eta     = np.exp(x[n_t + 3])
    gamma   = 1.0 / (1.0 + np.exp(-x[n_t + 4]))
    return theta_knots, rho_inf, rho_0, c_rho, eta, gamma


def essvi_residuals(x, data, T_knots, lam_bfly=25.0):
    n_t = len(T_knots)
    theta_knots, rho_inf, rho_0, c_rho, eta, gamma = _unpack_essvi_params(x, n_t)

    T     = data["T"].to_numpy(dtype=float)
    k     = data["k"].to_numpy(dtype=float)
    w_obs = data["total_variance"].to_numpy(dtype=float)
    wts   = data["weight"].to_numpy(dtype=float)

    theta_T = np.interp(T, T_knots, theta_knots)
    w_model = essvi_total_variance(k, theta_T, rho_inf, rho_0, c_rho, eta, gamma)
    res     = (w_model - w_obs) * np.sqrt(wts)

    # Butterfly no-arb penalty at each knot
    rho_k     = rho_of_theta(theta_knots, rho_inf, rho_0, c_rho)
    phi_knots = eta / np.maximum(theta_knots, WEIGHT_EPS) ** gamma
    arb       = theta_knots * phi_knots * (1.0 + np.abs(rho_k)) - 4.0
    arb_pen   = np.sqrt(lam_bfly) * np.maximum(0.0, arb)

    return np.concatenate([res, arb_pen])