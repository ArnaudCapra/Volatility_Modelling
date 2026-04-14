from src.volatility.models.essiv import essvi_total_variance
import numpy as np
from src.volatility.config.config import WEIGHT_EPS


def _iv_essvi(fit, T, k_grid):
    theta = float(np.interp(T, fit["T_knots"], fit["theta_knots"]))
    w = essvi_total_variance(k_grid, theta,
                             fit["rho_inf"], fit["rho_0"],
                             fit["c_rho"],   fit["eta"], fit["gamma"])
    return np.sqrt(np.clip(w, WEIGHT_EPS, None) / T)

