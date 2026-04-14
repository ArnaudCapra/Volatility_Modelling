import numpy as np
from src.volatility.config.config import WEIGHT_EPS

def _iv_local(fit, T, k_grid):
    return np.sqrt(np.clip(fit["predict"](T, k_grid), WEIGHT_EPS, None) / T)
