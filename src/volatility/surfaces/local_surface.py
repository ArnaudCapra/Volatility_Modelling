import numpy as np
from src.volatility.market.parity import EPS

def _iv_local(fit, T, k_grid):
    return np.sqrt(np.clip(fit["predict"](T, k_grid), EPS, None) / T)
