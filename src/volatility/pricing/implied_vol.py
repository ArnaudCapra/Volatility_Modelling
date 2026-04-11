import numpy as np
import pandas as pd
from src.volatility.pricing.BlackScholes import black_price_forward
from scipy.optimize import brentq

def get_iv_from_forward(price, F, K, T, D, option_type):
    """Invert Black-76 implied volatility from an option price."""
    if any(pd.isna(x) for x in [price, F, K, T, D]):
        return np.nan
    if price <= 0 or F <= 0 or K <= 0 or T <= 0 or D <= 0:
        return np.nan

    def objective(sigma):
        return black_price_forward(F, K, T, D, sigma, option_type) - price

    try:
        f_low = objective(1e-6)
        f_high = objective(5.0)

        if not np.isfinite(f_low) or not np.isfinite(f_high):
            return np.nan

        if f_low * f_high > 0:
            for upper in [7.5, 10.0, 15.0]:
                f_high = objective(upper)
                if np.isfinite(f_high) and f_low * f_high <= 0:
                    return brentq(objective, 1e-6, upper, maxiter=200)
            return np.nan

        return brentq(objective, 1e-6, 5.0, maxiter=200)

    except Exception:
        return np.nan

