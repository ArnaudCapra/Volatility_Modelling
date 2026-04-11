import numpy as np
import pandas as pd
from src.volatility.market.parity import EPS

def quote_weight(iv, T, rel_spread, oi=None):
    """Maturity-aware weight: tighter spreads, higher OI, longer tenor."""
    iv = max(float(iv), EPS)
    T  = max(float(T),  EPS)

    spread_term = 1.0 / (float(rel_spread) + EPS) if pd.notna(rel_spread) and rel_spread > 0 else 1.0

    oi_term = 1.0
    if oi is not None and pd.notna(oi):
        oi_term = float(np.clip(float(oi) / 100.0, 0.0, 1.0))

    time_term = T / (T + 0.05)
    iv_term   = 1.0 / iv

    return float(np.clip(spread_term * oi_term * time_term * iv_term, 1e-8, 1e8))