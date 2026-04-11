import numpy as np
import pandas as pd
from src.volatility.calibration.weights import quote_weight

def compute_fd_atm_skew(df_smile, dk_band=0.05):
    """
    Per-maturity finite-difference ATM skew from raw market IVs.

    For each maturity, selects all strikes within ±dk_band of ATM (k=0),
    fits a weighted linear regression IV ~ alpha + skew*k,
    and returns the slope as the FD skew estimate.

    Parameters
    ----------
    df_smile  : output of build_ssvi_dataset (has columns k, IV, T, T_days, weight)
    dk_band   : half-width in log-moneyness around ATM used for the regression

    Returns
    -------
    pd.DataFrame with columns [T_days, T, fd_atm_skew, n_strikes]
    """
    rows = []
    for tdays, sl in sorted(df_smile.groupby("T_days"), key=lambda x: x[0]):
        T   = float(sl["T"].iloc[0])
        sl  = sl.copy()
        sl  = sl[np.isfinite(sl["k"]) & np.isfinite(sl["IV"])].copy()

        near = sl[np.abs(sl["k"]) <= dk_band].copy()
        if len(near) < 3:
            rows.append({"T_days": tdays, "T": T, "fd_atm_skew": np.nan, "n_strikes": len(near)})
            continue

        k_n  = near["k"].astype(float).to_numpy()
        iv_n = near["IV"].astype(float).to_numpy()

        # Reuse quote_weight if available, otherwise fall back to uniform
        if "weight" in near.columns:
            wts = near["weight"].astype(float).to_numpy()
        else:
            wts = np.array([
                quote_weight(r["IV"], r["T"], r.get("REL_SPREAD"), r.get("OI"))
                for _, r in near.iterrows()
            ], dtype=float)
        wts = np.clip(wts, 1e-8, 1e8)

        # Weighted linear regression: IV = a + skew * k
        X   = np.column_stack([np.ones_like(k_n), k_n])
        W   = np.diag(wts)
        XtW = X.T @ W
        try:
            beta = np.linalg.solve(XtW @ X, XtW @ iv_n)
            skew = float(beta[1])
        except np.linalg.LinAlgError:
            skew = np.nan

        rows.append({"T_days": tdays, "T": T, "fd_atm_skew": skew, "n_strikes": len(near)})

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)