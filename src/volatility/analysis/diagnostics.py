import numpy as np
import pandas as pd
from src.volatility.calibration.weights import quote_weight
from src.volatility.models.essiv import rho_of_theta
from src.volatility.market.parity import EPS
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

def compute_atm_forward_skew(fit_essvi, fit_local, df_smile=None, dk_band=0.05):
    """
    ATM forward skew between consecutive maturity pairs.

    Forward skew definition (T1 < T2, τ = T2 - T1):
        w_fwd(k) = w(k, T2) - w(k, T1)          forward total variance
        σ_fwd    = sqrt(w_fwd(0) / τ)            forward ATM vol
        fwd_skew = [∂w_fwd/∂k|_{k=0}] / [2·σ_fwd·τ]

    For SSVI/eSSVI at k=0:
        w(0, T)       = θ(T)                     [exact]
        ∂w/∂k|_{k=0} = θ(T)·ρ(T)·φ(T)          [exact]

    For Local Quad  w(k,T) = a(T) + b(T)·k + c(T)·k²:
        w(0, T)       = a(T)
        ∂w/∂k|_{k=0} = b(T)

    For FD market: per-slice weighted quadratic fit to raw total variance
        near k=0 to extract w(0) and ∂w/∂k|_{k=0} from market data.

    Parameters
    ----------
    fit_essvi : output of fit_essvi_surface
    fit_local : output of fit_localized_surface
    df_smile  : output of build_ssvi_dataset (required for FD method)
    dk_band   : half-width in k used for the FD per-slice local fit

    Returns
    -------
    dict with keys 'eSSVI', 'local', 'fd' (optional),
    each a DataFrame with [T1, T2, T_mid, fwd_skew].
    T_mid = √(T1·T2) is used as the x-axis (geometric midpoint).
    """
    results = {}

    # ── eSSVI ────────────────────────────────────────────────────────────
    T_k     = fit_essvi["T_knots"]
    th_k    = fit_essvi["theta_knots"]
    rho_inf = fit_essvi["rho_inf"]
    rho_0   = fit_essvi["rho_0"]
    c_rho   = fit_essvi["c_rho"]
    eta     = fit_essvi["eta"]
    gamma   = fit_essvi["gamma"]

    rows_e = []
    for i in range(1, len(T_k)):
        T1, T2   = float(T_k[i - 1]), float(T_k[i])
        th1, th2 = float(th_k[i - 1]), float(th_k[i])
        tau = T2 - T1
        if tau <= 0:
            continue

        rh1 = float(rho_of_theta(th1, rho_inf, rho_0, c_rho))
        rh2 = float(rho_of_theta(th2, rho_inf, rho_0, c_rho))
        ph1 = eta / max(th1, EPS) ** gamma
        ph2 = eta / max(th2, EPS) ** gamma

        # ∂w/∂k|_{k=0} = θ·ρ·φ  for each knot
        dw1 = th1 * rh1 * ph1
        dw2 = th2 * rh2 * ph2

        # w(0,T) = θ(T), so w_fwd_atm = θ2 - θ1
        w_fwd_atm = th2 - th1
        if w_fwd_atm <= 0:
            continue

        sigma_fwd = np.sqrt(w_fwd_atm / tau)
        fwd_skew  = (dw2 - dw1) / (2.0 * sigma_fwd * tau)

        rows_e.append({"T1": T1, "T2": T2,
                        "T_mid": np.sqrt(T1 * T2), "fwd_skew": fwd_skew})

    results["eSSVI"] = pd.DataFrame(rows_e)

    # ── Local Quadratic ──────────────────────────────────────────────────
    lsum = fit_local["summary"].sort_values("T").reset_index(drop=True)
    T_l  = lsum["T"].to_numpy(dtype=float)
    a_l  = lsum["a"].to_numpy(dtype=float)   # w(0,T) = a
    b_l  = lsum["b"].to_numpy(dtype=float)   # ∂w/∂k|_{k=0} = b

    rows_l = []
    for i in range(1, len(T_l)):
        T1, T2 = T_l[i - 1], T_l[i]
        tau = T2 - T1
        if tau <= 0:
            continue

        w_fwd_atm = a_l[i] - a_l[i - 1]
        if w_fwd_atm <= 0:
            continue

        sigma_fwd = np.sqrt(w_fwd_atm / tau)
        fwd_skew  = (b_l[i] - b_l[i - 1]) / (2.0 * sigma_fwd * tau)

        rows_l.append({"T1": T1, "T2": T2,
                        "T_mid": np.sqrt(T1 * T2), "fwd_skew": fwd_skew})

    results["local"] = pd.DataFrame(rows_l)

    # ── FD Market ────────────────────────────────────────────────────────
    if df_smile is not None:
        # Fit a local weighted quadratic to raw total variance near k=0
        # for each maturity slice to extract w(0) and ∂w/∂k|_{k=0}
        slice_params = []
        for tdays, sl in sorted(df_smile.groupby("T_days"), key=lambda x: x[0]):
            T    = float(sl["T"].iloc[0])
            sl   = sl[np.isfinite(sl["k"]) & np.isfinite(sl["total_variance"])].copy()
            near = sl[np.abs(sl["k"]) <= dk_band].copy()
            if len(near) < 3:
                continue

            k_n = near["k"].astype(float).to_numpy()
            w_n = near["total_variance"].astype(float).to_numpy()
            wts = np.clip(
                np.array([
                    quote_weight(r["IV"], r["T"], r.get("REL_SPREAD"), r.get("OI"))
                    for _, r in near.iterrows()
                ], dtype=float),
                1e-8, 1e8,
            )

            X   = np.column_stack([np.ones_like(k_n), k_n, k_n ** 2])
            XtW = X.T * wts
            try:
                beta = np.linalg.lstsq(XtW @ X, XtW @ w_n, rcond=None)[0]
                a_fd, b_fd = float(beta[0]), float(beta[1])
            except Exception:
                continue

            if a_fd <= 0:
                continue

            slice_params.append({"T": T, "w_atm": a_fd, "dw_dk": b_fd})

        sp = pd.DataFrame(slice_params).sort_values("T").reset_index(drop=True)

        rows_fd = []
        for i in range(1, len(sp)):
            T1, T2   = float(sp.loc[i - 1, "T"]),     float(sp.loc[i, "T"])
            wa1, wa2 = float(sp.loc[i - 1, "w_atm"]), float(sp.loc[i, "w_atm"])
            dw1, dw2 = float(sp.loc[i - 1, "dw_dk"]), float(sp.loc[i, "dw_dk"])
            tau = T2 - T1
            if tau <= 0:
                continue

            w_fwd_atm = wa2 - wa1
            if w_fwd_atm <= 0:
                continue

            sigma_fwd = np.sqrt(w_fwd_atm / tau)
            fwd_skew  = (dw2 - dw1) / (2.0 * sigma_fwd * tau)

            rows_fd.append({"T1": T1, "T2": T2,
                             "T_mid": np.sqrt(T1 * T2), "fwd_skew": fwd_skew})

        results["fd"] = pd.DataFrame(rows_fd)

    return results

