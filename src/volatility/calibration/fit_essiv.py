import numpy as np
import pandas as pd
from src.volatility.config.config import WEIGHT_EPS
from src.volatility.calibration.weights import quote_weight
from src.volatility.calibration.loss import essvi_residuals, _unpack_essvi_params
from scipy.optimize import least_squares
from src.volatility.models.essiv import essvi_total_variance, rho_of_theta
from src.volatility.config.config import ESSVI_LAMBDA_BUTTERFLY




def estimate_theta_knots(df_smile, max_nearest=7):
    """Initial guess for theta(T) = ATM total variance per maturity."""
    rows = []
    for tdays, sl in df_smile.groupby("T_days"):
        sl = sl.copy()
        sl["abs_k"] = np.abs(sl["k"].astype(float))
        sl = sl.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_k", "total_variance", "IV", "T"]).copy()

        if len(sl) < 3:
            T_val = float(sl["T"].iloc[0]) if len(sl) > 0 else np.nan
            rows.append({"T_days": int(tdays), "T": T_val, "theta_init": np.nan})
            continue

        sl = sl.sort_values("abs_k").head(min(max_nearest, len(sl)))
        weights = [
            quote_weight(iv=r["IV"], T=r["T"], rel_spread=r.get("REL_SPREAD"), oi=r.get("OI"))
            for _, r in sl.iterrows()
        ]
        weights = np.asarray(weights, dtype=float)
        theta_init = np.average(sl["total_variance"].to_numpy(dtype=float), weights=weights)

        rows.append({
            "T_days":     int(tdays),
            "T":          float(sl["T"].iloc[0]),
            "theta_init": float(theta_init),
        })

    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    if out["theta_init"].notna().sum() >= 2:
        valid = out["theta_init"].notna()
        out["theta_init"] = np.interp(
            out["T"].to_numpy(),
            out.loc[valid, "T"].to_numpy(),
            out.loc[valid, "theta_init"].to_numpy(),
        )

    # FIX: use ffill/bfill (pandas ≥ 2.2 deprecates fillna(method=...))
    # FIX: fill NaN before np.maximum.accumulate, which doesn't skip NaN
    theta_vals = out["theta_init"].ffill().bfill().fillna(1e-8).to_numpy(dtype=float)
    theta_vals = np.maximum(theta_vals, 1e-8)
    out["theta_init"] = np.maximum.accumulate(theta_vals)

    return out

def fit_essvi_surface(df_smile):
    """Joint eSSVI calibration across all maturities."""
    if df_smile.empty:
        raise ValueError("df_smile is empty.")

    theta_df = (
        estimate_theta_knots(df_smile)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["T", "theta_init"])
        .sort_values("T")
        .reset_index(drop=True)
    )
    if len(theta_df) < 2:
        raise ValueError("Need at least two maturities.")

    T_knots    = theta_df["T"].to_numpy(dtype=float)
    theta_init = np.maximum.accumulate(np.maximum(theta_df["theta_init"].to_numpy(), 1e-8))
    theta_inc  = np.maximum(np.diff(np.concatenate([[0.0], theta_init])), 1e-8)

    x0 = np.concatenate([
        np.log(theta_inc),
        [np.arctanh(-0.4)],   # rho_inf  (long-term correlation)
        [np.arctanh(-0.7)],   # rho_0    (short-term correlation, steeper skew)
        [np.log(1.0)],        # log c_rho
        [np.log(1.0)],        # log eta
        [0.0],                # logit gamma → 0.5
    ])

    data = df_smile.copy()
    data["weight"] = np.clip(
        data.apply(
            lambda r: quote_weight(r["IV"], r["T"], r.get("REL_SPREAD"), r.get("OI")),
            axis=1,
        ).to_numpy(dtype=float),
        1e-8, 1e8,
    )

    res = least_squares(
        essvi_residuals, x0=x0,
        args=(data, T_knots, ESSVI_LAMBDA_BUTTERFLY),
        loss="soft_l1", f_scale=1.0,
        max_nfev=80000, method="trf",
    )

    theta_knots, rho_inf, rho_0, c_rho, eta, gamma = _unpack_essvi_params(res.x, len(T_knots))

    theta_T = np.interp(data["T"].to_numpy(dtype=float), T_knots, theta_knots)
    w_fit   = essvi_total_variance(data["k"].to_numpy(dtype=float),
                                   theta_T, rho_inf, rho_0, c_rho, eta, gamma)
    w_obs   = data["total_variance"].to_numpy(dtype=float)

    per_mat = []
    for t, theta in zip(T_knots, theta_knots):
        rho_t  = float(rho_of_theta(theta, rho_inf, rho_0, c_rho))
        phi_t  = eta / max(theta, WEIGHT_EPS) ** gamma
        atm_iv = np.sqrt(theta / t)
        per_mat.append({
            "T":        float(t),
            "theta":    float(theta),
            "rho_T":    rho_t,
            "phi_T":    float(phi_t),
            "atm_iv":   float(atm_iv),
            "atm_skew": float(theta * rho_t * phi_t / (2.0 * atm_iv * t)) if atm_iv > 0 else np.nan,
        })

    return {
        "method":         "eSSVI",
        "T_knots":        T_knots,
        "theta_knots":    theta_knots,
        "rho_inf":        float(rho_inf),
        "rho_0":          float(rho_0),
        "c_rho":          float(c_rho),
        "eta":            float(eta),
        "gamma":          float(gamma),
        "rmse_total_var": float(np.sqrt(np.mean((w_fit - w_obs) ** 2))),
        "mae_total_var":  float(np.mean(np.abs(w_fit - w_obs))),
        "summary":        pd.DataFrame(per_mat),
    }