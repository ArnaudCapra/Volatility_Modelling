import numpy as np
import pandas as pd
from src.volatility.calibration.weights import quote_weight
from src.volatility.models.local_quad import _fit_quadratic_slice
from src.volatility.config.config import WEIGHT_EPS


def fit_localized_surface(df_smile, min_strikes=5):
    """
    Calibrate an independent quadratic total-variance smile per maturity.
    Returns a dict with per-slice params, a predict(T, k) interpolator,
    and per-slice diagnostics.
    """
    slices = []
    for tdays, sl in sorted(df_smile.groupby("T_days"), key=lambda x: x[0]):
        sl = (
            sl.dropna(subset=["k", "total_variance", "IV", "T"])
              .pipe(lambda d: d[np.isfinite(d["k"]) & np.isfinite(d["total_variance"])])
              .copy()
        )
        if len(sl) < min_strikes:
            continue

        T   = float(sl["T"].iloc[0])
        k   = sl["k"].astype(float).to_numpy()
        w   = sl["total_variance"].astype(float).to_numpy()
        wts = np.clip(
            np.array([
                quote_weight(r["IV"], r["T"], r.get("REL_SPREAD"), r.get("OI"))
                for _, r in sl.iterrows()
            ], dtype=float),
            1e-8, 1e8,
        )

        a, b, c = _fit_quadratic_slice(k, w, wts)
        w_fit   = a + b * k + c * k ** 2
        atm_iv  = np.sqrt(max(a, WEIGHT_EPS) / T)

        slices.append({
            "T_days":    int(tdays),
            "T":         T,
            "a":         a,
            "b":         b,
            "c":         c,
            "atm_iv":    atm_iv,
            "atm_skew":  b / (2.0 * atm_iv * T) if atm_iv > 0 else np.nan,
            "rmse":      float(np.sqrt(np.mean((w_fit - w) ** 2))),
            "mae":       float(np.mean(np.abs(w_fit - w))),
            "n_strikes": len(sl),
        })

    summary = pd.DataFrame(slices).sort_values("T").reset_index(drop=True)

    T_grid = summary["T"].to_numpy()

    def predict(T_query, k_query):
        """Linearly interpolate (a, b, c) across T, then evaluate the quadratic."""
        k_query = np.asarray(k_query, dtype=float)
        a_i = float(np.interp(T_query, T_grid, summary["a"].to_numpy()))
        b_i = float(np.interp(T_query, T_grid, summary["b"].to_numpy()))
        c_i = float(np.interp(T_query, T_grid, summary["c"].to_numpy()))
        return np.clip(a_i + b_i * k_query + c_i * k_query ** 2, WEIGHT_EPS, None)

    return {
        "method":         "Localized Quadratic",
        "summary":        summary,
        "predict":        predict,
        "rmse_total_var": float(summary["rmse"].mean()) if len(summary) else np.nan,
        "mae_total_var":  float(summary["mae"].mean())  if len(summary) else np.nan,
    }