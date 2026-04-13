import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.volatility.models.essiv import essvi_total_variance

def _make_total_var_predictor(surface_result: dict):
    method = surface_result["method"]

    if method == "Localized Quadratic":
        return surface_result["predict"]

    elif method == "eSSVI":
        T_knots     = surface_result["T_knots"]
        theta_knots = surface_result["theta_knots"]
        rho_inf     = surface_result["rho_inf"]
        rho_0       = surface_result["rho_0"]
        c_rho       = surface_result["c_rho"]
        eta         = surface_result["eta"]
        gamma       = surface_result["gamma"]

        def predict(T, k_array):
            theta_T = float(np.interp(T, T_knots, theta_knots))
            return essvi_total_variance(k_array, theta_T, rho_inf, rho_0, c_rho, eta, gamma)

        return predict

    elif method in ("Heston-I0", "Heston-I", "Heston-MC"):
        # Map method label to the result key inside heston_malliavin_surface output
        iv_key = {"Heston-I0": "I0", "Heston-I": "I", "Heston-MC": "mc_iv"}[method]
        raw    = surface_result["raw"]   # {(T, K): {...}} from heston_malliavin_surface

        # Build sorted unique grids once
        T_grid = np.array(sorted({T for T, K in raw}), dtype=float)
        K_grid = np.array(sorted({K for T, K in raw}), dtype=float)
        k_grid = np.log(K_grid)           # log-moneyness already baked in (S0-normalised)

        # 2-D IV array  [len(T_grid), len(k_grid)]
        iv_array = np.full((len(T_grid), len(k_grid)), np.nan)
        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                iv_val = raw.get((T, K), {}).get(iv_key, np.nan)
                if iv_val is not None and np.isfinite(iv_val) and iv_val > 0:
                    iv_array[i, j] = iv_val

        def predict(T_query, k_query):
            """Bilinear interpolation over the (T, k) grid, returns total variance."""
            k_query  = np.asarray(k_query, dtype=float)
            # Clamp to avoid silent extrapolation
            T_clamped = np.clip(T_query, T_grid[0], T_grid[-1])
            i_lo = np.searchsorted(T_grid, T_clamped, side="right") - 1
            i_lo = int(np.clip(i_lo, 0, len(T_grid) - 2))
            i_hi = i_lo + 1

            alpha = (T_clamped - T_grid[i_lo]) / (T_grid[i_hi] - T_grid[i_lo])
            iv_lo = np.interp(k_query, k_grid, iv_array[i_lo])
            iv_hi = np.interp(k_query, k_grid, iv_array[i_hi])
            iv    = (1 - alpha) * iv_lo + alpha * iv_hi
            return iv ** 2 * T_query   # IV → total variance

        return predict

    else:
        raise ValueError(f"Unsupported surface method: '{method}'")


def plot_iv_surface_interactive(df_smile, surface_result,
                                n_k=250, n_t=250,
                                k_clip=0.45,
                                iv_clip=(0.0, 1.5),
                                k_range=None):
    if surface_result is None:
        return

    summary = surface_result.get("summary", pd.DataFrame())
    if isinstance(summary, pd.DataFrame):
        summary = summary.dropna(subset=["T"])

    if isinstance(summary, pd.DataFrame) and len(summary) >= 2:
        t_min = float(summary["T"].min())
        t_max = float(summary["T"].max())
    elif surface_result["method"].startswith("Heston"):
        raw   = surface_result["raw"]
        t_min = min(T for T, K in raw)
        t_max = max(T for T, K in raw)
    else:
        return

    T_grid = np.linspace(t_min, t_max, n_t)

    if k_range is not None:
        k_lo, k_hi = float(k_range[0]), float(k_range[1])
    elif df_smile is not None and not df_smile.empty:
        k_obs = np.log(
            df_smile["STRIKE"].astype(float) / df_smile["FORWARD_AT_TARGET"].astype(float)
        )
        k_obs = k_obs[np.isfinite(k_obs)]
        if len(k_obs) == 0:
            return
        k_lo = max(-k_clip, float(np.quantile(k_obs, 0.02)) - 0.05)
        k_hi = min( k_clip, float(np.quantile(k_obs, 0.98)) + 0.05)
        if k_lo >= k_hi:
            k_lo, k_hi = -k_clip, k_clip
    else:
        return

    k_grid  = np.linspace(k_lo, k_hi, n_k)
    predict = _make_total_var_predictor(surface_result)

    iv_floor = iv_clip[0]
    iv_ceil  = iv_clip[1]   # hard ceiling: never display above this

    IV_mesh = np.full((len(k_grid), len(T_grid)), np.nan)

    for j, T in enumerate(T_grid):
        total_var = predict(T, k_grid)
        if total_var is None:
            continue
        total_var = np.clip(total_var, 1e-12, (iv_ceil ** 2) * T)
        iv = np.sqrt(total_var / T)
        IV_mesh[:, j] = np.clip(iv, iv_floor, iv_ceil)

    for i in range(IV_mesh.shape[0]):
        row = IV_mesh[i, :]
        ok  = np.isfinite(row)
        if ok.sum() >= 2:
            IV_mesh[i, :] = np.interp(T_grid, T_grid[ok], row[ok])

    # Derive the actual upper bound from the data
    iv_max = float(np.nanmax(IV_mesh)) if np.any(np.isfinite(IV_mesh)) else iv_ceil
    iv_min= float(np.nanmin(IV_mesh)) if np.any(np.isfinite(IV_mesh)) else iv_floor

    fig = go.Figure(data=[
        go.Surface(
            z=IV_mesh,
            x=T_grid[None, :].repeat(len(k_grid), axis=0),
            y=k_grid[:, None].repeat(len(T_grid), axis=1),
            colorscale="Viridis",
            cmin=iv_min,
            cmax=iv_max,
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title="Time to Maturity (T)",
            yaxis_title="Log-Moneyness k = log(K/F)",
            zaxis_title="Implied Volatility",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[float(T_grid.min()), float(T_grid.max())]),
            yaxis=dict(range=[float(k_grid.min()), float(k_grid.max())]),
            zaxis=dict(range=[iv_min, iv_max]),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig