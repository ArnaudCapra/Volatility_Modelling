import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.volatility.models.essiv import essvi_total_variance
from src.volatility.analysis.diagnostics import compute_fd_atm_skew
from src.volatility.visualization.smiles import _C
from  src.volatility.analysis.term_structure import _bic_piecewise


def plot_rmse_comparison(df_smile, fit_essvi, fit_local):
    """
    Per-maturity RMSE in total-variance units (×10⁴).
    Short-maturity bars are most revealing: local quad has an unfair advantage
    (zero regularisation) while eSSVI imposes cross-maturity smoothness.
    """
    rows = []
    for tdays, sl in sorted(df_smile.groupby("T_days"), key=lambda x: x[0]):
        T  = float(sl["T"].iloc[0])
        k  = sl["k"].astype(float).to_numpy()
        w  = sl["total_variance"].astype(float).to_numpy()
        ok = np.isfinite(k) & np.isfinite(w)
        k, w = k[ok], w[ok]
        if len(k) < 3:
            continue

        try:
            theta = float(np.interp(T, fit_essvi["T_knots"], fit_essvi["theta_knots"]))
            w_e   = essvi_total_variance(k, theta, fit_essvi["rho_inf"], fit_essvi["rho_0"],
                                         fit_essvi["c_rho"], fit_essvi["eta"], fit_essvi["gamma"])
            re    = float(np.sqrt(np.mean((w_e - w) ** 2)))
        except Exception:
            re = np.nan

        try:
            w_l = fit_local["predict"](T, k)
            rl  = float(np.sqrt(np.mean((w_l - w) ** 2)))
        except Exception:
            rl = np.nan

        rows.append({"T_days": tdays, "T": T, "eSSVI": re, "Local": rl})

    df_r = pd.DataFrame(rows)
    if df_r.empty:
        return

    x     = np.arange(len(df_r))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(df_r) * 0.7), 5))
    ax.bar(x - width / 2, df_r["eSSVI"] * 1e4, width,
           color=_C["eSSVI"], alpha=0.85, label="eSSVI")
    ax.bar(x + width / 2, df_r["Local"] * 1e4, width,
           color=_C["local"], alpha=0.85, label="Local Quad")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}d" for d in df_r["T_days"]], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("RMSE (total variance × 10⁴)")
    ax.set_title("Per-Maturity RMSE — eSSVI vs Localized Quadratic\n"
                 "(Local Quad wins in-sample by design; gap is largest at short maturities)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_term_structure_comparison(fit_essvi, fit_local, df_smile=None, dk_band=0.05):
    T_e  = fit_essvi["summary"]["T"].to_numpy()
    iv_e = fit_essvi["summary"]["atm_iv"].to_numpy()
    sk_e = fit_essvi["summary"]["atm_skew"].to_numpy()
    rh_e = fit_essvi["summary"]["rho_T"].to_numpy()

    T_l  = fit_local["summary"]["T"].to_numpy()
    iv_l = fit_local["summary"]["atm_iv"].to_numpy()
    sk_l = fit_local["summary"]["atm_skew"].to_numpy()

    # Finite-difference skew from raw data (optional)
    fd = None
    if df_smile is not None:
        fd = compute_fd_atm_skew(df_smile, dk_band=dk_band)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- ATM IV ---
    ax = axes[0]
    ax.plot(T_e, iv_e * 100, "o-",  color=_C["eSSVI"], lw=2, label="eSSVI")
    ax.plot(T_l, iv_l * 100, "s--", color=_C["local"],  lw=2, label="Local Quad")
    ax.set(xlabel="T (years)", ylabel="ATM IV (%)", title="ATM IV Term Structure")
    ax.legend(); ax.grid(True, alpha=0.25)

    # --- ATM Skew ---
    ax = axes[1]
    ax.plot(T_e, sk_e, "o-",  color=_C["eSSVI"], lw=2,   label="eSSVI")
    ax.plot(T_l, sk_l, "s--", color=_C["local"],  lw=2,   label="Local Quad")
    if fd is not None:
        ok = fd["fd_atm_skew"].notna()
        ax.scatter(
            fd.loc[ok, "T"], fd.loc[ok, "fd_atm_skew"],
            marker="^", s=55, color="#e69c3c", zorder=6,
            label=f"FD market (±{dk_band:.2f})",
        )
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(xlabel="T (years)", ylabel="∂IV/∂k at k=0", title="ATM Skew Term Structure")
    ax.legend(); ax.grid(True, alpha=0.25)

    # --- eSSVI effective rho ---
    ax = axes[2]
    ax.plot(T_e, rh_e, "o-", color=_C["eSSVI"], lw=2)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(xlabel="T (years)", ylabel="ρ(θ(T))",
           title="eSSVI Effective Correlation ρ vs Maturity\n(short → ρ₀, long → ρ∞)")
    ax.grid(True, alpha=0.25)

    plt.suptitle("Term Structure Diagnostics", fontsize=12)
    plt.tight_layout()
    plt.show()

    
_DEFAULT_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]
_DEFAULT_COLORS_FIT = [
    "#a0a8fc", "#f0a09a", "#80e8c0", "#d4a0fc",
    "#ffd0a0", "#80e8f8", "#ffaabb", "#d4f0b0",
]

def plot_power_law_term_structure(
    *methods,
    T_col,
    y_col,
    title="Power-Law Term Structure",
    x_label="Log(T)",
    y_label="Log(|y|)",
    labels=None,
    colors=None,
    colors_fit=None,
    markers=None,
    dashes=None,
    min_pts=3,
):
    """
    BIC-selected piecewise power-law plot in log-log space.

    Parameters
    ----------
    *methods : DataFrames
        Each positional argument is a DataFrame (e.g. fit_essvi["summary"],
        fwd["local"], fd_skew). NaN rows are silently dropped.

    T_col : str | list[str]
        Column name(s) for maturities. A single string is broadcast to all
        methods; a list must match len(methods).

    y_col : str | list[str]
        Column name(s) for the values (absolute value + log applied internally).
        A single string is broadcast; a list must match len(methods).

    title, x_label, y_label : str
        Plot labels.

    labels : list[str] | None
        Legend names, one per method. Defaults to "Series 0", "Series 1", …

    colors : list[str] | None
        Dot colours. Defaults to _DEFAULT_COLORS cycling.

    colors_fit : list[str] | None
        Regression-line colours. Defaults to _DEFAULT_COLORS_FIT cycling.

    markers : list[str] | None
        Plotly marker symbols. Defaults to "circle".

    dashes : list[str] | None
        Plotly dash styles for the single-fit line. Defaults to "solid".

    min_pts : int
        Minimum points per segment in BIC search.

    Returns
    -------
    plotly Figure

    Examples
    --------
    # Spot skew
    plot_power_law_term_structure(
        fit_essvi["summary"], fit_local["summary"], fd_skew,
        T_col="T",
        y_col=["atm_skew", "atm_skew", "fd_atm_skew"],
        labels=["eSSVI", "Local Quad", "FD Market"],
        colors=[_C["eSSVI"], _C["local"], "#e69c3c"],
        colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
        title="ATM Spot Skew — Power-Law Term Structure",
        x_label="Log(T)", y_label="Log(|ATM Skew|)",
    )

    # Forward skew  (uniform columns → single string)
    plot_power_law_term_structure(
        fwd["eSSVI"], fwd["local"], fwd["fd"],
        T_col="T_mid",
        y_col="fwd_skew",
        labels=["eSSVI fwd", "Local Quad fwd", "FD Market fwd"],
        colors=[_C["eSSVI"], _C["local"], "#e69c3c"],
        colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
        title="ATM Forward Skew — Power-Law Term Structure",
        x_label="Log(T_mid = √(T₁·T₂))", y_label="Log(|Fwd Skew|)",
    )
    """
    n = len(methods)

    # --- broadcast scalar → list ------------------------------------------------
    def _broadcast(val, name):
        if isinstance(val, str):
            return [val] * n
        if val is None or len(val) == n:
            return val
        raise ValueError(f"`{name}` must be a str or a list of length {n}.")

    T_cols     = _broadcast(T_col, "T_col")
    y_cols     = _broadcast(y_col, "y_col")
    labels     = labels     or [f"Series {i}" for i in range(n)]
    colors     = colors     or [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]     for i in range(n)]
    colors_fit = colors_fit or [_DEFAULT_COLORS_FIT[i % len(_DEFAULT_COLORS_FIT)] for i in range(n)]
    markers    = markers    or ["circle"] * n
    dashes     = dashes     or ["solid"]  * n

    # ---------------------------------------------------------------------------
    fig = go.Figure()

    for df, tc, yc, label, c_dot, c_fit, marker, base_dash in zip(
        methods, T_cols, y_cols, labels, colors, colors_fit, markers, dashes
    ):
        T_vals = np.asarray(df[tc], dtype=float)
        y_vals = np.abs(np.asarray(df[yc], dtype=float))

        mask = (T_vals > 0) & (y_vals > 0) & np.isfinite(T_vals) & np.isfinite(y_vals)
        if mask.sum() < 4:
            continue

        lt = np.log(T_vals[mask])
        ly = np.log(y_vals[mask])
        order = np.argsort(lt)
        lt, ly = lt[order], ly[order]

        fig.add_trace(go.Scatter(
            x=lt, y=ly, mode="markers",
            name=label,
            marker=dict(size=8, color=c_dot, symbol=marker),
            legendgroup=label,
        ))

        use_split, sp, single, seg1, seg2 = _bic_piecewise(lt, ly, min_pts=min_pts)

        if use_split:
            for seg_lt, seg_ly, p, tag, dash in [
                (lt[:sp], ly[:sp], seg1, "short", "solid"),
                (lt[sp:], ly[sp:], seg2, "long",  "dash"),
            ]:
                fig.add_trace(go.Scatter(
                    x=seg_lt,
                    y=p["intercept"] + p["slope"] * seg_lt,
                    mode="lines",
                    name=f"{label} — {tag} (β={p['slope']:.3f}, R²={p['r2']:.3f})",
                    line=dict(color=c_fit, width=2, dash=dash),
                    legendgroup=label,
                ))
            split_x = (lt[sp - 1] + lt[sp]) / 2
            fig.add_vline(
                x=split_x,
                line_dash="dot", line_color=c_dot, opacity=0.55,
                annotation_text=f"{label} split T≈{np.exp(split_x):.2f}y",
                annotation_font_color=c_dot,
                annotation_font_size=10,
            )
        else:
            p = single
            fig.add_trace(go.Scatter(
                x=lt,
                y=p["intercept"] + p["slope"] * lt,
                mode="lines",
                name=f"{label} — single (β={p['slope']:.3f}, R²={p['r2']:.3f})",
                line=dict(color=c_fit, width=2, dash=base_dash),
                legendgroup=label,
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=20, b=40, t=50),
    )
    fig.show()
    return fig