import numpy as np
import plotly.graph_objects as go


def _bic_piecewise(log_t, log_y, min_pts=3):
    """
    BIC-selected single vs piecewise linear fit in log-log space.
    Returns (use_split, split_idx, single_params, seg1_params, seg2_params).
    Each params dict: {slope, intercept, r2}.
    """
    from scipy.stats import linregress

    n = len(log_t)
    if n < 6:
        s, i, r, *_ = linregress(log_t, log_y)
        return False, -1, {"slope": s, "intercept": i, "r2": r**2}, None, None

    s0, i0, r0, *_ = linregress(log_t, log_y)
    rss0       = np.sum((log_y - (i0 + s0 * log_t)) ** 2)
    bic_single = n * np.log(rss0 / n) + 3 * np.log(n)

    best_bic, best_i = np.inf, -1
    for i in range(min_pts, n - min_pts + 1):
        s1, i1, *_ = linregress(log_t[:i], log_y[:i])
        s2, i2, *_ = linregress(log_t[i:], log_y[i:])
        rss = (
            np.sum((log_y[:i] - (i1 + s1 * log_t[:i])) ** 2)
            + np.sum((log_y[i:] - (i2 + s2 * log_t[i:])) ** 2)
        )
        bic = n * np.log(rss / n) + 6 * np.log(n)
        if bic < best_bic:
            best_bic, best_i = bic, i

    use_split = (best_bic < bic_single) and (best_i > 0)

    single = {"slope": s0, "intercept": i0, "r2": r0**2}
    if not use_split:
        return False, -1, single, None, None

    s1, i1, r1, *_ = linregress(log_t[:best_i], log_y[:best_i])
    s2, i2, r2, *_ = linregress(log_t[best_i:], log_y[best_i:])
    seg1 = {"slope": s1, "intercept": i1, "r2": r1**2}
    seg2 = {"slope": s2, "intercept": i2, "r2": r2**2}
    return True, best_i, single, seg1, seg2


def plot_power_law_term_structure(
    series,
    title="Power-Law Term Structure",
    x_label="Log(T)",
    y_label="Log(|y|)",
    min_pts=3,
):
    """
    Generic BIC-selected piecewise power-law plot in log-log space.

    Parameters
    ----------
    series : list of dicts, each with keys:
        - "T"      : array-like of maturities (linear scale, positive)
        - "y"      : array-like of values     (linear scale, will take abs+log)
        - "label"  : str, legend name
        - "color"  : str, colour for dots
        - "color_fit" : str, colour for regression lines (optional, defaults to "color")
        - "marker" : str, matplotlib/plotly marker symbol (optional, default "circle")
        - "dash"   : plotly dash style for the single-fit line (optional, default "solid")

    title, x_label, y_label : plot labels
    min_pts : minimum points per segment in BIC search

    Returns
    -------
    plotly Figure (also calls fig.show())

    Example — spot skew
    --------------------
    series = [
        {"T": essvi_sum["T"].to_numpy(), "y": essvi_sum["atm_skew"].to_numpy(),
         "label": "eSSVI",      "color": _C["eSSVI"],  "color_fit": "#f0a09a"},
        {"T": local_sum["T"].to_numpy(), "y": local_sum["atm_skew"].to_numpy(),
         "label": "Local Quad", "color": _C["local"],  "color_fit": "#9ac4f0"},
        {"T": fd_ok["T"].to_numpy(),     "y": fd_ok["fd_atm_skew"].to_numpy(),
         "label": "FD Market",  "color": "#e69c3c",    "color_fit": "#f0cfa0"},
    ]
    plot_power_law_term_structure(series, title="ATM Skew Power-Law", ...)

    Example — forward skew
    -----------------------
    fwd = compute_atm_forward_skew(fit_essvi, fit_local, df_smile)
    series = [
        {"T": fwd["eSSVI"]["T_mid"].to_numpy(), "y": fwd["eSSVI"]["fwd_skew"].to_numpy(),
         "label": "eSSVI fwd",  "color": _C["eSSVI"],  "color_fit": "#f0a09a"},
        ...
    ]
    plot_power_law_term_structure(series, title="ATM Forward Skew Power-Law", ...)
    """
    fig = go.Figure()

    for s in series:
        T_vals    = np.asarray(s["T"],     dtype=float)
        y_vals    = np.abs(np.asarray(s["y"], dtype=float))
        label     = s["label"]
        c_dot     = s["color"]
        c_fit     = s.get("color_fit", c_dot)
        marker    = s.get("marker", "circle")
        base_dash = s.get("dash", "solid")

        # Filter and sort
        mask = (T_vals > 0) & (y_vals > 0) & np.isfinite(T_vals) & np.isfinite(y_vals)
        if mask.sum() < 4:
            continue

        lt = np.log(T_vals[mask])
        ly = np.log(y_vals[mask])
        order = np.argsort(lt)
        lt, ly = lt[order], ly[order]

        # Raw dots
        fig.add_trace(go.Scatter(
            x=lt, y=ly, mode="markers",
            name=label,
            marker=dict(size=8, color=c_dot, symbol=marker),
            legendgroup=label,
        ))

        use_split, sp, single, seg1, seg2 = _bic_piecewise(lt, ly, min_pts=min_pts)

        if use_split:
            segments = [
                (lt[:sp], ly[:sp], seg1, "short", "solid"),
                (lt[sp:], ly[sp:], seg2, "long",  "dash"),
            ]
            for seg_lt, seg_ly, p, tag, dash in segments:
                fig.add_trace(go.Scatter(
                    x=seg_lt,
                    y=p["intercept"] + p["slope"] * seg_lt,
                    mode="lines",
                    name=(f"{label} — {tag} "
                          f"(β={p['slope']:.3f}, R²={p['r2']:.3f})"),
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
                name=(f"{label} — single "
                      f"(β={p['slope']:.3f}, R²={p['r2']:.3f})"),
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