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