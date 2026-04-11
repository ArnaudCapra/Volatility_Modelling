import numpy as np

def _fit_quadratic_slice(k, w_obs, weights):
    """
    Weighted least-squares fit of w(k) = a + b·k + c·k²
    with positivity constraints a > 0, c > 0 enforced via bounded L-BFGS-B.
    """
    from scipy.optimize import minimize

    k   = np.asarray(k,       dtype=float)
    w   = np.asarray(w_obs,   dtype=float)
    wts = np.asarray(weights, dtype=float)

    X    = np.column_stack([np.ones_like(k), k, k ** 2])
    XtW  = X.T * wts
    beta_ols, *_ = np.linalg.lstsq(XtW @ X, XtW @ w, rcond=None)

    def obj(p):
        return float(np.sum(wts * (p[0] + p[1]*k + p[2]*k**2 - w) ** 2))

    res = minimize(
        obj,
        x0=[max(float(beta_ols[0]), 1e-5), float(beta_ols[1]), max(float(beta_ols[2]), 1e-5)],
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
        method="L-BFGS-B",
    )
    return tuple(float(v) for v in res.x)   # (a, b, c)