from src.volatility.models.heston import simulate_heston, effective_vol, malliavin_phi
from src.volatility.pricing.BlackScholes import black_price_forward,bs_vega, H_func
from src.volatility.pricing.implied_vol import bs_implied_vol
import numpy as np
import pandas as pd

def heston_malliavin_surface(
    kappa, theta, xi, rho,
    S0, v0,
    maturities, strikes,
    N_steps=200, N_paths=5_000, r=0.0, seed=42,
):
    """
    Returns
    -------
    dict  {(T, K) : {'I0', 'I', 'correction', 'mc_iv'}}
    """
    results = {}

    for T in np.asarray(maturities, dtype=float):
        # ── 1. Simulate per maturity ──────────────────────────────────────────
        t, X, var = simulate_heston(
            kappa, theta, xi, rho,
            S0, v0, T, N_steps, N_paths, r, seed
        )
        dt = T / N_steps
        x0 = np.log(S0)

        # ── 2. Path-wise quantities ───────────────────────────────────────────
        v, _  = effective_vol(var, dt, T)
        Phi   = malliavin_phi(var, xi, kappa, theta, dt)
        disc  = np.exp(-r * t)

        sl      = slice(1, N_steps)
        tau_sl  = T - t[sl]
        disc_sl = disc[sl]
        X_sl    = X[:,   sl]
        v_sl    = v[:,   sl]
        Phi_sl  = Phi[:, sl]

        for K in np.asarray(strikes, dtype=float):
            k = np.log(K)

            # V0 and I0 
            V0 = float(np.mean(black_price_forward( F=np.exp(x0), K=np.exp(k), T=T, D=1.0, sigma=v[:, 0], option_type="call")))
            I0 = bs_implied_vol(T, x0, k, V0)

            # MC benchmark
            payoffs  = np.maximum(np.exp(X[:, -1]) - K, 0.0)
            mc_price = np.exp(-r * T) * float(np.mean(payoffs))
            mc_iv    = bs_implied_vol(T, x0, k, mc_price)

            # Correction
            vega_sl  = bs_vega(tau_sl, X_sl, k, v_sl)
            H_sl     = H_func(tau_sl, X_sl, k, v_sl)
            inv_vega = np.where(vega_sl > 1e-12, 1.0 / vega_sl, 0.0)

            integrand = disc_sl * inv_vega * Phi_sl * H_sl * V0
            integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

            integral = np.trapezoid(integrand, t[sl], axis=1)
            integral = integral[~np.isnan(integral)]
            corr     = (rho / 2.0) * float(np.mean(integral))

            results[(T, K)] = {
                "I0": I0,
                "I": I0 + corr,
                "correction": corr,
                "mc_iv": mc_iv,
            }

    return results

def heston_surface_result(raw_heston: dict, surface: str = "I"):
    """
    surface : 'I0' | 'I' | 'mc_iv'
    """
    assert surface in ("I0", "I", "mc_iv")
    method_map = {"I0": "Heston-I0", "I": "Heston-I", "mc_iv": "Heston-MC"}
    return {
        "method":  method_map[surface],
        "raw":     raw_heston,
        "summary": pd.DataFrame({"T": sorted({T for T, K in raw_heston})}),
    }