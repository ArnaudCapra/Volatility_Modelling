from src.volatility.models.heston import simulate_heston, effective_vol, malliavin_phi
from src.volatility.pricing.BlackScholes import black_price_forward,bs_vega, H_func, dH_dk
from src.volatility.pricing.implied_vol import bs_implied_vol
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dataclasses import dataclass

@dataclass
class HestonPaths:
    t: np.ndarray
    X: np.ndarray
    var: np.ndarray
    v: np.ndarray
    Phi: np.ndarray
    disc: np.ndarray

def build_paths(kappa, theta, xi, rho, S0, v0, T, N_steps, N_paths, r, seed):
    t, X, var = simulate_heston(
        kappa, theta, xi, rho,
        S0, v0, T, N_steps, N_paths, r, seed
    )
    dt = T / N_steps

    v, _ = effective_vol(var, dt, T)
    Phi = malliavin_phi(var, xi, kappa, theta, dt)
    disc = np.exp(-r * t)

    return HestonPaths(t, X, var, v, Phi, disc)



def _compute_surface_for_T(
    T,
    kappa, theta, xi, rho,
    S0, v0,
    strikes,
    N_steps, N_paths,
    r, seed
):
    paths = build_paths(
        kappa, theta, xi, rho,
        S0, v0, T,
        N_steps, N_paths,
        r, seed
    )
    K_arr = np.asarray(strikes, dtype=float)
    k_arr = np.log(K_arr)
    x0 = np.log(S0)

    t = paths.t
    X = paths.X
    v = paths.v
    Phi = paths.Phi
    disc = paths.disc

    N_steps = len(t)
    sl = slice(1, N_steps)

    t_sl = t[sl]
    tau_sl = T - t_sl

    X_sl = X[:, sl]             # (P, S)
    v_sl = v[:, sl]
    Phi_sl = Phi[:, sl]
    disc_sl = disc[sl]

    # ── Broadcast shapes ──────────────────────────────────────────────────────
    X_b = X_sl[:, None, :]                      # (P, 1, S)
    v_b = v_sl[:, None, :]
    Phi_b = Phi_sl[:, None, :]
    disc_b = disc_sl[None, None, :]
    tau_b = tau_sl[None, None, :]
    k_b = k_arr[None, :, None]                  # (1, K, 1)

    # ── H and Vega ───────────────────────────────────────────────────────────
    H = H_func(tau_b, X_b, k_b, v_b)
    vega = bs_vega(tau_b, X_b, k_b, v_b)
    inv_vega = np.where(vega > 1e-12, 1.0 / vega, 0.0)

    # ── V0 (vectorized) ──────────────────────────────────────────────────────
    V0 = np.mean(
        black_price_forward(
            F=np.exp(x0),
            K=K_arr[None, :],
            T=T,
            D=1.0,
            sigma=v[:, 0, None],
            option_type="call"
        ),
        axis=0
    )

    # ── I0 ───────────────────────────────────────────────────────────────────
    I0 = bs_implied_vol(T, x0, k_arr, V0)

    # ── Correction ───────────────────────────────────────────────────────────
    integrand = disc_b * inv_vega * Phi_b * H * V0[None, :, None]
    integrand = np.nan_to_num(integrand)

    integral = np.trapezoid(integrand, t_sl, axis=2)   # (P, K)
    corr = (rho / 2.0) * np.mean(integral, axis=0)

    I = I0 + corr

    # ── MC benchmark ─────────────────────────────────────────────────────────
    payoffs = np.maximum(np.exp(X[:, -1, None]) - K_arr[None, :], 0.0)
    mc_price = np.exp(-r * T) * np.mean(payoffs, axis=0)
    mc_iv = bs_implied_vol(T, x0, k_arr, mc_price)

    return {
        "T": T,
        "K": K_arr,
        "I0": I0,
        "I": I,
        "mc_iv": mc_iv,
        "correction": corr,
    }

def heston_malliavin_surface(
    kappa, theta, xi, rho,
    S0, v0,
    maturities, strikes,
    N_steps=200, N_paths=5_000, r=0.0, seed=42,
):
    maturities = np.asarray(maturities, dtype=float)

    outputs = Parallel(n_jobs=-1)(
    delayed(_compute_surface_for_T)(
        T,
        kappa, theta, xi, rho,
        S0, v0,
        strikes,
        N_steps, N_paths,
        r, seed
    )
    for T in maturities
)

    # ── Repack to original dict format ───────────────────────────────────────
    results = {}
    for out in outputs:
        T = out["T"]
        for i, K in enumerate(out["K"]):
            results[(T, K)] = {
                "I0": out["I0"][i],
                "I": out["I"][i],
                "correction": out["correction"][i],
                "mc_iv": out["mc_iv"][i],
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



def heston_malliavin_skew(
    kappa, theta, xi, rho,
    S0, v0,
    maturities, strikes,
    N_steps=200, N_paths=5_000, r=0.0, seed=42,
    surface_raw=None,   # pass heston_malliavin_surface() output to skip recomputation
    dk_fd=1e-4,         # log-strike step for the central FD on I0
):
    """
    Compute ∂I/∂k via the Malliavin skew formula:

        ∂I/∂k = E[ ∫ (∂F/∂k − ½F) ds ] / Vega(T, x0, k, I(k))

    with  F := (ρ/2) e^{-r(s-t)} H(s, Xs, k, vs) Φs

    Parameters
    ----------
    surface_raw : dict, optional
        Output of heston_malliavin_surface().  If supplied, I0 and I are
        read from it directly (saves one full re-simulation per strike).
    dk_fd : float
        Central-difference step (in log-strike) used to estimate ∂I0/∂k.

    Returns
    -------
    dict  {(T, K): {'dI0_dk', 'dI_dk', 'skew_corr', 'numerator', 'denominator', 'I_k'}}
    """
    results      = {}
    strikes_arr  = np.asarray(strikes, dtype=float)
    maturities_arr = np.asarray(maturities, dtype=float)

    for T in maturities_arr:
        # ── 1. One simulation per maturity ───────────────────────────────────
        t, X, var = simulate_heston(
            kappa, theta, xi, rho,
            S0, v0, T, N_steps, N_paths, r, seed,
        )
        dt = T / N_steps
        x0 = np.log(S0)

        v, _   = effective_vol(var, dt, T)
        Phi    = malliavin_phi(var, xi, kappa, theta, dt)
        disc   = np.exp(-r * t)

        sl      = slice(1, N_steps)
        t_sl    = t[sl]
        tau_sl  = T - t_sl
        disc_sl = disc[sl]
        X_sl    = X[:, sl]
        v_sl    = v[:, sl]
        Phi_sl  = Phi[:, sl]

        for K in strikes_arr:
            k = np.log(K)

            # ── 2. Greek arrays (computed once, shared across both branches) ──
            H_sl    = H_func(tau_sl, X_sl, k, v_sl)
            dHdk_sl = dH_dk(tau_sl, X_sl, k, v_sl)

            # ── 3. Implied vol level: reuse surface or recompute ─────────────
            if surface_raw is not None and (T, K) in surface_raw:
                I0  = surface_raw[(T, K)]["I0"]
                I_k = surface_raw[(T, K)]["I"]
            else:
                V0 = float(np.mean(
                    black_price_forward(
                        F=np.exp(x0), K=K, T=T, D=1.0,
                        sigma=v[:, 0], option_type="call",
                    )
                ))
                I0 = bs_implied_vol(T, x0, k, V0)

                vega_sl  = bs_vega(tau_sl, X_sl, k, v_sl)
                inv_vega = np.where(vega_sl > 1e-12, 1.0 / vega_sl, 0.0)
                intg_0   = np.nan_to_num(
                    disc_sl * inv_vega * Phi_sl * H_sl * V0,
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                corr = (rho / 2.0) * float(np.mean(
                    np.trapezoid(intg_0, t_sl, axis=1)
                ))
                I_k = I0 + corr

            # ── 4. Malliavin skew numerator ──────────────────────────────────
            # integrand = (ρ/2) disc Φ (∂H/∂k − ½H)
            integrand = disc_sl * Phi_sl * (dHdk_sl - 0.5 * H_sl)
            integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

            numerator = (rho / 2.0) * float(np.mean(
                np.trapezoid(integrand, t_sl, axis=1)
            ))

            # ── 5. Denominator: scalar vega at (T, x0, k, I(k)) ─────────────
            denom     = float(bs_vega(T, x0, k, I_k))
            skew_corr = numerator / denom if abs(denom) > 1e-12 else 0.0

            # ── 6. ∂I0/∂k via central finite difference on E[BS(σ_eff)] ─────
            V0_up = float(np.mean(
                black_price_forward(F=np.exp(x0), K=np.exp(k + dk_fd), T=T,
                                    D=1.0, sigma=v[:, 0], option_type="call")
            ))
            V0_dn = float(np.mean(
                black_price_forward(F=np.exp(x0), K=np.exp(k - dk_fd), T=T,
                                    D=1.0, sigma=v[:, 0], option_type="call")
            ))
            I0_up  = bs_implied_vol(T, x0, k + dk_fd, V0_up)
            I0_dn  = bs_implied_vol(T, x0, k - dk_fd, V0_dn)
            dI0_dk = (I0_up - I0_dn) / (2.0 * dk_fd)

            results[(T, K)] = {
                "dI0_dk":      dI0_dk,
                "dI_dk":       dI0_dk + skew_corr,
                "skew_corr":   skew_corr,
                "numerator":   numerator,
                "denominator": denom,
                "I_k":         I_k,
            }

    return results


def heston_skew_result(raw_skew: dict, skew: str = "dI_dk"):
    """
    surface : 'dI0_dk' | 'dI_dk' | 'skew_corr'
    """
    assert skew in ("dI0_dk", "dI_dk", "skew_corr")
    method_map = {
        "dI0_dk":    "Heston-dI0_dk",
        "dI_dk":     "Heston-dI_dk",
        "skew_corr": "Heston-skew_corr",
    }
    return {
        "method":  method_map[skew],
        "raw":     raw_skew,
        "summary": pd.DataFrame({"T": sorted({T for T, K in raw_skew})}),
    }