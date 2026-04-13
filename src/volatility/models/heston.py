
import numpy as np
def simulate_heston(kappa, theta, xi, rho, S0, v0, T,
                    N_steps, N_paths, r=0.0, seed=None):
    """
    Heston dynamics:
      d(σ²_t) = κ(θ − σ²_t) dt + ξ σ_t dW_t
      dX_t    = (r − σ²_t/2) dt + σ_t dB_t,    corr(dB, dW) = ρ

    Parameters
    ----------
    v0 : float   initial variance σ₀²

    Returns
    -------
    t   : (N_steps+1,)
    X   : (N_paths, N_steps+1)  log-price paths
    var : (N_paths, N_steps+1)  variance paths σ²
    """
    rng = np.random.default_rng(seed)
    dt  = T / N_steps
    t   = np.linspace(0.0, T, N_steps + 1)

    X   = np.empty((N_paths, N_steps + 1))
    var = np.empty((N_paths, N_steps + 1))
    X[:, 0], var[:, 0] = np.log(S0), v0

    sq_dt = np.sqrt(dt)
    # Generate all random increments at once for efficiency
    Z1 = rng.standard_normal((N_paths, N_steps))
    Z2 = rng.standard_normal((N_paths, N_steps))
    dW = sq_dt * Z1
    dB = sq_dt * (rho * Z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * Z2)

    for i in range(N_steps):
        v  = np.maximum(var[:, i], 0.0)          # full truncation
        sv = np.sqrt(v)
        X[:,   i+1] = X[:,   i] + (r - 0.5*v)*dt + sv*dB[:, i]
        var[:, i+1] = np.maximum(
            v + kappa*(theta - v)*dt + xi*sv*dW[:, i], 0.0)

    return t, X, var


# ══════════════════════════════════════════════════════════════════════
#  Heston-specific quantities  (Image 2)
# ══════════════════════════════════════════════════════════════════════

def effective_vol(var, dt, T):
    """
    Y_t = ∫_t^T σ_s² ds  (integrated future variance)
    v_t = √(Y_t / (T−t))  (RMS effective vol)

    Fully vectorised backward cumulative sum.
    Returns v, Y both of shape (N_paths, N_steps+1).
    """
    N   = var.shape[1]
    tau = np.maximum(T - np.arange(N) * dt, dt)   # (N,)  avoid /0 at expiry
    Y   = np.flip(np.cumsum(np.flip(var, axis=1), axis=1), axis=1) * dt
    return np.sqrt(np.maximum(Y / tau, 1e-14)), Y


def malliavin_phi(var, xi, kappa, theta, dt):
    """
    Heston Malliavin derivative of variance (Image 2):
      D^W_{W,u}(σ²_t) = 1_{[t≥u]} · ξ σ_t · exp(−∫_u^t λ_s ds)
      λ_s = ξ/2 + (4κθ − ξ²) / (8σ_s²)

    Malliavin weight from Theorem 6.4.1 (Image 1):
      Φ_s = σ_s ∫_s^T D^W_s(σ_u²) du

    Substituting the Heston expression:
      Φ_s = ξ σ_s² · e^{Λ_s} · ∫_s^T e^{−Λ_u} du
    where  Λ_u = ∫_0^u λ_v dv  (forward cumulative integral).

    Fully vectorised — no Python loops.
    Returns Phi of shape (N_paths, N_steps+1).
    """
    eps   = 1e-12
    lam   = xi/2 + (4*kappa*theta - xi**2) / (8*np.maximum(var, eps))  # λ_s
    cum_L = np.cumsum(lam * dt, axis=1)                                  # Λ_u
    # ∫_s^T exp(−Λ_u) du  via backward cumulative sum
    back  = np.flip(
                np.cumsum(np.flip(np.exp(-cum_L), axis=1), axis=1),
                axis=1
            ) * dt
    return xi * var * np.exp(cum_L) * back    # (N_paths, N_steps+1)
