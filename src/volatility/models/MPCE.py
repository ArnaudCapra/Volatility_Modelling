import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd


def generate_correlated_noise(corr_matrix, T = 1,  N_sim = 252*5, M_factor = 100):
    dt = T / N_sim
    N_hist = int(M_factor * N_sim)
    N_total = N_hist + N_sim
    t_full = np.linspace(-M_factor * T, T, N_total)
    t_sim = t_full[N_hist:]
    L = np.linalg.cholesky(corr_matrix)
    Z_uncorr = np.random.normal(0, 1, (3, N_total))
    return dt, N_hist, N_sim, N_total, t_full, t_sim, L @ Z_uncorr

def generate_jacobi_process(dt, dW, x0, theta, kappa, sigma):
    """ Simulates Stochastic Hurst H_s over [-M, T] """
    n_steps = len(dW)
    path = np.zeros(n_steps)
    path[0] = x0
    sqrt_dt = np.sqrt(dt)
    
    for i in range(1, n_steps):
        drift = kappa * (theta - path[i-1]) * dt
        diff = sigma * np.sqrt(path[i-1] * (1 - path[i-1])) * (dW[i] * sqrt_dt)
        val = path[i-1] + drift + diff
        
        # Strict bounds for H in (0, 1)
        if val <= 0.01: val = 0.01
        if val >= 0.99: val = 0.99
        path[i] = val
    return path

def nu(H_s):
    return H_s/H_s  * np.sqrt(2 * H_s * np.sin(np.pi * H_s) * gamma(2 * H_s)) / gamma(H_s + 0.5)**2

def integrate_fractional_sde(dt, H_s, dW_cir, x0=1.0, Lambda=1, eta=0.2, zeta=0.1,
                              N_sim=252*5, t_sim=None, t_full=None, N_hist=None):
    """
    V_t = ∫_{-∞}^{t} K(t,s,H(s)) dY_s
    where Y_t is a standard CIR: dY = Λ(η−Y)dt + ζ√Y dW
    The Feller condition 2Λη ≥ ζ² now guarantees Y_t > 0 a.s.,
    making the kernel integral well-defined.
    """
    N_total = N_hist + N_sim

    if 2 * Lambda * eta >= zeta**2:
        print(f"Feller satisfied: 2*{Lambda}*{eta} >= {zeta}**2  ({2*Lambda*eta} >= {zeta**2})")
    else:
        print(f"Feller NOT satisfied: 2*{Lambda}*{eta} < {zeta}**2  ({2*Lambda*eta} < {zeta**2})")

    # Step 1: Simulate CIR Y_t over full history + simulation window
    Y_full = np.zeros(N_total)
    Y_full[0] = x0
    sqrt_dt = np.sqrt(dt)
    for i in range(1, N_total):
        y_prev = max(Y_full[i - 1], 0.0)
        Y_full[i] = y_prev + Lambda * (eta - y_prev) * dt + zeta * np.sqrt(y_prev) * dW_cir[i] * sqrt_dt

    # CIR increments dY
    dY_full = np.empty(N_total)
    dY_full[0] = 0.0
    dY_full[1:] = np.diff(Y_full)

    # Step 2: Apply multifractional kernel to dY
    nu_full = nu(H_s)
    path = np.zeros(N_sim)

    for i in range(N_sim):
        t = t_sim[i]

        valid_mask = t_full < t
        s_val  = t_full[valid_mask]
        H_val  = H_s[valid_mask]
        dY_val = dY_full[valid_mask]
        nu_val = nu_full[valid_mask]

        dist = t - s_val
        dist[dist < 1e-9] = dt / 100.0

        term_t = np.power(dist, H_val - 0.5)
        term_0 = np.zeros_like(s_val)
        neg_mask = s_val < 0
        term_0[neg_mask] = np.power(-s_val[neg_mask], H_val[neg_mask] - 0.5)

        kernel = (term_t - term_0) / gamma(H_val + 0.5)
        path[i] = np.sum(nu_val * kernel * dY_val)

        # Singular contribution: stochastic integral near the diagonal s → t
        # Var(∫_{t-dt}^t (t-s)^(H-½) dY_s) ≈ ζ² Y_t · dt^(2H) / (2H)
        h_curr  = H_s[N_hist + i]
        nu_curr = nu_full[N_hist + i]
        Y_curr  = max(Y_full[N_hist + i - 1], 1e-10)
        singular_std = (dt ** h_curr) / np.sqrt(2 * h_curr)
        singular_contribution = (nu_curr / gamma(h_curr + 0.5)) * singular_std \
                                * zeta * np.sqrt(Y_curr) * dW_cir[N_hist + i]

        path[i] += singular_contribution

    return path
def spot(dt, x_t, W, x0=1.0, r = .05):
    n_steps = len(x_t)
    x = np.zeros(n_steps)
    x[0] = x0
    dW = W * np.sqrt(dt)
    
    for i in range(1, n_steps):
        x[i] = x[i-1] + r *dt +  dW[i-1] * (x_t[i-1])**(1/2)
        
    return x