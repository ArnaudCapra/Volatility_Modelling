import numpy as np
from src.volatility.pricing.heston import heston_malliavin_surface, heston_surface_result
from src.volatility.visualization.interactive_surface import plot_iv_surface_interactive


# Heston parameters
kappa = 5.0    # mean-reversion speed
theta = 0.04   # long-run variance  → 20 % long-run vol
xi    = 0.3    # vol-of-vol
rho   = -0.1   # spot / vol Brownian correlation
S0    = 1.0    # normalised spot
v0    = 0.04   # initial variance  → 20 % initial vol
r     = 0.0

maturities = np.arange(0.075, 1.11, 0.1)
strikes    = np.linspace(0.8, 1.2, 9)

raw = heston_malliavin_surface(kappa, theta, xi, rho, S0, v0, maturities, strikes, N_paths= 5_000)

fig = plot_iv_surface_interactive(
    df_smile=None,
    surface_result=heston_surface_result(raw, surface="I"),
    k_range=(np.log(strikes[0]), np.log(strikes[-1])),   # required since no df_smile
)

fig.show()