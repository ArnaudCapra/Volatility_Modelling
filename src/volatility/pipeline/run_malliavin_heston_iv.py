from src.volatility.pricing.heston import heston_malliavin_surface, heston_surface_result
from src.volatility.visualization.interactive_surface import plot_iv_surface_interactive
from src.volatility.config.config import (
    HESTON_KAPPA, HESTON_THETA, HESTON_XI, HESTON_RHO,
    HESTON_S0, HESTON_V0, HESTON_R,
    HESTON_N_STEPS, HESTON_N_PATHS, HESTON_SEED,
    HESTON_MATURITIES, HESTON_STRIKES,
)
import numpy as np


def run_malliavin_heston(
    kappa=HESTON_KAPPA, theta=HESTON_THETA, xi=HESTON_XI, rho=HESTON_RHO,
    S0=HESTON_S0, v0=HESTON_V0,
    maturities=HESTON_MATURITIES, strikes=HESTON_STRIKES,
    N_steps=HESTON_N_STEPS, N_paths=HESTON_N_PATHS,
):
    raw = heston_malliavin_surface(
        kappa, theta, xi, rho, S0, v0,
        maturities, strikes, N_steps, N_paths,
    )
    return plot_iv_surface_interactive(
        df_smile=None,
        surface_result=heston_surface_result(raw, surface="I"),
        k_range=(np.log(strikes[0]), np.log(strikes[-1])),
    )


if __name__ == "__main__":
    fig = run_malliavin_heston()
    fig.show()