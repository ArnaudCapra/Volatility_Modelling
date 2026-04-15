from src.volatility.pricing.heston import (
    heston_malliavin_surface, heston_surface_result,
    heston_malliavin_skew, heston_skew_result,
)
from src.volatility.visualization.interactive_surface import plot_surface_interactive
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
    raw_surface = heston_malliavin_surface(
        kappa, theta, xi, rho, S0, v0,
        maturities, strikes, N_steps, N_paths,
    )
    fig_surface = plot_surface_interactive(
        df_smile=None,
        surface_result=heston_surface_result(raw_surface, surface="I"),
        k_range=(np.log(strikes[0]), np.log(strikes[-1])),
    )
    return raw_surface, fig_surface


def run_malliavin_heston_skew(
    kappa=HESTON_KAPPA, theta=HESTON_THETA, xi=HESTON_XI, rho=HESTON_RHO,
    S0=HESTON_S0, v0=HESTON_V0,
    maturities=HESTON_MATURITIES, strikes=HESTON_STRIKES,
    N_steps=HESTON_N_STEPS, N_paths=HESTON_N_PATHS,
    surface_raw=None,   # pass raw_surface from run_malliavin_heston() to avoid re-simulation
):
    raw_skew = heston_malliavin_skew(
        kappa, theta, xi, rho, S0, v0,
        maturities, strikes, N_steps, N_paths,
        surface_raw=surface_raw,
    )
    fig_skew = plot_surface_interactive(
        df_smile=None,
        surface_result=heston_skew_result(raw_skew, skew="dI_dk"),
        k_range=(np.log(strikes[0]), np.log(strikes[-1])), is_volatility = False
    )
    return raw_skew, fig_skew


if __name__ == "__main__":
    # surface first — raw_surface is reused by the skew to avoid re-simulation
    raw_surface, fig_surface = run_malliavin_heston()
    fig_surface.show()

    raw_skew, fig_skew = run_malliavin_heston_skew(surface_raw=raw_surface)
    fig_skew.show()