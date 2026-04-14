import numpy as np
from src.volatility.analysis.diagnostics import compute_fd_atm_skew, compute_atm_forward_skew
from src.volatility.visualization.diagnostics import plot_power_law_term_structure
from src.volatility.visualization.smiles import _C
from src.volatility.config.config import (
    DATA_PATH, SNAPSHOT_DATE,
    COLOR_ESSVI, COLOR_LOCAL, COLOR_FD_ATM,
    DIAG_DK_BAND,
)


def run_analysis(df_smile, fit_essvi, fit_local):
    fd_skew = compute_fd_atm_skew(df_smile, dk_band=DIAG_DK_BAND)
    fwd     = compute_atm_forward_skew(fit_essvi, fit_local, df_smile=df_smile, dk_band=DIAG_DK_BAND)

    fig_spot = plot_power_law_term_structure(
        fit_essvi["summary"], fit_local["summary"], fd_skew,
        T_col="T",
        y_col=["atm_skew", "atm_skew", "fd_atm_skew"],
        labels=["eSSVI", "Local Quad", "FD Market"],
        colors=[COLOR_ESSVI, COLOR_LOCAL, COLOR_FD_ATM],
        colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
        title="ATM Spot Skew — Power-Law Term Structure",
        x_label="Log(T)", y_label="Log(|ATM Skew|)",
    )

    fig_fwd = plot_power_law_term_structure(
        fwd["eSSVI"], fwd["local"], fwd["fd"],
        T_col="T_mid",
        y_col="fwd_skew",
        labels=["eSSVI fwd", "Local Quad fwd", "FD Market fwd"],
        colors=[COLOR_ESSVI, COLOR_LOCAL, COLOR_FD_ATM],
        colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
        title="ATM Forward Skew — Power-Law Term Structure",
        x_label="Log(T_mid = √(T₁·T₂))", y_label="Log(|Fwd Skew|)",
    )

    return fig_spot, fig_fwd


if __name__ == "__main__":
    from src.volatility.pipeline.build_surface import build_surface
    from src.volatility.pipeline.run_calibration import run_calibration
    df_smile = build_surface(DATA_PATH, SNAPSHOT_DATE)
    fit_essvi, fit_local, fig_essvi, fig_local = run_calibration(df_smile)
    run_analysis(df_smile, fit_local, fit_essvi)