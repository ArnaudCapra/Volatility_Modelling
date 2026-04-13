import numpy as np
from src.volatility.analysis.diagnostics import compute_fd_atm_skew, compute_atm_forward_skew
from src.volatility.visualization.diagnostics import plot_power_law_term_structure
from src.volatility.pipeline.build_surface import df_smile
from src.volatility.pipeline.run_calibration import fit_essvi, fit_local
from src.volatility.visualization.smiles import _C


fd_skew = compute_fd_atm_skew(df_smile)

plot_power_law_term_structure(
    fit_essvi["summary"], fit_local["summary"], fd_skew,
    T_col="T",
    y_col=["atm_skew", "atm_skew", "fd_atm_skew"],
    labels=["eSSVI", "Local Quad", "FD Market"],
    colors=[_C["eSSVI"], _C["local"], "#e69c3c"],
    colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
    title="ATM Spot Skew — Power-Law Term Structure",
    x_label="Log(T)", y_label="Log(|ATM Skew|)",
)
fwd = compute_atm_forward_skew(fit_essvi, fit_local, df_smile=df_smile)

plot_power_law_term_structure(
    fwd["eSSVI"], fwd["local"], fwd["fd"],
    T_col="T_mid",
    y_col="fwd_skew",
    labels=["eSSVI fwd", "Local Quad fwd", "FD Market fwd"],
    colors=[_C["eSSVI"], _C["local"], "#e69c3c"],
    colors_fit=["#f0a09a", "#9ac4f0", "#f0cfa0"],
    title="ATM Forward Skew — Power-Law Term Structure",
    x_label="Log(T_mid = √(T₁·T₂))", y_label="Log(|Fwd Skew|)",
)