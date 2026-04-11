import numpy as np
from src.volatility.analysis.diagnostics import compute_fd_atm_skew, compute_atm_forward_skew
from src.volatility.visualization.diagnostics import plot_power_law_term_structure
from src.volatility.pipeline.build_surface import df_smile
from src.volatility.pipeline.run_calibration import fit_essvi, fit_local
from src.volatility.visualization.smiles import _C


fd_skew = compute_fd_atm_skew(df_smile)

fd_ok = fd_skew[fd_skew["fd_atm_skew"].notna()]

f_series=[
        {"T": fit_essvi["summary"]["T"].to_numpy(),
         "y": fit_essvi["summary"]["atm_skew"].to_numpy(),
         "label": "eSSVI",      "color": _C["eSSVI"], "color_fit": "#f0a09a"},
        {"T": fit_local["summary"]["T"].to_numpy(),
         "y": fit_local["summary"]["atm_skew"].to_numpy(),
         "label": "Local Quad", "color": _C["local"], "color_fit": "#9ac4f0"},
        {"T": fd_ok["T"].to_numpy(),
         "y": fd_ok["fd_atm_skew"].to_numpy(),
         "label": "FD Market",  "color": "#e69c3c",   "color_fit": "#f0cfa0"},
    ]


plot_power_law_term_structure(series=f_series, title="ATM Spot Skew — Power-Law Term Structure", x_label="Log(T)", y_label="Log(|ATM Skew|)")

fwd = compute_atm_forward_skew(fit_essvi, fit_local, df_smile=df_smile)

fk_series=[
        {"T": fwd["eSSVI"]["T_mid"].to_numpy(),
         "y": fwd["eSSVI"]["fwd_skew"].to_numpy(),
         "label": "eSSVI fwd",      "color": _C["eSSVI"], "color_fit": "#f0a09a"},
        {"T": fwd["local"]["T_mid"].to_numpy(),
         "y": fwd["local"]["fwd_skew"].to_numpy(),
         "label": "Local Quad fwd", "color": _C["local"], "color_fit": "#9ac4f0"},
        {"T": fwd["fd"]["T_mid"].to_numpy(),
         "y": fwd["fd"]["fwd_skew"].to_numpy(),
         "label": "FD Market fwd",  "color": "#e69c3c",   "color_fit": "#f0cfa0"},
    ]

plot_power_law_term_structure(series=fk_series,title="ATM Forward Skew — Power-Law Term Structure",x_label="Log(T_mid = √(T₁·T₂))", y_label="Log(|Fwd Skew|)")