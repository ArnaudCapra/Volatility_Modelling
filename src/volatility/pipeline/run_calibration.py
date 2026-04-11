from src.volatility.calibration.fit_essiv import fit_essvi_surface
from src.volatility.calibration.fit_local import fit_localized_surface
from src.volatility.analysis.comparaison import print_comparison_summary
from src.volatility.visualization.diagnostics import plot_rmse_comparison, plot_term_structure_comparison
from src.volatility.visualization.smiles import plot_smile_comparison
from src.volatility.pipeline.build_surface import df_smile

# --- calibration ---
fit_essvi = fit_essvi_surface(df_smile)
fit_local = fit_localized_surface(df_smile)

# --- diagnostics ---
print_comparison_summary(fit_essvi, fit_local)
plot_smile_comparison(df_smile, fit_essvi, fit_local, n_slices=6)
plot_rmse_comparison(df_smile, fit_essvi, fit_local)
plot_term_structure_comparison(fit_essvi, fit_local, df_smile=df_smile, dk_band=0.05)