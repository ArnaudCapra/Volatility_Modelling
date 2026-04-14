from src.volatility.calibration.fit_essiv import fit_essvi_surface
from src.volatility.calibration.fit_local import fit_localized_surface
from src.volatility.analysis.comparaison import print_comparison_summary
from src.volatility.visualization.diagnostics import plot_rmse_comparison, plot_term_structure_comparison
from src.volatility.visualization.smiles import plot_smile_comparison
from src.volatility.visualization.interactive_surface import plot_iv_surface_interactive
from src.volatility.config.config import DATA_PATH, SNAPSHOT_DATE, VIZ_N_SLICES, DIAG_DK_BAND


def run_calibration(df_smile):
    fit_essvi = fit_essvi_surface(df_smile)
    fit_local = fit_localized_surface(df_smile)

    # --- diagnostics ---
    print_comparison_summary(fit_essvi, fit_local)
    plot_smile_comparison(df_smile, fit_essvi, fit_local, n_slices=VIZ_N_SLICES)
    plot_rmse_comparison(df_smile, fit_essvi, fit_local)
    plot_term_structure_comparison(fit_essvi, fit_local, df_smile=df_smile, dk_band=DIAG_DK_BAND)

    # --- surfaces ---
    return (
        fit_localized_surface(df_smile),
        fit_essvi_surface(df_smile),
        plot_iv_surface_interactive(df_smile, fit_essvi),
        plot_iv_surface_interactive(df_smile, fit_local),
    )


if __name__ == "__main__":
    from src.volatility.pipeline.build_surface import build_surface
    df_smile = build_surface(DATA_PATH, SNAPSHOT_DATE)
    fit_essvi, fit_local, fig_essvi, fig_local = run_calibration(df_smile)
    print(fit_essvi, fit_local)
    fig_essvi.show()
    fig_local.show()