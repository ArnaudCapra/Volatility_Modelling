def print_comparison_summary(fit_essvi, fit_local):
    sep = "=" * 58
    print(sep)
    print(f"{'Method':<24} {'RMSE (×10⁴)':>16} {'MAE (×10⁴)':>16}")
    print("-" * 58)
    for fit in [fit_essvi, fit_local]:
        print(f"{fit['method']:<24} "
              f"{fit['rmse_total_var']*1e4:>16.4f} "
              f"{fit['mae_total_var']*1e4:>16.4f}")
    print(sep)
    print(f"\neSSVI global params:")
    print(f"  ρ₀   = {fit_essvi['rho_0']:+.4f}  ← short-maturity correlation")
    print(f"  ρ∞   = {fit_essvi['rho_inf']:+.4f}  ← long-maturity correlation")
    print(f"  c    = {fit_essvi['c_rho']:.4f}   ← transition speed in θ-space")
    print(f"  η    = {fit_essvi['eta']:.4f}   ← vol-of-vol scaling")
    print(f"  γ    = {fit_essvi['gamma']:.4f}   ← power-law exponent")
    print("\neSSVI per-maturity:")
    print(fit_essvi["summary"].to_string(index=False, float_format="{:.4f}".format))
    print("\nLocalized Quadratic per-maturity:")
    print(fit_local["summary"][["T","atm_iv","atm_skew","rmse","n_strikes"]]
          .to_string(index=False, float_format="{:.4f}".format))
    
