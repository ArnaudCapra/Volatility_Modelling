import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.volatility.models.essiv import essvi_total_variance
from src.volatility.analysis.diagnostics import compute_fd_atm_skew
from src.volatility.visualization.smiles import _C


def plot_rmse_comparison(df_smile, fit_essvi, fit_local):
    """
    Per-maturity RMSE in total-variance units (×10⁴).
    Short-maturity bars are most revealing: local quad has an unfair advantage
    (zero regularisation) while eSSVI imposes cross-maturity smoothness.
    """
    rows = []
    for tdays, sl in sorted(df_smile.groupby("T_days"), key=lambda x: x[0]):
        T  = float(sl["T"].iloc[0])
        k  = sl["k"].astype(float).to_numpy()
        w  = sl["total_variance"].astype(float).to_numpy()
        ok = np.isfinite(k) & np.isfinite(w)
        k, w = k[ok], w[ok]
        if len(k) < 3:
            continue

        try:
            theta = float(np.interp(T, fit_essvi["T_knots"], fit_essvi["theta_knots"]))
            w_e   = essvi_total_variance(k, theta, fit_essvi["rho_inf"], fit_essvi["rho_0"],
                                         fit_essvi["c_rho"], fit_essvi["eta"], fit_essvi["gamma"])
            re    = float(np.sqrt(np.mean((w_e - w) ** 2)))
        except Exception:
            re = np.nan

        try:
            w_l = fit_local["predict"](T, k)
            rl  = float(np.sqrt(np.mean((w_l - w) ** 2)))
        except Exception:
            rl = np.nan

        rows.append({"T_days": tdays, "T": T, "eSSVI": re, "Local": rl})

    df_r = pd.DataFrame(rows)
    if df_r.empty:
        return

    x     = np.arange(len(df_r))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(df_r) * 0.7), 5))
    ax.bar(x - width / 2, df_r["eSSVI"] * 1e4, width,
           color=_C["eSSVI"], alpha=0.85, label="eSSVI")
    ax.bar(x + width / 2, df_r["Local"] * 1e4, width,
           color=_C["local"], alpha=0.85, label="Local Quad")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}d" for d in df_r["T_days"]], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("RMSE (total variance × 10⁴)")
    ax.set_title("Per-Maturity RMSE — eSSVI vs Localized Quadratic\n"
                 "(Local Quad wins in-sample by design; gap is largest at short maturities)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_term_structure_comparison(fit_essvi, fit_local, df_smile=None, dk_band=0.05):
    T_e  = fit_essvi["summary"]["T"].to_numpy()
    iv_e = fit_essvi["summary"]["atm_iv"].to_numpy()
    sk_e = fit_essvi["summary"]["atm_skew"].to_numpy()
    rh_e = fit_essvi["summary"]["rho_T"].to_numpy()

    T_l  = fit_local["summary"]["T"].to_numpy()
    iv_l = fit_local["summary"]["atm_iv"].to_numpy()
    sk_l = fit_local["summary"]["atm_skew"].to_numpy()

    # Finite-difference skew from raw data (optional)
    fd = None
    if df_smile is not None:
        fd = compute_fd_atm_skew(df_smile, dk_band=dk_band)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- ATM IV ---
    ax = axes[0]
    ax.plot(T_e, iv_e * 100, "o-",  color=_C["eSSVI"], lw=2, label="eSSVI")
    ax.plot(T_l, iv_l * 100, "s--", color=_C["local"],  lw=2, label="Local Quad")
    ax.set(xlabel="T (years)", ylabel="ATM IV (%)", title="ATM IV Term Structure")
    ax.legend(); ax.grid(True, alpha=0.25)

    # --- ATM Skew ---
    ax = axes[1]
    ax.plot(T_e, sk_e, "o-",  color=_C["eSSVI"], lw=2,   label="eSSVI")
    ax.plot(T_l, sk_l, "s--", color=_C["local"],  lw=2,   label="Local Quad")
    if fd is not None:
        ok = fd["fd_atm_skew"].notna()
        ax.scatter(
            fd.loc[ok, "T"], fd.loc[ok, "fd_atm_skew"],
            marker="^", s=55, color="#e69c3c", zorder=6,
            label=f"FD market (±{dk_band:.2f})",
        )
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(xlabel="T (years)", ylabel="∂IV/∂k at k=0", title="ATM Skew Term Structure")
    ax.legend(); ax.grid(True, alpha=0.25)

    # --- eSSVI effective rho ---
    ax = axes[2]
    ax.plot(T_e, rh_e, "o-", color=_C["eSSVI"], lw=2)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(xlabel="T (years)", ylabel="ρ(θ(T))",
           title="eSSVI Effective Correlation ρ vs Maturity\n(short → ρ₀, long → ρ∞)")
    ax.grid(True, alpha=0.25)

    plt.suptitle("Term Structure Diagnostics", fontsize=12)
    plt.tight_layout()
    plt.show()