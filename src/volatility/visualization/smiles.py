import numpy as np
import matplotlib.pyplot as plt
from src.volatility.surfaces.essiv_surface import _iv_essvi
from src.volatility.surfaces.local_surface import _iv_local


_C = {"eSSVI": "#e6453c", "local": "#3c7fe6", "mkt": "#222222"}

def plot_smile_comparison(df_smile, fit_essvi, fit_local, n_slices=6):
    """
    Grid of smile plots: market dots + eSSVI line + Local-Quad dashes.
    Slices are picked to cover short, mid, and long maturities uniformly.
    """
    grouped = sorted(df_smile.groupby("T_days"), key=lambda x: x[0])
    n = len(grouped)
    if n == 0:
        return

    # Evenly spaced indices, always including shortest and longest
    idx = sorted({0, n - 1} | {int(round(i * (n - 1) / (n_slices - 1))) for i in range(n_slices)})
    selected = [grouped[i] for i in idx[:n_slices]]

    ncols = 2
    nrows = (len(selected) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    for ax_i, (tdays, sl) in enumerate(selected):
        ax = axes[ax_i]
        T  = float(sl["T"].iloc[0])

        k_m  = sl["k"].astype(float).to_numpy()
        iv_m = sl["IV"].astype(float).to_numpy()
        ok   = np.isfinite(k_m) & np.isfinite(iv_m)
        k_m, iv_m = k_m[ok], iv_m[ok]
        if len(k_m) < 3:
            ax.set_visible(False)
            continue

        k_lo   = min(k_m.min() - 0.02, -0.05)
        k_hi   = max(k_m.max() + 0.02,  0.05)
        k_grid = np.linspace(k_lo, k_hi, 400)

        ax.scatter(k_m, iv_m * 100, s=22, color=_C["mkt"],
                   zorder=5, label="Market", alpha=0.8)

        try:
            ax.plot(k_grid, _iv_essvi(fit_essvi, T, k_grid) * 100,
                    color=_C["eSSVI"], lw=2.0, label="eSSVI")
        except Exception:
            pass

        try:
            ax.plot(k_grid, _iv_local(fit_local, T, k_grid) * 100,
                    color=_C["local"], lw=2.0, ls="--", label="Local Quad")
        except Exception:
            pass

        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("IV (%)")
        label = "SHORT" if ax_i == 0 else ("LONG" if ax_i == len(selected) - 1 else "")
        ax.set_title(f"T = {T:.3f}y  ({tdays}d)  {label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("eSSVI vs Localized Quadratic — Smile Fits", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.show()