from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def plot_rmse_summary(summary: pd.DataFrame, save_path: str | Path | None = None):
    """Plot raw vs corrected RMSE and RMSE improvement."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(summary["fhr"], summary["rmse_c_raw"], marker="o", label="Raw GEFS ensemble mean")
    axes[0].plot(summary["fhr"], summary["rmse_c_corrected"], marker="o", label="Bias corrected")
    axes[0].set_title("Raw vs bias-corrected GEFS T2M RMSE")
    axes[0].set_xlabel("Forecast hour")
    axes[0].set_ylabel("Area-weighted RMSE (°C)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, linewidth=1)
    axes[1].plot(summary["fhr"], summary["rmse_improvement_c"], marker="o")
    axes[1].set_title("Positive values mean the correction improved RMSE")
    axes[1].set_xlabel("Forecast hour")
    axes[1].set_ylabel("RMSE improvement (°C)")
    axes[1].grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_spatial_map(da: xr.DataArray, title: str, save_path: str | Path | None = None, cmap: str = "coolwarm"):
    """Plot a simple latitude-longitude map without Cartopy."""
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(da.values), 98))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = None

    da.plot(
        ax=ax, x="longitude", y="latitude", cmap=cmap,
        vmin=-vmax if vmax else None, vmax=vmax,
        cbar_kwargs={"label": da.attrs.get("units", "")},
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
