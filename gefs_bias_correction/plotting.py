from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


def _maybe_add_cartopy(ax):
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import cartopy.feature as cfeature

        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.STATES, linewidth=0.25)
    except Exception:
        # Plain matplotlib fallback. This keeps the notebook usable without cartopy.
        pass


def plot_map(
    da: xr.DataArray,
    title: str,
    cbar_label: str = "",
    save_path: str | Path | None = None,
    robust: bool = True,
):
    """Plot a latitude-longitude field with optional cartopy coastlines."""
    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(9, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="longitude",
            y="latitude",
            robust=robust,
            add_colorbar=True,
            cbar_kwargs={"label": cbar_label},
        )
        _maybe_add_cartopy(ax)
        ax.set_extent(
            [float(da.longitude.min()), float(da.longitude.max()), float(da.latitude.min()), float(da.latitude.max())],
            crs=ccrs.PlateCarree(),
        )
        ax.set_title(title)
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 5))
        da.plot.pcolormesh(
            ax=ax,
            x="longitude",
            y="latitude",
            robust=robust,
            add_colorbar=True,
            cbar_kwargs={"label": cbar_label},
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_raw_vs_corrected_by_lead(
    metrics_ds: xr.Dataset,
    metric_name: str = "rmse",
    save_path: str | Path | None = None,
):
    """Plot raw and corrected area-mean metrics by lead time."""
    raw_name = f"raw_{metric_name}"
    corrected_name = f"corrected_{metric_name}"
    if raw_name not in metrics_ds or corrected_name not in metrics_ds:
        raise KeyError(f"Expected {raw_name!r} and {corrected_name!r} in metrics dataset.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics_ds["fhr"], metrics_ds[raw_name], marker="o", label="Raw GEFS")
    ax.plot(metrics_ds["fhr"], metrics_ds[corrected_name], marker="o", label="Bias-corrected GEFS")
    ax.set_xlabel("Forecast lead time (hours)")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"Raw vs bias-corrected {metric_name.upper()} by lead time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
