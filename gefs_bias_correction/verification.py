from __future__ import annotations

import numpy as np
import xarray as xr


def forecast_error(forecast: xr.DataArray, analysis: xr.DataArray) -> xr.DataArray:
    """Forecast error, defined as forecast minus analysis."""
    err = forecast - analysis
    err.name = "forecast_error"
    return err


def bias(
    forecast: xr.DataArray,
    analysis: xr.DataArray,
    dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Mean error, defined as mean(forecast - analysis)."""
    out = (forecast - analysis).mean(dim=dim)
    out.name = "bias"
    return out


def rmse(
    forecast: xr.DataArray,
    analysis: xr.DataArray,
    dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Root mean square error."""
    out = np.sqrt(((forecast - analysis) ** 2).mean(dim=dim))
    out.name = "rmse"
    return out


def mae(
    forecast: xr.DataArray,
    analysis: xr.DataArray,
    dim: str | list[str] | None = None,
) -> xr.DataArray:
    """Mean absolute error."""
    out = np.abs(forecast - analysis).mean(dim=dim)
    out.name = "mae"
    return out


def area_weighted_mean(
    da: xr.DataArray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """Cosine-latitude-weighted spatial mean."""
    if lat_name not in da.coords:
        raise ValueError(f"Latitude coordinate {lat_name!r} not found.")
    if lon_name not in da.coords and lon_name in da.dims:
        pass

    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=[lat_name, lon_name])


def summarize_metrics_by_lead(
    raw_forecast: xr.DataArray,
    corrected_forecast: xr.DataArray,
    analysis: xr.DataArray,
) -> xr.Dataset:
    """Return area-mean bias/RMSE/MAE by forecast lead time."""
    raw_bias_map = bias(raw_forecast, analysis, dim="init_date")
    corrected_bias_map = bias(corrected_forecast, analysis, dim="init_date")

    raw_rmse_map = rmse(raw_forecast, analysis, dim="init_date")
    corrected_rmse_map = rmse(corrected_forecast, analysis, dim="init_date")

    raw_mae_map = mae(raw_forecast, analysis, dim="init_date")
    corrected_mae_map = mae(corrected_forecast, analysis, dim="init_date")

    return xr.Dataset(
        {
            "raw_bias": area_weighted_mean(raw_bias_map),
            "corrected_bias": area_weighted_mean(corrected_bias_map),
            "raw_rmse": area_weighted_mean(raw_rmse_map),
            "corrected_rmse": area_weighted_mean(corrected_rmse_map),
            "raw_mae": area_weighted_mean(raw_mae_map),
            "corrected_mae": area_weighted_mean(corrected_mae_map),
        }
    )
