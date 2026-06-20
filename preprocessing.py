from __future__ import annotations

import numpy as np
import xarray as xr


def infer_lat_lon_names(da: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    """Infer latitude and longitude coordinate names from an xarray object."""
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]

    lat_name = next((name for name in lat_candidates if name in da.coords), None)
    lon_name = next((name for name in lon_candidates if name in da.coords), None)

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not infer latitude/longitude coordinates. Available coords: {list(da.coords)}"
        )
    return lat_name, lon_name


def standardize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Rename common lat/lon coordinates and use 0-360 longitude convention.

    Output coordinates are named ``latitude`` and ``longitude``.
    """
    lat_name, lon_name = infer_lat_lon_names(da)

    rename_map = {}
    if lat_name != "latitude":
        rename_map[lat_name] = "latitude"
    if lon_name != "longitude":
        rename_map[lon_name] = "longitude"
    if rename_map:
        da = da.rename(rename_map)

    lon = da["longitude"]
    if float(lon.min()) < 0:
        da = da.assign_coords(longitude=(lon % 360))
        da = da.sortby("longitude")

    # Make latitude increasing so slices are intuitive.
    lat = da["latitude"]
    if lat.size > 1 and float(lat[0]) > float(lat[-1]):
        da = da.sortby("latitude")

    return da


def subset_box(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """Subset to a lat/lon box using 0-360 longitude convention."""
    da = standardize_lat_lon(da)
    return da.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )


def maybe_convert_temperature_units(da: xr.DataArray, units: str = "degC") -> xr.DataArray:
    """Convert Kelvin temperature to Celsius when requested."""
    units = units.lower()
    if units in {"degc", "c", "celsius", "°c"}:
        out = da - 273.15
        out.attrs.update(da.attrs)
        out.attrs["units"] = "degC"
        return out
    if units in {"k", "kelvin"}:
        da.attrs["units"] = "K"
        return da
    raise ValueError("temperature units must be one of: 'degC' or 'K'")


def interp_analysis_to_forecast_grid(
    analysis: xr.DataArray, forecast: xr.DataArray
) -> xr.DataArray:
    """Interpolate analysis to the forecast grid if coordinates differ.

    GEFS 0.25-degree and GFS 0.25-degree fields usually have matching grids.
    This function keeps the workflow robust if tiny coordinate differences occur.
    """
    analysis = standardize_lat_lon(analysis)
    forecast = standardize_lat_lon(forecast)

    same_lat = np.array_equal(analysis["latitude"].values, forecast["latitude"].values)
    same_lon = np.array_equal(analysis["longitude"].values, forecast["longitude"].values)

    if same_lat and same_lon:
        return analysis

    return analysis.interp(
        latitude=forecast["latitude"],
        longitude=forecast["longitude"],
        method="nearest",
    )
