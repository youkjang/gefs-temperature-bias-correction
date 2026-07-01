import xarray as xr


def standardize_lat_lon_names(da: xr.DataArray) -> xr.DataArray:
    """Rename common latitude/longitude coordinate names to latitude/longitude."""
    rename = {}
    if "lat" in da.coords:
        rename["lat"] = "latitude"
    if "lon" in da.coords:
        rename["lon"] = "longitude"
    return da.rename(rename) if rename else da


def convert_longitude_to_180(da: xr.DataArray) -> xr.DataArray:
    """Convert longitude coordinates from 0-360 to -180-180 if needed."""
    da = standardize_lat_lon_names(da)
    if float(da.longitude.max()) > 180.0:
        da = da.assign_coords(longitude=((da.longitude + 180.0) % 360.0) - 180.0)
        da = da.sortby("longitude")
    return da


def sort_latitude_longitude(da: xr.DataArray) -> xr.DataArray:
    """Sort latitude and longitude coordinates in increasing order."""
    da = standardize_lat_lon_names(da)
    if "latitude" in da.coords:
        da = da.sortby("latitude")
    if "longitude" in da.coords:
        da = da.sortby("longitude")
    return da


def subset_region(da: xr.DataArray, region) -> xr.DataArray:
    """Subset a DataArray to the configured region."""
    da = convert_longitude_to_180(da)
    da = sort_latitude_longitude(da)
    return da.sel(latitude=slice(region.south, region.north), longitude=slice(region.west, region.east))


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius when needed."""
    units = str(da.attrs.get("units", "")).lower()
    out = da - 273.15 if units in {"k", "kelvin"} or float(da.max()) > 150.0 else da.copy()
    out.attrs["units"] = "degC"
    return out


def select_t2m_variable(ds: xr.Dataset) -> xr.DataArray:
    """Select the likely 2-m temperature variable from a Herbie/cfgrib dataset."""
    for name in ["t2m", "tmp", "TMP", "2t"]:
        if name in ds.data_vars:
            return ds[name]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise ValueError(f"Could not identify T2M variable. Available variables: {list(ds.data_vars)}")


def remove_conflicting_scalar_coords(da: xr.DataArray) -> xr.DataArray:
    """Remove scalar GRIB coordinates that can cause concat conflicts."""
    drop_names = []
    for name in ["time", "step", "valid_time", "surface", "heightAboveGround"]:
        if name in da.coords and da[name].ndim == 0:
            drop_names.append(name)
    return da.drop_vars(drop_names) if drop_names else da


def regrid_analysis_to_forecast(analysis: xr.DataArray, forecast: xr.DataArray) -> xr.DataArray:
    """Interpolate analysis field to the GEFS forecast grid."""
    analysis = sort_latitude_longitude(convert_longitude_to_180(analysis))
    forecast = sort_latitude_longitude(convert_longitude_to_180(forecast))
    return analysis.interp(latitude=forecast["latitude"], longitude=forecast["longitude"], method="linear")
