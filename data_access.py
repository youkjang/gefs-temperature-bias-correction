from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import xarray as xr

from .config import ProjectConfig
from .paths import cached_path_for_url, gefs_ensemble_mean_url, gfs_analysis_url
from .preprocessing import (
    interp_analysis_to_forecast_grid,
    maybe_convert_temperature_units,
    standardize_lat_lon,
    subset_box,
)
from .time_utils import parse_init_time, valid_time_from_init


T2M_CANDIDATE_NAMES = ("t2m", "2t", "tmp", "TMP")


def _extract_t2m(ds: xr.Dataset) -> xr.DataArray:
    """Extract a 2-m temperature DataArray from a Herbie/cfgrib Dataset."""
    var_name = next((name for name in T2M_CANDIDATE_NAMES if name in ds.data_vars), None)
    if var_name is None:
        if len(ds.data_vars) == 1:
            var_name = list(ds.data_vars)[0]
        else:
            raise KeyError(
                "Could not identify 2-m temperature variable. "
                f"Available variables: {list(ds.data_vars)}"
            )
    da = ds[var_name]
    da.name = "t2m"
    return standardize_lat_lon(da)


def _herbie_object(*args, **kwargs):
    """Create a Herbie object while being tolerant of minor API changes."""
    from herbie import Herbie

    try:
        return Herbie(*args, priority=["aws", "nomads"], **kwargs)
    except TypeError:
        return Herbie(*args, **kwargs)


def open_gefs_mean_t2m_with_herbie(
    init_date: str,
    init_hour: str | int,
    fhr: int,
    cache_dir: str | Path,
    units: str = "degC",
) -> xr.DataArray:
    """Open GEFS ensemble-mean 2-m temperature using Herbie subsetting."""
    init_dt = parse_init_time(init_date, init_hour)
    H = _herbie_object(
        init_dt.strftime("%Y-%m-%d %H:00"),
        model="gefs",
        product="atmos.25",
        member="mean",
        fxx=int(fhr),
        save_dir=str(cache_dir),
    )
    ds = H.xarray("TMP:2 m")
    da = _extract_t2m(ds)
    return maybe_convert_temperature_units(da, units=units)


def open_gfs_analysis_t2m_with_herbie(
    valid_time: pd.Timestamp,
    cache_dir: str | Path,
    units: str = "degC",
) -> xr.DataArray:
    """Open GFS f000 2-m temperature using Herbie subsetting."""
    valid_time = pd.Timestamp(valid_time)
    H = _herbie_object(
        valid_time.strftime("%Y-%m-%d %H:00"),
        model="gfs",
        product="pgrb2.0p25",
        fxx=0,
        save_dir=str(cache_dir),
    )
    ds = H.xarray("TMP:2 m")
    da = _extract_t2m(ds)
    return maybe_convert_temperature_units(da, units=units)


def download_file(url: str, local_path: str | Path, overwrite: bool = False, timeout: int = 60) -> Path:
    """Download a remote file to a local cache path.

    This fallback downloads the full GRIB2 file. For routine use, prefer Herbie
    because it uses the GRIB index to download only the requested variable.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0 and not overwrite:
        return local_path

    with requests.get(url, stream=True, timeout=timeout) as response:
        if response.status_code != 200:
            raise FileNotFoundError(
                f"Could not download file. HTTP {response.status_code}: {url}\n"
                "Check the date, cycle, forecast hour, and product path."
            )
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return local_path


def open_t2m_from_grib(path: str | Path) -> xr.DataArray:
    """Open 2-m temperature from a local GRIB2 file using cfgrib."""
    backend_kwargs = {
        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2},
        "indexpath": "",
    }
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
    return _extract_t2m(ds)


def open_gefs_mean_t2m(
    init_date: str,
    init_hour: str | int,
    fhr: int,
    cache_dir: str | Path,
    units: str = "degC",
    overwrite: bool = False,
    prefer_herbie: bool = True,
) -> xr.DataArray:
    """Open GEFS ensemble-mean 2-m temperature for one forecast."""
    if prefer_herbie:
        try:
            return open_gefs_mean_t2m_with_herbie(init_date, init_hour, fhr, cache_dir, units)
        except ImportError as exc:
            raise ImportError(
                "Herbie is not installed. Install with: pip install herbie-data\n"
                "This project uses Herbie to download only the TMP:2 m GRIB message."
            ) from exc

    url = gefs_ensemble_mean_url(init_date, init_hour, fhr)
    subdir = f"gefs/{init_date}/{int(init_hour):02d}"
    local_path = cached_path_for_url(url, cache_dir, subdir=subdir)
    path = download_file(url, local_path, overwrite=overwrite)
    da = open_t2m_from_grib(path)
    return maybe_convert_temperature_units(da, units=units)


def open_gfs_analysis_t2m(
    valid_time: pd.Timestamp,
    cache_dir: str | Path,
    units: str = "degC",
    overwrite: bool = False,
    prefer_herbie: bool = True,
) -> xr.DataArray:
    """Open GFS f000 2-m temperature at a GEFS valid time."""
    valid_time = pd.Timestamp(valid_time)
    if prefer_herbie:
        try:
            return open_gfs_analysis_t2m_with_herbie(valid_time, cache_dir, units)
        except ImportError as exc:
            raise ImportError(
                "Herbie is not installed. Install with: pip install herbie-data\n"
                "This project uses Herbie to download only the TMP:2 m GRIB message."
            ) from exc

    url = gfs_analysis_url(valid_time.to_pydatetime())
    subdir = f"gfs/{valid_time:%Y%m%d}/{valid_time:%H}"
    local_path = cached_path_for_url(url, cache_dir, subdir=subdir)
    path = download_file(url, local_path, overwrite=overwrite)
    da = open_t2m_from_grib(path)
    return maybe_convert_temperature_units(da, units=units)


def build_matched_t2m_dataset(
    config: ProjectConfig,
    overwrite: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Build matched GEFS forecast and GFS analysis fields.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``forecast`` and ``analysis`` and dimensions
        ``init_date``, ``fhr``, ``latitude``, and ``longitude``.
    """
    forecast_by_date: list[xr.DataArray] = []
    analysis_by_date: list[xr.DataArray] = []

    for init_date in config.init_dates:
        init_dt = parse_init_time(init_date, config.init_hour)
        forecast_by_fhr: list[xr.DataArray] = []
        analysis_by_fhr: list[xr.DataArray] = []

        for fhr in config.forecast_hours:
            valid_dt = pd.Timestamp(valid_time_from_init(init_date, config.init_hour, fhr))
            if verbose:
                print(
                    f"Loading init={init_date} {config.init_hour}Z, "
                    f"f{fhr:03d}, valid={valid_dt:%Y-%m-%d %HZ}"
                )

            forecast = open_gefs_mean_t2m(
                init_date=init_date,
                init_hour=config.init_hour,
                fhr=fhr,
                cache_dir=config.cache_dir,
                units=config.temperature_units,
                overwrite=overwrite,
                prefer_herbie=config.prefer_herbie,
            )
            analysis = open_gfs_analysis_t2m(
                valid_time=valid_dt,
                cache_dir=config.cache_dir,
                units=config.temperature_units,
                overwrite=overwrite,
                prefer_herbie=config.prefer_herbie,
            )

            forecast = subset_box(
                forecast,
                lat_min=config.lat_min,
                lat_max=config.lat_max,
                lon_min=config.lon_min,
                lon_max=config.lon_max,
            )
            analysis = subset_box(
                analysis,
                lat_min=config.lat_min,
                lat_max=config.lat_max,
                lon_min=config.lon_min,
                lon_max=config.lon_max,
            )
            analysis = interp_analysis_to_forecast_grid(analysis, forecast)

            forecast_by_fhr.append(forecast.expand_dims(fhr=[int(fhr)]))
            analysis_by_fhr.append(analysis.expand_dims(fhr=[int(fhr)]))

        forecast_date = xr.concat(forecast_by_fhr, dim="fhr").expand_dims(
            init_date=[pd.Timestamp(init_dt)]
        )
        analysis_date = xr.concat(analysis_by_fhr, dim="fhr").expand_dims(
            init_date=[pd.Timestamp(init_dt)]
        )
        forecast_by_date.append(forecast_date)
        analysis_by_date.append(analysis_date)

    forecast_all = xr.concat(forecast_by_date, dim="init_date")
    analysis_all = xr.concat(analysis_by_date, dim="init_date")

    ds = xr.Dataset({"forecast": forecast_all, "analysis": analysis_all})
    ds["forecast"].attrs["long_name"] = "GEFS ensemble-mean 2-m temperature"
    ds["analysis"].attrs["long_name"] = "GFS f000 2-m temperature analysis"
    ds.attrs["project"] = "GEFS 2-m temperature bias correction"
    ds.attrs["verification_note"] = "GFS f000 fields are used as the verifying analysis."
    return ds
