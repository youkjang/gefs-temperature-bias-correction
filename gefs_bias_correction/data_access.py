from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .config import ProjectConfig
from .preprocessing import (
    remove_conflicting_scalar_coords,
    regrid_analysis_to_forecast,
    select_t2m_variable,
    subset_region,
    to_celsius,
)


def _safe_herbie_xarray(H, search: str) -> xr.Dataset:
    """Open a Herbie object as an xarray Dataset, handling list outputs."""
    out = H.xarray(search)
    if not isinstance(out, list):
        return out

    datasets = [item for item in out if isinstance(item, xr.Dataset) and len(item.data_vars) > 0]
    if not datasets:
        raise ValueError(f"Herbie returned no usable datasets for search={search!r}")

    for ds in datasets:
        try:
            select_t2m_variable(ds)
            return ds
        except ValueError:
            pass

    return datasets[0]


def load_gefs_mean_t2m(init_dt: pd.Timestamp, fhr: int, config: ProjectConfig) -> xr.DataArray:
    """Load GEFS ensemble-mean 2-m temperature forecast in Celsius."""
    from herbie import Herbie

    H = Herbie(
        init_dt.strftime("%Y-%m-%d %H:00"),
        model="gefs",
        product="atmos.5",
        member="mean",
        fxx=int(fhr),
        save_dir=str(config.cache_dir),
    )
    ds = _safe_herbie_xarray(H, "TMP:2 m")
    da = select_t2m_variable(ds)
    da = subset_region(to_celsius(da), config.region)
    da = remove_conflicting_scalar_coords(da)
    da.name = "forecast_t2m_c"
    da.attrs["units"] = "degC"
    return da


def load_gfs_analysis_t2m(valid_dt: pd.Timestamp, config: ProjectConfig) -> xr.DataArray:
    """Load GFS analysis 2-m temperature in Celsius at valid time."""
    from herbie import Herbie

    H = Herbie(
        valid_dt.strftime("%Y-%m-%d %H:00"),
        model="gfs",
        product="pgrb2.0p25",
        fxx=0,
        save_dir=str(config.cache_dir),
    )
    ds = _safe_herbie_xarray(H, "TMP:2 m")
    da = select_t2m_variable(ds)
    da = subset_region(to_celsius(da), config.region)
    da = remove_conflicting_scalar_coords(da)
    da.name = "analysis_t2m_c"
    da.attrs["units"] = "degC"
    return da


def build_matched_t2m_dataset(config: ProjectConfig) -> xr.Dataset:
    """Build matched GEFS forecast / GFS analysis dataset with case dimension."""
    forecast_cases, analysis_cases = [], []
    init_values, fhr_values, valid_values, errors = [], [], [], []

    for init_date in config.init_dates:
        init_dt = pd.to_datetime(f"{init_date} {config.init_hour}:00")
        for fhr in config.forecast_hours:
            valid_dt = init_dt + pd.Timedelta(hours=int(fhr))
            if config.verbose:
                print(f"Loading init={init_dt:%Y%m%d %HZ}, f{int(fhr):03d}, valid={valid_dt:%Y-%m-%d %HZ}")
            try:
                forecast = load_gefs_mean_t2m(init_dt, int(fhr), config)
                analysis = load_gfs_analysis_t2m(valid_dt, config)
                analysis = regrid_analysis_to_forecast(analysis, forecast)

                case_id = len(forecast_cases)
                forecast_cases.append(forecast.expand_dims(case=[case_id]))
                analysis_cases.append(analysis.expand_dims(case=[case_id]))
                init_values.append(init_dt.normalize())
                fhr_values.append(int(fhr))
                valid_values.append(valid_dt)
            except Exception as exc:
                message = f"Failed for init={init_dt:%Y%m%d %HZ}, f{int(fhr):03d}: {exc}"
                errors.append(message)
                if config.skip_missing:
                    print("WARNING:", message)
                    continue
                raise

    if not forecast_cases:
        raise ValueError("No forecast-analysis cases were loaded. Check dates, forecast hours, products, and network access.")

    forecast_all = xr.concat(forecast_cases, dim="case", compat="override", coords="minimal", combine_attrs="override")
    analysis_all = xr.concat(analysis_cases, dim="case", compat="override", coords="minimal", combine_attrs="override")

    ds = xr.Dataset({"forecast_t2m_c": forecast_all, "analysis_t2m_c": analysis_all})
    ds = ds.assign_coords(
        init_date=("case", pd.to_datetime(init_values)),
        fhr=("case", np.asarray(fhr_values, dtype=int)),
        valid_time=("case", pd.to_datetime(valid_values)),
    )
    if errors:
        ds.attrs["load_warnings"] = "\n".join(errors)
    return ds
