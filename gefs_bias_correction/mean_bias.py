import numpy as np
import pandas as pd
import xarray as xr


def compute_mean_bias_by_lead(train_ds: xr.Dataset) -> xr.DataArray:
    """Compute spatial mean-bias maps separately for each forecast lead time."""
    for name in ["forecast_t2m_c", "analysis_t2m_c", "fhr"]:
        if name not in train_ds:
            raise ValueError(f"train_ds must contain '{name}'.")
    if "case" not in train_ds.dims:
        raise ValueError("train_ds must have a 'case' dimension.")

    bias_maps = []
    for fhr in sorted(np.unique(train_ds["fhr"].values)):
        fhr_mask = train_ds["fhr"] == fhr
        forecast_fhr = train_ds["forecast_t2m_c"].where(fhr_mask, drop=True)
        analysis_fhr = train_ds["analysis_t2m_c"].where(fhr_mask, drop=True)
        if forecast_fhr.sizes.get("case", 0) == 0:
            continue
        error = forecast_fhr - analysis_fhr
        mean_bias = error.mean("case").expand_dims(fhr=[int(fhr)])
        bias_maps.append(mean_bias)

    if not bias_maps:
        raise ValueError("No bias maps were computed. Check training data and fhr values.")

    out = xr.concat(bias_maps, dim="fhr", compat="override", coords="minimal", combine_attrs="override")
    out.name = "mean_bias_c"
    out.attrs["long_name"] = "Training mean bias by forecast lead time"
    out.attrs["units"] = "degC"
    return out


def apply_mean_bias_correction(ds_input: xr.Dataset, mean_bias_by_lead: xr.DataArray) -> xr.DataArray:
    """Apply lead-time-dependent spatial mean-bias correction to each case."""
    for name in ["forecast_t2m_c", "fhr", "init_date", "valid_time"]:
        if name not in ds_input:
            raise ValueError(f"ds_input must contain '{name}'.")
    if "case" not in ds_input.dims:
        raise ValueError("ds_input must have a 'case' dimension.")

    available_fhrs = set(int(x) for x in mean_bias_by_lead["fhr"].values)
    needed_fhrs = set(int(x) for x in np.unique(ds_input["fhr"].values))
    missing = sorted(needed_fhrs - available_fhrs)
    if missing:
        raise ValueError(f"mean_bias_by_lead is missing forecast hours: {missing}")

    corrected_cases = []
    for case_id in ds_input["case"].values:
        fhr = int(ds_input["fhr"].sel(case=case_id).item())
        raw = ds_input["forecast_t2m_c"].sel(case=case_id)
        bias_map = mean_bias_by_lead.sel(fhr=fhr)
        corrected_cases.append((raw - bias_map).expand_dims(case=[case_id]))

    if not corrected_cases:
        raise ValueError("No corrected cases were created.")

    corrected_all = xr.concat(corrected_cases, dim="case", compat="override", coords="minimal", combine_attrs="override")
    corrected_all = corrected_all.assign_coords(
        init_date=("case", pd.to_datetime(ds_input["init_date"].values)),
        fhr=("case", ds_input["fhr"].values),
        valid_time=("case", pd.to_datetime(ds_input["valid_time"].values)),
    )
    corrected_all.name = "corrected_t2m_c"
    corrected_all.attrs["long_name"] = "Bias-corrected GEFS ensemble-mean 2-m temperature"
    corrected_all.attrs["units"] = "degC"
    return corrected_all
