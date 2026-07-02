"""Quantile mapping utilities for GEFS 2-m temperature bias correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


_REQUIRED_DATA_VARS = ["forecast_t2m_c", "analysis_t2m_c", "fhr"]


def _check_matched_dataset(ds_input: xr.Dataset) -> None:
    """Validate that a matched forecast-analysis dataset has required content."""
    if "case" not in ds_input.dims:
        raise ValueError("Dataset must have a 'case' dimension.")
    for name in _REQUIRED_DATA_VARS:
        if name not in ds_input:
            raise ValueError(f"Dataset must contain '{name}'.")


def _finite_values(da: xr.DataArray) -> np.ndarray:
    """Return finite values from a DataArray as a 1-D NumPy array."""
    values = np.asarray(da.values).ravel()
    return values[np.isfinite(values)]


def fit_quantile_mapping_by_lead(
    train_ds: xr.Dataset,
    n_quantiles: int = 101,
) -> xr.Dataset:
    """
    Estimate lead-time-dependent quantile mapping parameters.

    The method pools all grid points and training cases for each forecast lead
    time. For each lead time, it estimates the forecast distribution and the
    verifying analysis distribution using the same quantile levels.

    Parameters
    ----------
    train_ds:
        Matched training dataset with forecast_t2m_c, analysis_t2m_c, fhr,
        and case dimension.
    n_quantiles:
        Number of quantile levels between 0 and 1. Values such as 101 or 201
        are reasonable for an initial notebook experiment.

    Returns
    -------
    xr.Dataset
        Dataset with forecast and analysis quantile values by forecast lead time.
    """
    _check_matched_dataset(train_ds)

    if n_quantiles < 5:
        raise ValueError("n_quantiles should be at least 5.")

    quantile_levels = np.linspace(0.0, 1.0, int(n_quantiles))
    forecast_quantiles = []
    analysis_quantiles = []
    fhrs = []
    n_samples = []

    for fhr in sorted(np.unique(train_ds["fhr"].values)):
        mask = train_ds["fhr"] == fhr
        forecast_values = _finite_values(train_ds["forecast_t2m_c"].where(mask, drop=True))
        analysis_values = _finite_values(train_ds["analysis_t2m_c"].where(mask, drop=True))

        if forecast_values.size == 0 or analysis_values.size == 0:
            continue

        forecast_quantiles.append(np.quantile(forecast_values, quantile_levels))
        analysis_quantiles.append(np.quantile(analysis_values, quantile_levels))
        fhrs.append(int(fhr))
        n_samples.append(int(min(forecast_values.size, analysis_values.size)))

    if not fhrs:
        raise ValueError("No quantile mapping parameters were estimated. Check training data and fhr values.")

    qm = xr.Dataset(
        data_vars={
            "forecast_quantile_value_c": (("fhr", "quantile"), np.asarray(forecast_quantiles)),
            "analysis_quantile_value_c": (("fhr", "quantile"), np.asarray(analysis_quantiles)),
            "n_training_samples": ("fhr", np.asarray(n_samples, dtype=int)),
        },
        coords={
            "fhr": np.asarray(fhrs, dtype=int),
            "quantile": quantile_levels,
        },
        attrs={
            "method": "lead-time-dependent pooled quantile mapping",
            "description": (
                "Quantile mapping parameters estimated by pooling all training cases "
                "and grid points separately for each forecast lead time."
            ),
            "units": "degC",
        },
    )
    return qm


def apply_quantile_mapping_correction(
    ds_input: xr.Dataset,
    qm_params: xr.Dataset,
) -> xr.DataArray:
    """
    Apply lead-time-dependent quantile mapping to GEFS forecast cases.

    For each case, the raw forecast value is mapped from the training forecast
    distribution to the corresponding value in the training analysis distribution
    for that forecast lead time.

    Values outside the fitted forecast quantile range are clipped to the endpoint
    analysis quantile values by NumPy interpolation. This conservative behavior
    avoids unstable extrapolation in the initial notebook version.
    """
    _check_matched_dataset(ds_input)

    required_qm = ["forecast_quantile_value_c", "analysis_quantile_value_c"]
    for name in required_qm:
        if name not in qm_params:
            raise ValueError(f"qm_params must contain '{name}'.")

    available_fhrs = set(int(x) for x in qm_params["fhr"].values)
    needed_fhrs = set(int(x) for x in np.unique(ds_input["fhr"].values))
    missing = sorted(needed_fhrs - available_fhrs)
    if missing:
        raise ValueError(f"qm_params is missing forecast hours: {missing}")

    corrected_cases = []

    for case_id in ds_input["case"].values:
        fhr = int(ds_input["fhr"].sel(case=case_id).item())
        raw = ds_input["forecast_t2m_c"].sel(case=case_id)

        x_forecast = np.asarray(qm_params["forecast_quantile_value_c"].sel(fhr=fhr).values, dtype=float)
        y_analysis = np.asarray(qm_params["analysis_quantile_value_c"].sel(fhr=fhr).values, dtype=float)

        # np.interp requires monotonically increasing x-values. Quantile values
        # can occasionally repeat, so unique x-values are used for stability.
        x_unique, unique_index = np.unique(x_forecast, return_index=True)
        y_unique = y_analysis[unique_index]

        if x_unique.size < 2:
            raise ValueError(f"Not enough unique forecast quantile values for fhr={fhr}.")

        mapped_values = np.interp(raw.values, x_unique, y_unique)

        corrected = xr.DataArray(
            mapped_values,
            dims=raw.dims,
            coords=raw.coords,
            name="qm_corrected_t2m_c",
            attrs={
                "long_name": "Quantile-mapped GEFS ensemble-mean 2-m temperature",
                "units": "degC",
                "method": "lead-time-dependent pooled quantile mapping",
            },
        )
        corrected_cases.append(corrected.expand_dims(case=[case_id]))

    if not corrected_cases:
        raise ValueError("No quantile-mapped cases were created.")

    corrected_all = xr.concat(
        corrected_cases,
        dim="case",
        compat="override",
        coords="minimal",
        combine_attrs="override",
    )

    corrected_all = corrected_all.assign_coords(
        init_date=("case", pd.to_datetime(ds_input["init_date"].values)),
        fhr=("case", ds_input["fhr"].values),
        valid_time=("case", pd.to_datetime(ds_input["valid_time"].values)),
    )
    corrected_all.name = "qm_corrected_t2m_c"
    corrected_all.attrs["long_name"] = "Quantile-mapped GEFS ensemble-mean 2-m temperature"
    corrected_all.attrs["units"] = "degC"
    corrected_all.attrs["method"] = "lead-time-dependent pooled quantile mapping"
    return corrected_all


def make_quantile_mapping_summary_table(
    test_ds: xr.Dataset,
    mean_bias_corrected: xr.DataArray,
    quantile_mapped: xr.DataArray,
    summarize_func,
) -> pd.DataFrame:
    """
    Create a raw vs mean-bias vs quantile-mapping summary table.

    `summarize_func` should be a function with the same signature as
    gefs_bias_correction.verification.summarize_by_lead.
    """
    raw = summarize_func(test_ds["forecast_t2m_c"], test_ds["analysis_t2m_c"], test_ds["fhr"], label="raw")
    mean_bias = summarize_func(mean_bias_corrected, test_ds["analysis_t2m_c"], test_ds["fhr"], label="mean_bias")
    qm = summarize_func(quantile_mapped, test_ds["analysis_t2m_c"], test_ds["fhr"], label="quantile_mapping")

    long = pd.concat([raw, mean_bias, qm], ignore_index=True)
    return long.sort_values(["fhr", "method"]).reset_index(drop=True)
