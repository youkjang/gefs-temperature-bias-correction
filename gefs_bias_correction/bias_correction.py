from __future__ import annotations

import xarray as xr


def train_test_split_dates(
    dates: list | xr.DataArray,
    train_fraction: float = 0.67,
) -> tuple[list, list]:
    """Split initialization dates into training and testing groups.

    The split preserves chronological order. This is preferable to random splits
    for forecast verification because it mimics applying a correction learned
    from past dates to later independent forecasts.
    """
    values = list(dates.values) if hasattr(dates, "values") else list(dates)
    if len(values) < 3:
        raise ValueError("Use at least 3 dates for a train/test split.")

    n_train = int(round(len(values) * train_fraction))
    n_train = max(1, min(n_train, len(values) - 1))
    return values[:n_train], values[n_train:]


def compute_mean_bias(
    forecast: xr.DataArray,
    analysis: xr.DataArray,
    dim: str | list[str] = "init_date",
) -> xr.DataArray:
    """Estimate additive mean bias: forecast minus analysis."""
    mean_bias = (forecast - analysis).mean(dim=dim)
    mean_bias.name = "mean_bias"
    mean_bias.attrs["description"] = "Mean bias estimated as forecast minus analysis"
    return mean_bias


def apply_mean_bias_correction(
    forecast: xr.DataArray,
    mean_bias: xr.DataArray,
) -> xr.DataArray:
    """Apply additive mean-bias correction to a forecast field."""
    corrected = forecast - mean_bias
    corrected.name = "forecast_corrected"
    corrected.attrs.update(forecast.attrs)
    corrected.attrs["bias_correction"] = "forecast_corrected = forecast - mean_bias"
    return corrected


def percent_improvement(raw_metric: xr.DataArray, corrected_metric: xr.DataArray) -> xr.DataArray:
    """Return percentage improvement from raw to corrected metric.

    Positive values indicate that the corrected metric is smaller than the raw
    metric. This is most useful for error metrics such as RMSE or MAE.
    """
    out = 100.0 * (raw_metric - corrected_metric) / raw_metric
    out.name = "percent_improvement"
    return out
