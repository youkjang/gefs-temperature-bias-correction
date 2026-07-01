import numpy as np
import pandas as pd
import xarray as xr


def split_train_test_by_init_date(ds_input: xr.Dataset, train_fraction: float = 0.7):
    """Split matched dataset by unique initialization dates."""
    if "case" not in ds_input.dims:
        raise ValueError("Dataset must have a 'case' dimension.")
    if "init_date" not in ds_input.coords and "init_date" not in ds_input:
        raise ValueError("Dataset must contain 'init_date'.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    init_dates = pd.Index(pd.to_datetime(ds_input["init_date"].values)).unique().sort_values()
    n_dates = len(init_dates)
    if n_dates < 2:
        raise ValueError("Need at least two unique init_dates for train/test split.")

    n_train = int(np.floor(n_dates * train_fraction))
    n_train = max(1, min(n_train, n_dates - 1))

    train_dates = init_dates[:n_train]
    test_dates = init_dates[n_train:]

    init_values = pd.to_datetime(ds_input["init_date"].values).normalize()
    train_mask = np.isin(init_values, train_dates.normalize())
    test_mask = np.isin(init_values, test_dates.normalize())

    train_ds = ds_input.isel(case=train_mask)
    test_ds = ds_input.isel(case=test_mask)

    return (
        train_ds,
        test_ds,
        [d.strftime("%Y%m%d") for d in train_dates],
        [d.strftime("%Y%m%d") for d in test_dates],
    )
