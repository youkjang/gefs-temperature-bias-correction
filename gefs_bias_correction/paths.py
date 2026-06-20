from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .time_utils import yyyymmdd, hh

GEFS_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def gefs_ensemble_mean_url(init_date: str, init_hour: str | int, fhr: int) -> str:
    """Construct the GEFS 0.25-degree ensemble-mean GRIB2 URL.

    Product used:
        atmos/pgrb2sp25/geavg.tHHz.pgrb2s.0p25.fFFF
    """
    cycle = f"{int(init_hour):02d}"
    fff = f"{int(fhr):03d}"
    return (
        f"{GEFS_BASE_URL}/gefs.{init_date}/{cycle}/atmos/pgrb2sp25/"
        f"geavg.t{cycle}z.pgrb2s.0p25.f{fff}"
    )


def gfs_analysis_url(valid_time: datetime) -> str:
    """Construct the GFS 0.25-degree f000 analysis/initial-condition GRIB2 URL.

    The f000 file at the GEFS valid time is used as a practical verifying
    analysis for this first project version.
    """
    date = yyyymmdd(valid_time)
    cycle = hh(valid_time)
    return (
        f"{GFS_BASE_URL}/gfs.{date}/{cycle}/atmos/"
        f"gfs.t{cycle}z.pgrb2.0p25.f000"
    )


def cached_path_for_url(url: str, cache_dir: str | Path, subdir: str | None = None) -> Path:
    """Create a deterministic local cache path for a remote URL."""
    cache_dir = Path(cache_dir)
    filename = url.rstrip("/").split("/")[-1]
    if subdir:
        return cache_dir / subdir / filename
    return cache_dir / filename
