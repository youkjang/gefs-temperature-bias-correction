from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class Region:
    """Geographic domain in degrees east/north, using -180 to 180 longitude."""
    name: str = "CONUS"
    west: float = -130.0
    east: float = -60.0
    south: float = 20.0
    north: float = 55.0


@dataclass
class ProjectConfig:
    """Configuration for GEFS/GFS 2-m temperature bias correction."""
    init_dates: Sequence[str] = field(
        default_factory=lambda: pd.date_range(
            "2024-01-01", "2024-01-10", freq="D"
        ).strftime("%Y%m%d").tolist()
    )
    init_hour: str = "00"
    forecast_hours: Sequence[int] = field(default_factory=lambda: [24, 48])
    train_fraction: float = 0.7
    region: Region = field(default_factory=Region)
    cache_dir: str | Path = "data/herbie_cache"
    verbose: bool = True
    skip_missing: bool = True

    def __post_init__(self):
        self.init_dates = list(self.init_dates)
        self.forecast_hours = [int(x) for x in self.forecast_hours]
        self.cache_dir = Path(self.cache_dir)
