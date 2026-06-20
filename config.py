from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    """Configuration for a GEFS 2-m temperature bias-correction experiment.

    The default configuration is intentionally small so that the first run is
    practical in Google Colab. Increase the number of dates and lead times after
    the workflow succeeds.
    """

    init_dates: list[str] = field(
        default_factory=lambda: [
            "20260601",
            "20260602",
            "20260603",
            "20260604",
            "20260605",
            "20260606",
        ]
    )
    init_hour: str = "00"
    forecast_hours: list[int] = field(default_factory=lambda: [24, 48, 72])
    cache_dir: Path | str = Path("data/cache")
    region_name: str = "CONUS"
    temperature_units: str = "degC"
    use_gefs_ensemble_mean_product: bool = True
    prefer_herbie: bool = True

    # CONUS box in degrees east. Equivalent to roughly 125W-65W, 25N-50N.
    lon_min: float = 235.0
    lon_max: float = 295.0
    lat_min: float = 25.0
    lat_max: float = 50.0

    def __post_init__(self) -> None:
        self.init_hour = f"{int(self.init_hour):02d}"
        self.cache_dir = Path(self.cache_dir)
        self.forecast_hours = [int(fhr) for fhr in self.forecast_hours]
