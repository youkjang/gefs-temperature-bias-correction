from __future__ import annotations

from datetime import datetime, timedelta


def parse_init_time(init_date: str, init_hour: str | int) -> datetime:
    """Return a Python datetime from YYYYMMDD and cycle hour."""
    return datetime.strptime(f"{init_date}{int(init_hour):02d}", "%Y%m%d%H")


def valid_time_from_init(init_date: str, init_hour: str | int, fhr: int) -> datetime:
    """Return forecast valid time from initialization date/hour and lead time."""
    return parse_init_time(init_date, init_hour) + timedelta(hours=int(fhr))


def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def hh(dt: datetime) -> str:
    return dt.strftime("%H")
