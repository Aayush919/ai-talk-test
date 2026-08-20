"""Local India clock for the live coach. No network, no extra LLM call."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

LOCAL_TZ = timezone(timedelta(hours=5, minutes=30), name="IST")


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def local_now(*, now: datetime | None = None) -> datetime:
    stamp = now or datetime.now(LOCAL_TZ)
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=LOCAL_TZ)
    return stamp.astimezone(LOCAL_TZ)


def local_now_line(*, now: datetime | None = None) -> str:
    stamp = local_now(now=now)
    hour12 = stamp.strftime("%I").lstrip("0") or "0"
    pretty = (
        f"{stamp.strftime('%A')}, {stamp.day} {stamp.strftime('%B %Y')}, "
        f"{hour12}:{stamp.strftime('%M %p')}"
    )
    part = _part_of_day(stamp.hour)
    return (
        f"Local now (India): {pretty} ({part}). "
        "Use this for day, date, year, and time. Do not ask what day or time it is."
    )
