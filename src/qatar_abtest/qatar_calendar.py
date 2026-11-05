"""Calendar checks for a test window in Qatar.

Hijri dates come from the Umm al-Qura calendar through hijridate. Qatar announces
Ramadan and the Eids from moon sightings, so the real dates can differ by a day
from the calculated ones, and the official Eid holiday is often longer.

- Ramadan: Hijri month 9
- Eid al-Fitr: 1 to 3 Shawwal; Eid al-Adha: 9 (Arafat Day) to 12 Dhu al-Hijjah
- National Day: 18 December; National Sport Day: second Tuesday of February
- Weekend: Friday and Saturday
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from hijridate import Gregorian

NAMES = {
    "ramadan": "Ramadan",
    "eid_al_fitr": "Eid al-Fitr",
    "eid_al_adha": "Eid al-Adha",
    "national_day": "National Day",
    "sport_day": "National Sport Day",
}
SHARE_GAP = 0.10  # warn when two periods differ by this share of special days


@lru_cache(maxsize=None)
def day_label(day: dt.date) -> str:
    if day.month == 12 and day.day == 18:
        return "national_day"
    if day.month == 2 and day.weekday() == 1 and 8 <= day.day <= 14:
        return "sport_day"
    try:
        h = Gregorian(day.year, day.month, day.day).to_hijri()
    except (OverflowError, ValueError):
        return ""
    if h.month == 9:
        return "ramadan"
    if h.month == 10 and h.day <= 3:
        return "eid_al_fitr"
    if h.month == 12 and 9 <= h.day <= 12:
        return "eid_al_adha"
    return ""


def is_weekend(day: dt.date) -> bool:
    return day.weekday() in (4, 5)  # Friday, Saturday


@dataclass
class Period:
    name: str
    start: dt.date
    end: dt.date
    days: int = 0
    weekend_days: int = 0
    ramadan_days: int = 0
    special: dict[str, list[str]] = field(default_factory=dict)  # Eid and holiday dates by name

    @property
    def ramadan_share(self) -> float:
        return self.ramadan_days / self.days

    @property
    def holiday_days(self) -> int:
        return sum(len(v) for v in self.special.values())

    def to_dict(self) -> dict:
        out = asdict(self)
        out["start"], out["end"] = self.start.isoformat(), self.end.isoformat()
        return out


def describe(name: str, start: dt.date, end: dt.date) -> Period:
    if end < start:
        raise ValueError(f"{name} ends before it starts")
    p = Period(name, start, end)
    day = start
    while day <= end:
        p.days += 1
        p.weekend_days += is_weekend(day)
        label = day_label(day)
        if label == "ramadan":
            p.ramadan_days += 1
        elif label:
            p.special.setdefault(NAMES[label], []).append(day.isoformat())
        day += dt.timedelta(days=1)
    return p


def span(dates: list[str]) -> str:
    return dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"


def warnings(test: Period, pre: Period | None = None) -> list[str]:
    out = []
    if test.days % 7:
        out.append(
            f"{test.name} is {test.days} days, not whole weeks: {test.weekend_days} of them are Fridays or "
            f"Saturdays ({test.weekend_days / test.days:.0%}, against 29% in whole weeks)"
        )
    if 0 < test.ramadan_days < test.days:
        out.append(f"Ramadan covers {test.ramadan_days} of {test.days} days of {test.name} ({test.ramadan_share:.0%}), "
                   "so the result mixes Ramadan and ordinary days")
    elif test.ramadan_days == test.days:
        out.append(f"{test.name} is entirely in Ramadan; the effect may differ the rest of the year")
    for name, dates in test.special.items():
        out.append(f"{name} falls in {test.name} ({span(dates)})")
    if test.days >= 14:
        half = test.start + dt.timedelta(days=test.days // 2 - 1)
        first, second = describe("the first half", test.start, half), describe("the second half", half + dt.timedelta(days=1), test.end)
        out += _compare(first, second)
    if pre is not None:
        out += _compare(pre, test)
    return out


def _compare(a: Period, b: Period) -> list[str]:
    out = []
    if abs(a.ramadan_share - b.ramadan_share) >= SHARE_GAP:
        out.append(f"Ramadan days: {a.ramadan_share:.0%} of {a.name} against {b.ramadan_share:.0%} of {b.name}")
    holidays_a, holidays_b = a.holiday_days / a.days, b.holiday_days / b.days
    if (a.holiday_days == 0) != (b.holiday_days == 0) or abs(holidays_a - holidays_b) >= SHARE_GAP:
        out.append(f"Eid and holiday days: {a.holiday_days} in {a.name} against {b.holiday_days} in {b.name}")
    return out
