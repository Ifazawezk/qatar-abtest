import datetime as dt

import pytest

from qatar_abtest.qatar_calendar import day_label, describe, is_weekend, warnings

D = dt.date


@pytest.mark.parametrize(
    "day, label",
    [
        (D(2024, 3, 11), "ramadan"),  # 1 Ramadan 1445, the date announced in Qatar
        (D(2024, 4, 10), "eid_al_fitr"),  # 1 Shawwal 1445
        (D(2025, 3, 1), "ramadan"),  # 1 Ramadan 1446
        (D(2025, 3, 30), "eid_al_fitr"),  # 1 Shawwal 1446
        (D(2026, 2, 18), "ramadan"),  # 1 Ramadan 1447 in Umm al-Qura
        (D(2026, 5, 26), "eid_al_adha"),  # Arafat Day 1447
        (D(2026, 2, 10), "sport_day"),
        (D(2026, 12, 18), "national_day"),
        (D(2026, 7, 1), ""),
    ],
)
def test_labels(day, label):
    assert day_label(day) == label


def test_weekend_is_friday_and_saturday():
    week = [D(2026, 11, 1) + dt.timedelta(days=i) for i in range(7)]  # Sunday to Saturday
    assert [is_weekend(d) for d in week] == [False] * 5 + [True, True]


def test_describe_counts():
    p = describe("test", D(2026, 2, 8), D(2026, 2, 28))
    assert (p.days, p.weekend_days, p.ramadan_days) == (21, 6, 11)
    assert p.special == {"National Sport Day": ["2026-02-10"]}


def test_clean_window_has_no_warnings():
    assert warnings(describe("the test", D(2026, 4, 5), D(2026, 4, 25))) == []


def test_ramadan_partial_window_and_pre_period():
    test = describe("the test", D(2026, 2, 8), D(2026, 2, 28))
    pre = describe("the pre-period", D(2026, 1, 11), D(2026, 1, 31))
    found = warnings(test, pre)
    assert found[0] == "Ramadan covers 11 of 21 days of the test (52%), so the result mixes Ramadan and ordinary days"
    assert "National Sport Day falls in the test (2026-02-10)" in found
    assert "Ramadan days: 0% of the first half against 100% of the second half" in found
    assert "Ramadan days: 0% of the pre-period against 52% of the test" in found


def test_partial_weeks_and_eid():
    found = warnings(describe("the test", D(2026, 3, 15), D(2026, 3, 24)))
    assert found[0].startswith("the test is 10 days, not whole weeks: 2 of them are Fridays or Saturdays")
    assert "Eid al-Fitr falls in the test (2026-03-20 to 2026-03-22)" in found


def test_bad_window():
    with pytest.raises(ValueError):
        describe("the test", D(2026, 3, 2), D(2026, 3, 1))
