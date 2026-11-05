import json

import numpy as np
import pandas as pd
import pytest

from qatar_abtest.cli import main


@pytest.fixture
def users(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    variant = rng.choice(["control", "free_100"], n)
    pre = rng.gamma(2, 60, n)
    converted = rng.random(n) < np.where(variant == "free_100", 0.07, 0.05)
    revenue = np.where(converted, 0.5 * pre + rng.gamma(4, 40, n), 0.0).round(2)
    days = pd.Timestamp("2026-04-05") + pd.to_timedelta(rng.integers(0, 21, n), unit="D")
    df = pd.DataFrame({"variant": variant, "converted": converted.astype(int), "revenue_qar": revenue,
                       "pre_revenue_qar": pre.round(2), "first_seen": days.strftime("%Y-%m-%d")})
    path = tmp_path / "users.csv"
    df.to_csv(path, index=False)
    return path


def test_size_conversion(capsys):
    # statsmodels NormalIndPower gives 31217.4 users per group for 5.0% against 5.5%
    assert main(["size", "--baseline", "0.05", "--lift", "0.1", "--daily-users", "4000"]) == 0
    out = capsys.readouterr().out
    assert "users per group: 31,218 control" in out and "run 3 whole weeks (21 days)" in out


def test_size_mean_with_cuped(capsys):
    assert main(["size", "--sd", "140", "--mde", "8", "--cuped-r", "0.5", "--json", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cuped_control"] / data["control"] == pytest.approx(0.75, abs=0.002)


def test_analyze(users, capsys):
    code = main(["analyze", str(users), "--control", "control", "--converted", "converted", "--revenue", "revenue_qar",
                 "--pre", "pre_revenue_qar", "--date", "first_seen", "--looks", "4", "--look", "2", "--bootstrap", "500"])
    out = capsys.readouterr().out
    assert code == 0
    assert "sample ratio: control" in out and ", ok" in out
    assert "window: 2026-04-05 to 2026-04-25 (21 days)" in out
    assert "conversion" in out and "P(better)" in out and "bootstrap diff" in out and "CUPED diff" in out
    assert "look 2 of 4: O'Brien-Fleming bound |z| >= 2.963" in out
    assert "2 tests, p adjusted with holm" in out


def test_analyze_flags_sample_ratio_mismatch(users, tmp_path, capsys):
    df = pd.read_csv(users)
    skewed = pd.concat([df, df[df.variant == "free_100"].head(300)])
    path = tmp_path / "skewed.csv"
    skewed.to_csv(path, index=False)
    assert main(["analyze", str(path), "--converted", "converted"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_calendar_exit_codes(capsys):
    assert main(["calendar", "2026-04-05", "2026-04-25"]) == 0
    assert main(["calendar", "2026-02-08", "2026-02-28", "--pre-period", "2026-01-11", "2026-01-31"]) == 1
    out = capsys.readouterr().out
    assert "Ramadan covers 11 of 21 days" in out and "National Sport Day 2026-02-10" in out


def test_boundaries(capsys):
    assert main(["boundaries", "--looks", "5"]) == 0
    out = capsys.readouterr().out
    assert "4.877" in out and "2.031" in out and "false positive rate 0.142" in out


def test_compare(tmp_path, capsys):
    a = pd.DataFrame({"id": [f"q{i}" for i in range(30)], "hit": [i % 2 for i in range(30)]})
    b = a.assign(hit=1)
    a.to_csv(tmp_path / "old.csv", index=False)
    b.to_json(tmp_path / "new.jsonl", orient="records", lines=True)
    assert main(["compare", str(tmp_path / "old.csv"), str(tmp_path / "new.jsonl"), "--metric", "hit"]) == 0
    out = capsys.readouterr().out
    assert "diff +0.500" in out and "McNemar: 0 cases only old gets right, 15 only new" in out


def test_errors(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing.csv")]) == 2
    assert main(["size", "--baseline", "0.05"]) == 2
    assert "give --lift" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["calendar", "2026-02-30", "2026-03-01"])
    assert exc.value.code == 2
