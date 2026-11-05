import math

import pytest
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from qatar_abtest.power import (
    cohens_h,
    cuped_sample_size,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)


def test_cohens_h_matches_statsmodels():
    assert cohens_h(0.05, 0.06) == pytest.approx(proportion_effectsize(0.06, 0.05))


@pytest.mark.parametrize("p1, p2, ratio", [(0.032, 0.0352, 1.0), (0.10, 0.12, 1.0), (0.2, 0.25, 2.0)])
def test_proportions_match_statsmodels(p1, p2, ratio):
    h = proportion_effectsize(p2, p1)
    expected = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=ratio)
    assert sample_size_proportions(p1, p2, ratio=ratio).n1 == pytest.approx(expected, rel=1e-5)
    got = power_proportions(p1, p2, 5000, ratio=ratio)
    assert got == pytest.approx(NormalIndPower().power(h, 5000, 0.05, ratio=ratio), rel=1e-9)


def test_proportions_cohen_table():
    # Cohen (1988), Table 6.4.1: h = 0.20, power 0.80, two-sided alpha 0.05 gives n = 392 per group
    p2 = math.sin(0.1 + math.asin(math.sqrt(0.5))) ** 2  # p1 = 0.5 and h = 0.2
    assert cohens_h(0.5, p2) == pytest.approx(0.2)
    assert round(sample_size_proportions(0.5, p2).n1) == 392


def test_one_sided_proportions():
    h = proportion_effectsize(0.12, 0.10)
    expected = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, alternative="larger")
    assert sample_size_proportions(0.10, 0.12, alternative="larger").n1 == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("d, ratio", [(0.5, 1.0), (0.2, 1.0), (0.1, 1.5)])
def test_means_match_statsmodels(d, ratio):
    expected = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8, ratio=ratio)
    assert sample_size_means(d * 140, 140, ratio=ratio).n1 == pytest.approx(expected, rel=1e-5)
    assert power_means(d * 140, 140, 300, ratio=ratio) == pytest.approx(TTestIndPower().power(d, 300, 0.05, ratio=ratio))


def test_means_cohen_table():
    # Cohen (1988), Table 2.4.1: d = 0.50, power 0.80, two-sided alpha 0.05 gives n = 64 per group
    size = sample_size_means(0.5, 1.0)
    assert size.control == 64 and size.total == 128


def test_cuped_shrinks_by_one_minus_rho_squared():
    plain, adjusted = cuped_sample_size(8, 140, rho=0.6)
    assert adjusted.n1 / plain.n1 == pytest.approx(1 - 0.36, rel=1e-3)


def test_bad_inputs():
    with pytest.raises(ValueError):
        sample_size_proportions(0.1, 0.1)
    with pytest.raises(ValueError):
        sample_size_means(5, 10, power=1.2)
    with pytest.raises(ValueError):
        power_means(5, 10, 100, alternative="bigger")
