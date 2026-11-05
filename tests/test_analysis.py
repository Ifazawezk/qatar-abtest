import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from qatar_abtest.analysis import bootstrap_means, cuped, sample_ratio, two_proportions, welch, welch_samples

# conversions out of visitors, control then treatment
CASES = [(312, 10000, 361, 10050), (48, 1500, 70, 1480), (1210, 40000, 1199, 39800)]


@pytest.mark.parametrize("x_a, n_a, x_b, n_b", CASES)
def test_z_test_matches_statsmodels(x_a, n_a, x_b, n_b):
    est = two_proportions(x_a, n_a, x_b, n_b)
    z, p = proportions_ztest([x_b, x_a], [n_b, n_a])
    assert est.statistic == pytest.approx(z) and est.p_value == pytest.approx(p)


@pytest.mark.parametrize("x_a, n_a, x_b, n_b", CASES)
def test_intervals_match_statsmodels(x_a, n_a, x_b, n_b):
    est = two_proportions(x_a, n_a, x_b, n_b)
    low, high = confint_proportions_2indep(x_b, n_b, x_a, n_a, method="wald", compare="diff")
    assert est.diff_ci == pytest.approx((low, high))
    low, high = confint_proportions_2indep(x_b, n_b, x_a, n_a, method="log", compare="ratio")
    assert est.lift_ci == pytest.approx((low - 1, high - 1))


def test_welch_matches_scipy_and_statsmodels():
    rng = np.random.default_rng(3)
    a, b = rng.lognormal(5, 1.0, 400), rng.lognormal(5.1, 1.2, 380)
    est = welch_samples(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert est.statistic == pytest.approx(ref.statistic) and est.p_value == pytest.approx(ref.pvalue)
    low, high = CompareMeans(DescrStatsW(b), DescrStatsW(a)).tconfint_diff(usevar="unequal")
    assert est.diff_ci == pytest.approx((low, high))


def test_welch_from_summary_matches_scipy():
    est = welch(185.0, 140.0, 4000, 192.0, 150.0, 4050)
    ref = stats.ttest_ind_from_stats(192.0, 150.0, 4050, 185.0, 140.0, 4000, equal_var=False)
    assert est.p_value == pytest.approx(ref.pvalue)


def test_lift_interval_for_means_reduces_to_katz():
    # a 0/1 metric with the population variance p(1 - p) gives the Katz ratio interval
    x_a, n_a, x_b, n_b = CASES[0]
    p_a, p_b = x_a / n_a, x_b / n_b
    est = welch(p_a, (p_a * (1 - p_a)) ** 0.5, n_a, p_b, (p_b * (1 - p_b)) ** 0.5, n_b)
    assert est.lift_ci == pytest.approx(two_proportions(x_a, n_a, x_b, n_b).lift_ci)


def test_bootstrap_agrees_with_scipy():
    rng = np.random.default_rng(11)
    a, b = rng.lognormal(5, 1.1, 500), rng.lognormal(5.1, 1.1, 500)
    mine = bootstrap_means(a, b, rounds=4000, seed=1)
    ref = stats.bootstrap((a, b), lambda x, y, axis=-1: y.mean(axis) - x.mean(axis), vectorized=True,
                          paired=False, n_resamples=4000, method="percentile", rng=np.random.default_rng(2))
    width = mine.diff_ci[1] - mine.diff_ci[0]
    assert mine.diff_ci == pytest.approx(tuple(ref.confidence_interval), abs=0.05 * width)


def test_bootstrap_close_to_welch_for_normal_data():
    rng = np.random.default_rng(5)
    a, b = rng.normal(200, 40, 2000), rng.normal(204, 40, 2000)
    boot, w = bootstrap_means(a, b, rounds=3000), welch_samples(a, b)
    assert boot.diff_ci == pytest.approx(w.diff_ci, abs=0.3)


def test_cuped_reduces_variance_by_rho_squared():
    rng = np.random.default_rng(8)
    n, rho = 20000, 0.7
    x = rng.normal(0, 1, 2 * n)
    y = 100 + 20 * (rho * x + (1 - rho**2) ** 0.5 * rng.normal(0, 1, 2 * n))
    result = cuped(y[:n], x[:n], y[n:], x[n:])
    assert result.theta == pytest.approx(np.polyfit(x, y, 1)[0])  # theta is the OLS slope
    assert result.variance_reduction == pytest.approx(rho**2, abs=0.01)
    assert result.adjusted_a.mean() - result.adjusted_b.mean() == pytest.approx(y[:n].mean() - y[n:].mean(), abs=0.3)


def test_sample_ratio_by_hand():
    # (50300 - 50000)^2 / 50300 * 2 = 3.578, chi-square with 1 df gives p = 0.0586
    srm = sample_ratio({"control": 50000, "treatment": 50600})
    assert srm.chi2 == pytest.approx(3.5785, abs=1e-4) and srm.p_value == pytest.approx(0.05857, abs=1e-4)
    assert not srm.mismatch
    assert sample_ratio({"a": 50000, "b": 51500}).mismatch
    uneven = sample_ratio({"a": 2000, "b": 1000, "c": 1000}, {"a": 50, "b": 25, "c": 25})
    assert uneven.chi2 == 0 and uneven.expected_share["a"] == 0.5
