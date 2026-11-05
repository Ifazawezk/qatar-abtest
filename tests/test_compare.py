import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from qatar_abtest.compare import compare_systems, mcnemar, paired_bootstrap, paired_permutation


def pairs(table):
    """Per-item correctness from a 2x2 table [[both right, only A], [only B, both wrong]]."""
    (both, only_a), (only_b, neither) = table
    a = [1] * both + [1] * only_a + [0] * only_b + [0] * neither
    b = [1] * both + [0] * only_a + [1] * only_b + [0] * neither
    return a, b


def test_mcnemar_agresti_example():
    # Agresti (2002), Table 10.1: approval of the Prime Minister in two surveys of 1600 people.
    # 150 changed from approve to disapprove and 86 the other way; z = -4.17, z^2 = 17.36
    a, b = pairs([[794, 150], [86, 570]])
    result = mcnemar(a, b, exact=False, correction=False)
    assert (result.only_a, result.only_b) == (150, 86)
    assert result.statistic == pytest.approx(17.36, abs=0.01)


@pytest.mark.parametrize("table", [[[794, 150], [86, 570]], [[50, 9], [3, 38]], [[20, 2], [2, 10]]])
@pytest.mark.parametrize("exact", [True, False])
def test_mcnemar_matches_statsmodels(table, exact):
    ours = mcnemar(*pairs(table), exact=exact)
    ref = sm_mcnemar(np.array(table), exact=exact, correction=True)
    assert ours.p_value == pytest.approx(ref.pvalue) and ours.statistic == pytest.approx(ref.statistic)


def test_permutation_exact_known_value():
    # five positive differences: only all-plus and all-minus are as extreme, p = 2 / 32
    p, exact = paired_permutation([0] * 5, [1, 2, 3, 4, 5])
    assert exact and p == pytest.approx(0.0625)


def test_permutation_matches_scipy_exact():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 1, 12), rng.uniform(0.1, 1.1, 12)
    ref = stats.permutation_test((a, b), lambda x, y: np.mean(y - x), permutation_type="samples",
                                 n_resamples=np.inf, alternative="two-sided")
    p, exact = paired_permutation(a, b)
    assert exact and p == pytest.approx(ref.pvalue)


def test_permutation_monte_carlo_close_to_exact_scipy():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(0, 1, 18), rng.uniform(0.15, 1.15, 18)
    ref = stats.permutation_test((a, b), lambda x, y: np.mean(y - x), permutation_type="samples",
                                 n_resamples=np.inf, alternative="two-sided")
    p, exact = paired_permutation(a, b, rounds=40000)
    assert not exact and p == pytest.approx(ref.pvalue, abs=0.005)


def test_paired_bootstrap_matches_scipy():
    rng = np.random.default_rng(3)
    a, b = rng.binomial(1, 0.6, 60).astype(float), rng.binomial(1, 0.75, 60).astype(float)
    mean, low, high = paired_bootstrap(a, b)
    ref = stats.bootstrap((b - a,), np.mean, n_resamples=10000, method="percentile", rng=np.random.default_rng(9))
    assert mean == pytest.approx((b - a).mean())
    assert (low, high) == pytest.approx(tuple(ref.confidence_interval), abs=0.02)
    assert paired_bootstrap(np.zeros(5), np.full(5, 0.5)) == (0.5, 0.5, 0.5)


def test_compare_systems_with_holm():
    base = {f"q{i}": float(i % 2) for i in range(40)}
    better = {f"q{i}": 1.0 for i in range(40)}
    same = dict(base)
    rows = compare_systems(("base", base), [("better", better), ("same", same)], "hit@1")
    assert rows[0].diff == pytest.approx(0.5) and rows[0].mcnemar.only_b == 20
    assert rows[1].p_value == 1.0 and rows[1].ci == (0.0, 0.0)
    assert rows[0].p_adjusted == pytest.approx(min(1, 2 * rows[0].p_value))
    with pytest.raises(ValueError):
        compare_systems(("base", base), [("other", {"x": 1.0})], "hit@1")
