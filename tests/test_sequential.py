import numpy as np
import pytest

from qatar_abtest.sequential import crossing_probability, naive_peeking_alpha, obf_boundaries


def test_five_looks_match_published_bounds():
    # Lan-DeMets O'Brien-Fleming spending, five equal looks, 0.025 per side, as reported
    # by R's gsDesign (sfLDOF) and ldbounds: 4.877, 3.357, 2.680, 2.290, 2.031
    b = obf_boundaries(5)
    assert b.z == pytest.approx([4.877, 3.357, 2.680, 2.290, 2.031], abs=1e-3)
    assert b.spent[-1] == pytest.approx(0.05)


def test_two_looks():
    assert obf_boundaries([0.5, 1.0]).z == pytest.approx([2.963, 1.969], abs=1e-3)


@pytest.mark.parametrize("looks, published", [(1, 0.05), (2, 0.083), (3, 0.107), (5, 0.142), (10, 0.193), (20, 0.248)])
def test_naive_peeking_matches_armitage(looks, published):
    # Jennison and Turnbull (2000), Table 1.1, after Armitage, McPherson and Rowe (1969)
    assert naive_peeking_alpha(looks) == pytest.approx(published, abs=6e-4)


def test_boundaries_spend_exactly_alpha():
    b = obf_boundaries(4, alpha=0.05)
    assert crossing_probability(b.z) == pytest.approx(0.05, abs=1e-6)


def test_bad_fractions():
    with pytest.raises(ValueError):
        obf_boundaries([0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        obf_boundaries([0.5, 0.9])


def test_simulated_false_positive_rate():
    """A/A tests on a 5% conversion rate, looked at five times. Stopping at the
    first naive p < 0.05 gives about 14% false positives; the boundaries keep it near 5%."""
    rng = np.random.default_rng(2026)
    sims, looks, per_look, p = 20000, 5, 2000, 0.05
    conversions = rng.binomial(per_look, p, size=(sims, looks, 2)).cumsum(axis=1)
    n = per_look * np.arange(1, looks + 1)[None, :]
    rate_a, rate_b = conversions[..., 0] / n, conversions[..., 1] / n
    pooled = (rate_a + rate_b) / 2
    z = (rate_b - rate_a) / np.sqrt(pooled * (1 - pooled) * 2 / n)
    naive = (np.abs(z) >= 1.96).any(axis=1).mean()
    sequential = (np.abs(z) >= obf_boundaries(looks).z).any(axis=1).mean()
    assert naive == pytest.approx(0.142, abs=0.01)
    assert sequential == pytest.approx(0.05, abs=0.006)
