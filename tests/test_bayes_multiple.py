import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from qatar_abtest.bayes import beta_binomial, prob_b_beats_a
from qatar_abtest.multiple import adjust, benjamini_hochberg, holm


def test_prob_b_beats_a_known_cases():
    assert prob_b_beats_a(1, 1, 1, 1) == pytest.approx(0.5)
    # B ~ Beta(2, 1) has density 2p; against a uniform A, P(B > A) = integral of 2p * p = 2/3
    assert prob_b_beats_a(1, 1, 2, 1) == pytest.approx(2 / 3)
    assert prob_b_beats_a(5, 20, 5, 20) == pytest.approx(0.5)


def test_prob_b_beats_a_matches_simulation():
    rng = np.random.default_rng(4)
    a, b = (1 + 312, 1 + 10000 - 312), (1 + 361, 1 + 10050 - 361)
    simulated = (rng.beta(*b, 1_000_000) > rng.beta(*a, 1_000_000)).mean()
    assert prob_b_beats_a(*a, *b) == pytest.approx(simulated, abs=0.002)


def test_expected_loss_for_two_uniforms():
    # for two independent uniforms E[max(A - B, 0)] = E|A - B| / 2 = 1/6
    result = beta_binomial(0, 0, 0, 0, draws=1_000_000)
    assert result.loss_choosing_b == pytest.approx(1 / 6, abs=0.001)
    assert result.loss_choosing_a == pytest.approx(1 / 6, abs=0.001)


def test_beta_binomial_result():
    result = beta_binomial(312, 10000, 361, 10050)
    assert result.posterior_b == (362, 9690)
    assert 0.95 < result.prob_b_better < 0.99
    assert result.loss_choosing_b < result.loss_choosing_a
    assert result.lift_interval[0] < 0.157 < result.lift_interval[1]


# Benjamini and Hochberg (1995), section 4.1: 15 p-values from a study of myocardial infarction.
# At 0.05 their procedure rejects the four smallest, and Bonferroni style methods three.
BH_1995 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
           0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]


def test_published_example():
    assert (benjamini_hochberg(BH_1995) <= 0.05).sum() == 4
    assert (holm(BH_1995) <= 0.05).sum() == 3


@pytest.mark.parametrize("method, name", [("holm", "holm"), ("bh", "fdr_bh")])
def test_match_statsmodels(method, name):
    rng = np.random.default_rng(9)
    p = np.concatenate([rng.uniform(0, 0.02, 4), rng.uniform(0, 1, 12)])
    assert adjust(p, method) == pytest.approx(multipletests(p, method=name)[1])
    assert adjust(BH_1995, method) == pytest.approx(multipletests(BH_1995, method=name)[1])


def test_unknown_method():
    with pytest.raises(ValueError):
        adjust([0.1], "bonferroni")
