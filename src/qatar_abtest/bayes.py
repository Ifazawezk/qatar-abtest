"""Beta-binomial analysis of conversion rates.

With a Beta(a, b) prior and x conversions out of n, the posterior is
Beta(a + x, b + n - x). The probability that B beats A uses the closed form in
Evan Miller (2015), "Formulas for Bayesian A/B testing":

    P(pB > pA) = sum_{i=0}^{aB-1} B(aA + i, bA + bB) / ((bB + i) B(1 + i, bB) B(aA, bA))

which needs an integer aB, true for the default uniform prior. Expected loss and
the lift interval come from posterior draws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import betaln, logsumexp


def prob_b_beats_a(a_a: float, b_a: float, a_b: float, b_b: float) -> float:
    if float(a_b).is_integer():
        i = np.arange(int(a_b))
        terms = betaln(a_a + i, b_a + b_b) - np.log(b_b + i) - betaln(1 + i, b_b) - betaln(a_a, b_a)
        return float(min(1.0, np.exp(logsumexp(terms))))
    rng = np.random.default_rng(0)
    return float((rng.beta(a_b, b_b, 400_000) > rng.beta(a_a, b_a, 400_000)).mean())


@dataclass
class BayesResult:
    posterior_a: tuple[float, float]
    posterior_b: tuple[float, float]
    prob_b_better: float
    loss_choosing_b: float  # expected conversion rate given up by shipping B
    loss_choosing_a: float
    lift_interval: tuple[float, float]  # 95% credible interval for pB / pA - 1

    def to_dict(self) -> dict:
        return asdict(self)


def beta_binomial(x_a: int, n_a: int, x_b: int, n_b: int, prior: tuple[float, float] = (1, 1),
                  draws: int = 200_000, seed: int = 0) -> BayesResult:
    a_a, b_a = prior[0] + x_a, prior[1] + n_a - x_a
    a_b, b_b = prior[0] + x_b, prior[1] + n_b - x_b
    rng = np.random.default_rng(seed)
    p_a, p_b = rng.beta(a_a, b_a, draws), rng.beta(a_b, b_b, draws)
    low, high = np.percentile(p_b / p_a - 1, [2.5, 97.5])
    return BayesResult(
        (a_a, b_a), (a_b, b_b),
        prob_b_beats_a(a_a, b_a, a_b, b_b),
        float(np.maximum(p_a - p_b, 0).mean()),
        float(np.maximum(p_b - p_a, 0).mean()),
        (float(low), float(high)),
    )
