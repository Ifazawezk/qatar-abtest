"""Sample size and power for conversion rates and for means.

Conversion rates use Cohen's effect size h = 2 asin(sqrt(p2)) - 2 asin(sqrt(p1))
with a normal test, as in Cohen (1988), "Statistical Power Analysis for the
Behavioral Sciences", chapter 6, and statsmodels' NormalIndPower. Means use the
two-sample t test with the noncentral t distribution, as in Cohen's chapter 2 and
statsmodels' TTestIndPower.

`ratio` is the treatment group size divided by the control group size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

ALTERNATIVES = ("two-sided", "larger", "smaller")


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def _check(alpha: float, alternative: str) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {', '.join(ALTERNATIVES)}")


def power_proportions(p1: float, p2: float, n1: float, alpha: float = 0.05, ratio: float = 1.0,
                      alternative: str = "two-sided") -> float:
    _check(alpha, alternative)
    h = cohens_h(p1, p2)
    if alternative == "smaller":
        h = -h
    nc = h * math.sqrt(n1 / (1 + 1 / ratio))
    if alternative == "two-sided":
        crit = stats.norm.isf(alpha / 2)
        return float(stats.norm.sf(crit - nc) + stats.norm.cdf(-crit - nc))
    return float(stats.norm.sf(stats.norm.isf(alpha) - nc))


def power_means(delta: float, sd: float, n1: float, alpha: float = 0.05, ratio: float = 1.0,
                alternative: str = "two-sided") -> float:
    _check(alpha, alternative)
    n2 = n1 * ratio
    df = n1 + n2 - 2
    nc = (delta / sd) * math.sqrt(1 / (1 / n1 + 1 / n2))
    if alternative == "smaller":
        nc = -nc
    if alternative == "two-sided":
        crit = stats.t.isf(alpha / 2, df)
        return float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))
    return float(stats.nct.sf(stats.t.isf(alpha, df), df, nc))


def _solve(power_at, target: float) -> float:
    """Smallest n with power_at(n) = target. Power grows with n, so double an
    upper bound until it is reached, then bisect."""
    if not 0 < target < 1:
        raise ValueError("power must be between 0 and 1")
    high = 4.0
    while power_at(high) < target:
        high *= 2
        if high > 1e10:
            raise ValueError("the effect is too small to reach that power with any sample size")
    return float(optimize.brentq(lambda n: power_at(n) - target, max(2.0, high / 2), high, xtol=1e-6))


@dataclass
class SampleSize:
    n1: float  # exact solution, control group
    ratio: float

    @property
    def control(self) -> int:
        return math.ceil(self.n1 - 1e-9)

    @property
    def treatment(self) -> int:
        return math.ceil(self.n1 * self.ratio - 1e-9)

    @property
    def total(self) -> int:
        return self.control + self.treatment


def sample_size_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0,
                            alternative: str = "two-sided") -> SampleSize:
    """Users per group to detect a change from conversion rate p1 to p2."""
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("p1 and p2 must be different rates between 0 and 1")
    n1 = _solve(lambda n: power_proportions(p1, p2, n, alpha, ratio, alternative), power)
    return SampleSize(n1, ratio)


def sample_size_means(delta: float, sd: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0,
                      alternative: str = "two-sided") -> SampleSize:
    """Users per group to detect a difference `delta` in a mean with standard deviation `sd`."""
    if delta == 0 or sd <= 0:
        raise ValueError("delta must be non-zero and sd positive")
    n1 = _solve(lambda n: power_means(delta, sd, n, alpha, ratio, alternative), power)
    return SampleSize(n1, ratio)


def cuped_sample_size(delta: float, sd: float, rho: float, **kwargs) -> tuple[SampleSize, SampleSize]:
    """Sample size without and with CUPED. Adjusting for a pre-period metric with
    correlation rho multiplies the variance by 1 - rho^2 (Deng, Xu, Kohavi and
    Walker 2013, "Improving the sensitivity of online controlled experiments by
    utilizing pre-experiment data", WSDM)."""
    if not -1 < rho < 1:
        raise ValueError("rho must be between -1 and 1")
    plain = sample_size_means(delta, sd, **kwargs)
    adjusted = sample_size_means(delta, sd * math.sqrt(1 - rho**2), **kwargs)
    return plain, adjusted
