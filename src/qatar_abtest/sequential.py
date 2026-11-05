"""Group sequential boundaries, so a test can be looked at before it ends.

Looking at a fixed-sample test several times and stopping at the first p < 0.05
raises the real false positive rate: about 0.142 with five looks (Armitage,
McPherson and Rowe 1969; Jennison and Turnbull 2000, Table 1.1). An alpha
spending function decides in advance how much of alpha each look may use.

The boundaries use the Lan and DeMets (1983) spending function of O'Brien-Fleming
type, applied to each side with alpha / 2:

    spent(t) = 2 - 2 Phi(z_{1 - alpha/4} / sqrt(t))

where t is the share of the planned sample seen so far. Early looks get a very
high bar and the last one stays close to the fixed-sample 1.96. The crossing
probabilities are computed by recursive numerical integration over the Brownian
motion of the z statistic, as in Armitage et al.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

GRID = 801  # points per look; the integrand is smooth, so Simpson's rule is accurate to about 1e-7


def _simpson_weights(n: int, step: float) -> np.ndarray:
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * step / 3


class _Walk:
    """The density of B(t) = Z(t) sqrt(t) on paths that have not crossed yet."""

    def __init__(self) -> None:
        self.t = 0.0
        self.x: np.ndarray | None = None
        self.f: np.ndarray | None = None

    def crossing(self, t: float, bound: float) -> float:
        """Probability of first crossing |Z| >= bound at information time t."""
        b = bound * math.sqrt(t)
        if self.x is None:
            return float(2 * stats.norm.sf(bound))
        s = math.sqrt(t - self.t)
        outside = stats.norm.sf((b - self.x) / s) + stats.norm.cdf((-b - self.x) / s)
        return float(np.sum(self.w * self.f * outside))

    def advance(self, t: float, bound: float) -> None:
        b = bound * math.sqrt(t)
        y = np.linspace(-b, b, GRID)
        if self.x is None:
            f = stats.norm.pdf(y / math.sqrt(t)) / math.sqrt(t)
        else:
            s = math.sqrt(t - self.t)
            kernel = stats.norm.pdf((y[:, None] - self.x[None, :]) / s) / s
            f = kernel @ (self.w * self.f)
        self.t, self.x, self.f = t, y, f
        self.w = _simpson_weights(GRID, y[1] - y[0])


def _fractions(looks) -> np.ndarray:
    t = np.arange(1, looks + 1) / looks if isinstance(looks, int) else np.asarray(looks, float)
    if len(t) == 0 or np.any(np.diff(t) <= 0) or t[0] <= 0 or abs(t[-1] - 1) > 1e-9:
        raise ValueError("looks must be a count or increasing information fractions ending at 1")
    return t


def obf_spent(t, alpha: float = 0.05) -> np.ndarray:
    """Two-sided alpha spent by information fraction t."""
    t = np.asarray(t, float)
    return 2 * (2 - 2 * stats.norm.cdf(stats.norm.isf(alpha / 4) / np.sqrt(t)))


@dataclass
class Boundaries:
    fractions: np.ndarray
    z: np.ndarray  # reject at look k when |Z_k| >= z[k]
    alpha: float

    @property
    def nominal_p(self) -> np.ndarray:
        return 2 * stats.norm.sf(self.z)

    @property
    def spent(self) -> np.ndarray:
        return obf_spent(self.fractions, self.alpha)


def obf_boundaries(looks=5, alpha: float = 0.05) -> Boundaries:
    """O'Brien-Fleming type boundaries for `looks` equally spaced looks, or for
    a list of information fractions such as [0.25, 0.5, 1]."""
    t = _fractions(looks)
    spent = np.diff(np.concatenate([[0.0], obf_spent(t, alpha)]))
    walk, bounds = _Walk(), []
    for tk, budget in zip(t, spent):
        c = optimize.brentq(lambda c: walk.crossing(tk, c) - budget, 0.5, 40.0, xtol=1e-9)
        bounds.append(c)
        walk.advance(tk, c)
    return Boundaries(t, np.array(bounds), alpha)


def crossing_probability(bounds, looks=None) -> float:
    """Chance that |Z| crosses the given bound at some look when there is no
    effect. With 1.96 at every look this is the error rate of naive peeking."""
    bounds = np.atleast_1d(np.asarray(bounds, float))
    t = _fractions(looks if looks is not None else len(bounds))
    if len(bounds) == 1:
        bounds = np.repeat(bounds, len(t))
    walk, total = _Walk(), 0.0
    for tk, c in zip(t, bounds):
        total += walk.crossing(tk, c)
        walk.advance(tk, c)
    return total


def naive_peeking_alpha(looks: int, alpha: float = 0.05) -> float:
    return crossing_probability(stats.norm.isf(alpha / 2), looks)
