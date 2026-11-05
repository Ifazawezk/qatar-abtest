"""Compare two models or systems scored on the same test cases.

The cases are paired, so the tests work on per-case differences:
- McNemar's test for two classifiers right or wrong on the same items, exact
  binomial on the discordant pairs, or chi-square with continuity correction
  (McNemar 1947; Agresti 2002, "Categorical Data Analysis", section 10.1).
- A paired bootstrap interval for the mean difference: resample cases with
  replacement and take the 2.5th and 97.5th percentiles.
- A paired permutation test: flip the sign of each case's difference at random.
  Exact when 2^n is small enough, otherwise Monte Carlo with (hits + 1) / (rounds + 1).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .multiple import adjust

EXACT_LIMIT = 16  # enumerate all 2^n sign flips up to this many non-zero differences


@dataclass
class McNemar:
    only_a: int  # cases A gets right and B gets wrong
    only_b: int
    statistic: float
    p_value: float
    exact: bool


def mcnemar(a_correct, b_correct, exact: bool | None = None, correction: bool = True) -> McNemar:
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if exact is None:
        exact = n < 25
    if n == 0:
        return McNemar(0, 0, 0.0, 1.0, exact)
    if exact:
        p = min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5))
        return McNemar(only_a, only_b, float(min(only_a, only_b)), float(p), True)
    chi2 = (abs(only_a - only_b) - int(correction)) ** 2 / n
    return McNemar(only_a, only_b, chi2, float(stats.chi2.sf(chi2, 1)), False)


def paired_bootstrap(a, b, rounds: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of b - a and its percentile interval. With the defaults this draws the
    same resamples as qanun's evaluation, so the intervals match its report."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(rounds, len(diff)))
    means = diff[picks].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(low), float(high)


def paired_permutation(a, b, rounds: int = 10000, seed: int = 0) -> tuple[float, bool]:
    """Two-sided p-value for a mean difference of 0, and whether it is exact."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    diff = diff[diff != 0]  # zero differences do not change under a sign flip
    if len(diff) == 0:
        return 1.0, True
    observed = abs(diff.mean()) - 1e-12
    if len(diff) <= EXACT_LIMIT:
        signs = np.array(list(itertools.product([1, -1], repeat=len(diff))))
        return float((np.abs((signs * diff).mean(axis=1)) >= observed).mean()), True
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, rounds, 2000):
        size = min(2000, rounds - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(diff)))
        hits += int((np.abs((signs * diff).mean(axis=1)) >= observed).sum())
    return (hits + 1) / (rounds + 1), False


@dataclass
class Comparison:
    baseline: str
    system: str
    metric: str
    cases: int
    mean_baseline: float
    mean_system: float
    diff: float
    ci: tuple[float, float]
    p_value: float
    p_exact: bool
    p_adjusted: float = math.nan
    mcnemar: McNemar | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare_systems(baseline: tuple[str, dict], systems: list[tuple[str, dict]], metric: str,
                    rounds: int = 10000, seed: int = 0, correction: str = "holm") -> list[Comparison]:
    """Each system against the baseline on the cases they share. `baseline` and
    each system are (name, {case id: score}). With more than one system the
    permutation p-values are adjusted with Holm (or bh, or none)."""
    base_name, base_scores = baseline
    out = []
    for name, scores in systems:
        ids = sorted(set(base_scores) & set(scores))
        if not ids:
            raise ValueError(f"{name} shares no case ids with {base_name}")
        a = np.array([base_scores[i] for i in ids], float)
        b = np.array([scores[i] for i in ids], float)
        mean, low, high = paired_bootstrap(a, b, rounds, seed)
        p, exact = paired_permutation(a, b, rounds, seed)
        binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
        out.append(Comparison(base_name, name, metric, len(ids), float(a.mean()), float(b.mean()), mean, (low, high), p,
                              exact, mcnemar=mcnemar(a, b) if binary else None))
    for c, adjusted in zip(out, adjust([c.p_value for c in out], correction)):
        c.p_adjusted = float(adjusted)
    return out
