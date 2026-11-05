"""Frequentist analysis of a finished test.

Conversion: two-proportion z test with the pooled standard error, a Wald interval
for the difference and a log-scale (Katz) interval for the lift, the same methods
as statsmodels' proportions_ztest and confint_proportions_2indep.
Means: Welch's t test with Welch-Satterthwaite degrees of freedom, a delta-method
interval for the lift on the log scale, a percentile bootstrap for heavy tails,
and CUPED with a pre-period covariate.
Sample ratio mismatch: chi-square goodness of fit, flagged below p = 0.001 as
recommended by Fabijan et al. (2019), "Diagnosing sample ratio mismatch in online
controlled experiments", KDD.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

SRM_THRESHOLD = 0.001


@dataclass
class Estimate:
    metric: str
    test: str
    control: float
    treatment: float
    diff: float
    diff_ci: tuple[float, float]
    lift: float
    lift_ci: tuple[float, float]
    statistic: float
    p_value: float
    df: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _z(alpha: float) -> float:
    return float(stats.norm.isf(alpha / 2))


def ratio_ci(mean_a: float, var_a: float, n_a: int, mean_b: float, var_b: float, n_b: int,
             alpha: float = 0.05) -> tuple[float, float]:
    """Interval for mean_b / mean_a - 1 from the delta method on the log scale.
    With var = p(1 - p) this is the Katz interval for a ratio of proportions."""
    if mean_a <= 0 or mean_b <= 0:
        return (math.nan, math.nan)
    se = math.sqrt(var_a / (n_a * mean_a**2) + var_b / (n_b * mean_b**2))
    ratio = mean_b / mean_a
    z = _z(alpha)
    return (ratio * math.exp(-z * se) - 1, ratio * math.exp(z * se) - 1)


def two_proportions(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05, metric: str = "conversion") -> Estimate:
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se0 if se0 else 0.0
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff, crit = p_b - p_a, _z(alpha)
    return Estimate(
        metric, "z", p_a, p_b, diff, (diff - crit * se, diff + crit * se),
        p_b / p_a - 1 if p_a else math.nan,
        ratio_ci(p_a, p_a * (1 - p_a), n_a, p_b, p_b * (1 - p_b), n_b, alpha),
        z, float(2 * stats.norm.sf(abs(z))),
    )


def welch(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int, alpha: float = 0.05,
          metric: str = "mean") -> Estimate:
    va, vb = sd_a**2 / n_a, sd_b**2 / n_b
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
    diff = mean_b - mean_a
    t = diff / se
    crit = float(stats.t.isf(alpha / 2, df))
    return Estimate(
        metric, "welch", mean_a, mean_b, diff, (diff - crit * se, diff + crit * se),
        mean_b / mean_a - 1 if mean_a else math.nan,
        ratio_ci(mean_a, sd_a**2, n_a, mean_b, sd_b**2, n_b, alpha),
        t, float(2 * stats.t.sf(abs(t), df)), df,
    )


def welch_samples(a, b, alpha: float = 0.05, metric: str = "mean") -> Estimate:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return welch(a.mean(), a.std(ddof=1), len(a), b.mean(), b.std(ddof=1), len(b), alpha, metric)


@dataclass
class BootstrapResult:
    diff: float
    diff_ci: tuple[float, float]
    lift: float
    lift_ci: tuple[float, float]
    rounds: int


def bootstrap_means(a, b, rounds: int = 5000, seed: int = 0, alpha: float = 0.05, chunk: int = 500) -> BootstrapResult:
    """Percentile bootstrap for the difference and the lift in means, resampling
    each group on its own. Works in chunks so memory stays small."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means_a, means_b = np.empty(rounds), np.empty(rounds)
    for start in range(0, rounds, chunk):
        size = min(chunk, rounds - start)
        means_a[start:start + size] = a[rng.integers(0, len(a), (size, len(a)))].mean(axis=1)
        means_b[start:start + size] = b[rng.integers(0, len(b), (size, len(b)))].mean(axis=1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    diffs = means_b - means_a
    with np.errstate(divide="ignore", invalid="ignore"):
        lifts = means_b / means_a - 1
    d_low, d_high = np.percentile(diffs, q)
    l_low, l_high = np.percentile(lifts, q)
    return BootstrapResult(float(b.mean() - a.mean()), (float(d_low), float(d_high)),
                           float(b.mean() / a.mean() - 1), (float(l_low), float(l_high)), rounds)


@dataclass
class Cuped:
    theta: float
    rho: float
    variance_reduction: float
    adjusted_a: np.ndarray
    adjusted_b: np.ndarray


def cuped(y_a, x_a, y_b, x_b) -> Cuped:
    """Y - theta (X - mean X) with theta = cov(X, Y) / var(X) over both groups
    (Deng et al. 2013). The pre-period X must not be affected by the treatment."""
    y_a, x_a, y_b, x_b = (np.asarray(v, float) for v in (y_a, x_a, y_b, x_b))
    y, x = np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b])
    var_x = x.var(ddof=1)
    if var_x == 0:
        raise ValueError("the pre-period metric is constant, so it cannot reduce variance")
    theta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    centre = x.mean()
    adj_a, adj_b = y_a - theta * (x_a - centre), y_b - theta * (x_b - centre)
    adjusted = np.concatenate([adj_a, adj_b])
    return Cuped(theta, float(np.corrcoef(x, y)[0, 1]), float(1 - adjusted.var(ddof=1) / y.var(ddof=1)), adj_a, adj_b)


@dataclass
class SampleRatio:
    observed: dict[str, int]
    expected_share: dict[str, float]
    chi2: float
    p_value: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < SRM_THRESHOLD


def sample_ratio(observed: dict[str, int], expected_share: dict[str, float] | None = None) -> SampleRatio:
    names = list(observed)
    if expected_share is None:
        expected_share = {k: 1 / len(names) for k in names}
    if set(expected_share) != set(names):
        raise ValueError("expected shares must name the same variants as the data")
    total_share = sum(expected_share.values())
    counts = np.array([observed[k] for k in names], float)
    expected = counts.sum() * np.array([expected_share[k] / total_share for k in names])
    chi2, p = stats.chisquare(counts, expected)
    return SampleRatio(dict(observed), {k: expected_share[k] / total_share for k in names}, float(chi2), float(p))
