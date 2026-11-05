"""The per-case files in examples/qanun must reproduce the intervals qanun publishes
in eval/results.md ("Change from Hybrid RRF (BM25 + e5-small)" and "What each stage
adds"), each a 95% paired bootstrap with 10000 resamples and seed 0."""
import json
import re
from pathlib import Path

import pytest

from qatar_abtest.compare import paired_bootstrap

HERE = Path(__file__).resolve().parent.parent / "examples" / "qanun"
METRICS = ("recall@4", "recall@5", "mrr@10")
BASELINE = "Hybrid RRF (BM25 + e5-small)"
PUBLISHED = [
    (BASELINE, "BM25", ["-0.067 (-0.167 to +0.017)", "-0.100 (-0.200 to -0.017)", "-0.058 (-0.114 to -0.004)"]),
    (BASELINE, "Dense (e5-small)", ["-0.017 (-0.067 to +0.033)", "-0.067 (-0.150 to +0.000)", "+0.052 (-0.020 to +0.124)"]),
    (BASELINE, "Dense (minilm)", ["-0.058 (-0.183 to +0.067)", "-0.058 (-0.192 to +0.067)", "-0.127 (-0.245 to -0.013)"]),
    (BASELINE, "Hybrid RRF (BM25 + minilm)", ["+0.025 (-0.042 to +0.092)", "+0.033 (-0.033 to +0.100)", "-0.044 (-0.123 to +0.031)"]),
    (BASELINE, "Cross-language, no translation", ["+0.050 (+0.000 to +0.117)", "+0.017 (-0.033 to +0.083)", "+0.042 (+0.012 to +0.077)"]),
    (BASELINE, "Cross-language + translation", ["+0.100 (+0.000 to +0.217)", "+0.083 (-0.033 to +0.200)", "+0.060 (+0.001 to +0.126)"]),
    (BASELINE, "BM25 + translation", ["+0.033 (-0.067 to +0.133)", "+0.017 (-0.083 to +0.117)", "-0.007 (-0.073 to +0.059)"]),
    (BASELINE, "Plain RRF over both queries", ["-0.033 (-0.133 to +0.050)", "-0.050 (-0.150 to +0.050)", "-0.070 (-0.157 to +0.010)"]),
    (BASELINE, "Cross-language + mmarco", ["+0.067 (-0.017 to +0.150)", "+0.017 (-0.067 to +0.100)", "+0.087 (+0.009 to +0.164)"]),
    (BASELINE, "Cross-language + translation + mmarco", ["+0.233 (+0.133 to +0.350)", "+0.183 (+0.083 to +0.300)", "+0.205 (+0.118 to +0.294)"]),
    ("Plain RRF over both queries", "Cross-language + translation", ["+0.133 (+0.000 to +0.267)", "+0.133 (+0.017 to +0.250)", "+0.130 (+0.040 to +0.225)"]),
    ("Cross-language, no translation", "Cross-language + translation", ["+0.050 (-0.067 to +0.167)", "+0.067 (-0.050 to +0.183)", "+0.018 (-0.051 to +0.091)"]),
    ("BM25", "BM25 + translation", ["+0.100 (+0.000 to +0.200)", "+0.117 (+0.017 to +0.217)", "+0.051 (+0.013 to +0.093)"]),
    ("Cross-language, no translation", "Cross-language + mmarco", ["+0.017 (-0.067 to +0.100)", "+0.000 (-0.083 to +0.083)", "+0.046 (-0.027 to +0.116)"]),
    ("Cross-language + translation", "Cross-language + translation + mmarco", ["+0.133 (+0.050 to +0.233)", "+0.100 (+0.017 to +0.183)", "+0.145 (+0.063 to +0.229)"]),
]


def scores(method: str, metric: str) -> list[float]:
    slug = re.sub(r"[^a-z0-9]+", "-", method.lower()).strip("-")
    rows = [json.loads(line) for line in (HERE / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r[metric] for r in rows]


@pytest.mark.parametrize("old, new, cells", PUBLISHED, ids=[f"{n} vs {o}" for o, n, _ in PUBLISHED])
def test_reproduces_published_interval(old, new, cells):
    for metric, cell in zip(METRICS, cells):
        mean, low, high = paired_bootstrap(scores(old, metric), scores(new, metric))
        assert f"{mean:+.3f} ({low:+.3f} to {high:+.3f})" == cell, metric
