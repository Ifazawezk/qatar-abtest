"""Adjusted p-values when several metrics, variants or systems are tested at once.

Holm (1979) controls the family-wise error rate: the chance of any false positive.
Benjamini and Hochberg (1995) control the false discovery rate: the expected share
of false positives among the results called significant.
"""
from __future__ import annotations

import numpy as np

METHODS = ("holm", "bh", "none")


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(stepped, 1.0)
    return out


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)[::-1]  # largest first
    stepped = np.minimum.accumulate(p[order] * m / np.arange(m, 0, -1))
    out = np.empty(m)
    out[order] = np.minimum(stepped, 1.0)
    return out


def adjust(p_values, method: str = "holm") -> np.ndarray:
    if method == "holm":
        return holm(p_values)
    if method == "bh":
        return benjamini_hochberg(p_values)
    if method == "none":
        return np.asarray(p_values, float)
    raise ValueError(f"method must be one of {', '.join(METHODS)}")
