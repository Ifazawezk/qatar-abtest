"""Generate examples/checkout_test.csv. Every number is made up.

A Doha grocery app tests its free delivery threshold: QAR 150 today (free_150, the
control) against QAR 120 and QAR 100. 12,000 users enter between Sunday
8 February and Saturday 28 February 2026, three whole weeks that run into
Ramadan 1447 (calculated start 18 February). Revenue per user is in QAR and
heavy tailed. pre_revenue_qar is the same user's spend from 11 to 31 January,
for CUPED.

    python examples/make_checkout_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(1447)
n = 12000
variant = rng.choice(["free_150", "free_120", "free_100"], n)
appetite = rng.normal(0, 1, n)  # how much a household buys, shared by both periods

pre_orders = rng.poisson(np.exp(-0.8 + 0.9 * appetite))
pre_revenue = pre_orders * rng.lognormal(np.log(180) + 0.3 * appetite, 0.5, n)

first_seen = pd.Timestamp("2026-02-08") + pd.to_timedelta(rng.integers(0, 21, n), unit="D")
ramadan = np.asarray(first_seen >= pd.Timestamp("2026-02-18"))
uplift = pd.Series(variant).map({"free_150": 1.0, "free_120": 1.06, "free_100": 1.15}).to_numpy()
p_order = np.clip(0.09 * np.exp(0.8 * appetite) * uplift * np.where(ramadan, 1.15, 1.0), 0, 0.95)
converted = rng.random(n) < p_order
basket_shift = pd.Series(variant).map({"free_150": 0.0, "free_120": -0.03, "free_100": -0.07}).to_numpy()
basket = rng.lognormal(np.log(205) + 0.3 * appetite + basket_shift, 0.65, n)
revenue = np.where(converted, basket, 0.0)

df = pd.DataFrame(
    {
        "user_id": [f"u{i:05d}" for i in range(1, n + 1)],
        "variant": variant,
        "first_seen": first_seen.strftime("%Y-%m-%d"),
        "converted": converted.astype(int),
        "revenue_qar": revenue.round(2),
        "pre_revenue_qar": pre_revenue.round(2),
    }
)
df.to_csv(Path(__file__).parent / "checkout_test.csv", index=False)
print(df.groupby("variant")[["converted", "revenue_qar"]].mean().round(4))
