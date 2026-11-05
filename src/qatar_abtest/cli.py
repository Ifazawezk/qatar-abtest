"""Command line entry point: qabtest size, analyze, calendar, boundaries, compare."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import pandas as pd

from . import qatar_calendar
from .analysis import bootstrap_means, cuped, sample_ratio, two_proportions, welch_samples
from .bayes import beta_binomial
from .compare import compare_systems
from .multiple import METHODS, adjust
from .power import cuped_sample_size, sample_size_means, sample_size_proportions
from .sequential import naive_peeking_alpha, obf_boundaries

EXIT_OK, EXIT_WARN, EXIT_ERROR = 0, 1, 2


class UsageError(Exception):
    pass


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:+.{digits}f}%"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {text}") from None


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"file not found: {path}")
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True, dtype=False)
    return pd.read_csv(p)


def _emit(args, text: str, data: dict) -> None:
    if getattr(args, "json", None) == "-":
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return
    print(text)
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


# size


def cmd_size(args) -> int:
    common = dict(alpha=args.alpha, power=args.power, ratio=args.ratio)
    lines, data = [], {"alpha": args.alpha, "power": args.power, "ratio": args.ratio}
    if args.baseline is not None:
        if (args.lift is None) == (args.mde is None):
            raise UsageError("give --lift (relative) or --mde (absolute) with --baseline")
        target = args.baseline * (1 + args.lift) if args.lift is not None else args.baseline + args.mde
        size = sample_size_proportions(args.baseline, target, **common)
        lines.append(f"conversion {args.baseline:.2%} -> {target:.2%} ({_pct(target / args.baseline - 1)}), "
                     f"alpha {args.alpha}, power {args.power:.0%}")
        lines.append(f"users per group: {size.control:,} control, {size.treatment:,} treatment ({size.total:,} in total)")
        data.update(kind="conversion", baseline=args.baseline, target=target, control=size.control, treatment=size.treatment)
        totals = [("", size.total)]
    elif args.sd is not None:
        if args.mde is None:
            raise UsageError("give --mde, the difference in the mean to detect, with --sd")
        lines.append(f"mean difference {args.mde:g} with sd {args.sd:g} (d = {args.mde / args.sd:.3f}), "
                     f"alpha {args.alpha}, power {args.power:.0%}")
        if args.cuped_r:
            plain, adjusted = cuped_sample_size(args.mde, args.sd, args.cuped_r, **common)
            lines.append(f"users per group: {plain.control:,} control, {plain.treatment:,} treatment ({plain.total:,} in total)")
            lines.append(f"with CUPED (pre-period r = {args.cuped_r}): {adjusted.control:,} and {adjusted.treatment:,} "
                         f"({adjusted.total:,}, {adjusted.total / plain.total - 1:+.0%})")
            totals = [("", plain.total), (" with CUPED", adjusted.total)]
            data.update(cuped_control=adjusted.control, cuped_treatment=adjusted.treatment)
        else:
            plain = sample_size_means(args.mde, args.sd, **common)
            lines.append(f"users per group: {plain.control:,} control, {plain.treatment:,} treatment ({plain.total:,} in total)")
            totals = [("", plain.total)]
        data.update(kind="mean", sd=args.sd, mde=args.mde, control=plain.control, treatment=plain.treatment)
    else:
        raise UsageError("give --baseline for a conversion rate or --sd for a mean")
    if args.daily_users:
        for label, total in totals:
            days = math.ceil(total / args.daily_users)
            weeks = math.ceil(days / 7)
            lines.append(f"at {args.daily_users:,} users a day{label}: {days} days, so run {weeks} whole weeks ({weeks * 7} days)")
    _emit(args, "\n".join(lines), data)
    return EXIT_OK


# analyze


def _variants(df: pd.DataFrame, column: str, control: str | None) -> tuple[str, list[str]]:
    names = [str(v) for v in pd.unique(df[column].astype(str))]
    if control is None:
        control = sorted(names)[0]
    if control not in names:
        raise UsageError(f"control {control!r} is not a value of {column}; found {', '.join(names)}")
    if len(names) < 2:
        raise UsageError("need at least two variants")
    return control, [n for n in sorted(names) if n != control]


def cmd_analyze(args) -> int:
    df = _read(args.data)
    for col in [args.variant, args.converted, args.revenue, args.pre, args.date]:
        if col and col not in df.columns:
            raise UsageError(f"column {col!r} not in {args.data}; columns are {', '.join(map(str, df.columns))}")
    df[args.variant] = df[args.variant].astype(str)
    control, treatments = _variants(df, args.variant, args.control)
    groups = {name: part for name, part in df.groupby(args.variant)}
    out = [f"{Path(args.data).name}: {len(df):,} users in {len(groups)} variants, control {control}"]
    data: dict = {"users": len(df), "control": control}
    status = EXIT_OK

    counts = {name: len(groups[name]) for name in [control] + treatments}
    expected = None
    if args.expected:
        shares = [float(s) for s in args.expected.split(",")]
        if len(shares) != len(counts):
            raise UsageError(f"--expected needs {len(counts)} shares, in the order {', '.join(counts)}")
        expected = dict(zip(counts, shares))
    srm = sample_ratio(counts, expected)
    verdict = "MISMATCH, do not trust the results" if srm.mismatch else "ok"
    out.append("sample ratio: " + ", ".join(f"{k} {v:,}" for k, v in counts.items())
               + f"; chi2 {srm.chi2:.2f}, p {_p(srm.p_value)}, {verdict}")
    data["sample_ratio"] = {"counts": counts, "chi2": srm.chi2, "p_value": srm.p_value, "mismatch": srm.mismatch}
    if srm.mismatch:
        status = EXIT_WARN

    notes = []
    if args.date:
        days = pd.to_datetime(df[args.date]).dt.date
        test = qatar_calendar.describe("the test", days.min(), days.max())
        pre = qatar_calendar.describe("the pre-period", *args.pre_period) if args.pre_period else None
        notes = qatar_calendar.warnings(test, pre)
        out.append(f"window: {test.start} to {test.end} ({test.days} days)")
        data["calendar"] = {"test": test.to_dict(), "pre": pre.to_dict() if pre else None, "warnings": notes}
    out += [f"  ! {n}" for n in notes]

    boundary = None
    if args.looks:
        if not 1 <= args.look <= args.looks:
            raise UsageError("--look must be between 1 and --looks")
        bounds = obf_boundaries(args.looks, args.alpha)
        boundary = float(bounds.z[args.look - 1])
        out.append(f"look {args.look} of {args.looks}: O'Brien-Fleming bound |z| >= {boundary:.3f} "
                   f"(p < {_p(float(bounds.nominal_p[args.look - 1]))}); stopping at the first p < {args.alpha} "
                   f"over {args.looks} looks has a false positive rate of {naive_peeking_alpha(args.looks, args.alpha):.1%}")
        data["sequential"] = {"looks": args.looks, "look": args.look, "boundary": boundary}

    tests, results = [], []
    for name in treatments:
        a, b = groups[control], groups[name]
        block, row = [f"", f"{name} vs {control}"], {"variant": name}
        if args.converted:
            xa, xb = int(a[args.converted].astype(bool).sum()), int(b[args.converted].astype(bool).sum())
            est = two_proportions(xa, len(a), xb, len(b), args.alpha)
            bayes = beta_binomial(xa, len(a), xb, len(b), seed=args.seed)
            block.append(f"  conversion   {est.control:.2%} -> {est.treatment:.2%}  diff {100 * est.diff:+.2f} pp "
                         f"[{100 * est.diff_ci[0]:+.2f}, {100 * est.diff_ci[1]:+.2f}]  lift {_pct(est.lift)} "
                         f"[{_pct(est.lift_ci[0])}, {_pct(est.lift_ci[1])}]  z {est.statistic:.2f}  p {_p(est.p_value)}")
            block.append(f"  bayesian     P(better) {bayes.prob_b_better:.3f}  expected loss if shipped "
                         f"{100 * bayes.loss_choosing_b:.3f} pp  lift [{_pct(bayes.lift_interval[0])}, {_pct(bayes.lift_interval[1])}]")
            tests.append((name, "conversion", est.p_value, est.statistic))
            row.update(conversion=est.to_dict(), bayesian=bayes.to_dict())
        if args.revenue:
            ya, yb = a[args.revenue].to_numpy(float), b[args.revenue].to_numpy(float)
            w = welch_samples(ya, yb, args.alpha, args.revenue)
            boot = bootstrap_means(ya, yb, rounds=args.bootstrap, seed=args.seed, alpha=args.alpha)
            block.append(f"  {args.revenue}  {w.control:.2f} -> {w.treatment:.2f}  Welch diff {w.diff:+.2f} "
                         f"[{w.diff_ci[0]:+.2f}, {w.diff_ci[1]:+.2f}]  lift {_pct(w.lift)} [{_pct(w.lift_ci[0])}, "
                         f"{_pct(w.lift_ci[1])}]  p {_p(w.p_value)}")
            block.append(f"  {'':{len(args.revenue)}}  bootstrap diff [{boot.diff_ci[0]:+.2f}, {boot.diff_ci[1]:+.2f}]  "
                         f"lift [{_pct(boot.lift_ci[0])}, {_pct(boot.lift_ci[1])}]  ({boot.rounds} resamples)")
            row.update(revenue_welch=w.to_dict(), revenue_bootstrap=boot.__dict__)
            primary = w
            if args.pre:
                c = cuped(ya, a[args.pre].to_numpy(float), yb, b[args.pre].to_numpy(float))
                adj = welch_samples(c.adjusted_a, c.adjusted_b, args.alpha, args.revenue + " (CUPED)")
                block.append(f"  {'':{len(args.revenue)}}  CUPED diff {adj.diff:+.2f} [{adj.diff_ci[0]:+.2f}, "
                             f"{adj.diff_ci[1]:+.2f}]  p {_p(adj.p_value)}  (r {c.rho:.2f}, variance "
                             f"{-c.variance_reduction:+.0%}, interval {1 - (adj.diff_ci[1] - adj.diff_ci[0]) / (w.diff_ci[1] - w.diff_ci[0]):.0%} narrower)")
                row["revenue_cuped"] = {**adj.to_dict(), "theta": c.theta, "rho": c.rho, "variance_reduction": c.variance_reduction}
                primary = adj
            tests.append((name, args.revenue, primary.p_value, primary.statistic))
        out += block
        results.append(row)

    if len(tests) > 1:
        adjusted = adjust([t[2] for t in tests], args.correct)
        out += ["", f"{len(tests)} tests, p adjusted with {args.correct}:"]
        out += [f"  {n:<16} {m:<14} p {_p(p)}  adjusted {_p(q)}" for (n, m, p, _), q in zip(tests, adjusted)]
        data["adjusted"] = [{"variant": n, "metric": m, "p_value": p, "adjusted": float(q)} for (n, m, p, _), q in zip(tests, adjusted)]

    if boundary is not None:
        crossed = [f"{n} {m}" for n, m, _, z in tests if abs(z) >= boundary]
        naive = [f"{n} {m}" for n, m, p, z in tests if p < args.alpha and abs(z) < boundary]
        out.append("")
        if crossed:
            out.append(f"  crosses the boundary at look {args.look}: {', '.join(crossed)}")
        if naive and args.look < args.looks:
            out.append(f"  ! p < {args.alpha} but inside the boundary: {', '.join(naive)}. Stopping now would be peeking")
        if not crossed and args.look < args.looks:
            out.append(f"  nothing crosses the boundary; keep the test running to look {args.look + 1}")
    data["results"] = results
    _emit(args, "\n".join(out), data)
    return status


# calendar and boundaries


def cmd_calendar(args) -> int:
    test = qatar_calendar.describe("the test", args.start, args.end)
    pre = qatar_calendar.describe("the pre-period", *args.pre_period) if args.pre_period else None
    notes = qatar_calendar.warnings(test, pre)
    lines = []
    for p in [x for x in (pre, test) if x]:
        extra = "; ".join(f"{k} {qatar_calendar.span(v)}" for k, v in p.special.items())
        lines.append(f"{p.name}: {p.start} to {p.end}, {p.days} days, {p.weekend_days} on Friday or Saturday, "
                     f"{p.ramadan_days} in Ramadan" + (f"; {extra}" if extra else ""))
    lines += [f"  ! {n}" for n in notes] or ["  no calendar warnings"]
    _emit(args, "\n".join(lines), {"test": test.to_dict(), "pre": pre.to_dict() if pre else None, "warnings": notes})
    return EXIT_WARN if notes else EXIT_OK


def cmd_boundaries(args) -> int:
    fractions = [float(x) for x in args.fractions.split(",")] if args.fractions else args.looks
    b = obf_boundaries(fractions, args.alpha)
    lines = [f"O'Brien-Fleming type alpha spending (Lan-DeMets), two-sided alpha {args.alpha}", "",
             "look  share of sample  bound |z|  nominal p  alpha spent"]
    for k, (t, z, p, s) in enumerate(zip(b.fractions, b.z, b.nominal_p, b.spent), start=1):
        lines.append(f"{k:>4}  {t:>15.0%}  {z:>9.3f}  {p:>9.5f}  {s:>11.5f}")
    naive = naive_peeking_alpha(len(b.fractions), args.alpha)
    lines += ["", f"stopping at the first p < {args.alpha} over {len(b.fractions)} looks instead: false positive rate {naive:.3f}"]
    _emit(args, "\n".join(lines), {"fractions": b.fractions.tolist(), "z": b.z.tolist(), "nominal_p": b.nominal_p.tolist(),
                                   "naive_alpha": naive})
    return EXIT_OK


# compare


def _scores(path: str, id_column: str, metric: str) -> dict:
    df = _read(path)
    for col in (id_column, metric):
        if col not in df.columns:
            raise UsageError(f"{path} has no column {col!r}; columns are {', '.join(map(str, df.columns))}")
    if df[id_column].duplicated().any():
        raise UsageError(f"{path} repeats case ids in {id_column!r}")
    return dict(zip(df[id_column].astype(str), df[metric].astype(float)))


def cmd_compare(args) -> int:
    base = (Path(args.baseline).stem, _scores(args.baseline, args.id, args.metric))
    others = [(Path(p).stem, _scores(p, args.id, args.metric)) for p in args.systems]
    rows = compare_systems(base, others, args.metric, rounds=args.rounds, seed=args.seed, correction=args.correct)
    width = max(len(r.system) for r in rows)
    lines = [f"{args.metric} on {rows[0].cases} paired cases, baseline {base[0]} ({rows[0].mean_baseline:.3f})",
             f"95% paired bootstrap interval ({args.rounds} resamples, seed {args.seed}); "
             f"sign-flip permutation p, adjusted with {args.correct}", ""]
    for r in rows:
        exact = "" if r.p_exact else "~"
        lines.append(f"{r.system:<{width}}  {r.mean_system:.3f}  diff {r.diff:+.3f} ({r.ci[0]:+.3f} to {r.ci[1]:+.3f})  "
                     f"p {exact}{_p(r.p_value)}  adjusted {_p(r.p_adjusted)}")
        if r.mcnemar:
            m = r.mcnemar
            kind = "exact" if m.exact else "chi2 with continuity correction"
            lines.append(f"{'':<{width}}  McNemar: {m.only_a} cases only {base[0]} gets right, {m.only_b} only "
                         f"{r.system}; p {_p(m.p_value)} ({kind})")
    _emit(args, "\n".join(lines), {"metric": args.metric, "baseline": base[0], "rows": [r.to_dict() for r in rows]})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qabtest", description="A/B test planning and analysis for experiments run in Qatar.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, alpha=True):
        if alpha:
            p.add_argument("--alpha", type=float, default=0.05, help="two-sided significance level (default 0.05)")
        p.add_argument("--json", metavar="PATH", help="write JSON to PATH; - prints it instead of text")

    size = sub.add_parser("size", help="users needed per group")
    size.add_argument("--baseline", type=float, help="control conversion rate, such as 0.032")
    size.add_argument("--lift", type=float, help="relative change to detect, such as 0.10 for +10%%")
    size.add_argument("--sd", type=float, help="standard deviation of the metric, for a mean such as revenue per user")
    size.add_argument("--mde", type=float, help="absolute change to detect: rate points with --baseline, metric units with --sd")
    size.add_argument("--cuped-r", type=float, help="correlation with a pre-period metric, to show the CUPED saving")
    size.add_argument("--power", type=float, default=0.8)
    size.add_argument("--ratio", type=float, default=1.0, help="treatment size over control size (default 1)")
    size.add_argument("--daily-users", type=int, help="users entering the test per day, to turn the size into whole weeks")
    common(size)
    size.set_defaults(func=cmd_size)

    an = sub.add_parser("analyze", help="analyse a finished or running test from one row per user")
    an.add_argument("data", help="csv or jsonl with one row per user")
    an.add_argument("--variant", default="variant", help="variant column (default variant)")
    an.add_argument("--control", help="control variant (default: first in sort order)")
    an.add_argument("--converted", help="0/1 conversion column")
    an.add_argument("--revenue", help="numeric metric per user, such as revenue in QAR")
    an.add_argument("--pre", help="the same metric in the pre-period, for CUPED")
    an.add_argument("--date", help="date each user entered the test, for the calendar checks")
    an.add_argument("--pre-period", nargs=2, type=_date, metavar=("START", "END"), help="dates of the pre-period")
    an.add_argument("--expected", help="planned split in variant order, control first, such as 50,25,25 (default equal)")
    an.add_argument("--looks", type=int, help="planned number of looks, for the sequential boundary")
    an.add_argument("--look", type=int, default=1, help="which look this is (default 1)")
    an.add_argument("--correct", choices=METHODS, default="holm", help="correction across variants and metrics (default holm)")
    an.add_argument("--bootstrap", type=int, default=5000, help="bootstrap resamples (default 5000)")
    an.add_argument("--seed", type=int, default=0)
    common(an)
    an.set_defaults(func=cmd_analyze)

    cal = sub.add_parser("calendar", help="Ramadan, Eid, holiday and weekend checks for a test window")
    cal.add_argument("start", type=_date)
    cal.add_argument("end", type=_date)
    cal.add_argument("--pre-period", nargs=2, type=_date, metavar=("START", "END"))
    common(cal, alpha=False)
    cal.set_defaults(func=cmd_calendar)

    bd = sub.add_parser("boundaries", help="O'Brien-Fleming boundaries for planned looks")
    bd.add_argument("--looks", type=int, default=5)
    bd.add_argument("--fractions", help="unequal looks as shares of the sample, such as 0.25,0.5,1")
    common(bd)
    bd.set_defaults(func=cmd_boundaries)

    cmp_ = sub.add_parser("compare", help="compare systems scored on the same cases")
    cmp_.add_argument("baseline", help="per-case results of the baseline (csv or jsonl)")
    cmp_.add_argument("systems", nargs="+", help="per-case results of the systems to compare")
    cmp_.add_argument("--metric", required=True, help="column holding the per-case score, such as recall@5 or correct")
    cmp_.add_argument("--id", default="id", help="case id column (default id)")
    cmp_.add_argument("--rounds", type=int, default=10000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--correct", choices=METHODS, default="holm")
    common(cmp_, alpha=False)
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (UsageError, ValueError) as exc:
        print(f"qabtest: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
