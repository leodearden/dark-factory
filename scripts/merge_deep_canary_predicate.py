"""Read-only deep merge-ahead canary health predicate (task eta1's before_done, kind=predicate).

PRD: plans/deep-merge-ahead-prd.md (task eta1; predicate exit-code contract,
docs/task-authoring.md section 6). scripts/merge-deep-canary-predicate.sh is
the entry point gate 3190 runs; it execs this module with every argument.

Usage:
    merge-deep-canary-predicate.sh --runs-db PATH [--runs-db PATH ...]
        [--window-days N] [--deep-fail-rate-max R] [--items-per-verify-min R]
        [--verify-p90-max-secs S] [--p90-ratio-max R] [--min-deep-verifies N]
        [--now EPOCH_SECS]

STORES. Each ``--runs-db`` names one orchestrator runs.db, the append-only
sqlite event store. Every store is opened read-only (a ``mode=ro`` URI) and
nothing is written or migrated. Every leg pools across the stores, and each
store also gets its own ``store <path>: n_deep=<k>`` line. A store named twice
(by resolved path) is refused: it would double every count.

WINDOW. Rows count when ``now - N days < timestamp <= now``. ``--now`` is the
reference instant in epoch seconds and defaults to the current time, so a
``--now`` re-run reproduces a gate's verdict. The bounds are rendered in
EventStore.emit's own UTC ISO-8601 layout, so the lexical comparison against
events.timestamp is exact.

DEEP VERIFY. A merge_verify whose payload ``chain_items`` is an int >= 2: the
verified tree carried two or more chained items.

MINIMUM SAMPLE. When the pooled deep-verify count is below
``--min-deep-verifies`` (default 8, an integer >= 1), no leg is evaluated:
the final line (see STDOUT) is ``insufficient_signal n_deep_verifies=<n>
(< <N>)`` and the exit is 1. Too
little signal never promotes, and is labelled as what it is rather than as a
failed leg.

ITEM-ATTRIBUTED FAILURE. A failed deep verify (tip at T) whose own item was
broken says nothing about chain safety. It is item-attributed when a
merge_verify in the SAME store with the same events.task_id, ``chain_items``
1 and ``passed`` not true is stamped in (T, min(T + 6h, now)]. No event
records a chain's members, so the only known member is the dispatching item:
VerifyRunnerPool.dispatch stamps events.task_id with the task that dispatched
the chain (item #1). Task 7259's planned chain_truncated_reason /
chain_truncated_at name the first item NOT in the chain, so they would not
add membership either. Dispatcher-only attribution can only under-attribute,
which keeps a failure in the gated rate.

LEGS. Provisional thresholds, all overridable; task theta (the week-after
aggressiveness assessment) retunes them against real canary telemetry.
  deep_fail_rate   <= 0.35  (failed deep verifies - item-attributed ones) /
                            (deep verifies - item-attributed ones): an
                            attributed tip leaves the numerator AND the
                            population. The raw failed / deep rate is printed
                            as deep_fail_rate_raw. The replay study saw
                            2/59 ~= 0.034, so this is a loose gate.
  items_per_verify >= 1.3   the merge_finalized ``landed_via_chain`` sum (items
                            a chain landed) / passed deep ROUNDS: a chain must
                            land materially more than one item per round. A
                            passed deep round is a distinct (store, merge_sha)
                            over deep verifies with a str ``merge_sha``,
                            ``attempt`` 0 and ``passed`` true, so a re-dispatch
                            of the same tip and a failed tip, which land
                            nothing, do not dilute it. No passed round -> n/a.
  verify_p90_secs  <= 5400  nearest-rank p90 of the deep verifies' payload
                            ``duration_ms`` (75% of the 7200 s verify timeout);
                            the events.duration_ms column is NULL for this
                            event type. Reported only when the TIMED deep
                            verifies (those carrying a finite duration) reach
                            the minimum sample. An absolute ceiling, kept
                            beside the ratio below.
  verify_p90_ratio <= 1.5   nearest-rank p90 of per-deep-verify ratios: each
                            timed deep verify's duration / the p90 of its OWN
                            (store, runner) non-chain verifies in the window
                            (``chain_items`` 1, a str ``runner``, a finite
                            duration). Runner names are host labels reused
                            across projects, so baselines are keyed by store
                            too. For one runner this equals deep p90 /
                            baseline p90, because scaling preserves order. A
                            deep verify with no (or a zero) same-store,
                            same-runner baseline has no ratio, and the leg is
                            reported only when the ratios reach the minimum
                            sample, so a missing baseline can only HOLD.
A leg with no value FAILs: a canary that never exercised chains must not
silently promote.

STDOUT. The store lines (each with its ``baseline_p90_secs <runner>=<secs>
(n=<k>)`` baselines), then a ``populations: <name>=<value> ...`` line
carrying the pooled populations behind the legs, then either the
insufficient_signal line or the verdict line, always LAST:
``merge-deep-canary window_days=7 n_deep_verifies=N: <leg>=<value>(ok|FAIL
<bound>) ... -> PASS|HOLD``. DeterministicRunner keeps only that final line as
the done-provenance note.

EXIT CODES. The DeterministicRunner decides by exit code only:
  0  canary healthy          -> task done (promote 6->32 proceeds)
  1  unhealthy OR no signal  -> born-at-L2 milestone_check_failed (hold)
  2  fault                   -> never masquerades as a health verdict: a
                                usage or infra fault, or an internal error,
                                whose traceback goes to stderr
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ITEM_ATTRIBUTION_HORIZON = timedelta(hours=6)


def pct(xs: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile of *xs* (any order); None for an empty population."""
    if not xs:
        return None
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


class PredicateFault(Exception):
    """A usage or infra fault: reported on stderr, exit 2, never a verdict."""


@dataclass(frozen=True)
class Window:
    days: float
    as_of: datetime
    lo: str
    hi: str


@dataclass(frozen=True)
class Settings:
    stores: tuple[str, ...]
    window: Window
    deep_fail_rate_max: float
    items_per_verify_min: float
    verify_p90_max_secs: float
    p90_ratio_max: float
    min_deep_verifies: int


@dataclass(frozen=True)
class VerifyRow:
    store: str
    stamped_at: datetime | None
    task_id: str | None
    runner: str | None
    merge_sha: str | None
    passed: bool
    attempt: int | None
    chain_items: int | None
    duration_secs: float | None

    @property
    def is_deep(self) -> bool:
        return self.chain_items is not None and self.chain_items >= 2


@dataclass(frozen=True)
class Baseline:
    p90_secs: float
    n: int


@dataclass(frozen=True)
class StoreSample:
    """One store's window, reduced once to the populations its store line and the pool read."""

    path: str
    deep: tuple[VerifyRow, ...]
    passed_deep_rounds: int
    baselines: Mapping[str, Baseline]
    ratios: tuple[float, ...]
    landed_via_chain: int
    item_attributed_failures: int


@dataclass(frozen=True)
class Pool:
    """The deep-verify populations every leg reads, summed across stores."""

    n_deep: int
    failures: int
    item_attributed_failures: int
    landed_via_chain: int
    passed_deep_rounds: int
    durations: tuple[float, ...]
    ratios: tuple[float, ...]


def _rendered(value: float | None) -> str:
    return 'n/a' if value is None else str(round(value, 4))


@dataclass(frozen=True)
class Leg:
    name: str
    value: float | None
    bound: float
    at_most: bool

    @property
    def passed(self) -> bool:
        if self.value is None:
            return False
        return self.value <= self.bound if self.at_most else self.value >= self.bound

    def render(self) -> str:
        relation = '<=' if self.at_most else '>='
        return f"{self.name}={_rendered(self.value)}({'ok' if self.passed else 'FAIL'} {relation} {self.bound})"


def _finite(flag: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise PredicateFault(f'{flag} must be a finite number; got {raw!r}')
    return value


def _positive_int(flag: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise PredicateFault(f'{flag} must be an integer >= 1; got {raw!r}')
    return value


def _window(raw_days: str, raw_now: str | None) -> Window:
    days = _finite('--window-days', raw_days)
    if days <= 0:
        raise PredicateFault(f'--window-days must be > 0; got {raw_days!r}')
    now_epoch = time.time() if raw_now is None else _finite('--now', raw_now)
    try:
        as_of = datetime.fromtimestamp(now_epoch, UTC)
        lo = (as_of - timedelta(days=days)).isoformat(timespec='microseconds')
        hi = as_of.isoformat(timespec='microseconds')
    except (OverflowError, OSError, ValueError) as exc:
        raise PredicateFault(f'--now/--window-days put the window out of range: {exc}') from exc
    return Window(days=days, as_of=as_of, lo=lo, hi=hi)


def _stores(paths: Sequence[str]) -> tuple[str, ...]:
    seen: set[Path] = set()
    for path in paths:
        if not Path(path).is_file():
            raise PredicateFault(f'--runs-db not found: {path}')
        resolved = Path(path).resolve()
        if resolved in seen:
            raise PredicateFault(f'--runs-db names the same store twice: {path}')
        seen.add(resolved)
    return tuple(paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='merge-deep-canary-predicate',
        description='Read-only deep merge-ahead canary health predicate.',
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument('--runs-db', action='append', required=True, metavar='PATH')
    parser.add_argument('--window-days', default='7')
    parser.add_argument('--deep-fail-rate-max', default='0.35')
    parser.add_argument('--items-per-verify-min', default='1.3')
    parser.add_argument('--verify-p90-max-secs', default='5400')
    parser.add_argument('--p90-ratio-max', default='1.5')
    parser.add_argument('--min-deep-verifies', default='8')
    parser.add_argument('--now', default=None, metavar='EPOCH_SECS')
    return parser


def _settings(argv: Sequence[str]) -> Settings:
    args = _parser().parse_args(argv)
    deep_fail_rate_max = _finite('--deep-fail-rate-max', args.deep_fail_rate_max)
    items_per_verify_min = _finite('--items-per-verify-min', args.items_per_verify_min)
    verify_p90_max_secs = _finite('--verify-p90-max-secs', args.verify_p90_max_secs)
    p90_ratio_max = _finite('--p90-ratio-max', args.p90_ratio_max)
    min_deep_verifies = _positive_int('--min-deep-verifies', args.min_deep_verifies)
    window = _window(args.window_days, args.now)
    return Settings(
        stores=_stores(args.runs_db),
        window=window,
        deep_fail_rate_max=deep_fail_rate_max,
        items_per_verify_min=items_per_verify_min,
        verify_p90_max_secs=verify_p90_max_secs,
        p90_ratio_max=p90_ratio_max,
        min_deep_verifies=min_deep_verifies,
    )


def _payload(raw: object) -> dict[str, object]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _int_field(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) else None


def _str_field(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _payload_secs(data: dict[str, object]) -> float | None:
    ms = data.get('duration_ms')
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms):
        return None
    return ms / 1000.0


def _instant(timestamp: object) -> datetime | None:
    """An events.timestamp as an aware datetime; None when it is not one."""
    try:
        parsed = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
    except ValueError:
        return None
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def _verify_row(store: str, timestamp: object, task_id: object, raw: object) -> VerifyRow:
    data = _payload(raw)
    return VerifyRow(
        store=store,
        stamped_at=_instant(timestamp),
        task_id=None if task_id is None else str(task_id),
        runner=_str_field(data, 'runner'),
        merge_sha=_str_field(data, 'merge_sha'),
        passed=data.get('passed') is True,
        attempt=_int_field(data, 'attempt'),
        chain_items=_int_field(data, 'chain_items'),
        duration_secs=_payload_secs(data),
    )


def _landed_via_chain(raw: object) -> int:
    landed = _int_field(_payload(raw), 'landed_via_chain')
    return landed if landed is not None and landed >= 1 else 0


def _is_item_attributed(tip: VerifyRow, solo_failures: Sequence[VerifyRow], as_of: datetime) -> bool:
    if tip.task_id is None or tip.stamped_at is None:
        return False
    horizon_end = min(tip.stamped_at + ITEM_ATTRIBUTION_HORIZON, as_of)
    return any(
        solo.task_id == tip.task_id and solo.stamped_at is not None
        and tip.stamped_at < solo.stamped_at <= horizon_end
        for solo in solo_failures
    )


def _item_attributed_failures(verifies: Sequence[VerifyRow], as_of: datetime) -> int:
    solo_failures = [row for row in verifies if row.chain_items == 1 and not row.passed]
    failed_tips = [row for row in verifies if row.is_deep and not row.passed]
    return sum(1 for tip in failed_tips if _is_item_attributed(tip, solo_failures, as_of))


def _passed_deep_rounds(deep: Sequence[VerifyRow]) -> int:
    return len({row.merge_sha for row in deep if row.merge_sha is not None and row.attempt == 0 and row.passed})


def _baselines(verifies: Sequence[VerifyRow]) -> dict[str, Baseline]:
    """Each runner's non-chain verify p90 in one store, sorted by runner."""
    by_runner: dict[str, list[float]] = {}
    for row in verifies:
        if row.chain_items == 1 and row.runner is not None and row.duration_secs is not None:
            by_runner.setdefault(row.runner, []).append(row.duration_secs)
    return {
        runner: Baseline(p90_secs=p90, n=len(durations))
        for runner, durations in sorted(by_runner.items())
        if (p90 := pct(durations, 0.90)) is not None
    }


def _ratios(deep: Sequence[VerifyRow], baselines: Mapping[str, Baseline]) -> tuple[float, ...]:
    """Each timed deep verify's duration over its runner's baseline p90, when that baseline is non-zero."""
    ratios = []
    for row in deep:
        baseline = baselines.get(row.runner) if row.runner is not None else None
        if row.duration_secs is not None and baseline is not None and baseline.p90_secs > 0:
            ratios.append(row.duration_secs / baseline.p90_secs)
    return tuple(ratios)


def _sample(path: str, verifies: Sequence[VerifyRow], landed_via_chain: int, as_of: datetime) -> StoreSample:
    deep = tuple(row for row in verifies if row.is_deep)
    baselines = _baselines(verifies)
    return StoreSample(
        path=path,
        deep=deep,
        passed_deep_rounds=_passed_deep_rounds(deep),
        baselines=baselines,
        ratios=_ratios(deep, baselines),
        landed_via_chain=landed_via_chain,
        item_attributed_failures=_item_attributed_failures(verifies, as_of),
    )


def _window_rows(conn: sqlite3.Connection, event_type: str, window: Window) -> list[tuple[object, ...]]:
    return conn.execute(
        'SELECT timestamp, task_id, data FROM events '
        'WHERE event_type=? AND timestamp > ? AND timestamp <= ? ORDER BY id',
        (event_type, window.lo, window.hi),
    ).fetchall()


def _load_store(path: str, window: Window) -> StoreSample:
    uri = f'{Path(path).resolve().as_uri()}?mode=ro'
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            verify_events = _window_rows(conn, 'merge_verify', window)
            finalized = _window_rows(conn, 'merge_finalized', window)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise PredicateFault(f'cannot read runs.db {path}: {exc}') from exc
    return _sample(
        path,
        [_verify_row(path, ts, task_id, data) for ts, task_id, data in verify_events],
        landed_via_chain=sum(_landed_via_chain(data) for _, _, data in finalized),
        as_of=window.as_of,
    )


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sampled_p90(xs: Sequence[float], minimum: int) -> float | None:
    return pct(xs, 0.90) if len(xs) >= minimum else None


def _pool(samples: Sequence[StoreSample]) -> Pool:
    deep = [row for sample in samples for row in sample.deep]
    return Pool(
        n_deep=len(deep),
        failures=sum(1 for row in deep if not row.passed),
        item_attributed_failures=sum(sample.item_attributed_failures for sample in samples),
        landed_via_chain=sum(sample.landed_via_chain for sample in samples),
        passed_deep_rounds=sum(sample.passed_deep_rounds for sample in samples),
        durations=tuple(row.duration_secs for row in deep if row.duration_secs is not None),
        ratios=tuple(ratio for sample in samples for ratio in sample.ratios),
    )


def _legs(pool: Pool, settings: Settings) -> list[Leg]:
    attributed = pool.item_attributed_failures
    fail_rate = _ratio(pool.failures - attributed, pool.n_deep - attributed)
    items = _ratio(pool.landed_via_chain, pool.passed_deep_rounds)
    p90 = _sampled_p90(pool.durations, settings.min_deep_verifies)
    p90_ratio = _sampled_p90(pool.ratios, settings.min_deep_verifies)
    return [
        Leg('deep_fail_rate', fail_rate, settings.deep_fail_rate_max, at_most=True),
        Leg('items_per_verify', items, settings.items_per_verify_min, at_most=False),
        Leg('verify_p90_secs', p90, settings.verify_p90_max_secs, at_most=True),
        Leg('verify_p90_ratio', p90_ratio, settings.p90_ratio_max, at_most=True),
    ]


def _store_line(sample: StoreSample) -> str:
    baselines = ' '.join(
        f'{runner}={_rendered(baseline.p90_secs)}(n={baseline.n})' for runner, baseline in sample.baselines.items()
    )
    return (
        f'store {sample.path}: n_deep={len(sample.deep)} passed_deep_rounds={sample.passed_deep_rounds} '
        f'item_attributed_failures={sample.item_attributed_failures} '
        f'baseline_p90_secs {baselines or "none"}'
    )


def _populations_line(pool: Pool) -> str:
    return (
        f'populations: deep_fail_rate_raw={_rendered(_ratio(pool.failures, pool.n_deep))} '
        f'deep_failures={pool.failures} item_attributed_failures={pool.item_attributed_failures} '
        f'landed_via_chain={pool.landed_via_chain} passed_deep_rounds={pool.passed_deep_rounds} '
        f'timed_deep_verifies={len(pool.durations)} ratio_deep_verifies={len(pool.ratios)}'
    )


def _verdict_line(n_deep: int, legs: Sequence[Leg], window: Window) -> str:
    summary = ' '.join(leg.render() for leg in legs)
    outcome = 'PASS' if all(leg.passed for leg in legs) else 'HOLD'
    return f'merge-deep-canary window_days={window.days:g} n_deep_verifies={n_deep}: {summary} -> {outcome}'


def main(argv: Sequence[str]) -> int:
    try:
        settings = _settings(argv)
        samples = [_load_store(path, settings.window) for path in settings.stores]
    except PredicateFault as fault:
        print(f'predicate: {fault}', file=sys.stderr)
        return 2
    pool = _pool(samples)
    for sample in samples:
        print(_store_line(sample))
    print(_populations_line(pool))
    if pool.n_deep < settings.min_deep_verifies:
        print(f'insufficient_signal n_deep_verifies={pool.n_deep} (< {settings.min_deep_verifies})')
        return 1
    legs = _legs(pool, settings)
    print(_verdict_line(pool.n_deep, legs, settings.window))
    return 0 if all(leg.passed for leg in legs) else 1


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        sys.exit(2)
