#!/usr/bin/env python3
"""Census the verify-summary corpus, so a module's budget is measured not guessed.

Ruling D17 (task 3353) deliverable 2. Every verify attempt leaves an
`attempt-N[.<prefix>].summary.json` behind — in the worktree that ran it, and
in the durable archive under `data/verify-logs/<task_id>/`. Together those ARE
the duration distribution a per-module `verify_command_timeout_secs` should be
derived from, which is why this task does not run a measurement suite: the
production corpus already is the measurement, and (since deliverable 1) each
record carries the host load its command ran under.

STRICTLY READ-ONLY. This script opens every file `mode='r'`; it writes nothing,
files nothing, and emits no events. It is safe to run against a live tree while
the fleet is verifying.

WHAT THIS DOES NOT DO: it asserts no numeric target. It prints what it
measures, and every section is labelled with the concrete window it was
computed over — the same stance `scripts/merge_lane_throughput.py` takes, and
for the same reason. `--window 14d` resolves against the clock, so it covers a
different fortnight every day and cannot reproduce a table whose header carries
a fixed date; `--window <iso>..<iso>` is the mechanism for that. The GATE that
holds a budget up lives in tests/scripts/test_module_verify_budgets.py; do not
read this report as that gate.

THREE FACTS COME ONLY FROM THE PATH. A summary.json carries no task id, no
module prefix and no role, so `parse_record_path` recovers them from where the
file sits. The two corpora spell the filename differently and the difference is
easy to get wrong: the worktree side ends `.summary.json`, while the ARCHIVE
side carries its stamp AFTER that word — `.summary-20260914T123016_283575Z.json`
— so a `*.summary.json` glob selects zero archive records. D17's own text globs
that way; this script does not.

It also does not import orchestrator config code. It has to run read-only
against ANY project root, including one whose `.venv` is absent or whose
interpreter differs from the caller's, which is the normal state of a task
worktree — so `orchestrator.yaml` is read as plain YAML. The cost is that the
census cannot see config-layer precedence; it reports the raw values and says
so.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# `attempt-<N>[.<infix>].summary.json` — the live worktree spelling.
_WORKTREE_RE = re.compile(r'^attempt-(\d+)(?:\.(.+))?\.summary\.json$')
# `attempt-<N>[.<infix>].summary-<STAMP>.json` — the archive spelling. Both
# live stamp resolutions are accepted: `20260914T123016_283575Z` (microseconds)
# and the older `20260816T080255Z`.
_ARCHIVE_RE = re.compile(
    r'^attempt-(\d+)(?:\.(.+))?\.summary-(\d{8}T\d{6}(?:_\d+)?Z)\.json$',
)

# A non-task lane is `_<kind>-<hex>`. The two kinds D17 names get the ruling's
# vocabulary; any other kind is reported UNDER ITS OWN NAME rather than folded
# into a neighbour.
_LANE_RE = re.compile(r'^_([a-z]+)-')
_RULED_LANE_ROLES = {'mainprobe': 'probe', 'merge': 'merge'}

_WORKTREE_ANCHOR = '.worktrees'
_ARCHIVE_ANCHOR = 'verify-logs'

WORKTREE_GLOB = f'{_WORKTREE_ANCHOR}/*/.task/verify/*.summary.json'
ARCHIVE_GLOB = 'data/verify-logs/*/*.summary-*.json'

_RELATIVE_RE = re.compile(r'^(\d+)d$')
_RANGE_SEP = '..'

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WINDOW = '14d'


@dataclass(frozen=True)
class RecordPath:
    """The facts a summary.json's LOCATION carries, and nothing inferred.

    ``module_prefix`` is the filename infix VERBATIM, which is the sanitised
    form ``verify._make_infix`` writes (``/`` and space both become ``_``).
    That sanitisation has no inverse — ``tests/scripts`` and a literal
    ``tests_scripts`` both land on ``tests_scripts`` — so this type does not
    attempt one. Selection sanitises the REQUESTED prefix through the same rule
    and compares forward, which is unambiguous; see ``sanitise_prefix``.

    ``role`` is ``None`` when it is not knowable, which is every archive record
    (the archive carries both task-path and merge-path legs, and the path does
    not say which) and any unrecognised worktree lane name. ``None`` is the
    honest reading: defaulting it to ``'task'`` would fold merge legs, which
    run a different breadth on a different budget, into a task distribution and
    call the result measured.
    """

    path: Path
    task_id: str
    attempt: int
    module_prefix: str | None
    role: str | None
    corpus: str
    archived_at: str | None


def sanitise_prefix(prefix: str) -> str:
    """Map a module prefix to the filename infix ``verify._make_infix`` writes.

    Mirrors that function's rule rather than inventing a second convention, so
    ``--prefix tests/scripts`` matches the ``.tests_scripts`` records the fleet
    actually writes.
    """
    return prefix.replace('/', '_').replace(' ', '_')


def _lane_role(lane: str) -> str | None:
    """Classify a ``.worktrees/<lane>`` directory name into a verify role.

    Measured on the tree (2026-09-14): five lane classes exist — 774 bare
    numeric task ids, plus ``_mainsweep-``, ``_mainprobe-``, ``_merge-`` and
    ``_offline-``. A fall-through calling every non-merge, non-probe lane a
    ``task`` would file a sweep or offline-lane verify into the very
    distribution a budget is derived from, so an unruled ``_<kind>-`` lane is
    reported under ``<kind>`` instead (esc-3353-14). Anything matching neither
    shape is ``None`` — unknown, never guessed.
    """
    if lane.isdigit():
        return 'task'
    lane_match = _LANE_RE.match(lane)
    if lane_match is None:
        return None
    kind = lane_match.group(1)
    return _RULED_LANE_ROLES.get(kind, kind)


def parse_record_path(path: Path) -> RecordPath | None:
    """Recover the path-borne facts for *path*, or ``None`` if it is no record.

    ``None`` covers every non-record under these trees — the per-leg ``.log``
    files sitting beside each summary, a partial ``.tmp``, a marker file, and a
    summary that is not under either corpus anchor at all. Returning ``None``
    rather than raising is what lets the walker stay total: each rejection is
    counted with its reason by the caller.
    """
    parts = path.parts
    name = path.name

    archive_match = _ARCHIVE_RE.match(name)
    if archive_match and _ARCHIVE_ANCHOR in parts:
        attempt, infix, stamp = archive_match.groups()
        return RecordPath(
            path=path,
            task_id=path.parent.name,
            attempt=int(attempt),
            module_prefix=infix,
            # Not knowable from an archive path — see RecordPath.role.
            role=None,
            corpus='archive',
            archived_at=stamp,
        )

    worktree_match = _WORKTREE_RE.match(name)
    if worktree_match and _WORKTREE_ANCHOR in parts:
        attempt, infix = worktree_match.groups()
        lane = parts[parts.index(_WORKTREE_ANCHOR) + 1]
        return RecordPath(
            path=path,
            task_id=lane,
            attempt=int(attempt),
            module_prefix=infix,
            role=_lane_role(lane),
            corpus='worktree',
            archived_at=None,
        )

    return None


@dataclass(frozen=True)
class Record:
    """One loaded summary.json: where it sat, and what it said."""

    where: RecordPath
    payload: dict


@dataclass(frozen=True)
class Skip:
    """A file the globs selected and the walker could not use, with the reason.

    Carrying the reason BY VALUE is what makes the corpus reconcilable: the
    report prints these counts beside ``n``, so a reader can see that
    ``len(records) + len(skipped)`` accounts for every selected file rather
    than trusting that nothing was lost.
    """

    path: Path
    reason: str


@dataclass(frozen=True)
class Corpus:
    """Everything both globs found under the given roots, partitioned.

    Total by construction: every selected file lands in exactly one of the two
    tuples. A census that quietly shrinks its own corpus is the precise failure
    this deliverable exists to end — a budget derived from a distribution that
    dropped the records it could not read is not a measurement, and nothing in
    its output would say so.
    """

    records: tuple[Record, ...]
    skipped: tuple[Skip, ...]


def _read_payload(path: Path) -> tuple[dict | None, str | None]:
    """Load *path* as a JSON object, or return the reason it is unusable.

    Three distinguishable failures, because they mean different things to an
    operator: the file could not be READ at all (a directory in its place, a
    permission problem, a truncated mid-write read), it read but is not JSON (a
    partial write — the common one, since a summary is written while the fleet
    runs), or it is JSON but not an object (a shape this census cannot use).
    """
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None, 'unreadable'
    try:
        payload = json.loads(text)
    except ValueError:
        return None, 'not_json'
    if not isinstance(payload, dict):
        return None, 'not_an_object'
    return payload, None


def load_records(roots: Iterable[Path]) -> Corpus:
    """Walk both corpora under every *root* and load every summary record.

    The two globs are deliberately separate constants rather than one pattern:
    the worktree side ends ``.summary.json`` while the archive side carries its
    stamp AFTER that word, so no single glob selects both and a ``*.summary.json``
    glob selects zero archive records. Neither glob admits the per-leg ``.log``
    files sitting beside each summary, so those are not skips either — they were
    never selected.

    A root that does not exist contributes nothing and is not an error: the
    archive lives only in a project's main checkout, so a census run from a
    task worktree legitimately finds one corpus and not the other.
    """
    records: list[Record] = []
    skipped: list[Skip] = []
    for root in roots:
        for glob in (WORKTREE_GLOB, ARCHIVE_GLOB):
            for path in sorted(root.glob(glob)):
                where = parse_record_path(path)
                if where is None:
                    skipped.append(Skip(path, 'unparsed_path'))
                    continue
                payload, reason = _read_payload(path)
                if payload is None:
                    skipped.append(Skip(path, reason or 'unreadable'))
                    continue
                records.append(Record(where=where, payload=payload))
    return Corpus(records=tuple(records), skipped=tuple(skipped))


@dataclass(frozen=True)
class Leg:
    """One selected ``commands[]`` entry — the full-suite run of one attempt."""

    where: RecordPath
    label: str
    cmd: str
    rc: int
    timed_out: bool
    started_at: str
    duration_secs: float
    load: dict | None


@dataclass(frozen=True)
class Selection:
    """The selected legs, and why every other entry was not selected.

    ``rejected`` is a reason -> count mapping so ``n`` is always reconcilable
    against the corpus: ``len(legs) + sum(rejected.values())`` is the number of
    ``commands[]`` entries considered. A selector that reported only its
    positives could not distinguish "this module rarely runs the full suite"
    from "the shape filter stopped matching" — and those call for opposite
    responses.
    """

    legs: tuple[Leg, ...]
    rejected: dict[str, int]


def read_module_test_command(root: Path, prefix: str) -> str | None:
    """The ``test_command`` a module declares, read from its own yaml.

    Read as PLAIN YAML rather than through ``OrchestratorConfig``: this script
    has to run read-only against any project root, including one whose
    ``.venv`` is absent or whose interpreter differs from the caller's, which
    CLAUDE.md documents as the normal state of a task worktree. The cost is
    that the census cannot see config-layer precedence — it reports the raw
    declared value, and says so.

    ``None`` — never a fallback — for an absent file, an unparseable one, or a
    file with no ``test_command``. A guessed default here would be compared
    against every record in the corpus and reject them all, reporting ``n=0``,
    which reads exactly like "this module never ran the full suite".
    """
    path = root / prefix / 'orchestrator.yaml'
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    command = data.get('test_command')
    return command if isinstance(command, str) else None


def _normalise_command(cmd: str) -> str:
    """Collapse whitespace, and nothing else.

    The comparison is deliberately exact-modulo-whitespace. Anything fuzzier
    would readmit precisely the runs the filter exists to exclude: a
    file-scoped `pytest tests/test_foo.py`, or a `-k`/`--lf`-narrowed form,
    whose durations are not comparable to a full-suite run's. Whitespace alone
    is tolerated because a command that has been through a parse/render
    round-trip can differ in spacing while being the same invocation.
    """
    return ' '.join(cmd.split())


def select_full_suite_legs(
    corpus: Corpus,
    *,
    root: Path,
    prefix: str,
    label: str = 'test',
    role: str | None = None,
) -> Selection:
    """Select the ``label`` legs that ran *prefix*'s declared full suite.

    Reads each record's ``commands[]`` array, NEVER the top level. That
    distinction is load-bearing: ``_build_summary_payload`` fills the top-level
    rc/cmd/started_at/duration_secs from "the loudest raw exit code" (a
    negative rc — a signal kill — sorting above every non-negative one), so on
    an attempt whose lint leg was killed the top level describes the LINT leg.
    A per-module duration census reading it would report a 4-second lint
    command as the suite.

    ``role=None`` means "do not filter by role", which is the only usable
    default for the archive corpus, where the role is not knowable from the
    path at all (see ``RecordPath.role``).

    Every considered entry is either selected or counted under a reason, so the
    caller can always reconcile ``n`` against the corpus.
    """
    expected = read_module_test_command(root, prefix)
    wanted_infix = sanitise_prefix(prefix)
    legs: list[Leg] = []
    rejected: Counter[str] = Counter()

    for record in corpus.records:
        if record.where.module_prefix != wanted_infix:
            rejected['prefix_mismatch'] += 1
            continue
        if role is not None and record.where.role != role:
            rejected['role_mismatch'] += 1
            continue
        entries = record.payload.get('commands')
        if not isinstance(entries, list):
            rejected['no_commands_array'] += 1
            continue
        for entry in entries:
            reason = _reject_reason(entry, expected, label)
            if reason is not None:
                rejected[reason] += 1
                continue
            legs.append(
                Leg(
                    where=record.where,
                    label=entry['label'],
                    cmd=entry['cmd'],
                    rc=entry['rc'],
                    timed_out=bool(entry.get('timed_out')),
                    started_at=entry.get('started_at') or '',
                    duration_secs=float(entry['duration_secs']),
                    load=entry.get('load'),
                ),
            )

    return Selection(legs=tuple(legs), rejected=dict(rejected))


def _reject_reason(entry: object, expected: str | None, label: str) -> str | None:
    """Why *entry* is not a full-suite run of the wanted leg, or ``None``.

    Ordered cheapest-and-most-specific first, so a rejection is attributed to
    the most informative reason available rather than to whichever check
    happened to run first. ``segmented`` is kept distinct from
    ``command_mismatch`` because it is a different fact about the corpus: the
    command matched, but it ran as a sequence of separately-timed subprojects,
    so the duration describes a different execution topology of the same chain.
    """
    if not isinstance(entry, dict):
        return 'malformed_entry'
    if entry.get('label') != label:
        return 'label_mismatch'
    if entry.get('cmd') is None:
        return 'no_cmd'
    if entry.get('segments'):
        return 'segmented'
    if expected is None:
        return 'no_declared_command'
    if _normalise_command(str(entry['cmd'])) != _normalise_command(expected):
        return 'command_mismatch'
    if not isinstance(entry.get('duration_secs'), (int, float)):
        return 'no_duration'
    return None


# ---------------------------------------------------------------------------
# Numeric and temporal helpers. Shape mirrored from
# scripts/merge_lane_throughput.py (parse_window / _percentile / _series) —
# the same interpolation, the same None-on-empty, the same injected clock.
# COPIED, not imported: scripts/ modules do not import one another here, and
# that sibling is a 1899-line runs.db report whose sections are irrelevant.
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Return the *pct*-th percentile of *values*, or ``None`` when empty.

    Linear interpolation between the two nearest order statistics (the
    ``numpy.percentile`` default) on the ascending sort: with
    ``k = (n - 1) * pct / 100``, the result is
    ``s[floor(k)] + (k - floor(k)) * (s[ceil(k)] - s[floor(k)])``.

    ``None`` — never ``0.0`` — for an empty series. A ``0.0`` p50 would render
    "no full-suite run in this window" as an INSTANTANEOUS suite, which in a
    budget report is worse than merely wrong: it invites a reader to conclude
    the suite got faster on evidence that says nothing at all.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (k - lo) * (ordered[hi] - ordered[lo]))


def _series(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a duration series as n/p50/p90/max, the three stats ``None``
    together when the series is empty."""
    return {
        'n': len(values),
        'p50': _percentile(values, 50),
        'p90': _percentile(values, 90),
        'max': max(values) if values else None,
    }


def parse_instant(stamp: str) -> datetime | None:
    """Parse an ISO-8601 *stamp* to a tz-aware UTC instant, or ``None``.

    A naive stamp is read as UTC (every writer in this corpus emits UTC). An
    unparseable one is ``None`` rather than a substituted default: defaulting
    to the current clock would invent a run inside whatever window is being
    reported, in the one artifact whose job is to say what actually ran.
    """
    try:
        parsed = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def day_bucket(stamp: str) -> date | None:
    """The UTC calendar day *stamp* falls in, or ``None`` if unparseable.

    Normalised to UTC BEFORE taking the date. Bucketing on the string's leading
    ten characters would file ``2026-09-13T23:30:00-04:00`` under the 13th when
    it is the 14th in UTC — sliding runs across a day boundary and smearing the
    trailing-window edge a derived floor depends on.
    """
    instant = parse_instant(stamp)
    return None if instant is None else instant.date()


def parse_window(spec: str, now: datetime) -> tuple[datetime, datetime]:
    """Resolve a ``--window`` *spec* against an injected *now* into ``(lo, hi)``.

    Two forms::

        <N>d          -> (now - N days, now)
        <iso>..<iso>  -> exactly those two instants

    Both endpoints are tz-aware UTC. *now* is a parameter, never read inside,
    so every caller and every test fixes the clock explicitly — which is what
    makes a trailing window's boundary deterministic.

    The dated form is the mechanism for a report whose header carries a fixed
    date: ``14d`` covers a different fortnight every day and so cannot
    reproduce one.

    Raises :class:`argparse.ArgumentTypeError`, echoing the offending spec, for
    an empty, malformed, zero-length or reversed window. A reversed range is
    rejected rather than silently swapped: it far more often means the operator
    pasted the bounds backwards than that they wanted that window.
    """
    relative = _RELATIVE_RE.match(spec)
    if relative:
        days = int(relative.group(1))
        if days <= 0:
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: window must span at least one day.',
            )
        return (now - timedelta(days=days), now)

    if _RANGE_SEP in spec:
        parts = spec.split(_RANGE_SEP)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: the dated form takes exactly two '
                f'ISO-8601 endpoints separated by "..".',
            )
        lo = parse_instant(parts[0].strip())
        hi = parse_instant(parts[1].strip())
        if lo is None or hi is None:
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: both endpoints must be ISO-8601 instants.',
            )
        if lo >= hi:
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: start {lo.isoformat()} is not before '
                f'end {hi.isoformat()}.',
            )
        return (lo, hi)

    raise argparse.ArgumentTypeError(
        f'bad --window {spec!r}: expected "<N>d" or "<iso>..<iso>".',
    )


def within_window(
    legs: Iterable[Leg], window: tuple[datetime, datetime],
) -> tuple[Leg, ...]:
    """The legs whose ``started_at`` falls in ``[lo, hi)``.

    A leg whose timestamp does not parse is EXCLUDED: it cannot be placed in
    the window, so it cannot honestly be counted in it. The count still
    reconciles, because the selector already reported how many legs it found.
    """
    lo, hi = window
    kept = []
    for leg in legs:
        instant = parse_instant(leg.started_at)
        if instant is not None and lo <= instant < hi:
            kept.append(leg)
    return tuple(kept)


def summarise_legs(legs: Iterable[Leg]) -> dict[str, Any]:
    """Partition *legs* into clean durations, timeouts and failures.

    A TIMED-OUT leg's duration is the BUDGET, not the suite, so folding it into
    the percentiles would measure the ceiling and call it the workload — and a
    budget then derived from that distribution would be derived from itself,
    the one circularity a census must not have. A FAILED leg stopped at its
    first failure, so its duration is not the suite's either; it is a different
    fact, and counted apart.

    The two buckets are exclusive — a timeout carries a non-zero rc too, and if
    both claimed it ``n + timed_out + failed`` would stop reconciling against
    the number of legs.
    """
    durations: list[float] = []
    timed_out = 0
    failed = 0
    for leg in legs:
        if leg.timed_out:
            timed_out += 1
        elif leg.rc != 0:
            failed += 1
        else:
            durations.append(leg.duration_secs)
    return {
        'durations': _series(durations),
        'timed_out': timed_out,
        'failed': failed,
    }


def by_day(legs: Iterable[Leg]) -> dict[str, dict[str, Any]]:
    """Summarise *legs* per UTC calendar day, keyed by ISO date string.

    Keyed by string rather than ``date`` so the report dict is JSON-native with
    no serialisation step — the same flatness rule the summary schema itself
    follows.
    """
    buckets: dict[str, list[Leg]] = {}
    for leg in legs:
        day = day_bucket(leg.started_at)
        if day is None:
            continue
        buckets.setdefault(day.isoformat(), []).append(leg)
    return {day: summarise_legs(group) for day, group in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Load regimes. Deliverable 1 puts the pressure reading INSIDE the record being
# censused, which is why D1 is sequenced before D2 and why this does NOT join
# to the sampler's data/load-samples.db: a timestamp-range join into that
# long-format store has no precedent in the repo, would be new design surface
# with its own window-alignment and missing-sample semantics, and is
# unnecessary by construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PsiBand:
    """One half-open ``[lo, hi)`` band of host CPU ``some avg10`` pressure.

    ``why`` is not decoration: an edge nobody can argue with is an edge nobody
    can correct, and these are the boundaries a budget derivation will be read
    against.
    """

    name: str
    lo: float
    hi: float
    why: str


PSI_BANDS: tuple[PsiBand, ...] = (
    PsiBand(
        'idle', 0.0, 5.0,
        'the quiet host the old "measure on an idle box" advice asked for; '
        'below ~5% some-pressure nothing is waiting on CPU for long',
    ),
    PsiBand(
        'light', 5.0, 25.0,
        'one or two co-resident verifies — the fleet at low occupancy',
    ),
    PsiBand(
        'moderate', 25.0, 50.0,
        'the band the dispatch-admission gate is tuned around, so the '
        'common steady state of a busy fleet',
    ),
    PsiBand(
        'heavy', 50.0, 100.0,
        'sustained contention; a duration here is as much a statement about '
        'the host as about the suite',
    ),
    PsiBand(
        'saturated', 100.0, float('inf'),
        'pressure at or past the full-stall ceiling — kept as its own band '
        'rather than folded into heavy, because a budget derived from these '
        'runs is measuring the host',
    ),
)

UNSTAMPED = 'unstamped'
"""The bucket for a leg whose host load is NOT KNOWN.

Two populations land here and both belong: a record written before the load
stamp existed, and one whose PSI read degraded to null. Neither may be filed in
a zero-pressure band — that would put the busiest historical runs in the IDLE
band and then invite the conclusion that the suite is slow even on a quiet
host. Today this bucket is the whole corpus.
"""


def band_for(cpu_some10: float) -> str:
    """The band *cpu_some10* falls in. The bands tile ``[0, inf)``."""
    for band in PSI_BANDS:
        if band.lo <= cpu_some10 < band.hi:
            return band.name
    return PSI_BANDS[-1].name


def _start_pressure(load: dict | None) -> float | None:
    """The host CPU pressure a leg STARTED under, or ``None`` if not knowable.

    ``None`` covers an absent record, a malformed one, and a null reading
    inside a present one — the last being ``_load_sample``'s own
    "component degraded" encoding, which means "we could not tell" and must
    never be read as 0.0.
    """
    if not isinstance(load, dict):
        return None
    start = load.get('start')
    if not isinstance(start, dict):
        return None
    value = start.get('cpu_some10')
    return float(value) if isinstance(value, (int, float)) else None


def by_load_band(legs: Iterable[Leg]) -> dict[str, dict[str, Any]]:
    """Summarise *legs* per load band, with ``unstamped`` as its own row.

    Every band is present even when empty, so a reader can see that a band was
    measured and found empty rather than guessing whether it was reported at
    all. Banded counts plus ``unstamped`` reconcile against the selected ``n``.
    """
    buckets: dict[str, list[Leg]] = {band.name: [] for band in PSI_BANDS}
    buckets[UNSTAMPED] = []
    for leg in legs:
        pressure = _start_pressure(leg.load)
        key = UNSTAMPED if pressure is None else band_for(pressure)
        buckets[key].append(leg)
    return {name: summarise_legs(group) for name, group in buckets.items()}


def cold_separability(legs: Iterable[Leg]) -> dict[str, Any]:
    """State whether cold runs can be separated from warm ones. They cannot.

    A summary.json carries no is-cold flag. The only available inference —
    "attempt-1 in a worktree with no prior verify dir is cold" — is a guess:
    an attempt-1 record is also what a worktree that was RESET and re-verified
    warm leaves behind, and the marker that would settle it
    (``.task/verify_warmed``) is not part of the record and does not survive
    into the archive. Acting on the guess would mix warm reruns into a cold
    distribution and then freeze the result into a budget as if measured.

    D17 rules the fallback for exactly this case: label the cold value INTERIM
    with its basis. So this returns a FINDING — counts, and the reason — and
    deliberately contains NO duration series. A series here would be a cold
    distribution built on a guess, indistinguishable a month later from one
    that was measured.
    """
    legs = tuple(legs)
    first = sum(1 for leg in legs if leg.where.attempt == 1)
    return {
        'cold_separable': False,
        'first_attempt_records': first,
        'later_attempt_records': len(legs) - first,
        'basis': (
            'a summary.json carries no is-cold flag, and the only available '
            'inference (attempt-1 with no prior verify dir) cannot tell a cold '
            'first verify from a warm re-verify of a reset worktree: the '
            '.task/verify_warmed marker is not part of the record and does not '
            'survive into the archive. first_attempt_records is the count a '
            'cold inference would have claimed, reported so the size of the '
            'guess is visible rather than the guess being made.'
        ),
    }


_MERGE_BUDGET_KEY = 'merge_verify_cold_command_timeout_secs'
_MERGE_BUDGET_SOURCES = (
    Path('dark-factory-orchestrator.yaml'),
    Path('orchestrator/src/orchestrator/defaults.yaml'),
)


def merge_gate_budget(root: Path) -> dict[str, Any]:
    """Read the merge gate's cold budget from the config chain. REPORT ONLY.

    Resolved in the layering order an operator would read — the project's own
    top-level config, then the shipped defaults — as plain YAML, because this
    script must not import orchestrator config code (see the module docstring).
    The cost is that this cannot see full config-layer precedence; it reports
    the raw value and names the file it came from, so a reader can check.

    The merge gate is ALWAYS cold: it verifies a freshly-created worktree every
    time. So its budget is a different question from the task lane's, and this
    census does not answer it. Saying so in the report is what stops a reader
    applying a warm-derived figure to a path that is strictly costlier.
    """
    for relative in _MERGE_BUDGET_SOURCES:
        path = root / relative
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and _MERGE_BUDGET_KEY in data:
            return {
                _MERGE_BUDGET_KEY: data[_MERGE_BUDGET_KEY],
                'source': str(path),
                'always_cold': True,
                'note': (
                    'The merge gate verifies a freshly-created worktree every '
                    'time, so it is always COLD and its budget is a different '
                    'question from the task lane figures above. Reported for '
                    'context; this census changes nothing about it.'
                ),
            }
    return {
        _MERGE_BUDGET_KEY: None,
        'source': None,
        'always_cold': True,
        'note': (
            f'No {_MERGE_BUDGET_KEY} found in '
            f'{", ".join(str(s) for s in _MERGE_BUDGET_SOURCES)} under this '
            f'root. Reported as null rather than defaulted: a guessed budget '
            f'here reads exactly like a measured one. The merge gate is '
            f'always COLD either way, and this census changes nothing about it.'
        ),
    }


# ---------------------------------------------------------------------------
# CLI. One report dict, two renderings — the build_report / render split
# scripts/census_tagger_debris.py establishes, so `--json` and the text output
# can never describe different runs.
# ---------------------------------------------------------------------------


def resolve_roots(values: Sequence[str] | None) -> list[Path]:
    """Resolve the repeated ``--root`` values, defaulting to this checkout.

    argparse's ``append`` action leaves the destination at ``None`` (not
    ``[]``) when the flag never appears, so the default is applied HERE rather
    than via ``default=[...]``: an argparse list default is shared mutable
    state that ``append`` extends rather than replaces, which would silently
    add this checkout to every explicit invocation. (Copied from the sibling,
    which records the same trap.)

    Every root is ``.resolve()``d and the list de-duplicated order-preservingly,
    so two spellings of one root cannot be walked twice and double-count its
    records.
    """
    if not values:
        return [DEFAULT_PROJECT_ROOT]
    seen: set[Path] = set()
    roots: list[Path] = []
    for value in values:
        root = Path(value).resolve()
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return roots


def build_report(
    roots: Sequence[Path],
    *,
    module: str,
    window: tuple[datetime, datetime],
    label: str = 'test',
    role: str | None = None,
) -> dict[str, Any]:
    """Assemble the whole census as ONE JSON-native dict.

    This dict IS the ``--json`` payload and the text renderer's only input, so
    the two renderings cannot disagree. It carries its own provenance — the
    resolved window, the filters, and the command that was compared against —
    because a duration figure without those is not reproducible; and it carries
    the skip and rejection counts, so ``n`` can always be reconciled against
    the corpus rather than taken on trust.

    The expected command is resolved from the FIRST root that declares one:
    a multi-root run is comparing one module's suite across checkouts, and a
    per-root command would make the durations incomparable, which is the very
    thing the shape filter exists to prevent.
    """
    corpus = load_records(roots)
    expected = next(
        (cmd for cmd in (read_module_test_command(r, module) for r in roots) if cmd),
        None,
    )
    selection = select_full_suite_legs(
        corpus, root=roots[0], prefix=module, label=label, role=role,
    )
    legs = within_window(selection.legs, window)
    return {
        'module': module,
        'label': label,
        'role': role,
        'window': [window[0].isoformat(), window[1].isoformat()],
        'roots': [str(r) for r in roots],
        'expected_command': expected,
        'corpus': {
            'records': len(corpus.records),
            'skipped': dict(Counter(s.reason for s in corpus.skipped)),
        },
        'rejected': selection.rejected,
        'selected_outside_window': len(selection.legs) - len(legs),
        'overall': summarise_legs(legs),
        'by_day': by_day(legs),
        'by_load_band': by_load_band(legs),
        'cold_separability': cold_separability(legs),
        'merge_gate': merge_gate_budget(roots[0]),
    }


def _format_series(series: dict[str, Any]) -> str:
    """Render one duration series, printing ``-`` for a null rather than 0."""
    def show(key: str) -> str:
        value = series[key]
        return '-' if value is None else f'{value:.0f}'

    return (
        f"n={series['n']:<4} p50={show('p50'):>7} "
        f"p90={show('p90'):>7} max={show('max'):>7}"
    )


def _format_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"  {name:<14} {_format_series(row['durations'])}"
        f"  timed_out={row['timed_out']:<3} failed={row['failed']}"
    )


def format_report(report: dict[str, Any]) -> str:
    """Render *report* as text. Reads only the dict ``--json`` emits."""
    lo, hi = report['window']
    lines = [
        f"verify-budget census — module {report['module']!r}, "
        f"leg {report['label']!r}, role {report['role'] or 'any'}",
        f'  window   {lo} .. {hi}',
        f"  roots    {', '.join(report['roots'])}",
        f"  command  {report['expected_command'] or '<none declared>'}",
        '',
        f"  corpus   {report['corpus']['records']} records loaded, "
        f"skipped {report['corpus']['skipped'] or 'none'}",
        f"  rejected {report['rejected'] or 'none'}"
        f"  (+{report['selected_outside_window']} selected outside the window)",
        '',
        'FULL-SUITE DURATIONS (seconds; timed-out and failed legs counted, not averaged)',
        _format_row('overall', report['overall']),
    ]

    if report['overall']['durations']['n'] == 0:
        lines.append(
            '  NOTE: no full-suite run matched in this window. The nulls above '
            'are "not measured", NOT a fast suite.',
        )

    lines += ['', 'BY LOAD BAND (host cpu some avg10 at command START)']
    for band in PSI_BANDS:
        lines.append(_format_row(band.name, report['by_load_band'][band.name]))
    lines.append(_format_row(UNSTAMPED, report['by_load_band'][UNSTAMPED]))
    lines.append(
        '  NOTE: unstamped = load not knowable (record predates the stamp, or '
        'the PSI read degraded). NOT an idle host.',
    )

    if report['by_day']:
        lines += ['', 'BY DAY (UTC)']
        lines += [_format_row(day, row) for day, row in report['by_day'].items()]

    cold = report['cold_separability']
    lines += [
        '',
        'FINDING — cold runs are NOT separable from warm ones in this corpus.',
        f"  {cold['first_attempt_records']} first-attempt / "
        f"{cold['later_attempt_records']} later-attempt records.",
        f"  {cold['basis']}",
        '',
        'MERGE GATE (reported, not derived here)',
        f"  merge_verify_cold_command_timeout_secs = "
        f"{report['merge_gate']['merge_verify_cold_command_timeout_secs']}"
        f" (from {report['merge_gate']['source'] or '<unresolved>'})",
        f"  {report['merge_gate']['note']}",
        '',
        'This report asserts no numeric target. The gate that holds a budget '
        'up is tests/scripts/test_module_verify_budgets.py.',
    ]
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='verify_budget_census',
        description=(
            'Census the verify-summary corpus for one module: full-suite '
            'durations by day and by host-load band. STRICTLY READ-ONLY.'
        ),
    )
    parser.add_argument(
        '--root', action='append', dest='roots', metavar='PATH',
        help='project root to walk; repeatable (default: this checkout)',
    )
    parser.add_argument(
        '--module', default='orchestrator', metavar='PREFIX',
        help="module prefix whose suite to census (default: 'orchestrator')",
    )
    parser.add_argument(
        '--label', default='test', metavar='LEG',
        help="which check leg to census (default: 'test')",
    )
    parser.add_argument(
        '--role', default=None, metavar='ROLE',
        help='restrict to one verify role (task/merge/probe); default: any',
    )
    parser.add_argument(
        '--window', default=DEFAULT_WINDOW, metavar='SPEC',
        help=f"'<N>d' or '<iso>..<iso>' (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        '--json', action='store_true',
        help='emit the whole report as one JSON document',
    )
    return parser


def main(argv: Sequence[str], now: datetime | None = None) -> int:
    """CLI entry point.

    Exit codes, the vocabulary ``merge_lane_throughput.main`` documents:
    ``0`` on success, ``1`` when a NAMED root could not be read (the remaining
    roots still report, and the failure goes to stderr — one bad path must not
    cost the whole run), ``2`` on malformed arguments with nothing on stdout.

    *now* is a parameter so every caller and test fixes the clock explicitly.
    """
    args = build_parser().parse_args(argv)
    try:
        window = parse_window(args.window, now or datetime.now(UTC))
    except argparse.ArgumentTypeError as exc:
        print(f'verify_budget_census: {exc}', file=sys.stderr)
        return 2

    roots = resolve_roots(args.roots)
    readable = [root for root in roots if root.is_dir()]
    status = 0
    for root in roots:
        if root not in readable:
            print(
                f'verify_budget_census: cannot read root {root}; skipping it. '
                f'The remaining roots are reported below.',
                file=sys.stderr,
            )
            status = 1
    if not readable:
        return 1

    report = build_report(
        readable,
        module=args.module,
        window=window,
        label=args.label,
        role=args.role,
    )
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
