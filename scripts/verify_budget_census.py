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

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
