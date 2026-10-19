"""Flake ledger — occurrence trail and debt set (plans/flake-ledger-prd.md, task α).

Three tables in the project's existing ``data/orchestrator/runs.db``:

- ``flake_occurrence`` — append-only; MANY rows per test.  The evidence trail: every
  time a discriminator judged a failing test, what it concluded, where it ran, and
  what the host pressure was at that moment.
- ``flake_debt`` — ONE row per test.  The set the invariant governs: a test that has
  been suppressed owes a de-flake task, and the row carries the ledger-owned clock
  (``opened_at``/``resolved_at``) that ``.taskmaster/tasks/tasks.db`` cannot supply
  (§5.4 — that store has no ``created_at``, and ``planning_mode=True`` bypasses the
  curator ticket store entirely).
- ``flake_debt_due`` — the DUE LIST: one item per (test, observation) owed an
  :func:`open_debt` call.  An observation writes its occurrence rows and its items in
  one local transaction (:func:`enqueue_debt`), and
  ``orchestrator/src/orchestrator/flake_debt_drain.py`` files the owners afterwards,
  off the merge path (task 6934).  An item's context (:class:`DebtContext`) shapes the
  de-flake task it owes (plans/offline-lane-red-path-prd.md §5.6).

Living in the shared ``orchestrator`` package, on a DB every client project already
has, is what hands this facility to all 8 projects with ZERO provisioning: the tables
appear on first write via ``CREATE TABLE IF NOT EXISTS``, with no migration to sequence
(§5.2).

TWO BINDING CONTRACTS, both machine-checked:

1. **Writes go only through this API** — ``record_flake_occurrence`` / ``open_debt`` /
   ``resolve_debt``, and for the due list ``enqueue_debt`` / ``delete_due_item`` /
   ``record_failed_due_attempt`` / ``name_stale_due_items``, never raw SQL at a call
   site (§5.2).  Four call sites hand-rolling
   INSERTs is exactly how the two merge gates would drift into different notions of the
   verdict vocabulary.
2. **No public entry point ever raises** (§8.3, boundary row B12).  The merge path has
   no ``VerifyInfraError`` handler, so an uncaught raise here stalls the merge queue —
   a ledger failure must never fail a verify or a merge.  Every entry point degrades to
   an honest value (``None`` / ``[]`` / ``False``) and logs LOUDLY with ``exc_info``; it
   never fails silently.  This mirrors ``chronic_flake``'s catch-all-defensive contract.

RETENTION — a named, accepted position, not an oversight (§5.2, §11 Q1).  ``flake_debt``
is bounded by construction (one row per test); resolved rows are retained DELIBERATELY
because the recurrence trigger reads them, so deleting one would silently disarm it.
``flake_occurrence`` is append-only and UNPRUNED, bounded exactly the way ``events`` is
— i.e. it isn't.  That is consistent with the rest of ``runs.db``, where nothing is
pruned.  Do NOT add bespoke pruning for one table here; revisit repo-wide when ``events``
retention is addressed.  :func:`read_occurrences` therefore takes a ``limit`` — the read
side is bounded even though the table is not.

OWNERSHIP BOUNDARY vs ``chronic_flake.FilingLedger`` — a TRANSITIONAL overlap with a
named owner, not a permanent second store.  ``chronic_flake.FilingLedger``
(``data/orchestrator/chronic_flake_filings.json``, ``{test: last_filed_iso}``) is a
per-test store of "when did we last auto-file a de-flake task for this test", backing
the rate limit in ``maybe_file_chronic_flake_tasks``.  ``flake_debt`` covers
substantially the same ground with a different clock and different dedup semantics, so
during the overlap the boundary is:

- ``FilingLedger`` answers *"may I file AGAIN yet?"* — a rate limit over the 3-in-20
  chronic gate, keyed on a JSON file, still authoritative for that gate today.
- ``flake_debt`` answers *"does this test currently OWE a de-flake task?"* — one row per
  test with explicit open/resolve cycles and ``owner_task_id``.  It is the store the
  §5.9 write-time invariant is enforced against.

They converge in **task κ** (PRD §5.10, §10): κ migrates ``chronic_flake.py`` off JSONL
onto this ledger, retires 3-in-20 from a filing GATE to a severity input — which is what
makes ``FilingLedger``'s "may I file again yet?" question moot, since by then every
suppressed test owes a task unconditionally — and turns ``CHRONIC-FLAKY`` markers into
occurrence producers (``call_site='chronic_marker'``, which is why that member is already
in :class:`FlakeCallSite`).  Boundary row B14 is κ's acceptance check: a marker with no
JSONL present must produce an occurrence and **no second de-flake task**.  Until κ lands,
do NOT wire a second filing path through this module — ζ owns the single filing seam.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync
from shared.task_metadata import DoneProvenance, parse_metadata
from shared.task_statuses import TERMINAL, owner_liveness
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask, SubmittedTicket

logger = logging.getLogger(__name__)

# Sentinel test_id for an `unconfirmable` observation that identified no node-ids.
# The observation is still COUNTED — θ's class-1 health check is an `unconfirmable`
# RATE, so dropping the row because no node-ids were resolved would make "could not
# even determine which tests failed" invisible, reproducing the exact blindness this
# PRD exists to end.  The angle-bracket form cannot collide with a real pytest
# node-id.  It must NEVER be passed to `open_debt`: a sentinel names no test, so it
# can own no de-flake task.
UNKNOWN_TEST_ID = '<unknown>'

# PRD §5.3 verbatim.  Additive `CREATE TABLE IF NOT EXISTS` with no `PRAGMA
# user_version` ladder — the event_store.py:23-41 / run_store.py:18-76 idiom, and
# what makes the ledger a well-behaved fifth owner of an existing runs.db.
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS flake_occurrence (   -- append-only; many rows per test; the evidence trail
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT    NOT NULL,   -- ISO-8601 UTC, LEDGER-OWNED (§5.4)
    test_id         TEXT    NOT NULL,   -- pytest node-id, or script-suite test name
    project_id      TEXT    NOT NULL,   -- denormalised: the dashboard aggregates across runs.db files
    verdict         TEXT    NOT NULL,   -- passes_in_isolation | fails_in_isolation | unconfirmable
    call_site       TEXT    NOT NULL,   -- merge_gate | main_probe | chronic_marker | offline_lane
    runner          TEXT,               -- 'local' | remote host name — WHERE the discriminator ran
    merge_sha       TEXT,
    task_id         TEXT,
    psi_cpu_some10  REAL,               -- host pressure AT observation (shared.psi), NULL if read_ok=False
    detail          TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS flake_debt (         -- ONE row per test; the set the invariant governs
    test_id                 TEXT PRIMARY KEY,   -- runs.db is per-project, so test_id alone is the key
    project_id              TEXT NOT NULL,
    opened_at               TEXT NOT NULL,      -- LEDGER-OWNED
    resolved_at             TEXT,               -- NULL while open
    owner_task_id           TEXT,               -- the non-terminal de-flake task (§5.5)
    open_count              INTEGER NOT NULL DEFAULT 1,
    prior_resolved_at       TEXT,               -- previous cycle — feeds the recurrence trigger
    prior_resolving_commit  TEXT,               -- cited verbatim in the regressed_after_resolution L2
    last_occurrence_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flake_debt_due (     -- the drain's work list: one row per (test, observation) owed an open_debt call
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id             TEXT    NOT NULL,           -- never UNKNOWN_TEST_ID
    project_id          TEXT    NOT NULL,
    observed_at         TEXT    NOT NULL,           -- canonical ISO-8601 UTC, as flake_occurrence.observed_at
    call_site           TEXT    NOT NULL,           -- FlakeCallSite
    head                TEXT,                       -- the evidence's head (merge sha); NULL = unknown: no stale-head check
    observing_task_id   TEXT,
    context             TEXT    NOT NULL DEFAULT '{}',  -- JSON; '{}' until ε1 gives it meaning
    enqueued_at         TEXT    NOT NULL,           -- LEDGER-OWNED wall clock; the age the 30-minute bound reads
    attempts            INTEGER NOT NULL DEFAULT 0,
    next_attempt_at     TEXT    NOT NULL,           -- = enqueued_at until the first failed attempt
    last_error          TEXT,                       -- a flake_debt_drain.DueFailure value
    escalation_id       TEXT                        -- the stale-item escalation that named this item
);

-- §8.3's idempotency key, enforced declaratively: a replayed merge-path write must
-- land one row, not two.
CREATE UNIQUE INDEX IF NOT EXISTS idx_flake_occurrence_dedup
    ON flake_occurrence(test_id, observed_at, call_site);

-- Serves the windowed rate reads (unconfirmable rate, suppressions-in-window).
CREATE INDEX IF NOT EXISTS idx_flake_occurrence_observed
    ON flake_occurrence(observed_at);

-- The same idempotency key as flake_occurrence: a replayed write lands one item.
CREATE UNIQUE INDEX IF NOT EXISTS idx_flake_debt_due_dedup
    ON flake_debt_due(test_id, observed_at, call_site);
"""

#: A due item older than this, by ``enqueued_at``, is STALE: the drain escalates it and
#: the report flags it.  Age since ``enqueued_at``, never since ``next_attempt_at``, which
#: every failed attempt pushes forward.
DUE_ITEM_STALE_AFTER: Final = timedelta(minutes=30)
DUE_ITEM_STALE_MINUTES: Final = int(DUE_ITEM_STALE_AFTER.total_seconds() // 60)

# The one spelling of the stale predicate (heuristic 11, SPOT).
_IS_STALE_SQL = 'enqueued_at <= :stale_before'


class FlakeVerdict(StrEnum):
    """What a discriminator concluded about a failing test (PRD §8).

    §5.5: the vocabulary names the OBSERVATION, never the remedy — ``passes_in_isolation``,
    never ``flaky_test: true``.  One discriminator serves both merge gates, so this is
    what keeps them from drifting into different notions of "passes in isolation".
    """

    passes_in_isolation = 'passes_in_isolation'  # re-ran clean, isolated + serial → suppressible
    fails_in_isolation = 'fails_in_isolation'  # re-ran and failed → a real red
    unconfirmable = 'unconfirmable'  # could not map node-ids / could not re-run


class FlakeCallSite(StrEnum):
    """WHICH gate produced an observation (PRD §5.3's ``call_site`` vocabulary).

    An enum for the same reason :class:`FlakeVerdict` is one, and with a sharper edge:
    ``call_site`` is one third of §8.3's ``(test_id, observed_at, call_site)``
    idempotency key, so a single typo at one call site ('merge-gate' for 'merge_gate')
    would silently defeat dedup AND split θ's per-site rates into two half-populated
    buckets — a drift that reads as a data trend rather than as a bug.  Coerced in
    :func:`record_flake_occurrence`, so an unrecognised value degrades through B12
    instead of being persisted verbatim.
    """

    merge_gate = 'merge_gate'  # the pre-merge verify gate
    main_probe = 'main_probe'  # the periodic probe of main
    chronic_marker = 'chronic_marker'  # a reify CHRONIC-FLAKY marker parsed from verify stdout
    offline_lane = 'offline_lane'  # the offline integration lane (plans/offline-lane-red-path-prd.md D6)


class EvidenceKind(StrEnum):
    """How a test went red: an assertion failure or an error outside the assertion."""

    failed = 'failed'
    error = 'error'


@dataclass(frozen=True)
class FailureEvidence:
    """One test's failure, as the observer saw it: the kind, the first line of the
    report, and where the full record lives when one was kept."""

    kind: EvidenceKind
    first_line: str
    record_dir: str | None = None


@dataclass(frozen=True)
class FlakeSuppression:
    """Discriminator output. Produced WHEREVER the worktree is (local or remote);
    consumed ONLY on the dispatcher. Rides VerifyResult across the wire.

    Not hashable once *evidence* is set: the mapping is a plain ``dict``, so
    ``dataclasses.asdict`` and the runner codec flatten it."""

    verdict: FlakeVerdict
    test_ids: tuple[str, ...]  # node-ids examined — EMPTY is legal only for `unconfirmable`
    observed_at: str  # ISO-8601 UTC, stamped by the DISCRIMINATOR at observation
    call_site: FlakeCallSite  # WHICH gate observed this
    runner: str  # 'local' | remote host name — WHERE the re-run ran
    psi_cpu_some10: float | None  # shared.psi at observation; None when read_ok is False
    unconfirmable_reason: str | None  # populated iff verdict is unconfirmable
    evidence: Mapping[str, FailureEvidence] | None = None  # per test id, for those it covers


def _coerce_test_ids(value: object, *, context: str, warn: bool = True) -> tuple[str, ...]:
    """Normalize a ``test_ids`` payload to a tuple, stating the wrap-not-drop rule ONCE.

    A bare ``str`` where a tuple belongs is an easy mistake to make (one node-id passed
    unwrapped), it is TRUTHY, and ``tuple('a::t')`` explodes it into one entry PER
    CHARACTER — a dozen garbage test_ids silently written into the evidence trail this
    PRD exists to make trustworthy.  WRAP rather than drop: a single node-id string has
    exactly one honest reading, so the observation is preserved and the warning still
    surfaces the upstream bug.

    Both the WRITE path (:func:`record_flake_occurrence`) and the WIRE path
    (:func:`flake_suppression_from_wire`) route through here.  They had two separate
    copies of this rule with two separately-worded warnings, which could drift
    independently — e.g. a ``bytes`` or generator case added to one and not the other,
    which would then behave differently depending only on whether the observation
    crossed the wire.  *context* names the caller for the log line (a ``call_site``
    value at write time, ``'the wire'`` on the deserialization path) so one rule can
    still produce a locatable message.

    Raises ``TypeError`` on a non-iterable, deliberately: both callers wrap this in a
    catch-all (B12 at write time, the never-raise guard on the wire), and each already
    knows how to degrade loudly.

    *warn* ``False`` coerces without the warning, for a second coercion of a payload
    whose first coercion already warned.
    """
    if isinstance(value, str):
        if warn:
            logger.warning(
                'flake_ledger: test_ids arrived as a bare str (%r) from %s; treating it as '
                'ONE node-id — the producer should pass a list/tuple',
                value,
                context,
            )
        return (value,)
    return tuple(value)  # type: ignore[call-overload]


def flake_suppression_from_wire(d: object) -> FlakeSuppression | None:
    """Rebuild a :class:`FlakeSuppression` from its JSON-decoded wire form (§8, §8.4).

    The READ half of the ``VerifyResult.flake_suppression`` carrier.  ``result_to_dict``
    is ``dataclasses.asdict`` and ``result_from_dict`` is a bare ``VerifyResult(**d)``
    (``orchestrator/src/orchestrator/verify_runner.py::result_to_dict`` /
    ``::result_from_dict``), so the WRITE half needs nothing — ``json.dumps``
    flattens a ``StrEnum`` to its value and a tuple to an array losslessly — but the read
    half hands ``verdict``/``call_site`` back as plain ``str`` and ``test_ids`` as a
    ``list``.  Coercing here is what makes the field's annotation true on the
    deserialized path and makes the round-trip equality-preserving.

    It NEVER raises, and that is the load-bearing property, not politeness:
    ``RemoteRunner.run_merge_verify`` converts any ``TypeError``/``ValueError`` escaping
    ``result_from_json`` into a ``RunnerUnavailable``
    (``orchestrator/src/orchestrator/verify_runner.py::RemoteRunner.run_merge_verify``),
    which the pool pays for with a whole local re-verify.  Letting one malformed
    sub-payload cost a re-verify would be strictly worse than dropping the one
    observation it carried, so a
    bad payload degrades to ``None`` plus a LOUD warning (B12's discipline, applied to
    the wire).

    It is deliberately NOT a validator.  An unrecognised ``verdict``/``call_site`` string
    is PRESERVED on the field rather than rejected, so that
    :func:`record_flake_occurrence`'s existing coercion guard stays the SINGLE write-time
    vocabulary policy.  Duplicating that policy here would give a producer that went
    off-vocabulary two different failure modes depending on whether it ran locally or
    across the wire — and would delete the evidence before the log line that names the
    bug could be emitted.
    """
    if d is None:
        return None
    try:
        # `bool` is an `int` and neither is a payload; the isinstance check below rejects
        # every non-mapping uniformly, which is what B13's "old remote sent something
        # else entirely" case needs.
        if not isinstance(d, dict):
            logger.warning(
                'flake_ledger: flake_suppression arrived as %s, not a dict; dropping the '
                'observation (%r)',
                type(d).__name__,
                d,
            )
            return None

        # Project onto exactly the declared field names: an unknown key from a NEWER
        # producer must not cost the observation it rides along with (forward-compat),
        # and `FlakeSuppression(**d)` would `TypeError` on it.
        kwargs = _declared(FlakeSuppression, d)

        # Coerce the two vocabulary fields back to members, falling back to the RAW value
        # so an off-vocabulary string reaches `record_flake_occurrence`'s B12 guard
        # intact (see the docstring — one coercion policy, at write time).
        for name, enum_cls in (('verdict', FlakeVerdict), ('call_site', FlakeCallSite)):
            if name in kwargs:
                with contextlib.suppress(ValueError):
                    kwargs[name] = enum_cls(kwargs[name])

        # Same bare-str hazard `record_flake_occurrence` guards, and literally the same
        # code: `_coerce_test_ids` states the wrap-not-drop rule once so the two paths
        # cannot drift.  A MISSING key is left missing so the `FlakeSuppression(**kwargs)`
        # below still raises for a truncated payload (caught, warned, dropped).
        if kwargs.get('test_ids') is not None:
            kwargs['test_ids'] = _coerce_test_ids(kwargs['test_ids'], context='the wire')

        if 'evidence' in kwargs:
            kwargs['evidence'] = _evidence_from_wire(kwargs['evidence'])

        # A missing REQUIRED key raises TypeError here and is caught below — a truncated
        # payload is a producer bug worth a loud line, not a half-built observation.
        return FlakeSuppression(**kwargs)
    except Exception:
        logger.warning(
            'flake_ledger: could not rebuild a FlakeSuppression from the wire payload '
            '%r; dropping the observation (a malformed sub-payload must never cost a '
            're-verify)',
            d,
            exc_info=True,
        )
        return None


def _declared(cls: type, entry: Mapping[str, Any]) -> dict[str, Any]:
    """*entry* projected onto *cls*'s declared dataclass fields: an unknown key from a
    newer producer is dropped, a missing one is left missing."""
    return {f.name: entry[f.name] for f in fields(cls) if f.name in entry}


def _evidence_from_wire(value: object) -> dict[str, FailureEvidence] | None:
    """Rebuild a suppression's evidence from its JSON form, each entry projected onto
    :class:`FailureEvidence`'s declared fields as the observation itself is.  ``None``
    for none, and ``None`` plus a warning for anything malformed: the observation is
    never dropped for its evidence."""
    if value is None:
        return None
    try:
        if not isinstance(value, dict):
            raise TypeError(f'evidence is a {type(value).__name__}, not a dict')
        evidence: dict[str, FailureEvidence] = {}
        for test_id, entry in value.items():
            kwargs = _declared(FailureEvidence, entry)
            kwargs['kind'] = EvidenceKind(kwargs.get('kind'))
            evidence[test_id] = FailureEvidence(**kwargs)
        return evidence
    except Exception:
        logger.warning(
            'flake_ledger: could not rebuild the evidence of a wire observation (%r); '
            'keeping the observation without it',
            value,
            exc_info=True,
        )
        return None


@dataclass(frozen=True)
class FlakeOccurrenceRow:
    """One ``flake_occurrence`` row — a single test's observation at a single moment.

    A ``FlakeSuppression`` naming N tests fans out to N of these.
    """

    id: int
    observed_at: str
    test_id: str
    project_id: str
    verdict: str
    call_site: str
    runner: str | None
    merge_sha: str | None
    task_id: str | None
    psi_cpu_some10: float | None
    detail: str


@dataclass(frozen=True)
class DebtRow:
    """The single ``flake_debt`` row for one test — its current cycle plus the
    prior-cycle fields that feed η's recurrence trigger."""

    test_id: str
    project_id: str
    opened_at: str
    resolved_at: str | None
    owner_task_id: str | None
    open_count: int
    prior_resolved_at: str | None
    prior_resolving_commit: str | None
    last_occurrence_at: str


def ledger_db_path(project_root: Path) -> Path:
    """The project's ``runs.db``.

    One spelling of a literal that is hand-built at 5+ sites today (harness.py:2119 and
    friends) with no shared helper, so the ledger's consumers do not each re-derive it.
    """
    return project_root / 'data' / 'orchestrator' / 'runs.db'


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a ledger connection with the full durability pragma triad applied.

    Byte-identical to the two existing ``runs.db`` owners (event_store.py:512,
    run_store.py:90) — no new pragma policy and no second durability story.  The 5s
    ``busy_timeout`` is what absorbs concurrent merge-lane contention, in place of a
    hand-rolled lock.

    The close-on-failure guard exists because ``sqlite3.connect()`` succeeds LAZILY
    even against a non-DB file: the failure surfaces on the first real statement — the
    ``PRAGMA journal_mode=WAL`` inside ``apply_full_durability_pragmas_sync`` — by
    which point ``conn`` already owns an open file descriptor that nothing else will
    ever close.
    """
    conn = sqlite3.connect(str(db_path))
    # The two sibling stores share the connect-then-pragma idiom WITHOUT this guard,
    # and that is fine for them: they propagate the raise and the caller stops, so the
    # process is dying anyway.  Do not "align" this back to them.  B12 makes every
    # public entry point here deliberately SWALLOW the exception and return an honest
    # degrade value, so a long-lived orchestrator keeps calling on every merge —
    # turning a one-shot leak into an unbounded one.
    #
    # BaseException, not Exception: this is resource cleanup, so a KeyboardInterrupt or
    # SystemExit arriving mid-pragma must still release the fd.  The bare `raise`
    # preserves the original exception and traceback, so the caller's B12 catch-all
    # still logs the true cause with exc_info=True.
    try:
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
    except BaseException:
        conn.close()
        raise
    return conn


def _open(db_path: Path) -> sqlite3.Connection:
    """Connect AND provision the schema on ONE connection — the only way in.

    Every public entry point must guarantee the tables exist before its statement runs,
    because the API is free functions taking a ``db_path`` (§8.3): the recorder is called
    from stateless merge-path sites and a CLI, neither of which holds a long-lived store
    object, so there is no constructor to run the DDL once.

    The DDL is deliberately NOT memoized — a per-path cache would go stale the moment the
    DB file is replaced or removed — but it does NOT need its own connection.  Running
    ``CREATE TABLE IF NOT EXISTS`` on the connection the statement is about to use anyway
    is what keeps a call at ONE connection instead of two (``open_debt`` was four: its
    own pair plus ``read_debt``'s).  Each connection costs the full five-pragma
    durability triad, including a ``journal_mode=WAL`` switch, so this is per-merge cost,
    not a micro-optimisation.

    The close-on-failure guard mirrors :func:`_connect`'s, for the same B12 reason: a
    corrupt DB fails at ``executescript``, and a swallowed exception upstream would
    otherwise leak this fd on every merge.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_schema(db_path: Path) -> None:
    """Create the ledger tables and their indexes if absent.  Additive and idempotent.

    Public because ι and the tests want an explicit "provision this DB" verb; the other
    entry points do NOT call it, they go through :func:`_open`, which provisions on the
    connection they were opening regardless.
    """
    try:
        _open(db_path).close()
    except Exception:
        logger.warning('flake_ledger: schema init failed for %s', db_path, exc_info=True)


def _canonicalize_utc(dt: datetime, *, origin: str | None = None) -> str:
    """Coerce *dt* to aware-UTC and return its canonical ISO-8601 spelling.

    Shared by both ledger clocks: :func:`_normalize_observed_at` (the discriminator-
    supplied ``observed_at``) and ``open_debt``/``resolve_debt`` (the ledger-owned
    ``opened_at``/``resolved_at``/``last_occurrence_at``).  A naive value gets UTC
    ATTACHED, never ``.astimezone()``: every one of these columns is documented as UTC,
    and ``.astimezone()`` on a naive datetime applies the HOST's local offset, which
    would silently shift the stamp by the dispatcher's timezone.

    *origin* controls whether a naive *dt* is worth a loud warning, and is left
    ``None`` on the :func:`_normalize_observed_at` path deliberately: ``observed_at``
    arrives over the wire from a remote discriminator with no runtime validation, so a
    missing offset there is expected input, not a caller bug, and warning on it would
    just be noise on every legitimate remote observation.  ``open_debt``/``resolve_debt``
    pass their function name: their ``now`` is supplied by IN-REPO callers, where a
    naive datetime is a caller bug rather than untrusted input — the repo's own
    precedent for the analogous case is loud (``orchestrator/overrides.py:222`` raises
    on a naive ``ttl_until``) — and silent coercion would otherwise mask a future ζ/ε
    call site passing ``datetime.now()`` instead of ``datetime.now(UTC)``, producing a
    stamp that is wrong by the host offset while still looking canonical.
    """
    return _aware(dt, origin=origin).astimezone(UTC).isoformat()


def _aware(dt: datetime, *, origin: str | None = None) -> datetime:
    """*dt* with UTC ATTACHED if it is naive — the half of :func:`_canonicalize_utc` a
    caller needs when it must hand one instant on as a ``datetime``, so the naive
    warning (same *origin* rule) fires once rather than once per consumer."""
    if dt.tzinfo is None:
        if origin is not None:
            logger.warning(
                'flake_ledger: %s received a naive now; treating as UTC — callers '
                'should pass datetime.now(UTC)',
                origin,
            )
        dt = dt.replace(tzinfo=UTC)
    return dt


def _normalize_observed_at(raw: str) -> str:
    """Canonicalise a discriminator stamp to ISO-8601 UTC.

    ``observed_at`` is doubly load-bearing and arrives over the wire on a dataclass with
    no runtime validation, so it gets the same coercion treatment as ``verdict``: it is
    one third of §8.3's ``(test_id, observed_at, call_site)`` idempotency key, AND
    :func:`read_occurrences`'s ``since`` window plus ``ORDER BY observed_at`` rely on
    lexicographic order matching chronological order.  Two equally valid spellings of the
    same instant — ``'…T12:00:00Z'`` from a hand-built stamp and ``'…T12:00:00+00:00'``
    from ``datetime.isoformat()`` — would otherwise land the SAME observation twice and
    sort apart inside a window.

    Raises on a malformed stamp, deliberately — the caller's B12 catch-all turns that
    into a loud warning and a dropped write, which beats persisting a stamp that breaks
    both dedup and the window scan.

    BOTH SIDES OF THE COMPARISON MUST GO THROUGH HERE (or through :func:`_canonicalize_utc`
    for an already-parsed ``datetime``, as ``open_debt``/``resolve_debt`` do for their
    ``opened_at``/``resolved_at``/``last_occurrence_at`` columns).  This is used on the
    WRITE path (``record_flake_occurrence``) and on the READ path
    (:func:`read_occurrences`'s ``since`` boundary), and normalising only one side of a
    comparison is worse than normalising neither: it makes a correct-looking query
    against a correctly-stored row return NOTHING, silently.  Any future window bound (an
    ``until``, a debt-age cutoff) must be routed through here — or through
    :func:`_canonicalize_utc` — too.
    """
    return _canonicalize_utc(datetime.fromisoformat(raw))


_OccurrenceKey = tuple[str, str, str]


def _occurrence_rows(
    project_id: str,
    s: FlakeSuppression,
    *,
    merge_sha: str | None,
    task_id: str | None,
) -> list[tuple]:
    """The ``flake_occurrence`` rows *s* asks for, one per test it names, in INSERT
    column order; ``[]`` (logged) for an observation with nothing honest to record.

    *s* rides ``VerifyResult`` across the wire (§8) and carries NO runtime validation, so
    all three key-bearing fields — ``verdict``, ``call_site``, ``observed_at`` — plus
    ``test_ids`` are coerced/guarded here.  Raises on anything that does not survive
    coercion; the public callers' B12 catch-all turns that into a loud warning and a
    dropped write.
    """
    # `s` is a plain dataclass with NO runtime validation, and it rides VerifyResult
    # across the wire (§8): a JSON round-trip hands back `verdict` as a plain `str`,
    # not a `FlakeVerdict`.  Coerce ONCE here, so that (a) the `.value` accesses below
    # cannot AttributeError out of a public entry point, and (b) an unrecognised verdict
    # string degrades through B12's catch-all instead of being written verbatim into the
    # vocabulary column.  Everything downstream uses `verdict`, never `s.verdict`.
    verdict = FlakeVerdict(s.verdict)
    # Same hazard, same treatment: `call_site` is one third of the dedup key AND
    # θ's per-site grouping key, so an unrecognised spelling must not be persisted.
    call_site = FlakeCallSite(s.call_site)
    observed_at = _normalize_observed_at(s.observed_at)

    # The bare-`str` hazard (one node-id passed unwrapped, TRUTHY, and
    # `tuple('a::t')` explodes it into one row PER CHARACTER) is handled by the
    # shared `_coerce_test_ids` — same rule, same warning, as the wire path, stated
    # in exactly one place.
    supplied_test_ids: tuple[str, ...] = _coerce_test_ids(
        s.test_ids, context=f'call_site={call_site.value}',
    )

    # §8: EMPTY test_ids is legal only for `unconfirmable`.  An unconfirmable
    # observation that resolved no node-ids is still COUNTED, under the sentinel —
    # θ's class-1 health check is an unconfirmable RATE, so dropping the row would
    # make "could not even determine which tests failed" invisible.  UNKNOWN_TEST_ID
    # must never reach `open_debt`: it names no test, so it can own no de-flake task.
    #
    # `==`, never `is`: a wire-deserialized 'unconfirmable' str is EQUAL to the
    # member (StrEnum) but not IDENTICAL to it, and an identity test would misroute
    # it into the silent-drop branch below — deleting exactly the class-1 signal the
    # sentinel exists to preserve.  The coercion above already makes them identical;
    # `==` is the belt to that braces, and is correct either way.
    if supplied_test_ids:
        test_ids = tuple(supplied_test_ids)
    elif verdict == FlakeVerdict.unconfirmable:
        test_ids = (UNKNOWN_TEST_ID,)
    else:
        # A confirmed verdict about zero tests is meaningless.  Degrade loudly rather
        # than accept it: a sentinel row here would corrupt the denominator θ's
        # unconfirmable rate divides by.
        logger.warning(
            'flake_ledger: %s verdict with empty test_ids at call_site=%s; dropping '
            '(a confirmed verdict about zero tests is meaningless)',
            verdict.value,
            call_site.value,
        )
        return []

    evidence = _evidence_by_test(s)
    # Columns are named explicitly, never a positional `INSERT INTO t VALUES (...)`:
    # run_store.py:53-58's ALTER-parity note — a migrated DB and a fresh one must both
    # accept the same statement.
    return [
        (
            observed_at,
            test_id,
            project_id,
            verdict.value,
            call_site.value,
            s.runner,
            merge_sha,
            task_id,
            s.psi_cpu_some10,  # bound directly, so None becomes SQL NULL, never 0.0
            _row_detail(s.unconfirmable_reason, evidence.get(test_id)),
        )
        for test_id in test_ids
    ]


def _evidence_by_test(s: FlakeSuppression) -> Mapping[str, FailureEvidence]:
    """*s*'s evidence, read once and checked; ``{}`` (logged) when it is malformed, so
    bad evidence costs the evidence and never the rows."""
    try:
        evidence = s.evidence
        if evidence is None:
            return {}
        if not isinstance(evidence, Mapping):
            raise TypeError(f'evidence is a {type(evidence).__name__}, not a mapping')
        for test_id, entry in evidence.items():
            if not isinstance(entry, FailureEvidence):
                raise TypeError(f'evidence for {test_id!r} is a {type(entry).__name__}')
            EvidenceKind(entry.kind)
        return evidence
    except Exception:
        logger.warning(
            'flake_ledger: malformed evidence on an observation at call_site=%s; writing '
            'its rows without evidence',
            getattr(s, 'call_site', None),
            exc_info=True,
        )
        return {}


def _row_detail(reason: str | None, evidence: FailureEvidence | None) -> str:
    """One row's ``detail`` JSON.  A row with no evidence keeps the bytes it always had."""
    detail: dict[str, Any] = {'unconfirmable_reason': reason} if reason else {}
    if evidence is not None:
        detail['evidence'] = {
            'kind': EvidenceKind(evidence.kind).value,
            'first_line': evidence.first_line,
            'record_dir': evidence.record_dir,
        }
    return json.dumps(detail) if detail else '{}'


def _write_observation(conn: sqlite3.Connection, rows: Sequence[tuple]) -> set[_OccurrenceKey]:
    """INSERT *rows* (from :func:`_occurrence_rows`) on *conn*, uncommitted, and return the
    ``(test_id, observed_at, call_site)`` key of every row this call newly inserted.

    OR IGNORE, against the ``idx_flake_occurrence_dedup`` UNIQUE index, is §8.3's
    idempotency key enforced declaratively.  Deliberately NOT left to the catch-all: a
    plain INSERT would (1) abort the rest of the batch on the first duplicate, silently
    LOSING the genuinely new rows beside it, and (2) turn every legitimate merge-path
    retry into a loud warning, training operators to ignore the log line B12 relies on.

    One ``execute`` per row, not ``executemany``: the latter's ``rowcount`` is the SUM
    over the batch, and :func:`enqueue_debt` needs to know WHICH rows were new.
    """
    inserted: set[_OccurrenceKey] = set()
    for row in rows:
        cursor = conn.execute(
            'INSERT OR IGNORE INTO flake_occurrence '
            '(observed_at, test_id, project_id, verdict, call_site, runner, '
            ' merge_sha, task_id, psi_cpu_some10, detail) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            row,
        )
        if cursor.rowcount == 1:
            inserted.add(_row_key(row))
    return inserted


def _row_key(row: tuple) -> _OccurrenceKey:
    observed_at, test_id, _project_id, _verdict, call_site, *_rest = row
    return test_id, observed_at, call_site


def record_flake_occurrence(
    db_path: Path,
    project_id: str,
    s: FlakeSuppression,
    *,
    merge_sha: str | None,
    task_id: str | None,
) -> None:
    """Append one ``flake_occurrence`` row per test named in *s* (PRD §8.3).

    ``observed_at`` comes from *s* — the DISCRIMINATOR stamps it at observation, which
    is both semantically right for the remote path (the observation happens on the
    remote host; the write happens later on the dispatcher) and mechanically required:
    §8.3's idempotency key is ``(test_id, observed_at, call_site)``, and a write-time
    stamp would make every retry a distinct row.  It is CANONICALISED on the way in (see
    :func:`_normalize_observed_at`) so that two spellings of one instant cannot become
    two observations.

    The rows-only case of :func:`enqueue_debt`: anything that survives coercion
    (:func:`_occurrence_rows`) is written; anything that does not degrades through B12
    (loud warning, nothing written, no raise).
    """
    enqueue_debt(db_path, project_id, (), occurrence=s, merge_sha=merge_sha, task_id=task_id)


def _to_occurrence_row(row: sqlite3.Row) -> FlakeOccurrenceRow:
    return FlakeOccurrenceRow(
        id=row['id'],
        observed_at=row['observed_at'],
        test_id=row['test_id'],
        project_id=row['project_id'],
        verdict=row['verdict'],
        call_site=row['call_site'],
        runner=row['runner'],
        merge_sha=row['merge_sha'],
        task_id=row['task_id'],
        psi_cpu_some10=row['psi_cpu_some10'],
        detail=row['detail'],
    )


def read_occurrences(
    db_path: Path,
    *,
    test_id: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> list[FlakeOccurrenceRow]:
    """Recorded occurrences, oldest first, optionally filtered.

    Args:
        test_id: Only this test's observations — the recurrence chain for one test.
        since: Only observations at or after this ISO-8601 stamp.  INCLUSIVE at the
            boundary.  ANY valid ISO-8601 spelling of the instant works — ``'…T12:00:00Z'``,
            its ``'…T12:00:00+00:00'`` twin, and the equivalent ``'…T13:00:00+01:00'`` all
            select the same rows — because the boundary is canonicalised through
            :func:`_normalize_observed_at`, exactly as the stored stamp was on the way in.
            The comparison is then lexicographic on a TEXT column, which is sound ONLY
            because both sides are normalised to one fixed offset, at which point string
            order matches chronological order; ``idx_flake_occurrence_observed`` serves
            the window scan.  A malformed stamp degrades through B12 (warns, returns
            ``[]``) rather than silently selecting everything.
        limit: At most this many rows — the MOST RECENT ones, still returned oldest
            first.  Composes with the other filters.  ``0`` (or negative) yields ``[]``.

    ``limit`` selects the TAIL, not the head, and that is deliberate: ``flake_occurrence``
    is append-only and unpruned by design (see the module docstring), so the only bound
    a caller ever wants on it is "the recent end".  Truncating the OLDEST *limit* rows —
    what a bare ``LIMIT`` on this ascending order would give — would silently answer a
    dashboard's "what happened lately?" with year-old rows, which is worse than no bound.
    Callers doing rate math should still pass ``since``: a count over a ``limit``-capped
    read is a count over an unknown window, and dividing by it is meaningless.

    Ordering is part of the contract, not incidental: ``observed_at`` then ``id``, so
    the sequence reads chronologically even though rows arrive out of order (a remote
    observation is written on the dispatcher after a local one that followed it).
    """
    try:
        # Values are BOUND, never interpolated — a test_id is an arbitrary pytest node-id
        # arriving from a remote host.
        clauses: list[str] = []
        params: list[Any] = []
        if test_id is not None:
            clauses.append('test_id = ?')
            params.append(test_id)
        if since is not None:
            # Canonicalised with the SAME function the write path uses, and that pairing
            # is the whole point: `observed_at` is stored normalised, so comparing a raw
            # caller-supplied spelling against it is comparing two different alphabets.
            # A `'…T12:00:00Z'` window boundary against a stored `'…T12:00:00+00:00'`
            # matches NOTHING — and it fails SILENT-EMPTY, which is the dangerous
            # direction here: θ divides windowed rates by this count, so a zero-row
            # answer reads as "healthy" rather than as a broken query.  Inside the try,
            # so a malformed `since` degrades through B12 (loud warning, `[]`) instead of
            # raising into a caller.
            clauses.append('observed_at >= ?')
            params.append(_normalize_observed_at(since))
        where = f' WHERE {" AND ".join(clauses)}' if clauses else ''

        sql = f'SELECT * FROM flake_occurrence{where} ORDER BY observed_at, id'
        if limit is not None:
            # Take the tail DESC, then re-sort ASC so the caller still gets the pinned
            # oldest-first contract.  `max(0, ...)` because SQLite reads a NEGATIVE LIMIT
            # as "no limit at all" — the exact silent opposite of what `limit=-1` asks
            # for, on the one read path whose whole point is to stay bounded.
            sql = (
                f'SELECT * FROM (SELECT * FROM flake_occurrence{where} '
                'ORDER BY observed_at DESC, id DESC LIMIT ?) ORDER BY observed_at, id'
            )
            params.append(max(0, limit))

        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [_to_occurrence_row(r) for r in rows]
    except Exception:
        logger.warning(
            'flake_ledger: failed to read occurrences from %s',
            db_path,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# The due list: what an observation owes open_debt, drained afterwards (task 6934)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DueItem:
    """What an observation asks the drain to do: one :func:`open_debt` call for
    *test_id*.  *observed_at* is raw (canonicalised on enqueue) and *call_site* is a
    :class:`FlakeCallSite` or its spelling."""

    test_id: str
    observed_at: str
    call_site: str
    head: str | None
    observing_task_id: str | None
    context: str = '{}'


@dataclass(frozen=True)
class DueRow:
    """One stored ``flake_debt_due`` item."""

    id: int
    test_id: str
    project_id: str
    observed_at: str
    call_site: str
    head: str | None
    observing_task_id: str | None
    context: str
    enqueued_at: str
    attempts: int
    next_attempt_at: str
    last_error: str | None
    escalation_id: str | None


@dataclass(frozen=True)
class DueSummary:
    """The due list at a glance, for the drain's stale-item escalation and the report.

    ``stale`` counts the items enqueued at or before the caller's *stale_before*, and
    ``unnamed_stale`` those of them no stale-item escalation has named yet;
    ``stale_sample`` is the oldest ten of them.  ``pending_since`` maps each test on the
    list to its OLDEST item's ``enqueued_at``.
    """

    size: int
    oldest: DueRow | None
    stale: int
    unnamed_stale: int
    stale_sample: tuple[DueRow, ...]
    pending_since: Mapping[str, str]


_STALE_SAMPLE_SIZE = 10


def _to_due_row(row: sqlite3.Row) -> DueRow:
    return DueRow(
        id=row['id'],
        test_id=row['test_id'],
        project_id=row['project_id'],
        observed_at=row['observed_at'],
        call_site=row['call_site'],
        head=row['head'],
        observing_task_id=row['observing_task_id'],
        context=row['context'],
        enqueued_at=row['enqueued_at'],
        attempts=row['attempts'],
        next_attempt_at=row['next_attempt_at'],
        last_error=row['last_error'],
        escalation_id=row['escalation_id'],
    )


def named_tests(s: FlakeSuppression) -> tuple[str, ...]:
    """The tests *s* names, the sentinel excepted: the ids a debt can be opened for.

    Verdict-agnostic: whether the observation makes its tests debt-eligible is the
    caller's decision.  Never raises; a malformed *s* is logged and names nothing.

    A bare-``str`` ``test_ids`` is coerced here WITHOUT the warning: the same payload's
    rows are derived by :func:`enqueue_debt`, which warns once for it.
    """
    try:
        test_ids = _coerce_test_ids(
            s.test_ids, context=f'call_site={s.call_site}', warn=False,
        )
        return tuple(test_id for test_id in test_ids if test_id != UNKNOWN_TEST_ID)
    except Exception:
        logger.warning(
            'flake_ledger: could not read the tests an observation names', exc_info=True,
        )
        return ()


@dataclass(frozen=True)
class _ValidItem:
    key: _OccurrenceKey
    head: str | None
    observing_task_id: str | None
    context: str


def _validated(item: DueItem) -> _ValidItem:
    """*item* canonicalised to its stored form; raises on an item no drain could act on."""
    if not isinstance(item.test_id, str) or not item.test_id or item.test_id == UNKNOWN_TEST_ID:
        raise ValueError(f'not a debt-eligible test id: {item.test_id!r}')
    if not isinstance(item.context, str):
        raise TypeError(f'context must be JSON text, got {type(item.context).__name__}')
    call_site = FlakeCallSite(item.call_site)
    observed_at = _normalize_observed_at(item.observed_at)
    return _ValidItem(
        key=(item.test_id, observed_at, call_site.value),
        head=item.head or None,
        observing_task_id=item.observing_task_id,
        context=item.context,
    )


def _valid_items(items: Sequence[DueItem]) -> list[_ValidItem]:
    valid: list[_ValidItem] = []
    for item in items:
        try:
            valid.append(_validated(item))
        except Exception:
            logger.warning(
                'flake_ledger: skipping a due item no drain could act on: %r',
                item,
                exc_info=True,
            )
    return valid


def enqueue_debt(
    db_path: Path,
    project_id: str,
    items: Sequence[DueItem],
    *,
    occurrence: FlakeSuppression | None = None,
    merge_sha: str | None = None,
    task_id: str | None = None,
    now: datetime | None = None,
) -> None:
    """Write *occurrence*'s ``flake_occurrence`` rows (as :func:`record_flake_occurrence`
    writes them) and *items* on the due list, in ONE transaction on ONE connection.

    So no restart can leave the rows without their items or the items without their
    rows: both land, or neither does.  *occurrence* is optional — a lane recovery owes
    items and has no rows.

    A REPLAY LANDS NO ITEM.  An item whose key ``(test_id, observed_at, call_site)`` is
    one of THIS call's rows is enqueued only if that row was newly inserted here.  The
    occurrence table's dedup already makes a replay write no row; this makes a replay
    after the drain deleted the item write no item either, which would otherwise re-run
    :func:`open_debt` for an old observation.  The item table's own unique index on the
    same key folds every other repeat.

    An item that fails validation (the sentinel, an unknown call site, a malformed
    stamp) is skipped with a warning and the rest still land.  ``enqueued_at`` and
    ``next_attempt_at`` are *now*, the ledger-owned wall clock by default.

    Never raises (B12): a failure logs with ``exc_info`` and writes nothing.
    """
    try:
        valid = _valid_items(items)
        rows = (
            []
            if occurrence is None
            else _occurrence_rows(project_id, occurrence, merge_sha=merge_sha, task_id=task_id)
        )
        if not rows and not valid:
            return
        enqueued_at = _canonicalize_utc(now or datetime.now(UTC), origin='enqueue_debt')
        conn = _open(db_path)
        try:
            inserted = _write_observation(conn, rows)
            replayed = {_row_key(row) for row in rows} - inserted
            for item in valid:
                if item.key in replayed:
                    continue
                test_id, observed_at, call_site = item.key
                conn.execute(
                    'INSERT OR IGNORE INTO flake_debt_due '
                    '(test_id, project_id, observed_at, call_site, head, observing_task_id, '
                    ' context, enqueued_at, next_attempt_at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        test_id, project_id, observed_at, call_site, item.head,
                        item.observing_task_id, item.context, enqueued_at, enqueued_at,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning(
            'flake_ledger: failed to record an observation (%d due item(s), occurrence '
            'call_site=%s); nothing was written',
            len(items),
            getattr(occurrence, 'call_site', None),
            exc_info=True,
        )


def read_due_items(db_path: Path, *, now: datetime, limit: int) -> list[DueRow] | None:
    """The due items whose ``next_attempt_at`` has passed at *now*, at most *limit*, in
    DRAIN ORDER: tests by their earliest ``enqueued_at``, a test's items by
    ``observed_at``.

    A test's items at or after its first item that is NOT yet due wait behind it, so a
    backed-off item is never overtaken by a later item of the same test.

    ``None`` when the read failed (B12): a failed read is not an empty list.
    """
    try:
        stamp = _canonicalize_utc(now, origin='read_due_items')
        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'WITH tests AS ('
                '    SELECT test_id, '
                '           MIN(enqueued_at) AS first_enqueued, '
                '           MIN(CASE WHEN next_attempt_at > :now THEN observed_at END) '
                '               AS waiting_from '
                '      FROM flake_debt_due GROUP BY test_id) '
                'SELECT d.* FROM flake_debt_due d JOIN tests t USING (test_id) '
                ' WHERE d.next_attempt_at <= :now '
                '   AND (t.waiting_from IS NULL OR d.observed_at < t.waiting_from) '
                ' ORDER BY t.first_enqueued, d.test_id, d.observed_at, d.id '
                ' LIMIT :limit',
                {'now': stamp, 'limit': max(0, limit)},
            ).fetchall()
        finally:
            conn.close()
        return [_to_due_row(r) for r in rows]
    except Exception:
        logger.warning('flake_ledger: failed to read due items from %s', db_path, exc_info=True)
        return None


def _write_due(db_path: Path, sql: str, params: Mapping[str, Any]) -> int:
    conn = _open(db_path)
    try:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def delete_due_item(db_path: Path, item_id: int) -> bool:
    """Remove the due item *item_id*; ``True`` iff a row was deleted.  Never raises."""
    try:
        return _write_due(db_path, 'DELETE FROM flake_debt_due WHERE id = :id', {'id': item_id}) == 1
    except Exception:
        logger.warning('flake_ledger: failed to delete due item %s', item_id, exc_info=True)
        return False


def record_failed_due_attempt(
    db_path: Path, item_id: int, *, reason: str, next_attempt_at: datetime,
) -> bool:
    """Count one failed attempt on *item_id*, with its *reason*, and hold it until
    *next_attempt_at*.  ``True`` iff the item was updated.  Never raises."""
    try:
        return _write_due(
            db_path,
            'UPDATE flake_debt_due SET attempts = attempts + 1, '
            '    next_attempt_at = :next_attempt_at, last_error = :reason '
            'WHERE id = :id',
            {
                'id': item_id,
                'reason': reason,
                'next_attempt_at': _canonicalize_utc(
                    next_attempt_at, origin='record_failed_due_attempt',
                ),
            },
        ) == 1
    except Exception:
        logger.warning(
            'flake_ledger: failed to record a failed attempt on due item %s',
            item_id,
            exc_info=True,
        )
        return False


def name_stale_due_items(db_path: Path, escalation_id: str, *, stale_before: datetime) -> int:
    """Stamp every stale item no escalation has named yet with *escalation_id*, and
    return how many were stamped.  Never raises; ``0`` on failure."""
    try:
        return _write_due(
            db_path,
            'UPDATE flake_debt_due SET escalation_id = :escalation_id '
            f'WHERE escalation_id IS NULL AND {_IS_STALE_SQL}',
            {
                'escalation_id': escalation_id,
                'stale_before': _canonicalize_utc(stale_before, origin='name_stale_due_items'),
            },
        )
    except Exception:
        logger.warning(
            'flake_ledger: failed to name stale due items for %s',
            escalation_id,
            exc_info=True,
        )
        return 0


def summarize_due(db_path: Path, *, stale_before: datetime) -> DueSummary | None:
    """The due list's :class:`DueSummary`, with items enqueued at or before
    *stale_before* counted as stale.  ``None`` when the read failed.  Never raises."""
    try:
        params = {
            'stale_before': _canonicalize_utc(stale_before, origin='summarize_due'),
            'sample': _STALE_SAMPLE_SIZE,
        }
        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            size, stale, unnamed_stale = conn.execute(
                'SELECT COUNT(*), '
                f'       COALESCE(SUM({_IS_STALE_SQL}), 0), '
                f'       COALESCE(SUM({_IS_STALE_SQL} AND escalation_id IS NULL), 0) '
                '  FROM flake_debt_due',
                params,
            ).fetchone()
            oldest = conn.execute(
                'SELECT * FROM flake_debt_due ORDER BY enqueued_at, id LIMIT 1'
            ).fetchone()
            sample = conn.execute(
                f'SELECT * FROM flake_debt_due WHERE {_IS_STALE_SQL} '
                'ORDER BY enqueued_at, id LIMIT :sample',
                params,
            ).fetchall()
            pending_since = conn.execute(
                'SELECT test_id, MIN(enqueued_at) FROM flake_debt_due GROUP BY test_id'
            ).fetchall()
        finally:
            conn.close()
        return DueSummary(
            size=size,
            oldest=None if oldest is None else _to_due_row(oldest),
            stale=stale,
            unnamed_stale=unnamed_stale,
            stale_sample=tuple(_to_due_row(r) for r in sample),
            pending_since={test_id: since for test_id, since in pending_since},
        )
    except Exception:
        logger.warning('flake_ledger: failed to summarize the due list in %s', db_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# The de-flake task ζ files at write time (§5.5, §5.9)
# ---------------------------------------------------------------------------

# The binding corollary of §5.5, carried IN the filed task rather than left for the
# de-flake agent to already know.  Task 1836 widened a 10s timeout to 30s and thereby
# MASKED a real SIGHUP bug for a day, until task 1841 found it; `esc-3650-2` was itself
# a real production bug (SIGPIPE under `pipefail`, task 3552), not a bad test.
#
# DELIBERATE NON-DEDUP with `chronic_flake.py::_ROOT_CAUSE_INSTRUCTION`, which says the
# same thing in remedy framing ('ROOT-CAUSE this...').  The two texts differ in FRAMING,
# which is the entire point of §5.5: chronic_flake's presupposes a fix, and this one must
# not.  Sharing one string would force one framing on both.  Importing `chronic_flake`
# here would also break this module's import discipline: the dependency runs ledger ->
# nothing-but-`shared`, and chronic_flake is the layer ABOVE, so the edge would point
# backwards.  (It would NOT drag in `orchestrator.config`, as this comment previously
# claimed: chronic_flake imports that only under `if TYPE_CHECKING`, and at runtime pulls
# stdlib plus `shared.safe_io`/`shared.task_statuses` only -- measured, not assumed.  The
# layering reason is real; the import-weight one was not.)
# Task κ — which migrates chronic_flake onto this ledger — OWNS converging the two texts.
NEVER_WIDEN_A_TIMEOUT = (
    'CONSTRAINT, binding: do NOT widen a timeout, lengthen a sleep, or add a retry as '
    'the remedy. That makes the observation rarer and harder to reproduce without '
    'removing its cause, and it has already masked a real defect here (task 1836 '
    'widened 10s->30s and hid a genuine SIGHUP bug for a day, until task 1841 found '
    'it). Prefer condition-polling (wait for the state the test actually depends on) '
    'and structural asserts (assert on state, not on log/prose output).'
)

# Rationale: plans/flake-ledger-prd.md §5.9 ("Removing the test from the ledger").
RESPONSIBILITY_FIX_ROOT_DEFECT = (
    'You are responsible for finding and fixing the ROOT DEFECT behind this observation'
)
RESPONSIBILITY_REMOVE_FROM_LEDGER = (
    'You do NOT write to the flake ledger, and must not ask anyone to: the ledger '
    'closes this debt cycle itself once it observes this task done. Your part is to '
    'keep the test\'s pytest node-id byte-identical (no rename, no parametrize, no '
    'move into a class), because the ledger keys its debt row on that id and a '
    'changed id orphans the row and hides a recurrence'
)


class DebtTrigger(StrEnum):
    """What made a test debt-eligible on the offline lane (offline-lane PRD D6).  The
    merge gate's value is a follow-up's to add."""

    cleared = 'cleared'
    recovered = 'recovered'


@dataclass(frozen=True)
class RecentOccurrence:
    """One recent red of the test, as its de-flake task lists it."""

    pass_id: str | None
    head: str | None
    kind: EvidenceKind
    first_line: str


DEBT_CONTEXT_EVIDENCE_LIMIT: Final = 5


@dataclass(frozen=True)
class DebtContext:
    """Why an observation made a test owe debt.  It shapes the de-flake task's text and
    metadata (offline-lane PRD D6, §5.6), never whether the task is filed."""

    call_site: FlakeCallSite
    trigger: DebtTrigger
    command: str | None = None
    working_directory: str | None = None
    reproduce: str | None = None
    heads: tuple[str, ...] = ()
    pass_ids: tuple[str, ...] = ()
    evidence: tuple[RecentOccurrence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'call_site', FlakeCallSite(self.call_site))
        object.__setattr__(self, 'trigger', DebtTrigger(self.trigger))
        if len(self.evidence) > DEBT_CONTEXT_EVIDENCE_LIMIT:
            raise ValueError(
                f'a DebtContext carries at most {DEBT_CONTEXT_EVIDENCE_LIMIT} evidence '
                f'entries; got {len(self.evidence)}'
            )


def debt_context_to_json(ctx: DebtContext) -> str:
    """*ctx* as the due list's ``context`` text."""
    return json.dumps(asdict(ctx))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError(f'expected a list of strings, got {value!r}')
    return tuple(value)


def _recent_occurrence_from_json(entry: Mapping[str, Any]) -> RecentOccurrence:
    kwargs = _declared(RecentOccurrence, entry)
    kwargs['kind'] = EvidenceKind(kwargs.get('kind'))
    return RecentOccurrence(**kwargs)


def debt_context_from_json(text: str) -> DebtContext | None:
    """The :class:`DebtContext` a due item's *text* holds: ``None`` for ``'{}'``, and
    ``None`` plus a warning for anything malformed.  Never raises."""
    try:
        raw = json.loads(text)
        if raw == {}:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f'a debt context is a JSON object, got {type(raw).__name__}')
        return DebtContext(
            call_site=raw['call_site'],
            trigger=raw['trigger'],
            command=raw.get('command'),
            working_directory=raw.get('working_directory'),
            reproduce=raw.get('reproduce'),
            heads=_strings(raw.get('heads', [])),
            pass_ids=_strings(raw.get('pass_ids', [])),
            evidence=tuple(_recent_occurrence_from_json(e) for e in raw.get('evidence', [])),
        )
    except Exception:
        logger.warning(
            'flake_ledger: could not decode a debt context (%r); filing without it',
            text,
            exc_info=True,
        )
        return None


#: What each lane trigger observed, worded as the observation (§5.5, §5.6).
LANE_TRIGGER_OBSERVATIONS: Final[Mapping[DebtTrigger, str]] = {
    DebtTrigger.cleared: (
        'The test failed in an offline-lane detection run and passed an isolated, serial '
        're-run at the same head.'
    ),
    DebtTrigger.recovered: (
        'The test failed its offline-lane re-run and passed on a later lane pass at the '
        'same head.'
    ),
}
LANE_NOTHING_SUPPRESSED = 'Nothing was suppressed, and no merge depended on this result.'
LIVE_DEPENDENCY_ROOT_DEFECT = (
    'If the test exercises a live dependency, the root defect may be in how the product '
    'handles a slow or failed call to it: changing that handling is a fix; relaxing the '
    'test is not.'
)
LANE_TEST_NOT_ON_MERGE_GATE = (
    'The merge gate does not run this test, so your own run of the reproduce line is '
    "this task's evidence, and the offline lane will observe the test again after your "
    'change lands.'
)

_NOT_RECORDED = '(not recorded)'


def _deflake_title(row: DebtRow, observation: str) -> str:
    """The one shape of a de-flake title: the test, what was observed, and the cycle."""
    return f'Flake debt: {row.test_id} — {observation}, cycle {row.open_count}'


def _evidence_block(evidence: Sequence[RecentOccurrence]) -> str:
    """The recent occurrences, in the order given (most recent last); ``''`` for none."""
    if not evidence:
        return ''
    lines = [f'Recent occurrences (at most {DEBT_CONTEXT_EVIDENCE_LIMIT}, most recent last):\n']
    lines += [
        f'  pass {e.pass_id or _NOT_RECORDED}  head {(e.head or _NOT_RECORDED)[:12]}  '
        f'{EvidenceKind(e.kind).value}: {e.first_line}\n'
        for e in evidence
    ]
    return ''.join(lines)


def _field_lines(fields_: Sequence[tuple[str, object]]) -> str:
    return ''.join(f'  {label + ":":<20}{value}\n' for label, value in fields_)


def _joined(values: Sequence[str]) -> str:
    return ', '.join(values) if values else _NOT_RECORDED


def _lane_task_text(row: DebtRow, ctx: DebtContext) -> tuple[str, str]:
    """The title and description of a lane debt's de-flake task (offline-lane PRD §5.6)."""
    on = 'on the offline lane' if ctx.command is None else f'on the offline lane ({ctx.command})'
    evidence = _evidence_block(ctx.evidence)
    description = (
        f'{LANE_TRIGGER_OBSERVATIONS[ctx.trigger]} {LANE_NOTHING_SUPPRESSED}\n'
        f'\n'
        + _field_lines([
            ('test', row.test_id),
            ('project', row.project_id),
            ('opened_at', row.opened_at),
            ('open_count', row.open_count),
            ('trigger', ctx.trigger.value),
            ('command', ctx.command or _NOT_RECORDED),
            ('working directory', ctx.working_directory or _NOT_RECORDED),
            ('reproduce', ctx.reproduce or _NOT_RECORDED),
            ('heads', _joined(ctx.heads)),
            ('pass ids', _joined(ctx.pass_ids)),
        ])
        + (f'\n{evidence}' if evidence else '')
        + f'\n'
        f'{RESPONSIBILITY_FIX_ROOT_DEFECT}.\n'
        f'{RESPONSIBILITY_REMOVE_FROM_LEDGER}.\n'
        f'\n'
        f'{NEVER_WIDEN_A_TIMEOUT}\n'
        f'\n'
        f'{LIVE_DEPENDENCY_ROOT_DEFECT}\n'
        f'\n'
        f'{LANE_TEST_NOT_ON_MERGE_GATE}\n'
        f'\n'
        f'Auto-filed by the flake ledger when the debt row was opened (PRDs '
        f'plans/flake-ledger-prd.md §5.9 and plans/offline-lane-red-path-prd.md §5.6).'
    )
    return _deflake_title(row, f'{ctx.trigger.value} {on}'), description


def _merge_gate_task_text(row: DebtRow, evidence: Sequence[RecentOccurrence]) -> tuple[str, str]:
    """The title and description of a merge-gate suppression's de-flake task, with an
    evidence block after the field block only when there is evidence."""
    block = _evidence_block(evidence)
    description = (
        f'The flake ledger recorded a `{FlakeVerdict.passes_in_isolation.value}` '
        f'observation for this test and SUPPRESSED the red it produced, so a merge '
        f'landed on a gate that had gone red:\n'
        f'\n'
        f'  test:        {row.test_id}\n'
        f'  project:     {row.project_id}\n'
        f'  opened_at:   {row.opened_at}\n'
        f'  open_count:  {row.open_count}\n'
        + (f'\n{block}' if block else '')
        + f'\n'
        f'What that verdict means, precisely: the test failed under load in the '
        f'merge verify and then PASSED on an isolated, serial re-run. Read that as '
        f'evidence about the SYSTEM, not as a verdict on the test — a test that '
        f'passes alone and fails under load has repeatedly turned out to be a real '
        f'production defect here (task 1836 -> 1841: a widened timeout masked a '
        f'genuine SIGHUP bug for a day; esc-3650-2 was a real SIGPIPE-under-pipefail '
        f'bug, task 3552). Diagnose before you conclude the test is at fault.\n'
        f'\n'
        f'{RESPONSIBILITY_FIX_ROOT_DEFECT}.\n'
        f'{RESPONSIBILITY_REMOVE_FROM_LEDGER}.\n'
        f'\n'
        f'{NEVER_WIDEN_A_TIMEOUT}\n'
        f'\n'
        f'Auto-filed by the flake ledger when the debt row was opened (PRD '
        f'plans/flake-ledger-prd.md §5.9). The merge already landed, so this is '
        f'visible, owned debt rather than an incident.'
    )
    title = _deflake_title(
        row, f'{FlakeVerdict.passes_in_isolation.value} after a merge-gate red was suppressed',
    )
    return title, description


def _deflake_metadata(row: DebtRow, ctx: DebtContext | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'spawn_context': 'flake_ledger_debt',
        'flake_debt_test': row.test_id,
        'flake_debt_open_count': row.open_count,
        'flake_debt_opened_at': row.opened_at,
    }
    if ctx is not None:
        metadata['flake_debt_call_site'] = ctx.call_site.value
        metadata['flake_debt_trigger'] = ctx.trigger.value
        if ctx.command is not None:
            metadata['flake_debt_command'] = ctx.command
    return metadata


def build_deflake_task_arguments(row: DebtRow, context: DebtContext | None = None) -> dict:
    """The ``submit_task`` argument block for the task that owns *row*'s debt (§5.9).

    Pure and side-effect free: :func:`open_debt` files the result, this only shapes it.

    Follows ``orchestrator/src/orchestrator/chronic_flake.py::build_chronic_flake_fix_task_arguments``'s
    block shape (title / description / priority / metadata), RE-FRAMED per §5.5 so the
    text records the OBSERVATION and never presupposes a remedy — see
    :data:`NEVER_WIDEN_A_TIMEOUT` for why the instruction text is re-derived here rather
    than shared with that module's ``_ROOT_CAUSE_INSTRUCTION``, and which task owns
    converging them.

    Two keys are load-bearing and easy to "clean up" wrongly:

    - ``planning_mode: True`` — this is what makes ``submit_task`` return a REAL task id
      SYNCHRONOUSLY (fused-memory ``task_interceptor.py::_submit_task_planning_mode``),
      bypassing the curator ticket store, which PRD §5.4 explicitly sanctions "for
      exactly the tasks this subsystem files".  The default two-phase path returns only
      ``{'ticket': 'tkt_...'}`` and the curator decides create/combine/drop
      ASYNCHRONOUSLY — so there would be no id to store in ``owner_task_id`` and the task
      might never exist at all, making §5.9's invariant silently hollow exactly where
      §5.7 has already landed the merge.  The task is born ``deferred``; the caller
      completes it with ``commit_planning``.
    - NO ``project_root`` — :func:`open_debt` takes a ``db_path``, not a project root
      (§8.3), and the only way to name one from here would be ``db_path.parent.parent
      .parent``, a silent position-dependent inversion of :func:`ledger_db_path` that
      breaks for every non-standard path.  The adapter already HOLDS the project root and
      injects it (``chronic_flake.py::SchedulerChronicFlakeTaskClient.submit_task``).

    The title names the cycle on every cycle, the first included: the task store folds
    a submission whose title matches an existing non-cancelled task, DONE ones
    included, so a re-entry can never be filed under its finished owner's title, folded
    onto that done task and refused as no owner.

    ``metadata.files`` is likewise omitted deliberately: a ``test_id`` may be a reify
    script-suite NAME rather than a path, and a wrong or over-wide concurrency lock
    derived from a guess is worse than no lock — the de-flake agent discovers the files.

    *context*, when given, words an offline-lane debt in the lane's terms (offline-lane
    PRD §5.6) and stamps its call site, trigger and command into the metadata; any other
    context adds only its evidence.  With none the block is the merge gate's.
    """
    if context is not None and context.call_site == FlakeCallSite.offline_lane:
        title, description = _lane_task_text(row, context)
    else:
        title, description = _merge_gate_task_text(row, () if context is None else context.evidence)
    return {
        'title': title,
        'description': description,
        'priority': 'medium',
        'planning_mode': True,
        'metadata': _deflake_metadata(row, context),
    }


def _to_debt_row(row: sqlite3.Row) -> DebtRow:
    return DebtRow(
        test_id=row['test_id'],
        project_id=row['project_id'],
        opened_at=row['opened_at'],
        resolved_at=row['resolved_at'],
        owner_task_id=row['owner_task_id'],
        open_count=row['open_count'],
        prior_resolved_at=row['prior_resolved_at'],
        prior_resolving_commit=row['prior_resolving_commit'],
        last_occurrence_at=row['last_occurrence_at'],
    )


def read_debt(db_path: Path, test_id: str) -> DebtRow | None:
    """The debt row for *test_id*, or ``None`` if the test has never been suppressed.

    A primary-key lookup — which is exactly the shape §5.3 says makes η's recurrence
    trigger a lookup rather than a scan.  RESOLVED rows are returned too: they are
    retained deliberately (§5.2) because the recurrence trigger reads them.
    """
    try:
        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM flake_debt WHERE test_id = ?', (test_id,)).fetchone()
        finally:
            conn.close()
        return _to_debt_row(row) if row is not None else None
    except Exception:
        logger.warning(
            'flake_ledger: failed to read debt for test_id=%s',
            test_id,
            exc_info=True,
        )
        return None


class FlakeLedgerTaskClient(Protocol):
    """The task seam :func:`open_debt` needs to enforce §5.9, and :func:`resolve_debt`
    needs to corroborate a resolution — declared STRUCTURALLY (``typing.Protocol``) so
    it is machine-checked (INV-1) rather than prose, and so no import edge is created to
    the module that satisfies it.

    ``orchestrator/src/orchestrator/chronic_flake.py::SchedulerChronicFlakeTaskClient``
    is the concrete adapter.  It is NOT imported here, deliberately: this module depends
    on ``shared`` alone and sits BELOW ``chronic_flake``, so importing it would point the
    dependency backwards through the layer that is meant to build on this one.

    Every method must degrade rather than raise where it can, but the ledger does not
    RELY on that — :func:`_ensure_owner_task` and :func:`resolve_debt` guard each call
    independently, because a partial or older adapter (one lacking a method entirely,
    hence ``AttributeError``) is a shape that really arrives.

    ``get_statuses`` and ``get_task`` go further and report their failures IN BAND,
    because degrading to a bare ``{}`` or ``None`` is not a safe degrade at this seam:
    an empty mapping and a missing task already MEAN something here (see their
    docstrings).
    """

    async def submit_task(self, arguments: dict) -> SubmitOutcome:
        """File the task described by *arguments* (see
        :func:`build_deflake_task_arguments`) and return its typed outcome.  The block
        carries ``planning_mode: True``, so a ``SubmittedTask`` is required: the
        invariant is stored against its id.  A ``SubmittedTicket``, a ``SubmitFailed``,
        and a ``combined`` task whose status is terminal or unknown are failed filings
        (:func:`_owner_from_submission`)."""
        ...

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        """Live ``{id: status}`` for *ids*, as a ``(statuses, error)`` pair —
        ``scheduler.py::SchedulerFacade.get_statuses``' convention, and the shape that
        makes the seam's FAILURE/ABSENCE distinction machine-checked (INV-1) instead of
        resting on an adapter's exception discipline.

        An unknown id is OMITTED from the mapping rather than reported as a status, and
        the ledger reads that omission as a CORROBORATED ABSENCE — the task was deleted
        — and files a replacement for it.  So a FAILED read must NEVER be reported as an
        empty mapping: put it in the error half.  An implementation that swallows a
        failure into ``{}`` is indistinguishable from a real absence at this seam, and
        the ledger will act on it — one duplicate de-flake task per suppression, for the
        length of the outage, with no rate limit to bound it.
        """
        ...

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        """The task *task_id*, read LIVE, as a ``(task, error)`` pair that keeps failure
        and absence machine-distinguishable (INV-1, the lesson of :meth:`get_statuses`):

        - ``(task, None)`` — a task was read;
        - ``(None, None)`` — a CORROBORATED ABSENCE: the server answered that no such
          task exists;
        - ``(None, exc)`` — a FAILED read: nothing is known about the task.

        :func:`resolve_debt` stamps a resolution ONLY from a task it actually read, so
        an absence and a failure both leave the cycle open; the pair is what lets its
        log say which of the two happened.  :func:`open_debt`'s owner enforcement acts
        on the same read, and there the distinction is :meth:`get_statuses`' again: an
        absence files a replacement owner, a failure files nothing.
        """
        ...

    async def commit_planning(self, task_ids: list[str]) -> None:
        """Release planning-mode tasks from ``deferred`` to ``pending`` — the second
        phase of the initial filing, without which the task never dispatches."""
        ...


@dataclass(frozen=True)
class _OwnerRead:
    """One SUCCESSFUL live ``get_task`` read of the owner *owner_task_id*.  *task* is
    ``None`` when the server answered that no such task exists: a corroborated absence.

    :func:`open_debt` takes it from its lazy close and hands it to ζ's enforcement, so
    an owner is read ONCE per call.  One live fact then has one source, and the
    resolution and the owner verdict cannot disagree about it.
    """

    owner_task_id: str
    task: dict | None

    @property
    def status(self) -> str | None:
        return None if self.task is None else self.task.get('status')


class _OwnerClaimLost(RuntimeError):
    """The conditional owner UPDATE matched no row — another lane won the claim, or the
    row is gone.  Raised by :func:`_write_owner_task_id` so the filed task is reported as
    the ORPHAN it is, instead of a success line for a pointer that never landed."""


def _write_owner_task_id(
    db_path: Path, test_id: str, owner_task_id: str, *, expected_owner: str | None,
) -> None:
    """CLAIM *test_id*'s debt row for *owner_task_id*, conditionally on
    *expected_owner* still being what is on disk.

    Exists so :func:`_ensure_owner_task` never hand-rolls an UPDATE at a call site —
    contract 1 of the module docstring ("writes go only through this API") applies
    inside the module too, since a second spelling of this statement is exactly how the
    column would drift.

    CONDITIONAL, AND THE CONDITION IS THE POINT.  The caller decided to write this
    pointer by READING ``row.owner_task_id`` (from ``open_debt``'s read-back), then
    awaiting a task filing — so between the read and this write sit at least two
    ``await`` points, and every ``open_debt`` call is one.  Two merge lanes suppressing
    the same test in one event loop genuinely interleave there.  An UNCONDITIONAL
    ``SET owner_task_id = ?`` would be exactly the read-modify-write race §5.1 chose
    SQLite to avoid (see ``open_debt``'s single-statement upsert, which carries that
    reasoning), reintroduced inside the ledger itself on the one column §5.9's whole
    dedup rests on: the loser would silently clobber the winner's pointer, and the
    winner's task — a real, non-terminal task — would never be referenced again.

    So the WHERE clause carries the expectation:

    * ``expected_owner is None`` — a FRESH claim, guarded by ``owner_task_id IS NULL``.
      A concurrent lane that already claimed the row wins, and this filing is the orphan.
    * ``expected_owner`` set — a REPLACEMENT for the terminal/absent owner the caller
      corroborated, guarded by ``owner_task_id = <that id>``.  If it changed underneath,
      what is there now was not corroborated by this call and must not be overwritten.

    Raises on a ledger failure, and on a matched-zero-rows claim
    (:class:`_OwnerClaimLost`), deliberately: the sole caller wraps the whole filing in
    its own guard and knows how to degrade loudly, and swallowing either would report a
    filing as recorded when the pointer never landed.
    """
    conn = _open(db_path)
    try:
        if expected_owner is None:
            cursor = conn.execute(
                'UPDATE flake_debt SET owner_task_id = ? '
                'WHERE test_id = ? AND owner_task_id IS NULL',
                (owner_task_id, test_id),
            )
        else:
            cursor = conn.execute(
                'UPDATE flake_debt SET owner_task_id = ? '
                'WHERE test_id = ? AND owner_task_id = ?',
                (owner_task_id, test_id, expected_owner),
            )
        conn.commit()
        if cursor.rowcount == 0:
            # Not a ledger failure — a LOST CLAIM.  Distinguished from an OperationalError
            # because the remedy differs: nothing is retried here, and the next
            # suppression will corroborate whatever owner did land rather than file again.
            raise _OwnerClaimLost(
                f'owner claim for test_id={test_id!r} matched no row '
                f'(expected owner {expected_owner!r}); another lane claimed it first, '
                f'or the row is gone'
            )
    finally:
        conn.close()


async def _owner_status(
    task_client: FlakeLedgerTaskClient, owner_task_id: str, owner_read: _OwnerRead | None,
) -> tuple[str | None, Exception | None]:
    """*owner_task_id*'s LIVE status as a ``(status, error)`` pair, the status ``None``
    for a corroborated absence.

    It is *owner_read*, the lazy close's read, when that read is of THIS owner, so the
    owner is not read twice.  Otherwise (that read failed or timed out, or a concurrent
    lane re-pointed the row since) it is a ``get_statuses`` read, whose failure arrives
    in TWO shapes that land on ONE path, so the fail-safe direction cannot drift between
    them.  The real adapter never raises: it reports the failure as the ERROR HALF of
    its pair, because a failure returned as a bare ``{}`` is byte-identical to a
    corroborated absence and would be acted on as one.  A partial or older adapter, or
    one returning a non-tuple that the unpack rejects, raises instead, and that is
    caught here.
    """
    if owner_read is not None and owner_read.owner_task_id == owner_task_id:
        return owner_read.status, None
    try:
        statuses, error = await task_client.get_statuses([owner_task_id])
    except Exception as exc:
        return None, exc
    if error is not None:
        return None, error
    return statuses.get(owner_task_id), None


@dataclass(frozen=True)
class _SubmittedOwner:
    task_id: str
    needs_commit_planning: bool
    combined: bool


@dataclass(frozen=True)
class _FailedFiling:
    reason: str


def _owner_from_submission(outcome: SubmitOutcome) -> _SubmittedOwner | _FailedFiling:
    """The task a ``submit_task`` *outcome* makes the debt row's owner, or the reason
    the filing failed.

    A ``combined`` outcome means the server folded the filing into an existing task.
    A live one owns the debt as it stands; it needs ``commit_planning`` only while it is
    still ``deferred``, since committing an open task would run ``set_task_status``
    gates on it.  A terminal or unknown one swallowed the filing: nothing open owns the
    debt.
    """
    match outcome:
        case SubmittedTask(task_id=task_id, combined=False):
            return _SubmittedOwner(task_id, needs_commit_planning=True, combined=False)
        case SubmittedTask(task_id=task_id, status=status) if status is None or status in TERMINAL:
            return _FailedFiling(
                f'submit_task combined the filing into task {task_id} '
                f'(status={status!r}), which cannot own it — the filing was swallowed'
            )
        case SubmittedTask(task_id=task_id, status=status):
            return _SubmittedOwner(
                task_id, needs_commit_planning=status == 'deferred', combined=True,
            )
        case SubmittedTicket(ticket_id=ticket_id):
            return _FailedFiling(
                f'submit_task returned ticket {ticket_id!r}, not a task — '
                'planning_mode did not take effect'
            )
        case SubmitFailed(reason=reason):
            return _FailedFiling(reason)
        case _:
            return _FailedFiling(
                f'submit_task answered a {type(outcome).__name__}, not a SubmitOutcome'
            )


async def _ensure_owner_task(
    db_path: Path,
    project_id: str,
    row: DebtRow,
    *,
    task_client: FlakeLedgerTaskClient | None,
    owner_read: _OwnerRead | None = None,
) -> DebtRow:
    """Enforce §5.9's invariant for *row*, returning the row as it stands afterwards.

    > Any test in the flaky ledger has a non-terminal de-flake task explicitly
    > responsible both for fixing the root defect and for removing the test from the
    > ledger.

    Enforced at WRITE TIME, inside :func:`open_debt`, so it is SELF-MAINTAINING rather
    than audited after the fact.  Dedup follows
    ``orchestrator/src/orchestrator/flake_recorder.py::_bump_suppression_streak_and_maybe_escalate``,
    which already dedupes on a fixed sentinel; here the sentinel is ``owner_task_id`` on
    the debt row.

    INV-3, verbatim and load-bearing: a stored ``owner_task_id`` is a SNAPSHOT.  The
    task behind it may have gone terminal, been cancelled, or been deleted since it was
    written, so it is re-read against LIVE status every time and NEVER assumed
    still-open.  Short-circuiting on a non-NULL ``owner_task_id`` would satisfy the
    invariant's letter while pointing stale rows at done tasks forever.  The live read is
    *owner_read* when :func:`open_debt`'s lazy close made one of this owner, so each call
    reads its owner once (:func:`_owner_status`).  Only a cancelled or absent owner is
    REPLACED: a done one is kept, because on an open row it means the lazy close did not
    land and the next suppression retries it (the ``done`` verdict below).

    Why :func:`shared.task_statuses.owner_liveness` has four verdicts and not a boolean.
    ``planning_mode=True`` creates the task ``deferred``, so between ``submit_task`` and
    ``commit_planning`` a crash or a failed second round-trip leaves a task that EXISTS
    but will never be dispatched.  A boolean "open" predicate classifies that as not
    open, so every subsequent suppression of the same test would file another orphan —
    a duplicate-generating loop that gets worse the more the test flakes.  The
    ``deferred`` branch finishes the half-done filing instead.

    ``done`` is split out of ``closed`` (task η) because its correct action differs.
    With a ``task_client`` wired, :func:`open_debt`'s lazy close resolves a done owner's
    cycle and the re-entry upsert discharges that owner, so a done owner on an OPEN row
    can only mean that resolution did not land: ``get_task`` failed, raised, timed out
    or is missing and the ``get_statuses`` fallback read the owner done, or the stamp
    itself failed.  Replacing it would overwrite the only pointer to the fix that did
    not hold, so the cycle would never close, ``prior_resolving_commit`` would never be
    written, and no ``regressed_after_resolution`` L2 would fire.

    A SUCCESSFUL read that finds no such task — ``get_task`` answering
    ``TaskNotFoundError``, or ``get_statuses`` silently OMITTING the id — is a
    corroborated absence (the task was deleted) and classifies ``closed``.  A FAILED read
    never reaches the classification: both reads report failure in band — ``get_task``'s
    ``(task, error)`` and :class:`FlakeLedgerTaskClient.get_statuses`' ``(statuses,
    error)`` pair — and the code below returns early on a non-``None`` error.  That was
    NOT true while the adapter swallowed failures into a bare ``{}``: it arrived
    indistinguishable from an absence and was classified ``closed``, filing a replacement
    task for an owner that was alive and merely unread.

    COUPLING RULE, binding (§5.9): the ledger READS task status but never WRITES it,
    except the initial filing.  It never marks a task done, never blocks one, never
    reprioritises one — a de-flake task's lifecycle belongs to the orchestrator, and the
    ledger only observes it.  That is what keeps INV-6 (``status-matches-liveness``) N/A
    by construction rather than merely satisfied.

    Never raises: it runs under :func:`open_debt`'s catch-all (B12) and step-8 adds its
    own inner per-call guards so ONE failing client call costs only that signal.
    """
    if task_client is None:
        # A legitimate configuration (a CLI or a test) — but it is also the one way the
        # invariant silently stops being enforced in production, so it is stated rather
        # than assumed.
        logger.info(
            'flake_ledger: no task_client wired — debt for test_id=%s is opened with no '
            'owner and §5.9 is NOT enforced for it (ι renders it as an invariant breach)',
            row.test_id,
        )
        return row

    if row.owner_task_id:
        # INV-3: corroborate BEFORE acting.  The stored id is a snapshot; it is re-read
        # against live status on every suppression and never assumed still-open.
        #
        # On a FAILED read the fail-safe direction is do-NOT-file.  This is the
        # opposite of `chronic_flake._has_open_dedup_match`'s fail-open-towards-filing,
        # deliberately: there a duplicate is bounded by `FilingLedger`'s multi-day
        # per-test rate limit, here there is none, so a transient MCP outage during a
        # suppression burst would file one duplicate PER SUPPRESSION.  An unreadable
        # status is also simply not evidence that the owner went terminal, and INV-3
        # says corroborate before acting.  A missed filing is cheaply recoverable — the
        # next suppression of the same test retries, and θ's age backstop catches a row
        # that stays stuck; a duplicate task tree is not.
        #
        # It is the TYPED PAIR, not the caller's exception discipline, that makes "a
        # FAILED read never reaches the classification" true through the production
        # composition rather than merely aspirational.
        status, read_error = await _owner_status(task_client, row.owner_task_id, owner_read)
        if read_error is not None:
            logger.warning(
                'flake_ledger: could not corroborate owner %s for test_id=%s — KEEPING '
                'the stored owner and filing nothing (an unreadable status is not '
                'evidence the task went terminal)',
                row.owner_task_id,
                row.test_id,
                exc_info=read_error,
            )
            return row
        liveness = owner_liveness(status)
        if liveness == 'open':
            logger.debug(
                'flake_ledger: debt for test_id=%s is already owned by live task %s',
                row.test_id,
                row.owner_task_id,
            )
            return row
        if liveness == 'deferred':
            # A half-completed initial filing, not a closed one.  Finish it — filing a
            # SECOND task here is the duplicate-generating loop the deferred
            # verdict exists to prevent.  Completing the initial filing is explicitly exempted from
            # §5.9's coupling rule; this is still never lifecycle management.
            logger.info(
                'flake_ledger: completing the half-filed de-flake task %s for test_id=%s '
                '(deferred — its commit_planning never landed)',
                row.owner_task_id,
                row.test_id,
            )
            try:
                await task_client.commit_planning([row.owner_task_id])
            except Exception:
                logger.warning(
                    'flake_ledger: could not complete the half-filed de-flake task %s '
                    'for test_id=%s — it stays deferred and the next suppression retries',
                    row.owner_task_id,
                    row.test_id,
                    exc_info=True,
                )
            return row
        if liveness == 'done':
            # The resolution did not land this call (see this function's docstring).  Keep the
            # pointer and file nothing, the unreadable-status branch's fail-safe
            # direction: the next lazy resolve_debt closes the cycle through it, where a
            # replacement would erase it for good.
            logger.warning(
                'flake_ledger: owner %s of test_id=%s is done but its debt cycle could not '
                'be closed — KEEPING the owner rather than replacing it; '
                'regressed_after_resolution detection is DEFERRED to the next suppression '
                "(or θ's sweep)",
                row.owner_task_id,
                row.test_id,
            )
            return row
        logger.info(
            'flake_ledger: de-flake task %s owning test_id=%s was cancelled or no longer '
            'exists (status=%r) — filing a replacement',
            row.owner_task_id,
            row.test_id,
            status,
        )

    # The filing gets its OWN guard for the same reason the corroboration does: one
    # failing client call must cost exactly that signal, never the merge (B12) and never
    # the other side-effects — the "independently guarded side-effects" discipline
    # `orchestrator/src/orchestrator/flake_recorder.py::_guarded` established.
    try:
        outcome = await task_client.submit_task(build_deflake_task_arguments(row))
    except Exception:
        logger.warning(
            'flake_ledger: failed to file a de-flake task for test_id=%s (project_id=%s) '
            '— the debt row stays UNOWNED and §5.9 is breached for it until the next '
            'suppression retries',
            row.test_id,
            project_id,
            exc_info=True,
        )
        return row
    owner = _owner_from_submission(outcome)
    if isinstance(owner, _FailedFiling):
        logger.warning(
            'flake_ledger: failed to file a de-flake task for test_id=%s (project_id=%s): '
            '%s — the debt row stays UNOWNED and §5.9 is breached for it until the next '
            'suppression retries',
            row.test_id,
            project_id,
            owner.reason,
        )
        return row
    new_id = owner.task_id

    # The pointer is written BEFORE commit_planning is awaited, and the order picks
    # which failure is survivable.  Written last, a commit_planning failure would orphan
    # a task that really was created — invisible, unreferenced, and re-created on every
    # subsequent suppression.  Written first, the worst case is a row pointing at a
    # `deferred` task, which ι renders (owner shown) and which the `deferred` branch
    # above repairs on the next suppression.  Same principle as the recorder's "durable
    # row first, lose the recoverable half".
    #
    # The claim is CONDITIONAL on what this call actually corroborated — NULL for a fresh
    # filing, the terminal id for a replacement — because `row` is a snapshot taken
    # before two awaits (the status read and the filing) and a concurrent lane may have
    # claimed the row in between.  See `_write_owner_task_id`.
    prior_owner = row.owner_task_id or None
    try:
        _write_owner_task_id(db_path, row.test_id, new_id, expected_owner=prior_owner)
    except _OwnerClaimLost:
        # The task exists server-side and nothing will ever reference it, so the warning
        # NAMES it: that string is the only trace a human can search for.  Deliberately
        # not a clobber and deliberately not a retry — the row now carries some other
        # lane's owner (or is gone), and the next suppression corroborates THAT rather
        # than filing again, so the leak is one task, not a loop.
        logger.warning(
            'flake_ledger: filed de-flake task %s for test_id=%s but LOST the owner claim '
            '(expected owner %r) — another merge lane owns the row, so this task is '
            'ORPHANED; the winner is left in place rather than clobbered',
            new_id,
            row.test_id,
            prior_owner,
            exc_info=True,
        )
        return row
    except Exception:
        logger.warning(
            'flake_ledger: filed de-flake task %s for test_id=%s but could not store the '
            'pointer — the task is ORPHANED and the next suppression will file another',
            new_id,
            row.test_id,
            exc_info=True,
        )
        return row

    # `planning_mode=True` created the task `deferred`; commit_planning is the SECOND
    # PHASE of the initial filing, not lifecycle management, and without it the task
    # exists but is never dispatched.
    try:
        if owner.needs_commit_planning:
            await task_client.commit_planning([new_id])
    except Exception:
        logger.warning(
            'flake_ledger: filed de-flake task %s for test_id=%s but could not release it '
            'from deferred — the pointer IS stored, so the next suppression completes it '
            'rather than filing a duplicate',
            new_id,
            row.test_id,
            exc_info=True,
        )
    if owner.combined:
        logger.info(
            'flake_ledger: de-flake filing for test_id=%s was combined into live task %s, '
            'now its owner (project_id=%s)',
            row.test_id,
            new_id,
            project_id,
        )
    else:
        logger.info(
            'flake_ledger: filed de-flake task %s owning debt for test_id=%s (project_id=%s)',
            new_id,
            row.test_id,
            project_id,
        )
    # No re-read.  After a ROWCOUNT-CHECKED conditional UPDATE, `new_id` is exactly what
    # is in that column on disk — so patching the one column this function owns is not a
    # "local copy" of the row, it is the row.
    #
    # This used to `read_debt` and fall back to `replace(...)` only on a `None` return,
    # which was wrong in both directions: `read_debt` returns `None` for a DEGRADED READ
    # and for a GENUINELY ABSENT row alike (it catches and logs), so the fallback also
    # fired when the row had been deleted underneath us — handing the caller a row
    # asserting an ownership the database does not hold.  That is the same
    # present-vs-absent conflation the `(statuses, error)` pair was introduced to
    # eliminate one function away.  It also cost a THIRD connection on the owned path
    # (upsert, UPDATE, re-read), each paying the full five-pragma durability triad on
    # the merge path — the cost `TestOneConnectionPerCall` exists to bound.
    return replace(row, owner_task_id=new_id)
# Stay well under SQLite's SQLITE_MAX_VARIABLE_NUMBER (999 on older builds, 32766 on
# modern ones -- measured 32766 on SQLite 3.50.4 here) -- the test universe a report
# covers is unbounded, so a single `IN (...)` over it is not safely bounded.  500 and the
# chunking shape are the in-repo precedent at
# `evals/reviewer_trial/__main__.py::_FETCH_TITLES_CHUNK_SIZE`; one spelling of this
# pattern in the orchestrator package rather than a second idiom.
_READ_DEBT_CHUNK_SIZE = 500


def read_debt_many(db_path: Path, test_ids: Iterable[str]) -> dict[str, DebtRow]:
    """The debt rows for *test_ids*, keyed by ``test_id`` — one connection for the batch.

    The batched form of :func:`read_debt`, and deliberately nothing more: it calls the
    same :func:`_to_debt_row` on the same columns, so the two cannot drift in how a row
    becomes a :class:`DebtRow`.  It exists because ι's report builds one chain per test
    and a per-test :func:`read_debt` pays :func:`_open`'s full cost EACH TIME —
    ``parent.mkdir`` → ``connect`` → the five-pragma durability triad (including a
    ``journal_mode=WAL`` switch and ``synchronous=FULL``) → ``executescript(_SCHEMA)`` —
    against the live ``runs.db`` the merge lane holds a 5s ``busy_timeout`` on.  That is
    DDL per test on a nominally read-only report; here it is DDL once per report.

    RESOLVED rows are returned, exactly as :func:`read_debt` returns them: §5.2 retains
    them deliberately because η's recurrence trigger reads them, and they are the whole
    reason a batched reader is wanted — ι's chain goes blank precisely when a test is
    BETWEEN cycles, which is the PRD's motivating case.

    A test_id with NO debt row is ABSENT from the result, never present with a ``None``
    value.  That makes ``read_debt_many(db, ids).get(t)`` reproduce
    ``read_debt(db, t)``'s ``DebtRow | None`` contract with zero adaptation at the call
    site, and it keeps "we looked and found nothing" distinguishable from "we never
    asked" — the collapse a ``dict[str, DebtRow | None]`` would force on every consumer.

    B12: never raises.  A failure degrades the WHOLE call to ``{}`` rather than returning
    whatever was read so far, uniform with every sibling reader here (``[]`` / ``None``)
    — the realistic failures are per-DATABASE (corrupt, truncated, lock-contended), not
    per-row, so a partial result is a state that essentially cannot arise.  The report
    already renders a missing debt row honestly, and the loud ``exc_info`` warning still
    names the cause.
    """
    # Deduped because the caller passes a set difference and a repeat would waste one of
    # SQLite's bounded variable slots; SORTED because a `set` iterates in arbitrary
    # order, and without this the emitted SQL — and, once chunked, the chunk CONTENTS —
    # would vary run to run in a subsystem whose renderer is contractually byte-stable.
    # Bound BEFORE the try so the handler's `len(ids)` can never itself raise NameError,
    # mirroring `record_flake_occurrence`'s `test_ids: tuple[str, ...] = ()` precedent.
    ids: tuple[str, ...] = ()
    try:
        ids = tuple(sorted(set(test_ids)))
        # Short-circuit BEFORE `_open`: ι's contract is that printing a report never
        # provisions a DB, and `_open` would `mkdir` + `connect` + run the schema DDL.
        if not ids:
            return {}

        found: dict[str, DebtRow] = {}
        # ONE connection for the whole batch, hoisted OUTSIDE the chunk loop.  A
        # per-chunk connection would each pay the five-pragma durability triad and
        # re-run the schema DDL -- the same amplification this reader exists to remove,
        # merely at 1/500th the rate -- and would break the module's machine-checked
        # one-connection-per-entry-point contract for any batch over the chunk size.
        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            for i in range(0, len(ids), _READ_DEBT_CHUNK_SIZE):
                chunk = ids[i : i + _READ_DEBT_CHUNK_SIZE]
                placeholders = ','.join('?' for _ in chunk)
                rows = conn.execute(
                    f'SELECT * FROM flake_debt WHERE test_id IN ({placeholders})', chunk
                ).fetchall()
                found.update((row['test_id'], _to_debt_row(row)) for row in rows)
        finally:
            conn.close()
        return found
    except Exception:
        logger.warning(
            'flake_ledger: failed to read debt for %d test(s) from %s',
            len(ids),
            db_path,
            exc_info=True,
        )
        return {}


def _this_call_reentered(row: DebtRow, *, stamp: str) -> bool:
    """Whether the upsert that read back *row* RE-ENTERED a resolved cycle at *stamp*.

    The upsert's re-entry branch is the only writer that moves ``opened_at`` on an
    existing row, and it writes THIS call's *stamp*; ``open_count > 1`` excludes the
    fresh insert, which writes the stamp too.  Two calls sharing one canonical stamp —
    only an injected clock does that — would both read as re-entries.
    """
    return row.open_count > 1 and row.opened_at == stamp


def _report_regression(row: DebtRow, hook: Callable[[DebtRow], None] | None) -> None:
    """Hand the re-entered *row* to *hook* (``open_debt``'s
    ``on_regressed_after_resolution``), or log the regression LOUDLY when none is wired.
    Never raises: a failing hook costs only the report."""
    if hook is None:
        logger.warning(
            'flake_ledger: test_id=%s re-entered debt after a resolution '
            '(regressed_after_resolution, cycle %d, prior resolving commit %s) but no hook '
            'is wired, so it is visible only in `orchestrator flake-ledger`',
            row.test_id,
            row.open_count,
            row.prior_resolving_commit or '(none recorded)',
        )
        return
    try:
        hook(row)
    except Exception:
        logger.warning(
            'flake_ledger: reporting regressed_after_resolution for test_id=%s raised — '
            'the re-entered debt row IS written and its owner is still enforced',
            row.test_id,
            exc_info=True,
        )


# How long, in seconds, the owner read that closes a cycle may take.  A healthy
# `get_task` answers in well under a second, but `mcp_call` retries through a
# fused-memory restart for ~120s
# (`orchestrator/src/orchestrator/mcp_lifecycle.py::McpSession._retry_backoffs`).
_RESOLVE_BUDGET_SECS = 10.0


@dataclass(frozen=True)
class _CloseVerdict:
    """What :func:`_resolve` did: whether this call stamped the resolution, the live
    owner read it made when one succeeded, and — when *halt* is set — the row
    :func:`open_debt` returns at once, with no upsert and no owner filing."""

    closed: bool
    owner_read: _OwnerRead | None
    halt: DebtRow | None = None


async def _ask_containment(
    head_contains_fix: Callable[[Mapping[str, Any]], Awaitable[bool | None]],
    task: Mapping[str, Any],
    test_id: str,
) -> bool | None:
    """*head_contains_fix*'s answer for *task*; a raise is ``None``, undecided (§5.9: a
    failed ancestry check is a named state, never a negative answer)."""
    try:
        return await head_contains_fix(task)
    except Exception:
        logger.warning(
            'flake_ledger: the stale-head check for test_id=%s raised — treated as '
            'undecided',
            test_id,
            exc_info=True,
        )
        return None


async def open_debt(
    db_path: Path,
    project_id: str,
    test_id: str,
    *,
    task_client: FlakeLedgerTaskClient | None = None,
    now: datetime | None = None,
    on_regressed_after_resolution: Callable[[DebtRow], None] | None = None,
    resolve_budget_secs: float = _RESOLVE_BUDGET_SECS,
    head_contains_fix: Callable[[Mapping[str, Any]], Awaitable[bool | None]] | None = None,
) -> DebtRow | None:
    """Open (or advance) the single ``flake_debt`` row for *test_id* (PRD §8.3).

    Returns the resulting row, or ``None`` if the ledger was unavailable — the never-
    raises invariant and a ``-> DebtRow`` return type are in direct conflict, so the
    honest degrade is ``None`` rather than a fabricated row, and consumers must handle
    ledger unavailability explicitly.

    ``task_client`` is where §5.9's invariant is ENFORCED (see
    :func:`_ensure_owner_task`): after the upsert, ``owner_task_id`` is re-corroborated
    against live task status and a de-flake task is filed if there is no owner, or it was
    cancelled or deleted.  Pass
    ``None`` — the CLI and storeless callers do — and the row is written exactly as α
    wrote it, with no owner; that is a legitimate degrade, logged, and rendered by ι as
    an invariant breach rather than hidden.

    CLOSE, THEN UPSERT (task η).  With a ``task_client`` wired, the current cycle is
    first closed via :func:`resolve_debt` if its stored owner reads back live-done, so
    this observation RE-ENTERS the debt (``open_count`` +1, the fix's commit carried
    forward) instead of ζ quietly replacing a finished owner.  That makes recurrence
    detection race-free with no sweep at all: ``owner_task_id`` changes only inside this
    function, so the done owner is still on the row when the next suppression arrives,
    whether or not an eager sweep has run.  The eager sweep is task θ's.  The close's
    live ``get_task`` read of the owner is also the read ζ's enforcement acts on, so one
    call reads its owner once; ``get_statuses`` is only the fallback when that read
    failed.

    The close's owner read is the only network wait BEFORE the upsert, so it is bounded
    by *resolve_budget_secs*: a slow or hung read fails like any other, costing the
    resolution and never the occurrence.  Only a caller's own budget that expires inside
    that bound can still cancel the call ahead of its upsert, which loses the same thing
    as a test that budget skipped outright.

    STALE HEAD (offline-lane red-path PRD D6).  The drain calls this after the
    observation, possibly long after it, so the stored owner can have gone done in
    between, and evidence from a head WITHOUT that owner's fix says nothing about
    whether the fix held.  *head_contains_fix*, when given, is asked about the owner the
    lazy close read as done: ``True`` is today's close-then-re-enter; ``False`` closes
    the cycle and STOPS — no upsert, no re-entry, no regression report, no owner filing,
    and the closed row is returned; ``None`` (undecided, or the check raised) decides
    nothing — no close, no upsert — and returns the row as read.  The check bounds its
    own wait.

    REGRESSION REPORT (task η).  A RE-ENTRY means the test flaked again after its fix
    landed, and *on_regressed_after_resolution* receives the re-entered row EXACTLY ONCE
    per re-entry: never for a first suppression, never for a repeat inside a cycle.  It
    is called synchronously right after the upsert commits, with no await in between and
    BEFORE ζ's owner filing awaits anything, so a filing that hangs, or that the caller
    cancels, cannot lose the report; that is also why the row it
    receives has no owner yet.  A raising hook costs only the report, never the row or
    its owner.  Unwired, the re-entry is logged as a WARNING instead: a legitimate
    configuration, never a silent one.

    COUPLING RULE, binding: the ledger READS task status but never WRITES it, except
    for the initial filing (``submit_task`` plus the ``commit_planning`` that completes
    it).  It never marks a task done, never blocks one, never reprioritises one.

    ``UNKNOWN_TEST_ID`` must never be passed here: a sentinel names no test, so it can
    own no de-flake task.  That is REFUSED, not merely documented — ε/ζ plausibly iterate
    the test_ids of a recorded occurrence batch, which is exactly where the sentinel
    lives, and an accepted ``<unknown>`` debt row would surface in ι's report and, in ζ,
    file a de-flake task against a test that does not exist.

    *now* defaults to the current UTC time; it is injectable so §5.4's ledger-owned
    timestamps are deterministically testable.  Whatever is passed — naive or any
    offset — is routed through :func:`_canonicalize_utc` before being stamped, so an
    injected clock cannot write ``opened_at``/``last_occurrence_at`` in a spelling that
    breaks :func:`list_open_debt`'s ``ORDER BY opened_at`` contract against rows written
    by the default aware-UTC path.  In-repo callers should always pass aware-UTC
    (``datetime.now(UTC)``) — a naive ``now`` is coerced rather than rejected, to honour
    the never-raises invariant, but it also logs a loud warning, because here (unlike
    :func:`_normalize_observed_at`'s wire-supplied ``observed_at``) a missing offset is
    a caller bug, not untrusted input.  It is ONE instant for the whole call: a
    lazily-closed cycle's ``resolved_at`` and the re-entered ``opened_at`` are the same
    observation.
    """
    if test_id == UNKNOWN_TEST_ID:
        logger.warning(
            'flake_ledger: refusing to open debt for the %s sentinel — it names no '
            'test, so it can own no de-flake task',
            UNKNOWN_TEST_ID,
        )
        return None

    # Resolved ahead of the lazy close, which needs the same instant, so it gets its own
    # guard: B12 holds for a `now` that is not a datetime at all.
    try:
        observed = _aware(now or datetime.now(UTC), origin='open_debt')
        stamp = _canonicalize_utc(observed)
    except Exception:
        logger.warning(
            'flake_ledger: failed to open debt for test_id=%s (project_id=%s) — now=%r '
            'is not a usable observation instant',
            test_id,
            project_id,
            now,
            exc_info=True,
        )
        return None

    owner_read: _OwnerRead | None = None
    if task_client is not None:
        # Outside the upsert's guard, and safe there: the close never raises and its
        # read is bounded, so a failed or slow resolution costs only the resolution.
        close = await _resolve(
            db_path, project_id, test_id,
            task_client=task_client, now=observed, read_budget_secs=resolve_budget_secs,
            head_contains_fix=head_contains_fix,
        )
        if close.halt is not None:
            return close.halt
        owner_read = close.owner_read

    try:
        conn = _open(db_path)
        try:
            # ONE statement, deliberately.  SQL evaluates every SET right-hand side against
            # the PRE-UPDATE row, so all four CASE guards observe the OLD `resolved_at`
            # even though the same clause sets it to NULL — which is what lets a re-entry
            # (was resolved) and an ordinary repeat (still open) be distinguished without a
            # prior SELECT.  Verified empirically on SQLite 3.50.4: open_count=2,
            # prior_resolved_at=the old resolved_at, resolved_at=NULL, opened_at advanced.
            #
            # Splitting this into SELECT-then-UPDATE would reintroduce exactly the
            # read-modify-write race §5.1 chose SQLite to avoid, inside the ledger itself:
            # two merge lanes suppressing the same test could interleave and lose a cycle.
            #
            # `prior_resolving_commit` is deliberately NOT touched — `resolve_debt` already
            # wrote it, and it must survive the re-open verbatim for η's
            # regressed_after_resolution citation.
            #
            # A re-entry DISCHARGES the resolved cycle's owner (`owner_task_id` -> NULL);
            # a repeat keeps it.  Left on the re-opened row, the done owner would let a
            # concurrent lane's `resolve_debt` — a compare-and-set on exactly that owner —
            # close the new cycle it never worked on: a phantom cycle, open_count +2, and a
            # duplicate L2.  With the column NULL, ζ's fresh claim (`owner_task_id IS
            # NULL`) assigns the new cycle its own owner.
            conn.execute(
                'INSERT INTO flake_debt (test_id, project_id, opened_at, open_count, '
                ' last_occurrence_at) '
                'VALUES (?, ?, ?, 1, ?) '
                'ON CONFLICT(test_id) DO UPDATE SET '
                '    last_occurrence_at = excluded.last_occurrence_at, '
                '    open_count        = open_count '
                '                        + (CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END), '
                '    opened_at         = CASE WHEN resolved_at IS NOT NULL '
                '                        THEN excluded.opened_at ELSE opened_at END, '
                '    prior_resolved_at = CASE WHEN resolved_at IS NOT NULL '
                '                        THEN resolved_at ELSE prior_resolved_at END, '
                '    owner_task_id     = CASE WHEN resolved_at IS NOT NULL '
                '                        THEN NULL ELSE owner_task_id END, '
                '    resolved_at       = NULL',
                (test_id, project_id, stamp, stamp),
            )
            conn.commit()
            # Read back on the SAME connection rather than re-entering `read_debt`,
            # which would open two more (its `_open` plus its SELECT's).  Inside the
            # same connection this also reads the row this transaction just committed,
            # with no window for another lane to interleave a resolve between them.
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM flake_debt WHERE test_id = ?', (test_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        debt_row = _to_debt_row(row)
    except Exception:
        logger.warning(
            'flake_ledger: failed to open debt for test_id=%s (project_id=%s)',
            test_id,
            project_id,
            exc_info=True,
        )
        return None

    if _this_call_reentered(debt_row, stamp=stamp):
        _report_regression(debt_row, on_regressed_after_resolution)

    # ENSURING THE OWNER IS OUTSIDE THE TRY ABOVE, AND THE BOUNDARY IS THE COMMIT.
    # `None` is a contract -- it means "the ledger was unavailable, the measurement is
    # lost" (see this function's docstring), and `record_merge_flake_suppression` logs it
    # as exactly that.  By the time we get here the upsert has COMMITTED, so the debt row
    # is durably on disk and that statement would be false.  `_ensure_owner_task` claims
    # never to raise, but it drives a duck-typed `task_client` and does its own sqlite
    # write, so "claims" is not "cannot": were it inside the try, one raise would report a
    # successful write as a lost one, and the row would be re-opened as if fresh.
    # A filing failure must degrade to "row WITHOUT an owner" -- which `flake_report`
    # already renders as the §5.9 breach it is -- never to "no row at all".
    try:
        return await _ensure_owner_task(
            db_path, project_id, debt_row, task_client=task_client, owner_read=owner_read
        )
    except Exception:
        logger.warning(
            'flake_ledger: opened debt for test_id=%s (project_id=%s) but enforcing the '
            'owner invariant raised -- the debt row IS durably written and is returned '
            'UNOWNED, which flake_report renders as the breach it is',
            test_id,
            project_id,
            exc_info=True,
        )
        return debt_row


async def _read_owner(
    task_client: FlakeLedgerTaskClient, test_id: str, owner_task_id: str, *, budget_secs: float,
) -> _OwnerRead | None:
    """*owner_task_id*'s task, read LIVE; ``None`` when the read FAILED, which is logged
    and is evidence of nothing.

    Its own guard, like :func:`_owner_status`'s fallback read: a raising or partial
    adapter (no ``get_task`` at all, hence ``AttributeError``) lands on the SAME path as
    the real adapter's in-band error, so the two failure shapes cannot drift.  So does a
    read that outlives *budget_secs* (``TimeoutError``): see :data:`_RESOLVE_BUDGET_SECS`.
    """
    try:
        task, error = await asyncio.wait_for(
            task_client.get_task(owner_task_id), timeout=budget_secs,
        )
    except Exception as exc:
        task, error = None, exc
    if error is not None:
        logger.warning(
            'flake_ledger: could not read owner %s of test_id=%s — the cycle stays open '
            '(an unreadable owner is not evidence the fix landed)',
            owner_task_id,
            test_id,
            exc_info=error,
        )
        return None
    return _OwnerRead(owner_task_id, task)


def _finished_task(owner: _OwnerRead, test_id: str) -> dict | None:
    """*owner*'s task if and only if it reads back ``done``, the one status that closes
    a cycle; ``None`` otherwise, the reason logged at the level it deserves."""
    if owner.task is None:
        logger.info(
            'flake_ledger: owner %s of test_id=%s no longer exists — the cycle stays open',
            owner.owner_task_id,
            test_id,
        )
        return None
    if owner.status != 'done':
        logger.debug(
            'flake_ledger: owner %s of test_id=%s is %r, not done — the cycle stays open',
            owner.owner_task_id,
            test_id,
            owner.status,
        )
        return None
    return owner.task


def resolving_commit(task: Mapping[str, Any]) -> str | None:
    """The commit that landed *task*'s fix, from its ``metadata.done_provenance``, read
    with the one typed metadata parser.  ``None`` when none was recorded: a task that
    predates ``done_provenance``, or a commitless kind.

    ``isinstance`` rather than ``is not None``: a blob that fails whole-model validation
    comes back from ``parse_metadata`` UNVALIDATED, and then ``done_provenance`` is the
    raw dict (measured).
    """
    metadata, _warnings = parse_metadata(task.get('metadata'), direction='read')
    provenance = metadata.done_provenance
    return provenance.commit if isinstance(provenance, DoneProvenance) else None


def fix_contained(task: Mapping[str, Any], ancestry: bool | None) -> bool | None:
    """Whether *task*'s fix is contained in the head the caller asked about: ``True``,
    ``False``, or ``None`` when that is undecided.

    ``True`` when *task* recorded no fix commit (:func:`resolving_commit` is ``None``):
    a done task with nothing to be missing from the head is contained.  Otherwise
    *ancestry* decides: it is the caller's already-performed
    ``is_ancestor_checked(commit, head)`` read for that commit, and ``None`` there (the
    read failed) leaves the answer ``None``.
    """
    if resolving_commit(task) is None:
        return True
    return ancestry


def _stamp_resolution(
    db_path: Path,
    test_id: str,
    *,
    owner_task_id: str,
    resolving_commit: str | None,
    stamp: str,
) -> bool:
    """Close *test_id*'s open cycle for *owner_task_id* — the ONE spelling of the
    resolution UPDATE.  ``True`` iff this call closed it.

    A compare-and-set on the owner, for the reason :func:`_write_owner_task_id` is
    conditional: the caller decided to stamp from a live read of that owner, the read
    was an await, and another lane may have re-pointed, re-opened or closed the row
    since.  A row the WHERE clause no longer matches was not corroborated by this call.
    Raises on a ledger failure; the caller's B12 guard owns that.
    """
    conn = _open(db_path)
    try:
        cursor = conn.execute(
            'UPDATE flake_debt SET resolved_at = ?, prior_resolving_commit = ? '
            'WHERE test_id = ? AND resolved_at IS NULL AND owner_task_id = ?',
            (stamp, resolving_commit, test_id, owner_task_id),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


async def resolve_debt(
    db_path: Path,
    project_id: str,
    test_id: str,
    *,
    task_client: FlakeLedgerTaskClient,
    now: datetime | None = None,
) -> bool:
    """Close *test_id*'s current debt cycle if its owning de-flake task is live-done
    (PRD §8.3).  ``True`` iff THIS call closed it.

    CORROBORATED, NEVER TRUSTED (INV-3).  A stored ``owner_task_id`` is a snapshot, and
    so is any status a caller could hand in.  The decision is therefore taken here, from
    a live ``get_task`` read of the stored owner, and the stamp is a compare-and-set on
    that owner (:func:`_stamp_resolution`).  The resolving commit comes from the same
    read (``metadata.done_provenance.commit``), so status and commit cannot disagree and
    no caller can supply a commit for a task this function never read.

    ONLY ``done`` RESOLVES.  ``prior_resolving_commit`` is what the
    ``regressed_after_resolution`` L2 cites as the fix that did not hold, and a
    cancelled de-flake task landed no fix: resolving on it would manufacture a false
    regression.  An absent or unreadable owner is evidence of nothing, so the cycle
    stays open and ζ's owner enforcement deals with it.  A done task with no recorded
    commit (a legacy task, a commitless kind) does resolve, with a NULL commit.

    ``resolved_at`` is the LEDGER's observation time (§5.4): the task store keeps no
    timestamps, so it records when the ledger SAW the owner done.

    CALLERS.  :func:`open_debt` is the LAZY one, before its upsert, so a recurrence after
    a landed fix re-enters the cycle instead of quietly replacing the owner.  Task θ's
    periodic sweep is the intended EAGER one.

    COUPLING RULE (§5.9): the ledger reads task state and never writes it; the live read
    is the only client call made here.

    The row is RETAINED (§5.2): the recurrence trigger reads resolved rows.  An absent,
    unowned or already-closed row returns ``False`` with no live read, so a replayed
    resolution never walks ``resolved_at`` forward or overwrites
    ``prior_resolving_commit``; :func:`open_debt` sets ``resolved_at`` back to NULL on
    re-entry, so each cycle resolves once.

    ONE connection when nothing is stamped, TWO when it stamps: the row read and the
    stamp straddle a network await, so they cannot share a transaction, and the
    compare-and-set is what makes the split safe.

    Keyed on ``test_id`` alone (§5.3); *project_id* is used only in log messages.  *now*
    goes through :func:`_canonicalize_utc` with the same loud-on-naive treatment as
    ``open_debt``.  The live read is bounded by :data:`_RESOLVE_BUDGET_SECS`, and one
    that outlives it is a failed read.  Never raises (B12): a failure logs with
    ``exc_info`` and returns ``False``.
    """
    close = await _resolve(
        db_path, project_id, test_id,
        task_client=task_client, now=now, read_budget_secs=_RESOLVE_BUDGET_SECS,
    )
    return close.closed


async def _resolve(
    db_path: Path,
    project_id: str,
    test_id: str,
    *,
    task_client: FlakeLedgerTaskClient,
    now: datetime | None,
    read_budget_secs: float,
    head_contains_fix: Callable[[Mapping[str, Any]], Awaitable[bool | None]] | None = None,
) -> _CloseVerdict:
    """:func:`resolve_debt`'s body: whether this call closed the cycle, and the live
    owner read it made when one SUCCEEDED, so :func:`open_debt` acts on that read rather
    than repeating it.  The read's bound is the caller's: :func:`open_debt` passes its
    own ``resolve_budget_secs``.

    *head_contains_fix* is :func:`open_debt`'s stale-head check, asked only about an
    owner read as done; its ``False`` and ``None`` set :attr:`_CloseVerdict.halt`."""
    try:
        stamp = _canonicalize_utc(now or datetime.now(UTC), origin='resolve_debt')
        row = read_debt(db_path, test_id)
        if row is None or row.resolved_at is not None:
            return _CloseVerdict(False, None)
        owner_task_id = row.owner_task_id
        if not owner_task_id:
            logger.info(
                'flake_ledger: debt for test_id=%s has no owner, so no finished task can '
                'close it',
                test_id,
            )
            return _CloseVerdict(False, None)
        owner = await _read_owner(
            task_client, test_id, owner_task_id, budget_secs=read_budget_secs,
        )
        task = None if owner is None else _finished_task(owner, test_id)
        if task is None:
            return _CloseVerdict(False, owner)
        contained = (
            True if head_contains_fix is None
            else await _ask_containment(head_contains_fix, task, test_id)
        )
        if contained is None:
            logger.info(
                'flake_ledger: cannot tell whether the evidence for test_id=%s contains the '
                'fix of owner %s — the cycle is neither closed nor re-opened',
                test_id,
                owner_task_id,
            )
            return _CloseVerdict(False, owner, halt=row)
        commit = resolving_commit(task)
        closed = _stamp_resolution(
            db_path, test_id, owner_task_id=owner_task_id, resolving_commit=commit, stamp=stamp,
        )
        if not closed:
            logger.info(
                'flake_ledger: did not resolve debt for test_id=%s — the row changed while '
                'owner %s was being read (re-pointed, re-opened or already closed)',
                test_id,
                owner_task_id,
            )
        else:
            logger.info(
                'flake_ledger: resolved debt for test_id=%s (project_id=%s): owner %s is '
                'done, resolving commit %s',
                test_id,
                project_id,
                owner_task_id,
                commit or '(none recorded)',
            )
        if contained:
            return _CloseVerdict(closed, owner)
        logger.info(
            'flake_ledger: evidence for test_id=%s predates the fix of owner %s: cycle '
            'closed, not re-opened',
            test_id,
            owner_task_id,
        )
        halt = replace(row, resolved_at=stamp, prior_resolving_commit=commit) if closed else row
        return _CloseVerdict(closed, owner, halt=halt)
    except Exception:
        logger.warning(
            'flake_ledger: failed to resolve debt for test_id=%s (project_id=%s)',
            test_id,
            project_id,
            exc_info=True,
        )
        return _CloseVerdict(False, None)


def list_open_debt(db_path: Path) -> list[DebtRow]:
    """Every test currently in debt, oldest first.

    Ordering is part of the contract (``opened_at`` then ``test_id``) because ι prints
    this list.
    """
    try:
        conn = _open(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM flake_debt WHERE resolved_at IS NULL ORDER BY opened_at, test_id'
            ).fetchall()
        finally:
            conn.close()
        return [_to_debt_row(r) for r in rows]
    except Exception:
        logger.warning(
            'flake_ledger: failed to list open debt from %s',
            db_path,
            exc_info=True,
        )
        return []
