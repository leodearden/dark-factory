#!/usr/bin/env python3
"""Audit whether an admitted model is actually being DISPATCHED, and safely.

Answers one question in five parts, for a model M admitted to a set of roles R
at a known time: is M actually being resolved and run for R, are its runs
completing, are the per-model ceiling and the scoped account cap behaving, and
has M leaked onto a role outside R.

Written for the D6 claude-fable-5-1 admission milestone checks (tasks 5440 and
5441), but parameterized by --model / --expect-roles / --since / --window /
--ceiling so any future model admission is the same command with different
arguments, rather than a fresh set of hand-written SQL that quietly disagrees
with the last one.

STRICTLY READ-ONLY.  Every connection is a `mode=ro` SQLite URI
(:func:`_connect_ro`); this script writes nothing, files nothing, and emits no
events.  It is safe to run against a live store while the orchestrator is
merging.

WHAT THIS DOES NOT DO: it asserts no threshold.  It prints what it measures and
FLAGS observations — at-or-over-ceiling, over-the-flat-role-ceiling,
unexpected-role — for a human narrative to interpret.  In particular, exceeding
the flat per-role ceiling is NOT a failure (see `DEFAULT_ROLE_CEILINGS_SECS`);
`timed_out` is the field that says a run was killed.  The close-or-escalate
judgement belongs
to the reader and to the milestone task's own conditional, not to a pass/fail
hardcoded inside a read-only audit.  Each rendered section is labelled with the
concrete resolved window and target model it was computed over, so a report that
pastes the output cannot drift from the query that produced it.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# The orchestrator's own data/ is gitignored (.gitignore:9 "/data/"), so a task
# worktree has no store of its own and a worktree-relative default would never
# resolve.  The live store exists only in the main checkout.
DEFAULT_RUNS_DB = Path('/home/leo/src/dark-factory/data/orchestrator/runs.db')

# A trailing-window spec: N hours or N days, N a positive integer. Anchored so
# '24' and '24x' are both rejected rather than silently truncated to 24.
_WINDOW_RE = re.compile(r'^(\d+)([hd])$')

# orchestrator/src/orchestrator/routing.py::_model_rejection_reason returns
# exactly these three; resolve_route namespaces each as "<layer>:<reason>"
# before appending it to the decision's `rejected` list.  Kept here as the one
# place the vocabulary is written down.
#
# NOT included, deliberately: 'model-not-in-ladder'.  resolve_route appends that
# one when a '+N' ladder-relative spec cannot be resolved, BEFORE any candidate
# model is formed or validated — so it is not evidence that a model was
# attempted and refused, which is the only thing this audit reads rejections for.
MODEL_REJECTION_REASONS = frozenset({
    'model-not-in-allowlist',
    'model-ceiling-exhausted',
    'model-capacity-exhausted',
})


def _connect_ro(path: str | Path) -> sqlite3.Connection:
    """Open *path* strictly read-only, via a `mode=ro` SQLite URI.

    `mode=ro` (rather than a bare `sqlite3.connect`) is the house convention for
    a script that reads a LIVE store — see `scripts/merge_lane_throughput.py`,
    `scripts/audit_wiped_metadata_files.py` and `scripts/census_tagger_debris.py`.
    It buys two things a bare connect does not: a write attempted through this
    connection fails loudly instead of mutating an orchestrator's event store,
    and a typo'd path raises rather than silently creating an empty database
    that would then report every section as "no data".
    """
    uri = f'file:{Path(path).resolve()}?mode=ro'
    return sqlite3.connect(uri, uri=True)


def _iso(moment: datetime) -> str:
    """Format *moment* in the exact ISO-8601 spelling the store writes.

    `event_store.py::EventStore.emit` and the cost store both write
    ``datetime.now(UTC).isoformat()``, which renders the offset as ``+00:00``
    (never ``Z``).  Those columns are TEXT and the spelling is fixed, so the
    result can be used directly as a SQL comparand and the comparison is a
    correct chronological one.
    """
    return moment.astimezone(UTC).isoformat()


def _loads_object(raw: Any) -> dict[str, Any] | None:
    """Parse *raw* as a JSON object, or return None if it is not one.

    Tolerant by design: the live store holds at least one ``account_events.
    details`` that is a bare non-JSON string (``'Escalation watcher (auto)'``),
    and a 181 MB store must not be abortable by one malformed row.  Every
    caller counts what it skips, so a dropped row stays visible in the output.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_model_rejection(entry: Any) -> bool:
    """True when *entry* is a ``"<layer>:<reason>"`` naming a model rejection.

    CAVEAT, and it decides the shape of this whole check: the producer's
    rejection string names the LAYER and the REASON but never the candidate
    MODEL (routing.py::resolve_route appends ``f'config:{reason}'`` and
    friends).  So a rejection cannot be attributed to a particular model from
    the event payload alone.  This audit therefore reports every model
    rejection on every role and leaves attribution to the reader — deliberately
    over-inclusive, because the failure that matters is missing one.
    """
    if not isinstance(entry, str):
        return False
    _, _, reason = entry.rpartition(':')
    return reason in MODEL_REJECTION_REASONS


@dataclass(frozen=True)
class RoutingSelection:
    """One decision that resolved to the target model."""

    timestamp: str
    task_id: str | None
    role: str
    source_layer: str
    rule_id: str | None
    routing_tier: int | None


@dataclass(frozen=True)
class RoutingRejection:
    """One decision that recorded a model rejection, and what it resolved to instead."""

    timestamp: str
    task_id: str | None
    role: str
    resolved_model: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoutingScan:
    """What the resolver DECIDED since a given moment (task checks 1 and 3)."""

    selections: tuple[RoutingSelection, ...]
    rejections: tuple[RoutingRejection, ...]
    skipped_rows: int


def scan_routing_decisions(
    conn: sqlite3.Connection, *, model: str, since: datetime
) -> RoutingScan:
    """Scan `routing_decision` events at or after *since*, once, for two answers.

    SELECTIONS are the decisions that resolved to *model*, each carrying the
    ``source_layer``/``rule_id``/``routing_tier`` that say WHY — which is the
    only way to answer "did rule <X> actually match?".

    REJECTIONS span ALL roles, not just the roles that selected *model*: a
    rejection is recorded on a decision that then resolved to a DIFFERENT
    model, so filtering to selections would systematically miss every rejection
    there is.  See :func:`_is_model_rejection` for why they cannot be filtered
    to *model* itself.
    """
    cursor = conn.execute(
        'SELECT timestamp, task_id, data FROM events '
        'WHERE event_type = ? AND timestamp >= ? ORDER BY timestamp, id',
        ('routing_decision', _iso(since)),
    )
    selections: list[RoutingSelection] = []
    rejections: list[RoutingRejection] = []
    skipped = 0
    for timestamp, task_id, raw in cursor:
        payload = _loads_object(raw)
        if payload is None:
            skipped += 1
            continue
        role = payload.get('role') or ''
        resolved = payload.get('model') or ''
        if resolved == model:
            selections.append(RoutingSelection(
                timestamp=timestamp,
                task_id=task_id,
                role=role,
                source_layer=payload.get('source_layer') or '',
                rule_id=payload.get('rule_id'),
                routing_tier=payload.get('routing_tier'),
            ))
        rejected = payload.get('rejected')
        reasons = tuple(
            entry for entry in rejected if _is_model_rejection(entry)
        ) if isinstance(rejected, list) else ()
        if reasons:
            rejections.append(RoutingRejection(
                timestamp=timestamp,
                task_id=task_id,
                role=role,
                resolved_model=resolved,
                reasons=reasons,
            ))
    return RoutingScan(
        selections=tuple(selections),
        rejections=tuple(rejections),
        skipped_rows=skipped,
    )


# timeouts.<role>, from orchestrator/src/orchestrator/defaults.yaml::timeouts.
# Passed IN rather than read from config: scripts/tests/ imports no first-party
# package (dark-factory-orchestrator.yaml:111-112), so importing orchestrator
# config here would break test collection outright.
#
# THIS IS NOT A TOTAL WALL CLOCK, and the distinction is the whole reason the
# flag derived from it is named for the FLAT CEILING rather than for a timeout.
# workflow.py's working-regime progress extension (task 2360) enforces this
# number flatly only until the transcript proves liveness; from turn 1 onward
# the bound becomes max(timeouts.working_idle_secs, this) as an IDLE bound,
# itself capped by invocation_timeout. At stock dark-factory config that is
# 600 s flat -> 1800 s idle -> 7200 s absolute. A healthy merger that keeps
# producing turns therefore runs well past 600 s BY DESIGN — measured: three
# opus merger runs at 642-1103 s and one claude-fable-5-1 run at 1149 s, all
# successful, none timed out. Read InvocationRecord.timed_out for the
# producer's own kill verdict; never infer one from duration.
DEFAULT_ROLE_CEILINGS_SECS: dict[str, int] = {'merger': 600}

# The only role a merge outcome can be attributed to: a merge is resolved by a
# merger run, and `merge_finalized` events carry no role of their own.
MERGER_ROLE = 'merger'


@dataclass(frozen=True)
class MergeOutcome:
    """How the merge this invocation was working on ultimately finished."""

    timestamp: str
    state: str
    merge_sha: str | None
    reason: str | None


@dataclass(frozen=True)
class InvocationRecord:
    """One run of the target model, with what the invocations table cannot say.

    ``end_event_model`` is the model string as the `invocation_end` event
    records it — an independent second witness to the `invocations.model`
    column, which is what makes "is this a lineage alias or the literal string?"
    answerable rather than assumed.

    ``at_or_over_flat_role_ceiling`` is None, not False, for a role with no
    configured ceiling: False would assert "ran under the ceiling" for a ceiling
    we do not know.  It is NOT a failure signal — see
    :data:`DEFAULT_ROLE_CEILINGS_SECS` for why a healthy run exceeds it.
    ``timed_out`` is the producer's own kill verdict, and is the field to read
    for "did this run die at a wall clock".
    """

    task_id: str | None
    project_id: str
    role: str
    account_name: str
    cost_usd: float
    duration_ms: int
    capped: bool
    started_at: str
    completed_at: str
    turns: int | None
    succeeded: bool | None
    timed_out: bool | None
    end_event_model: str | None
    merge_outcome: MergeOutcome | None
    at_or_over_flat_role_ceiling: bool | None


@dataclass(frozen=True)
class EventRow:
    """One parsed `events` row, reduced to the four fields this audit reads."""

    timestamp: str
    task_id: str | None
    role: str
    payload: dict[str, Any]


def _load_events(
    conn: sqlite3.Connection, event_type: str, since: datetime
) -> list[EventRow]:
    """Load *event_type* rows at or after *since*, in (timestamp, id) order.

    Rows whose payload is not a JSON object are skipped — see
    :func:`_loads_object` for why tolerance is the right posture against a live
    store.
    """
    cursor = conn.execute(
        'SELECT timestamp, task_id, role, data FROM events '
        'WHERE event_type = ? AND timestamp >= ? ORDER BY timestamp, id',
        (event_type, _iso(since)),
    )
    rows = []
    for timestamp, task_id, role, raw in cursor:
        payload = _loads_object(raw)
        if payload is not None:
            rows.append(EventRow(timestamp, task_id, role or '', payload))
    return rows


def _by_task(rows: Iterable[EventRow]) -> dict[str | None, list[EventRow]]:
    """Group *rows* by task_id, preserving each task's chronological order.

    Grouping in Python rather than issuing a correlated subquery per invocation:
    the whole of :func:`scan_invocations` is three table reads regardless of how
    many runs match, which is what makes it safe against a 181 MB live store.
    """
    grouped: dict[str | None, list[EventRow]] = {}
    for row in rows:
        grouped.setdefault(row.task_id, []).append(row)
    return grouped


def _merger_starts_by_task(
    conn: sqlite3.Connection, *, model: str, since: datetime
) -> dict[str | None, list[str]]:
    """When each task's merger runs started — ON ANY MODEL — in chronological order.

    Deliberately UNFILTERED by model, which is the entire point: what ends one
    merger run's claim on a task's merge outcome is the NEXT merger run,
    whoever ran it.  A run on the audited model that leaves a merge blocked,
    retried to done by a merger on a different model, must not read as the
    audited model's success.

    Bounded below by the earliest audited merger start (computed in SQL, so the
    bound cannot drift from the rows it bounds), because no boundary earlier
    than that can end any audited run's window.  One extra scan of the same
    table, not a correlated subquery per audited row.
    """
    grouped: dict[str | None, list[str]] = {}
    cursor = conn.execute(
        'SELECT task_id, started_at FROM invocations WHERE role = ? AND started_at > '
        '(SELECT MIN(started_at) FROM invocations '
        'WHERE model = ? AND role = ? AND completed_at >= ?) '
        'ORDER BY started_at, id',
        (MERGER_ROLE, model, MERGER_ROLE, _iso(since)),
    )
    for task_id, started_at in cursor:
        grouped.setdefault(task_id, []).append(started_at)
    return grouped


def _merge_outcome(
    finalized: Sequence[EventRow], *, after: str, before: str | None
) -> MergeOutcome | None:
    """The LAST `merge_finalized` inside one merger run's attribution window.

    The window is ``[after, before)`` — at or after this run started, and
    strictly before the next merger run on the same task took over.  *before*
    is None when no later merger run exists, leaving the window open-ended.

    LAST rather than first, because a post-merge verification failure blocks a
    task and the same merge is retried to done within the one run's window;
    reporting the first would read as the merger having failed to resolve a
    merge it did resolve.  BOUNDED rather than open-ended, because the symmetric
    error is worse: an unbounded join hands a later merger's success to the run
    being audited, reporting a failure by the model under audit as a success.
    """
    inside = [
        row for row in finalized
        if row.timestamp >= after and (before is None or row.timestamp < before)
    ]
    if not inside:
        return None
    last = inside[-1]
    return MergeOutcome(
        timestamp=last.timestamp,
        state=last.payload.get('state') or '',
        merge_sha=last.payload.get('merge_sha'),
        reason=last.payload.get('reason'),
    )


def scan_invocations(
    conn: sqlite3.Connection,
    *,
    model: str,
    since: datetime,
    role_ceilings_secs: dict[str, int] | None = None,
) -> tuple[InvocationRecord, ...]:
    """Every run of *model* completed at or after *since*, with its outcome.

    Two enrichments the `invocations` table cannot supply on its own:

    TURNS, from the matching `invocation_end` event — `invocations` has no turns
    column, so this join is the only way to answer the question at all.  Matched
    on task_id + role, taking the first such event at or after the invocation's
    ``started_at``.

    MERGE OUTCOME, for merger runs, from the `merge_finalized` events inside
    this run's attribution window — see :func:`_merge_outcome` for the window
    and :func:`_merger_starts_by_task` for the boundary that closes it.
    Matched on task_id ALONE — the producer leaves these events' `role` column
    empty — which is exactly why the window has to do the attributing.
    """
    ceilings = DEFAULT_ROLE_CEILINGS_SECS if role_ceilings_secs is None else role_ceilings_secs
    ends = _by_task(_load_events(conn, 'invocation_end', since))
    merges = _by_task(_load_events(conn, 'merge_finalized', since))
    merger_starts = _merger_starts_by_task(conn, model=model, since=since)
    cursor = conn.execute(
        'SELECT task_id, project_id, role, account_name, cost_usd, duration_ms, '
        'capped, started_at, completed_at FROM invocations '
        'WHERE model = ? AND completed_at >= ? ORDER BY completed_at, id',
        (model, _iso(since)),
    )
    records = []
    for (task_id, project_id, role, account_name, cost_usd, duration_ms,
         capped, started_at, completed_at) in cursor:
        end = next(
            (row.payload for row in ends.get(task_id, ())
             if row.role == role and row.timestamp >= started_at),
            None,
        )
        merge = None
        if role == MERGER_ROLE:
            later_starts = merger_starts.get(task_id, ())
            merge = _merge_outcome(
                merges.get(task_id, ()),
                after=started_at,
                before=next((s for s in later_starts if s > started_at), None),
            )
        ceiling_secs = ceilings.get(role)
        records.append(InvocationRecord(
            task_id=task_id,
            project_id=project_id,
            role=role,
            account_name=account_name,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            capped=bool(capped),
            started_at=started_at,
            completed_at=completed_at,
            turns=end.get('turns') if end else None,
            succeeded=end.get('success') if end else None,
            timed_out=end.get('timed_out') if end else None,
            end_event_model=end.get('model') if end else None,
            merge_outcome=merge,
            at_or_over_flat_role_ceiling=(
                None if ceiling_secs is None else duration_ms >= ceiling_secs * 1000
            ),
        ))
    return tuple(records)


@dataclass(frozen=True)
class ScopedCapHit:
    """One cap_hit that capped a single model scope rather than the account."""

    created_at: str
    account_name: str
    reason: str


@dataclass(frozen=True)
class ServiceRestart:
    """One service restart, named — the service is what makes it evidence or not."""

    timestamp: str
    service: str
    reason: str | None


@dataclass(frozen=True)
class ScopedCapScan:
    """Whether the model's cap scope has been hit, and whether it can yet be (check 4)."""

    scoped_hits: tuple[ScopedCapHit, ...]
    unscoped_cap_hit_count: int
    restarts: tuple[ServiceRestart, ...]


def scan_scoped_cap(
    conn: sqlite3.Connection, *, model: str, since: datetime
) -> ScopedCapScan:
    """Cap hits attributable to *model*'s scope, plus the restarts since *since*.

    The writer of the shape read here is ``shared/src/shared/usage_gate.py::
    AccountPool`` — its scoped path emits ``{"reason": ..., "scope": <model>}``
    and deliberately bypasses the account-level site, which emits a payload with
    no ``scope`` key at all.  So the KEY'S PRESENCE is the discriminator; a
    substring scan for the model name is not, since it would also match
    unrelated cap-message prose.

    Cap hits partition three ways, not two: scoped to *model*, scoped to some
    OTHER model, and unscoped.  Unscoped hits are counted rather than dropped
    because ``usage_cap.scoped_cap_models`` is restart-tier — before the
    orchestrator restarts, a cap on *model* lands on the account-level path, and
    dropping scope-less rows would hide precisely that.

    Restarts are returned NAMED for the same reason: only an orchestrator
    restart reloads a restart-tier leaf, so a count that lumps in dashboard and
    fused-memory restarts answers "has it had a chance to take effect?" wrongly.
    """
    cursor = conn.execute(
        'SELECT created_at, account_name, details FROM account_events '
        'WHERE event_type = ? AND created_at >= ? ORDER BY created_at, id',
        ('cap_hit', _iso(since)),
    )
    scoped: list[ScopedCapHit] = []
    unscoped = 0
    for created_at, account_name, raw in cursor:
        details = _loads_object(raw)
        scope = details.get('scope') if details else None
        if scope is None:
            unscoped += 1
        elif scope == model:
            scoped.append(ScopedCapHit(
                created_at=created_at,
                account_name=account_name,
                reason=(details or {}).get('reason') or '',
            ))
    # Read FLAT, not grouped by task: every service_restart row in the live
    # store carries a task_id (the merge that triggered it), so bucketing these
    # by task and reading one bucket would silently report zero restarts.
    restarts = [
        ServiceRestart(
            timestamp=row.timestamp,
            service=row.payload.get('service') or '',
            reason=row.payload.get('reason'),
        )
        for row in _load_events(conn, 'service_restart', since)
    ]
    return ScopedCapScan(
        scoped_hits=tuple(scoped),
        unscoped_cap_hit_count=unscoped,
        restarts=tuple(restarts),
    )


@dataclass(frozen=True)
class SpendInWindow:
    """The model's spend over one window, against its configured ceiling (check 5)."""

    window_start: str
    window_end: str
    total_usd: float
    invocation_count: int
    ceiling_usd: float | None
    headroom_usd: float | None
    at_or_over_ceiling: bool | None


def spend_in_window(
    conn: sqlite3.Connection,
    *,
    model: str,
    window_start: datetime,
    window_end: datetime,
    ceiling_usd: float | None,
) -> SpendInWindow:
    """Sum *model*'s cost over the HALF-OPEN window ``[window_start, window_end)``.

    Half-open, on ``completed_at``: a row exactly at the start is in, a row
    exactly at the end is out.  Stated because the report's trailing-24h figure
    is only reproducible if the convention is fixed, and because task 5441
    re-runs this over a wider window and diffs the two.

    Half-open DELIBERATELY DIFFERS from the producer of the figure the resolver
    actually enforces against: ``shared/src/shared/cost_store.py::CostStore.
    model_cost_in_window`` sums with SQLite ``BETWEEN``, which is INCLUSIVE at
    both ends.  The two therefore disagree about a row stamped exactly at the
    window end.  Reproducibility wins here — two adjacent audit windows must not
    both count the row on their shared boundary — so read the headroom below as
    this audit's measurement of the same rows, not as the number routing
    compared against.

    ``at_or_over_ceiling`` is at-or-ABOVE — spend exactly equal to the ceiling
    counts as exhausted, matching ``routing.py::_model_rejection_reason``, which
    rejects on ``spend >= ceiling``.  That comparison IS the resolver's rule.

    With no ceiling supplied, ``ceiling_usd``, ``headroom_usd`` and
    ``at_or_over_ceiling`` are all None rather than 0.0 / negative / True: a
    zero default would render every non-empty window as a ceiling breach, which
    is indistinguishable from a real one on the column most likely to be read as
    an alarm.  Same convention as ``InvocationRecord.
    at_or_over_flat_role_ceiling`` — a limit we were not given is unknown, never
    a plausible-looking False.
    """
    total, count = conn.execute(
        'SELECT COALESCE(SUM(cost_usd), 0.0), COUNT(*) FROM invocations '
        'WHERE model = ? AND completed_at >= ? AND completed_at < ?',
        (model, _iso(window_start), _iso(window_end)),
    ).fetchone()
    return SpendInWindow(
        window_start=_iso(window_start),
        window_end=_iso(window_end),
        total_usd=total,
        invocation_count=count,
        ceiling_usd=ceiling_usd,
        headroom_usd=None if ceiling_usd is None else ceiling_usd - total,
        at_or_over_ceiling=None if ceiling_usd is None else total >= ceiling_usd,
    )


@dataclass(frozen=True)
class RoleUsage:
    """One role's run count and spend on the model."""

    role: str
    count: int
    total_usd: float


@dataclass(frozen=True)
class RoleContainment:
    """Which roles actually ran on the model, against the roles admitted (check 6)."""

    expected_roles: tuple[str, ...]
    by_role: tuple[RoleUsage, ...]
    unexpected_roles: tuple[str, ...]


def roles_on_model(
    conn: sqlite3.Connection,
    *,
    model: str,
    since: datetime,
    expected_roles: Sequence[str],
) -> RoleContainment:
    """Group *model*'s runs by role and compare against *expected_roles*.

    An expected-vs-observed comparison rather than a bare GROUP BY, so a
    containment regression fails LOUDLY on a later run instead of depending on
    someone eyeballing a table.  ``by_role`` is seeded from *expected_roles*
    first, so an admitted role that never ran renders as a visible zero — "the
    merger never ran on this model at all" is the loudest finding this check can
    make, and an omitted row would render it as silence.
    """
    observed = {
        role: (count, total)
        for role, count, total in conn.execute(
            'SELECT role, COUNT(*), COALESCE(SUM(cost_usd), 0.0) FROM invocations '
            'WHERE model = ? AND completed_at >= ? GROUP BY role ORDER BY role',
            (model, _iso(since)),
        )
    }
    ordered = list(expected_roles) + [r for r in observed if r not in expected_roles]
    return RoleContainment(
        expected_roles=tuple(expected_roles),
        by_role=tuple(
            RoleUsage(role=role, count=observed.get(role, (0, 0.0))[0],
                      total_usd=observed.get(role, (0, 0.0))[1])
            for role in ordered
        ),
        unexpected_roles=tuple(r for r in observed if r not in expected_roles),
    )


@dataclass(frozen=True)
class AuditResult:
    """Every measurement the report renders, frozen against one store read."""

    model: str
    since: str
    window_start: str
    window_end: str
    expected_roles: tuple[str, ...]
    routing: RoutingScan
    invocations: tuple[InvocationRecord, ...]
    scoped_cap: ScopedCapScan
    spend: SpendInWindow
    containment: RoleContainment

    @property
    def tier_escalations(self) -> tuple[RoutingSelection, ...]:
        """Selections made at retry tier 1 or above — the retry-ladder question.

        Derived rather than stored: it is a VIEW of ``routing.selections``, and
        storing it too would let the two disagree.
        """
        return tuple(
            s for s in self.routing.selections
            if s.routing_tier is not None and s.routing_tier >= 1
        )


def audit(
    conn: sqlite3.Connection,
    *,
    model: str,
    since: datetime,
    expected_roles: Sequence[str],
    window: tuple[datetime, datetime],
    ceiling_usd: float | None,
    role_ceilings_secs: dict[str, int] | None = None,
) -> AuditResult:
    """Run all five scans against one connection and freeze the results.

    *since* anchors the "has anything happened since the admission?" sections;
    *window* is the separate, usually shorter, half-open span the spend-versus-
    ceiling section is computed over (the ceiling is a trailing-24h rule, while
    the admission may be weeks old).  Keeping them separate is why the report
    can label each section with the window it actually used.
    """
    return AuditResult(
        model=model,
        since=_iso(since),
        window_start=_iso(window[0]),
        window_end=_iso(window[1]),
        expected_roles=tuple(expected_roles),
        routing=scan_routing_decisions(conn, model=model, since=since),
        invocations=scan_invocations(
            conn, model=model, since=since, role_ceilings_secs=role_ceilings_secs,
        ),
        scoped_cap=scan_scoped_cap(conn, model=model, since=since),
        spend=spend_in_window(
            conn, model=model, window_start=window[0],
            window_end=window[1], ceiling_usd=ceiling_usd,
        ),
        containment=roles_on_model(
            conn, model=model, since=since, expected_roles=expected_roles,
        ),
    )


def render_json(result: AuditResult) -> str:
    """Emit *result* as one JSON object, one key per rendered section."""
    return json.dumps({
        'meta': {
            'model': result.model,
            'since': result.since,
            'window_start': result.window_start,
            'window_end': result.window_end,
            'expected_roles': list(result.expected_roles),
        },
        'routing_decisions': asdict(result.routing),
        'invocations': [asdict(r) for r in result.invocations],
        'steward_tier_escalation': {
            'exercised': bool(result.tier_escalations),
            'dispatches': [asdict(s) for s in result.tier_escalations],
        },
        'scoped_cap': asdict(result.scoped_cap),
        'spend': asdict(result.spend),
        'role_containment': asdict(result.containment),
    }, indent=2)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """A markdown table, or a single italic line when there are no rows.

    The empty case is spelled out rather than emitted as a headed table with no
    body, because an empty table reads as a rendering glitch while "none" reads
    as a measurement.
    """
    body = [f"| {' | '.join(str(cell) for cell in row)} |" for row in rows]
    if not body:
        return ['_none_']
    return [
        f"| {' | '.join(header)} |",
        f"|{'|'.join('---' for _ in header)}|",
        *body,
    ]


def render_markdown(result: AuditResult) -> str:
    """Render the six sections, each labelled with the window it was computed over.

    Every number here comes from the frozen *result*; nothing is recomputed, so
    a report that pastes this output cannot disagree with the queries that
    produced it.
    """
    model, since = result.model, result.since
    out: list[str] = []

    out += [f'### 1. Routing decisions for `{model}` since {since}', '']
    out += _table(
        ['timestamp', 'task', 'role', 'source_layer', 'rule_id', 'tier'],
        [(s.timestamp, s.task_id or '-', s.role, s.source_layer, s.rule_id or '-',
          s.routing_tier) for s in result.routing.selections],
    )
    out += ['', f'Rejections naming a model, any role, since {since}:', '']
    out += _table(
        ['timestamp', 'task', 'role', 'resolved to', 'reasons'],
        [(r.timestamp, r.task_id or '-', r.role, r.resolved_model, ', '.join(r.reasons))
         for r in result.routing.rejections],
    )
    out += ['', f'Unparseable payloads skipped: {result.routing.skipped_rows}', '']

    out += [f'### 2. Invocations on `{model}` and how they ended, since {since}', '']
    out += _table(
        ['task', 'project', 'role', 'account', 'cost $', 'turns', 'ok', 'timed out',
         'model @end', 'duration ms', 'over flat ceiling', 'merge'],
        [(r.task_id or '-', r.project_id, r.role, r.account_name, f'{r.cost_usd:.2f}',
          '-' if r.turns is None else r.turns, r.succeeded, r.timed_out,
          r.end_event_model or '-', r.duration_ms, r.at_or_over_flat_role_ceiling,
          _merge_cell(r.merge_outcome)) for r in result.invocations],
    )
    out += ['']

    out += [f'### 3. Dispatches at retry tier >= 1 since {since}', '']
    if result.tier_escalations:
        out += _table(
            ['timestamp', 'task', 'role', 'tier', 'rule_id'],
            [(s.timestamp, s.task_id or '-', s.role, s.routing_tier, s.rule_id or '-')
             for s in result.tier_escalations],
        )
    else:
        out += [f'_Not yet exercised: no dispatch resolved to `{model}` at tier >= 1 '
                f'since {since}. Absence of a tier-escalated dispatch is not a '
                f'failure of the rule; it means the rule has not been reached._']
    out += ['']

    out += [f'### 4. Scoped cap posture for `{model}` since {since}', '']
    out += _table(
        ['created_at', 'account', 'reason'],
        [(h.created_at, h.account_name, h.reason) for h in result.scoped_cap.scoped_hits],
    )
    out += ['', f'Account-level (unscoped) cap hits in the same period: '
                f'{result.scoped_cap.unscoped_cap_hit_count}', '',
            'Service restarts since then (only an orchestrator restart reloads a '
            'restart-tier leaf):', '']
    out += _table(
        ['timestamp', 'service', 'reason'],
        [(r.timestamp, r.service, r.reason or '-') for r in result.scoped_cap.restarts],
    )
    out += ['']

    spend = result.spend
    # '-' for the three ceiling cells when no ceiling was supplied: there is no
    # comparison to render, and printing 0.00 / -3.86 / True would read as a
    # breach. (Section 2's `None` cells are a different thing — a measured
    # tri-state, rendered like the True/False in the same column.)
    ceiling = '-' if spend.ceiling_usd is None else f'{spend.ceiling_usd:.2f}'
    headroom = '-' if spend.headroom_usd is None else f'{spend.headroom_usd:.2f}'
    over_ceiling = '-' if spend.at_or_over_ceiling is None else spend.at_or_over_ceiling
    out += [f'### 5. Spend on `{model}` over [{spend.window_start}, {spend.window_end})', '']
    out += _table(
        ['invocations', 'total $', 'ceiling $', 'headroom $', 'at/over ceiling'],
        [(spend.invocation_count, f'{spend.total_usd:.2f}', ceiling, headroom,
          over_ceiling)],
    )
    out += ['']

    containment = result.containment
    out += [f'### 6. Roles observed on `{model}` since {since}', '',
            f'Admitted roles: {", ".join(containment.expected_roles)}', '']
    out += _table(
        ['role', 'invocations', 'total $', 'admitted'],
        [(u.role, u.count, f'{u.total_usd:.2f}', u.role in containment.expected_roles)
         for u in containment.by_role],
    )
    out += ['', f'Roles outside the admitted set: '
                f'{", ".join(containment.unexpected_roles) or "none"}', '']
    return '\n'.join(out)


def _merge_cell(outcome: MergeOutcome | None) -> str:
    if outcome is None:
        return '-'
    detail = outcome.merge_sha or outcome.reason or ''
    return f'{outcome.state} ({detail})' if detail else outcome.state


def _parse_window(spec: str) -> timedelta:
    """Parse a `<N>h` / `<N>d` trailing-window spec.

    Anchored: '24' and '24x' must both be rejected rather than silently
    truncated to 24 hours.
    """
    match = _WINDOW_RE.match(spec)
    if not match:
        raise argparse.ArgumentTypeError(
            f'bad --window {spec!r}: expected <N>h or <N>d, e.g. 24h or 14d.'
        )
    size = int(match.group(1))
    return timedelta(hours=size) if match.group(2) == 'h' else timedelta(days=size)


def _parse_moment(spec: str) -> datetime:
    """Parse an ISO-8601 instant, reading a naive one as UTC.

    Naive-means-UTC rather than naive-means-local: the store is UTC throughout,
    and silently shifting a hand-typed bound by the host's offset would move
    the window without saying so.
    """
    try:
        parsed = datetime.fromisoformat(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'bad ISO-8601 instant {spec!r}: {exc}') from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Audit whether an admitted model is actually being dispatched, '
                    'and whether its ceiling and scoped cap are behaving. '
                    'Strictly read-only.',
    )
    parser.add_argument('--model', required=True, help='the model string to audit')
    parser.add_argument(
        '--expect-roles', required=True,
        help='comma-separated roles the model was admitted for, e.g. merger,steward',
    )
    parser.add_argument(
        '--since', required=True, type=_parse_moment,
        help='ISO-8601 instant the admission was applied; anchors sections 1-4 and 6',
    )
    parser.add_argument(
        '--window', default='24h', type=_parse_window,
        help='trailing window for the spend-vs-ceiling section (default: 24h)',
    )
    parser.add_argument(
        '--ceiling', default=None, type=float,
        help='per-model daily ceiling in USD to measure spend against; omitted '
             'means no ceiling was supplied, and the ceiling, headroom and '
             'at/over cells render as "-" rather than as a spurious breach',
    )
    parser.add_argument('--runs-db', default=DEFAULT_RUNS_DB, type=Path)
    parser.add_argument('--format', default='markdown', choices=('markdown', 'json'))
    args = parser.parse_args(argv)

    window_end = datetime.now(UTC)
    conn = _connect_ro(args.runs_db)
    try:
        result = audit(
            conn,
            model=args.model,
            since=args.since,
            expected_roles=tuple(
                r.strip() for r in args.expect_roles.split(',') if r.strip()
            ),
            window=(window_end - args.window, window_end),
            ceiling_usd=args.ceiling,
        )
    finally:
        conn.close()

    render = render_json if args.format == 'json' else render_markdown
    print(render(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
