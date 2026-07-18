"""Escalation-lifecycle integration gate — boundary matrix end-to-end (task 2662 / θ).

INTEGRATION GATE for plans/escalation-lifecycle-dashboard-prd.md ("Boundary-test
sketch" rows 1–11). Adds no product behaviour; it proves the
α (escalation pkg) → archive → γ (dashboard aggregator + endpoint) →
δ/ε/ζ (frontend payload) chain agrees end-to-end.

Unlike the existing γ suites (dashboard/tests/test_escalation_analytics.py,
which hand-write golden JSON dicts), this module drives a LIVE round-trip: it
FILES + RESOLVES escalations through the real escalation server/queue
chokepoints, archives them, then runs the real aggregator + FastAPI route over
THAT archive. It lives in dashboard/tests/ (not escalation/tests/) because the
γ aggregator already imports escalation.classify/queue — a dashboard test that
imports escalation.server/queue follows the natural dependency direction, and
both packages are collected by the single root ``pytest`` invocation.

Row coverage: rows 1–6, 9, 10, 11 end-to-end here; row 7 (synthetic-scale
perf) is asserted by γ's own TestBuildEscalationAnalyticsPerf in the same
root-pytest invocation and is deliberately NOT duplicated here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient  # noqa: F401  (route tests use the `client` conftest fixture)

from escalation.classify import effective_benign
from escalation.models import RESOLUTION_CLASSES, Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

from dashboard.config import DashboardConfig
from dashboard.data.escalation_analytics import (
    _aggregate_project,
    build_escalation_analytics,
)

# ---------------------------------------------------------------------------
# Per-row source (agent_role) labels — one distinct source per boundary row so
# the per-source origin aggregates can be asserted against the round-tripped
# truth without cross-row aliasing.
# ---------------------------------------------------------------------------

SRC_ROW1 = 'gate-src-row1'          # actionable, stamped (L1, explicit class via server)
SRC_ROW2_MEMBER = 'gate-src-row2-member'  # benign, stamped (L1 cascade members)
SRC_ROW2_L2 = 'gate-src-row2-l2'    # benign, stamped (L2 resolved with class='benign')
SRC_ROW3 = 'gate-src-row3'          # actionable, inferred (L1, no class → proxy)
SRC_ROW4 = 'gate-src-row4'          # PENDING (rejected class) → unclassified
SRC_ROW5 = 'gate-src-row5'          # benign, stamped (L0 age-out auto-dismiss)

# Deterministic escalation ids (task_id embedded per the esc-{task}-{seq} form).
ID_ROW1 = 'esc-g1-1'
ID_ROW2_M1 = 'esc-g2m1-1'
ID_ROW2_M2 = 'esc-g2m2-1'
ID_ROW2_L2 = 'esc-g2l2-1'
ID_ROW3 = 'esc-g3-1'
ID_ROW4 = 'esc-g4-1'
ID_ROW5 = 'esc-g5-1'
CORRUPT_ID = 'esc-999-1'            # dropped as unparseable JSON at the queue root

# Filed-at timestamps: anchored well before the (real wall-clock) resolved_at
# the chokepoint stamps, so every terminal record has a positive lifespan and
# the L2's timestamp is >= its members' (non-negative l1_to_l2_promotion delta).
_BASE = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    """Fixed aggregation clock — deterministic ``now`` for every aggregator call."""
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _live_queue(tmp_path: Path) -> EscalationQueue:
    """An EscalationQueue rooted at the exact dir DashboardConfig.escalations_dir derives.

    ``DashboardConfig(project_root=tmp_path).escalations_dir`` is
    ``tmp_path/'data'/'escalations'`` — so a queue built here writes to the same
    archive the route/aggregator will later read for ``project_root=tmp_path``.
    """
    return EscalationQueue(tmp_path / 'data' / 'escalations')


def _make_config(tmp_path: Path, *, known_project_roots: list[Path] | None = None) -> DashboardConfig:
    """DashboardConfig pointed at *tmp_path* (mirrors test_tab_escalation_analytics._make_config)."""
    return DashboardConfig(project_root=tmp_path, known_project_roots=known_project_roots or [])


async def _resolve_via_server(server: Any, esc_id: str, **kw: Any) -> dict[str, Any]:
    """Drive the REAL resolve_issue MCP tool (chokepoint for rows 1/3/4).

    Mirrors escalation/tests/test_server.py's ``_blocker``/``_get_pending``
    idiom: ``get_tool`` is awaited, but ``resolve_issue`` is a sync ``def`` so
    ``tool.fn(...)`` returns its dict directly (no await on the call).
    """
    tool = await server.get_tool('resolve_issue')
    return tool.fn(esc_id, **kw)


@dataclass
class _RecordExpectation:
    """Round-tripped truth for one archived record (the builder's bookkeeping unit).

    ``cls``/``prov`` are the expected :func:`escalation.classify.effective_benign`
    pair — the SAME predicate γ's aggregator reads — so a test can pin the exact
    ``(class, provenance)`` the origin/flow blocks will compute (INV-5: one
    classification site shared by the α write path and the γ read path).
    """

    id: str
    agent_role: str
    level: int
    expected_status: str        # 'resolved' | 'dismissed' | 'pending'
    cls: str | None             # effective_benign class ('benign'|'actionable'|None)
    prov: str                   # effective_benign provenance ('stamped'|'inferred'|'excluded')
    resolved_by: str | None = None


@dataclass
class _LiveArchive:
    """Bookkeeping returned by :func:`_build_live_boundary_archive`."""

    records: list[_RecordExpectation] = field(default_factory=list)
    corrupt_id: str = CORRUPT_ID
    triaged_id: str = ID_ROW1
    row4_reject_result: dict[str, Any] | None = None

    def terminal(self) -> list[_RecordExpectation]:
        return [r for r in self.records if r.expected_status in ('resolved', 'dismissed')]

    def expected_origin_by_source(self) -> dict[str, dict[str, int]]:
        """Per-source (agent_role) benign/actionable/stamped truth over terminal records."""
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            bucket = out.setdefault(r.agent_role, {'benign': 0, 'actionable': 0, 'stamped': 0})
            if r.cls == 'benign':
                bucket['benign'] += 1
            elif r.cls == 'actionable':
                bucket['actionable'] += 1
            if r.prov == 'stamped':
                bucket['stamped'] += 1
        return out


def _submit_pending(
    queue: EscalationQueue, esc_id: str, task_id: str, agent_role: str, *,
    level: int, filed_at: datetime, members: list[str] | None = None,
) -> None:
    queue.submit(Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=agent_role,
        severity='blocking',
        category='cleanup_needed',
        summary=f'boundary gate {esc_id}',
        timestamp=_iso(filed_at),
        status='pending',
        level=level,
        members=members or [],
    ))


async def _build_live_boundary_archive(queue: EscalationQueue, server: Any) -> _LiveArchive:
    """File + resolve the rows-1–5 boundary set through the REAL chokepoints.

    Every terminal write goes through α's production terminal-write path
    (server ``resolve_issue`` → ``queue.resolve`` for rows 1/3, ``queue.resolve``
    cascade for row 2, ``queue.dismiss_all_pending`` for row 5); row 4 exercises
    the reject-before-mutate path (record stays pending). A single corrupt
    ``esc-999-1.json`` is dropped at the queue root (INV-4 parse-failure surface).

    Returns an :class:`_LiveArchive` carrying the round-tripped truth for every
    record so callers can pin the aggregator/route output against it.
    """
    arch = _LiveArchive()

    # --- Row 1: L1, explicit class='actionable' via the server (stamped). Also
    #     triage-stamped while pending so the aggregator emits triage_segments. ---
    _submit_pending(queue, ID_ROW1, 'g1', SRC_ROW1, level=1, filed_at=_BASE)
    queue.stamp_triage(ID_ROW1, triaged_by='escalation-watcher-auto', triage_note='gate triage')
    await _resolve_via_server(
        server, ID_ROW1, resolution='fix applied', action='resume',
        resolved_by='interactive', resolution_class='actionable',
    )
    arch.records.append(_RecordExpectation(
        ID_ROW1, SRC_ROW1, 1, 'resolved', 'actionable', 'stamped', resolved_by='interactive',
    ))

    # --- Row 2: L2 cluster resolved with class='benign'; members inherit the
    #     stamp via the cascade (resolved_by='l2-cascade:<L2 id>'). ---
    _submit_pending(queue, ID_ROW2_M1, 'g2m1', SRC_ROW2_MEMBER, level=1, filed_at=_BASE + timedelta(hours=1))
    _submit_pending(queue, ID_ROW2_M2, 'g2m2', SRC_ROW2_MEMBER, level=1, filed_at=_BASE + timedelta(hours=2))
    _submit_pending(
        queue, ID_ROW2_L2, 'g2l2', SRC_ROW2_L2, level=2, filed_at=_BASE + timedelta(hours=3),
        members=[ID_ROW2_M1, ID_ROW2_M2],
    )
    queue.resolve(ID_ROW2_L2, 'cluster ruling', resolution_class='benign')
    cascade_by = f'l2-cascade:{ID_ROW2_L2}'
    arch.records.append(_RecordExpectation(
        ID_ROW2_L2, SRC_ROW2_L2, 2, 'resolved', 'benign', 'stamped', resolved_by=None,
    ))
    arch.records.append(_RecordExpectation(
        ID_ROW2_M1, SRC_ROW2_MEMBER, 1, 'resolved', 'benign', 'stamped', resolved_by=cascade_by,
    ))
    arch.records.append(_RecordExpectation(
        ID_ROW2_M2, SRC_ROW2_MEMBER, 1, 'resolved', 'benign', 'stamped', resolved_by=cascade_by,
    ))

    # --- Row 3: L1 resolved via the server with NO class → unstamped, proxy
    #     infers 'actionable' from the resolved status. ---
    _submit_pending(queue, ID_ROW3, 'g3', SRC_ROW3, level=1, filed_at=_BASE + timedelta(hours=4))
    await _resolve_via_server(
        server, ID_ROW3, resolution='resumed', action='resume', resolved_by='interactive',
    )
    arch.records.append(_RecordExpectation(
        ID_ROW3, SRC_ROW3, 1, 'resolved', 'actionable', 'inferred', resolved_by='interactive',
    ))

    # --- Row 4: unknown class rejected before any mutation → record stays pending. ---
    _submit_pending(queue, ID_ROW4, 'g4', SRC_ROW4, level=1, filed_at=_BASE + timedelta(hours=5))
    arch.row4_reject_result = await _resolve_via_server(
        server, ID_ROW4, resolution='should not apply', resolution_class='meh',
    )
    arch.records.append(_RecordExpectation(
        ID_ROW4, SRC_ROW4, 1, 'pending', None, 'excluded', resolved_by=None,
    ))

    # --- Row 5: age-out auto-dismiss of a stale pending L0 (only L0 is swept;
    #     the row-4 L1 above is preserved). ---
    _submit_pending(queue, ID_ROW5, 'g5', SRC_ROW5, level=0, filed_at=_BASE + timedelta(hours=6))
    queue.dismiss_all_pending('age-out sweep')
    arch.records.append(_RecordExpectation(
        ID_ROW5, SRC_ROW5, 0, 'dismissed', 'benign', 'stamped', resolved_by='auto-dismissed',
    ))

    # --- Corrupt file (INV-4): a non-JSON esc-*.json at the queue root. ---
    (queue.queue_dir / f'{CORRUPT_ID}.json').write_text('{ this is not valid json ,,,')

    return arch


def _make_server(queue: EscalationQueue) -> Any:
    """create_server harness with startup_sweep disabled for a controlled archive."""
    return create_server(queue, startup_sweep=False)


# ---------------------------------------------------------------------------
# step-1 — TestAlphaChokepointRoundTrip: boundary rows 1–5 at the record level.
# ---------------------------------------------------------------------------


class TestAlphaChokepointRoundTrip:
    """Boundary rows 1–5 at the record level, over the live-chokepoint archive.

    Every row is filed + resolved through α's REAL terminal-write path by
    :func:`_build_live_boundary_archive`; here we reload each archived record
    via ``queue.get()`` and pin its ``(resolution_class, effective_benign)``
    against the PRD Seam-1 contract (per-path default, cascade inheritance,
    proxy fallback, reject-before-mutate). A disagreement here is a real
    α-seam regression — assertions must never be weakened to force green.
    """

    async def test_boundary_rows_1_through_5(self, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        server = _make_server(queue)
        arch = await _build_live_boundary_archive(queue, server)

        # --- Row 1: an explicit 'actionable' stamp survives the
        #     server → queue.resolve round-trip and is read stamped. ---
        r1 = queue.get(ID_ROW1)
        assert r1 is not None and r1.status == 'resolved'
        assert r1.resolution_class == 'actionable'
        assert effective_benign(r1) == ('actionable', 'stamped')

        # --- Row 3: no class passed → record stays unstamped; effective_benign's
        #     read-time proxy infers 'actionable' from the resolved status. ---
        r3 = queue.get(ID_ROW3)
        assert r3 is not None and r3.status == 'resolved'
        assert r3.resolution_class is None
        assert effective_benign(r3) == ('actionable', 'inferred')

        # --- Row 4: an unknown class is rejected BEFORE any mutation, at both
        #     the server tool and the queue chokepoint (INV-1 reject-before-
        #     mutate). We pin the rejection CONTRACT (typed code + unchanged
        #     record + ValueError), NOT the exact legal-value set — that set
        #     legitimately grows (e.g. 'moot-terminal-subject', task 2724). ---
        assert {'benign', 'actionable'} <= RESOLUTION_CLASSES
        assert 'meh' not in RESOLUTION_CLASSES
        assert arch.row4_reject_result is not None
        assert arch.row4_reject_result.get('code') == 'invalid_resolution_class'
        r4 = queue.get(ID_ROW4)
        assert r4 is not None and r4.status == 'pending'   # unchanged by the server reject
        assert r4.resolution_class is None                 # never stamped
        assert r4.resolved_at is None and r4.resolved_by is None
        with pytest.raises(ValueError):
            queue.resolve(ID_ROW4, 'still rejected', resolution_class='meh')
        assert queue.get(ID_ROW4).status == 'pending'      # queue reject left it untouched too

        # --- Row 2: resolving the L2 with class='benign' cascades the stamp to
        #     every member L1 (stamped-inherited via resolved_by='l2-cascade:<id>'). ---
        l2 = queue.get(ID_ROW2_L2)
        assert l2 is not None and l2.status == 'resolved'
        assert l2.resolution_class == 'benign'
        assert effective_benign(l2) == ('benign', 'stamped')
        for member_id in (ID_ROW2_M1, ID_ROW2_M2):
            m = queue.get(member_id)
            assert m is not None, f'cascade member {member_id} missing from archive'
            assert m.status in ('resolved', 'dismissed')
            assert m.resolution_class == 'benign'
            assert effective_benign(m) == ('benign', 'stamped')
            assert m.resolved_by == f'l2-cascade:{ID_ROW2_L2}'

        # --- Row 5: age-out auto-dismiss → the reaper-sweep per-path benign
        #     default is stamped (resolved_by='auto-dismissed'). ---
        r5 = queue.get(ID_ROW5)
        assert r5 is not None
        assert r5.status == 'dismissed'
        assert r5.resolved_by == 'auto-dismissed'
        assert r5.resolution_class == 'benign'
        assert effective_benign(r5) == ('benign', 'stamped')


# ---------------------------------------------------------------------------
# step-3 — TestAggregateOverLiveArchive: rows 6 + 11 (real aggregator).
# ---------------------------------------------------------------------------


class TestAggregateOverLiveArchive:
    """Rows 6 + 11: the REAL aggregator over the live round-tripped archive.

    The α↔γ seam — γ reads the records α wrote through its production
    chokepoints (not hand-written golden dicts) via the SAME
    ``effective_benign``/``classify_resolver_tier`` predicates, so a
    disagreement in record shape or classification surfaces here.
    """

    async def test_row6_origin_and_parse_failures(self, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        server = _make_server(queue)
        arch = await _build_live_boundary_archive(queue, server)

        esc_dir = _make_config(tmp_path).escalations_dir
        entry, pf = _aggregate_project(tmp_path.name, esc_dir, tmp_path / 'runs.db', now=_now())

        # INV-4: the one corrupt esc-999-1.json is counted, never silently dropped.
        assert pf == 1

        # Per-source origin benign/actionable/stamped == the round-tripped truth
        # (cascade members + age-out record → benign & stamped; the unstamped
        # resume → actionable & inferred; the rejected row-4 record → pending,
        # unclassified). effective_benign is the shared predicate (INV-5).
        expected = arch.expected_origin_by_source()
        sources_by_name = {s['source']: s for s in entry['origin']['sources']}
        assert set(sources_by_name) == set(expected)
        for name, exp in expected.items():
            s = sources_by_name[name]
            assert s['benign'] == exp['benign'], f'{name} benign'
            assert s['actionable'] == exp['actionable'], f'{name} actionable'
            # No 'moot-terminal-subject' records in this archive, so the
            # classified count is exactly benign+actionable; derive the expected
            # stamped_share from the stamped truth.
            classified = exp['benign'] + exp['actionable']
            expected_share = exp['stamped'] / classified if classified else 0.0
            assert s['stamped_share'] == pytest.approx(expected_share), f'{name} stamped_share'

        # build_escalation_analytics wraps the same archive: parse_failures
        # surfaces the corrupt file (+0 for the committed regime-markers file),
        # and generated_at is a non-empty iso string.
        payload = build_escalation_analytics(
            [(tmp_path.name, esc_dir, tmp_path / 'runs.db')], now=_now(),
        )
        assert payload['parse_failures'] >= 1
        assert isinstance(payload['generated_at'], str) and payload['generated_at']

    async def test_row11_flow_cube_reconciles_over_live_archive(self, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        server = _make_server(queue)
        arch = await _build_live_boundary_archive(queue, server)

        esc_dir = _make_config(tmp_path).escalations_dir
        entry, _pf = _aggregate_project(tmp_path.name, esc_dir, tmp_path / 'runs.db', now=_now())

        samples = entry['lifespan']['samples']
        flow_daily = entry['workflow']['flow_daily']
        tier_weekly = entry['workflow']['tier_weekly']
        sources = entry['origin']['sources']

        # sum(flow_daily.n) == len(samples) == terminal-with-valid-times count.
        # Holds exactly because queue.resolve always stamps resolved_at, and the
        # flow cube and samples share one population (both gate on parseable
        # timestamp AND resolved_at).
        expected_terminal = len(arch.terminal())
        total_flow_n = sum(row['n'] for row in flow_daily)
        assert total_flow_n == len(samples) == expected_terminal

        # Per-date marginals match exactly (both keyed by date(resolved_at)) →
        # the identity survives summing over ANY date sub-window.
        sample_date_counts = Counter(row[0] for row in samples)
        flow_date_counts: Counter = Counter()
        for row in flow_daily:
            flow_date_counts[row['date']] += row['n']
        assert sample_date_counts == flow_date_counts

        # Per-tier == tier_weekly totals == samples-by-tier (same population/derivation).
        sample_tier_counts = Counter(row[1] for row in samples)
        flow_tier_counts: Counter = Counter()
        for row in flow_daily:
            flow_tier_counts[row['tier']] += row['n']
        tier_weekly_totals: Counter = Counter()
        for week_bucket in tier_weekly.values():
            for tier, n in week_bucket.items():
                tier_weekly_totals[tier] += n
        assert flow_tier_counts == tier_weekly_totals == sample_tier_counts

        # Per-(source, class) == sources[] benign/actionable.
        flow_source_class_counts: Counter = Counter()
        for row in flow_daily:
            flow_source_class_counts[(row['source'], row['class'])] += row['n']
        for s in sources:
            assert flow_source_class_counts.get((s['source'], 'benign'), 0) == s['benign']
            assert flow_source_class_counts.get((s['source'], 'actionable'), 0) == s['actionable']
