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

Since plans/dashboard-one-datum-one-path-prd.md leaf η (task 5596) the
aggregator reads the escalation CORPUS — one walk of every queue's root and
archive — so this gate walks the escalation-server-written tree with
:func:`dashboard.data.escalation_corpus.measure_corpus` rather than handing the
aggregator a directory. Two rows of that PRD run end-to-end here as well:
sketch #10 (one walk, two named views, one instant) and the complete
resolution-class split (every class shown, its parts summing to the terminal
population that ``action_mix`` also counts).
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP
from escalation.archive import archive_dir_for_date
from escalation.classify import effective_benign
from escalation.models import RESOLUTION_CLASSES, Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from starlette.testclient import (
    TestClient,  # noqa: F401  (route tests use the `client` conftest fixture)
)

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum
from dashboard.data.escalation_analytics import build_escalation_analytics
from dashboard.data.escalation_corpus import (
    EscalationCorpus,
    corpus_queues,
    measure_corpus,
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
SRC_STRAND = 'gate-src-strand'      # stale-strand, stamped (L0 strand sweep)
SRC_MOOT = 'gate-src-moot'          # moot-terminal-subject, stamped (via the server)
SRC_SKETCH = 'gate-src-sketch10'    # PENDING, root and archive (PRD sketch #10)

# Deterministic escalation ids (task_id embedded per the esc-{task}-{seq} form).
ID_ROW1 = 'esc-g1-1'
ID_ROW2_M1 = 'esc-g2m1-1'
ID_ROW2_M2 = 'esc-g2m2-1'
ID_ROW2_L2 = 'esc-g2l2-1'
ID_ROW3 = 'esc-g3-1'
ID_ROW4 = 'esc-g4-1'
ID_ROW5 = 'esc-g5-1'
ID_STRAND = 'esc-gs-1'
ID_MOOT = 'esc-gm-1'
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


def _clear_escalation_caches() -> None:
    """Clear every cache the two escalation routes read, so a route test walks its own tree.

    The corpus cache and the analytics memo, plus the task snapshot units and
    the per-id lookup cache the /escalations route's owner probe and cards read.
    """
    from dashboard.api.escalations import _analytics_memo_clear
    from dashboard.data.escalation_corpus import _corpus_cache_clear
    from dashboard.data.task_lookup import _lookup_cache_clear
    from dashboard.data.task_snapshot import _snapshot_cache_clear

    _corpus_cache_clear()
    _analytics_memo_clear()
    _snapshot_cache_clear()
    _lookup_cache_clear()


def _walk(tmp_path: Path) -> Datum[EscalationCorpus]:
    """The corpus over *tmp_path*'s queues, walked at the fixed aggregation clock."""
    return measure_corpus(corpus_queues(_make_config(tmp_path)), now=_now())


def _no_tasks() -> CannedMCP:
    """A fused-memory substrate holding no task, so no route read reaches a real server."""
    return CannedMCP(rows=[], status_map={}, status_page_size=2000)


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
    cls: str | None             # effective_benign class (a RESOLUTION_CLASSES member, or None)
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

    def expected_origin_by_source(self) -> dict[str, dict[str, Any]]:
        """Per-source (agent_role) class split and stamped count, over terminal records.

        ``classes`` is keyed by every :data:`RESOLUTION_CLASSES` member,
        zero-filled — the shape the aggregator serves.
        """
        out: dict[str, dict[str, Any]] = {}
        for r in self.records:
            bucket = out.setdefault(
                r.agent_role, {'classes': dict.fromkeys(RESOLUTION_CLASSES, 0), 'stamped': 0},
            )
            if r.cls is not None:
                bucket['classes'][r.cls] += 1
            if r.prov == 'stamped':
                bucket['stamped'] += 1
        return out


def _assert_origin_matches(sources: list[dict[str, Any]], arch: _LiveArchive) -> None:
    """Every source's served class split and stamped share equal the round-tripped truth."""
    expected = arch.expected_origin_by_source()
    sources_by_name = {s['source']: s for s in sources}
    assert set(sources_by_name) == set(expected)
    for name, exp in expected.items():
        s = sources_by_name[name]
        assert s['classes'] == exp['classes'], f'{name} classes'
        classified = sum(exp['classes'].values())
        assert s['classified'] == classified, f'{name} classified'
        expected_share = exp['stamped'] / classified if classified else 0.0
        assert s['stamped_share'] == pytest.approx(expected_share), f'{name} stamped_share'


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


async def _build_every_class_archive(queue: EscalationQueue, server: Any) -> _LiveArchive:
    """The boundary archive, plus one record in each class it does not already hold.

    All go through α's production terminal-write paths: the L0 strand sweep
    (``dismiss_all_pending`` with a strand threshold, the reaper's path) stamps
    ``'stale-strand'``, and the server's ``resolve_issue`` stamps
    ``'moot-terminal-subject'`` and every remaining class.
    """
    arch = await _build_live_boundary_archive(queue, server)

    _submit_pending(queue, ID_STRAND, 'gs', SRC_STRAND, level=0, filed_at=_BASE + timedelta(hours=7))
    queue.dismiss_all_pending('strand sweep', strand_age_secs=3600)
    arch.records.append(_RecordExpectation(
        ID_STRAND, SRC_STRAND, 0, 'dismissed', 'stale-strand', 'stamped', resolved_by='auto-dismissed',
    ))

    _submit_pending(queue, ID_MOOT, 'gm', SRC_MOOT, level=1, filed_at=_BASE + timedelta(hours=8))
    await _resolve_via_server(
        server, ID_MOOT, resolution='subject already terminal', resolved_by='interactive',
        resolution_class='moot-terminal-subject',
    )
    arch.records.append(_RecordExpectation(
        ID_MOOT, SRC_MOOT, 1, 'resolved', 'moot-terminal-subject', 'stamped', resolved_by='interactive',
    ))

    held = {r.cls for r in arch.records if r.cls is not None}
    for n, cls in enumerate(sorted(RESOLUTION_CLASSES - held), start=1):
        esc_id, source = f'esc-gx{n}-1', f'gate-src-{cls}'
        _submit_pending(queue, esc_id, f'gx{n}', source, level=0, filed_at=_BASE + timedelta(hours=8 + n))
        await _resolve_via_server(
            server, esc_id, resolution=f'ruled {cls}', resolved_by='interactive', resolution_class=cls,
        )
        arch.records.append(_RecordExpectation(
            esc_id, source, 0, 'resolved', cls, 'stamped', resolved_by='interactive',
        ))
    return arch


def _make_server(queue: EscalationQueue) -> Any:
    """create_server harness with startup_sweep disabled for a controlled archive."""
    return create_server(queue, startup_sweep=False)


def _live_archive_sync(base: Path) -> tuple[EscalationQueue, _LiveArchive]:
    """Build the live boundary archive under *base* synchronously.

    The builder awaits the async ``resolve_issue`` tool, but the FastAPI route
    tests drive the SYNC starlette ``TestClient`` (whose lifespan portal runs
    its own loop in a separate thread). Running the builder via ``asyncio.run``
    in the main thread — which has no running loop in a sync test — keeps the
    two off each other.
    """
    queue = EscalationQueue(base / 'data' / 'escalations')
    server = _make_server(queue)
    arch = asyncio.run(_build_live_boundary_archive(queue, server))
    return queue, arch


def _plain_terminal_archive(base: Path, *, source: str = 'gate-src-secondary') -> None:
    """Write one plain resolved record (terminal, valid times, NO triaged_at) under *base*.

    Used as the row-10 counter-example project: an archive with no
    ``triaged_at`` record must yield NO ``triage_segments`` key (render-when-
    present), never an error.
    """
    queue = EscalationQueue(base / 'data' / 'escalations')
    queue.submit(Escalation(
        id='esc-sec-1', task_id='sec', agent_role=source, severity='blocking',
        category='cleanup_needed', summary='secondary, no triage',
        timestamp=_iso(_BASE), status='pending', level=0,
    ))
    queue.resolve('esc-sec-1', 'closed', resolved_by='interactive')


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
        r4_after = queue.get(ID_ROW4)
        assert r4_after is not None and r4_after.status == 'pending'  # queue reject left it untouched too

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

        corpus = _walk(tmp_path)
        assert corpus.value is not None

        # INV-4: the one corrupt esc-999-1.json is counted, never silently dropped.
        (unreadable,) = corpus.value.scan(str(tmp_path)).unreadable
        assert unreadable.path.endswith(f'{CORRUPT_ID}.json')

        # Per-source origin class split + stamped share == the round-tripped
        # truth (cascade members + age-out record → benign & stamped; the
        # unstamped resume → actionable & inferred; the rejected row-4 record →
        # pending, unclassified). effective_benign is the shared predicate (INV-5).
        payload = build_escalation_analytics(corpus)
        (entry,) = payload['per_project']
        _assert_origin_matches(entry['origin']['sources'], arch)

        # The payload surfaces the corrupt file (+0 for the committed
        # regime-markers file), and answers for the walk it was derived from.
        assert payload['parse_failures'] >= 1
        assert payload['generated_at'] == _now().isoformat()

    async def test_row11_flow_cube_reconciles_over_live_archive(self, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        server = _make_server(queue)
        arch = await _build_live_boundary_archive(queue, server)

        (entry,) = build_escalation_analytics(_walk(tmp_path))['per_project']

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

        # Per-(source, class) == sources[] classes, for every resolution class.
        flow_source_class_counts: Counter = Counter()
        for row in flow_daily:
            flow_source_class_counts[(row['source'], row['class'])] += row['n']
        for s in sources:
            for cls in RESOLUTION_CLASSES:
                assert flow_source_class_counts.get((s['source'], cls), 0) == s['classes'][cls]


# ---------------------------------------------------------------------------
# step-5 — TestEndpointOverLiveArchive: rows 9 + 10 + row-6 through the real route.
# ---------------------------------------------------------------------------


class TestEndpointOverLiveArchive:
    """Rows 6 + 9 + 10 through the REAL FastAPI route over the live archive.

    Sync tests (like the existing route suite): the async builder runs via
    :func:`_live_archive_sync`; the route is driven by the sync starlette
    ``TestClient`` (``client`` conftest fixture). The route wraps the aggregator
    payload under the ``ESCALATION_ANALYTICS`` key.
    """

    def test_row6_route_contract_and_origin_stamps(self, client, tmp_path: Path) -> None:

        _queue, arch = _live_archive_sync(tmp_path)

        client.app.state.config = _make_config(tmp_path)
        _clear_escalation_caches()
        resp = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp.status_code == 200
        body = resp.json()
        assert 'ESCALATION_ANALYTICS' in body
        analytics = body['ESCALATION_ANALYTICS']
        assert set(analytics) == {
            'generated_at', 'parse_failures', 'regime_markers', 'per_project',
            # The two archive-reach signals, and the only fields that can tell
            # an absent archive from an empty one: ``archives_present`` (all)
            # is the completeness diagnostic, ``archives_reached`` (any) is
            # the corpus cache's own cacheability fact.
            'archives_present', 'archives_reached',
            # The corpus' named views across every orchestrator queue.
            'views',
        }
        assert isinstance(analytics['generated_at'], str) and analytics['generated_at']
        assert analytics['archives_present'] is True
        assert analytics['archives_reached'] is True
        assert isinstance(analytics['regime_markers'], list)
        assert len(analytics['per_project']) == 1
        entry = analytics['per_project'][0]
        assert entry['project'] == tmp_path.name

        # Origin reflects the round-tripped stamps (same per-source truth the
        # aggregator produced in step-3, now surfaced through the route/JSON layer).
        _assert_origin_matches(entry['origin']['sources'], arch)

    def test_row9_malformed_regime_markers_never_500s(self, client, tmp_path, monkeypatch) -> None:
        import dashboard.data.escalation_analytics as escalation_analytics_module

        _live_archive_sync(tmp_path)
        client.app.state.config = _make_config(tmp_path)

        # Pre-corruption baseline: parse_failures already >= 1 (the corrupt esc
        # file in the live archive), regime_markers loads cleanly.
        _clear_escalation_caches()
        resp1 = client.get('/api/v2/dashboard/escalation-analytics')
        assert resp1.status_code == 200
        pre = resp1.json()['ESCALATION_ANALYTICS']['parse_failures']
        assert pre >= 1

        # Corrupt the regime-markers file the route loads → must degrade loudly,
        # never 500: regime_markers empties and parse_failures strictly increases.
        bad_markers = tmp_path / 'bad-regime-markers.yaml'
        bad_markers.write_text('date: [unclosed')
        monkeypatch.setattr(
            escalation_analytics_module, '_DEFAULT_REGIME_MARKERS_PATH', bad_markers,
        )
        _clear_escalation_caches()
        resp2 = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp2.status_code == 200
        analytics2 = resp2.json()['ESCALATION_ANALYTICS']
        assert analytics2['regime_markers'] == []
        assert analytics2['parse_failures'] > pre

    def test_row10_triage_segments_render_when_present(self, client, tmp_path: Path) -> None:

        # Primary: the live archive (row 1 carries a round-tripped triaged_at).
        _live_archive_sync(tmp_path)
        # Secondary: a plain archive with NO triaged_at record.
        secondary = tmp_path / 'secondary'
        _plain_terminal_archive(secondary)

        client.app.state.config = _make_config(tmp_path, known_project_roots=[secondary])
        _clear_escalation_caches()
        resp = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp.status_code == 200
        per_project = resp.json()['ESCALATION_ANALYTICS']['per_project']
        by_project = {p['project']: p for p in per_project}
        assert tmp_path.name in by_project and 'secondary' in by_project

        # Primary carries a triaged terminal record → triage_segments present.
        primary_lifespan = by_project[tmp_path.name]['lifespan']
        assert 'triage_segments' in primary_lifespan
        assert primary_lifespan['triage_segments']['count'] >= 1

        # Secondary has no triaged_at record → the key is omitted entirely
        # (render-when-present), and the endpoint still returns 200 for it.
        assert 'triage_segments' not in by_project['secondary']['lifespan']


# ---------------------------------------------------------------------------
# step-7 — TestFrontendPayloadContract: the γ↔frontend seam's final agreement.
# ---------------------------------------------------------------------------


class TestFrontendPayloadContract:
    """Every leaf key the δ/ε/ζ components consume is present + typed over the live payload.

    Pins the producer (γ) to its three consumers without a JS runtime: the
    fields asserted here are exactly those read by the analytics tab (δ,
    test_tab_escalation_analytics.py), the StatTile strip (ε,
    test_tab_escalations.py), and the mini-Sankey flow diagram (ζ,
    test_esc_flow_diagram.py). A subset (``<=``) check per object so a
    render-when-present extra (e.g. ``triage_segments``, ``samples_downsampled``)
    never trips the contract, while a DROPPED consumed key fails loudly.
    """

    def test_served_payload_exposes_every_consumed_key(self, client, tmp_path: Path) -> None:

        _live_archive_sync(tmp_path)
        client.app.state.config = _make_config(tmp_path)
        _clear_escalation_caches()
        resp = client.get('/api/v2/dashboard/escalation-analytics')
        assert resp.status_code == 200
        analytics = resp.json()['ESCALATION_ANALYTICS']

        # Top-level regime_markers[] (δ timeline overlay).
        regime_markers = analytics['regime_markers']
        assert isinstance(regime_markers, list) and regime_markers, (
            'committed regime-markers seed should surface at least one marker'
        )
        for marker in regime_markers:
            assert {'date', 'label', 'tasks'} <= set(marker)

        entry = analytics['per_project'][0]

        # The project's terminal population (the action-mix donut's caption)
        # and its corpus views (the strip's open-in-history tile).
        assert isinstance(entry['terminal'], int)
        assert {'queue_pending', 'open_in_history'} <= set(entry['views'])

        # origin — δ donut/sparkline + ε StatTile strip. Every class is a key of
        # `classes`: the origin bar draws one segment per served key.
        origin = entry['origin']
        assert isinstance(origin['daily_by_source'], dict)
        assert origin['sources']
        for source in origin['sources']:
            assert {
                'source', 'filings', 'classes', 'classified', 'stamped_share',
                'benign_rate', 'predictably_benign', 'daily_spark',
            } <= set(source)
            assert set(RESOLUTION_CLASSES) <= set(source['classes'])

        # lifespan — δ percentiles + open-items table.
        lifespan = entry['lifespan']
        assert {
            'percentiles_by_level', 'samples', 'open_items', 'l1_to_l2_promotion',
        } <= set(lifespan)
        assert lifespan['open_items'], 'row-4 pending record guarantees an open item'
        for item in lifespan['open_items']:
            assert {'id', 'task_id', 'level', 'age_secs', 'breach_6h'} <= set(item)

        # workflow — δ throughput + ζ mini-Sankey flow cube.
        workflow = entry['workflow']
        assert {
            'tier_weekly', 'action_mix', 'churn_daily', 'esc_per_done_daily', 'flow_daily',
            'done_counts_read',
        } <= set(workflow)
        assert workflow['esc_per_done_daily']
        for row in workflow['esc_per_done_daily']:
            assert {'date', 'filings', 'done', 'ratio'} <= set(row)
        assert workflow['flow_daily']
        for row in workflow['flow_daily']:
            assert {'date', 'source', 'level', 'tier', 'class', 'n'} <= set(row)


# ---------------------------------------------------------------------------
# PRD leaf η (task 5596) — the corpus' views and the complete class split,
# end-to-end over a tree the escalation server wrote.
# ---------------------------------------------------------------------------


def _get_both_routes(client) -> tuple[dict, dict]:
    with patch('dashboard.data.tasks.mcp_tool_call', new=_no_tasks()):
        escalations = client.get('/api/v2/dashboard/escalations')
        analytics = client.get('/api/v2/dashboard/escalation-analytics')
    assert escalations.status_code == analytics.status_code == 200
    return escalations.json(), analytics.json()


class TestSketch10OverLiveQueue:
    """PRD sketch #10: 2 pending in the live queue and 3 more in the archive read 2 and 5.

    Filed through the REAL EscalationQueue, then three of the pending files
    moved under its archive the way a stranded record sits there. The
    /escalations pill counts the live queue, the analytics strip counts every
    open record — two named views of ONE walk, so they share its instant.
    """

    def test_queue_pending_and_open_in_history_come_from_one_walk(self, client, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        for n in range(1, 6):
            _submit_pending(
                queue, f'esc-sk{n}-1', f'sk{n}', SRC_SKETCH, level=1,
                filed_at=_BASE + timedelta(hours=n),
            )
        for n in (3, 4, 5):
            archived = archive_dir_for_date(queue.queue_dir, _iso(_BASE))
            archived.mkdir(parents=True, exist_ok=True)
            os.replace(queue.queue_dir / f'esc-sk{n}-1.json', archived / f'esc-sk{n}-1.json')

        config = _make_config(tmp_path)
        config.reconciliation_escalations_dir.mkdir(parents=True, exist_ok=True)
        client.app.state.config = config
        _clear_escalation_caches()
        try:
            escalations, analytics = _get_both_routes(client)
        finally:
            _clear_escalation_caches()

        queue_pending = escalations['ESCALATIONS']['views']['queue_pending']
        (project,) = analytics['ESCALATION_ANALYTICS']['per_project']
        open_in_history = project['views']['open_in_history']
        assert (queue_pending['value'], open_in_history['value']) == (2, 5)
        assert queue_pending['state'] == open_in_history['state'] == 'fresh'
        assert queue_pending['as_of'] == open_in_history['as_of'], (
            'the live-queue count and the history count came from two different walks'
        )


class TestEveryResolutionClassOverLiveQueue:
    """Every resolution class is served, and its parts sum to the terminal population.

    A strand-swept L0 and a record ruled moot join the boundary archive, so all
    four classes are present. Each project's ``classes`` (summed over sources)
    and its ``action_mix`` both count exactly its ``terminal`` records.
    """

    def test_classes_and_action_mix_sum_to_terminal(self, client, tmp_path: Path) -> None:
        queue = _live_queue(tmp_path)
        arch = asyncio.run(_build_every_class_archive(queue, _make_server(queue)))

        client.app.state.config = _make_config(tmp_path)
        _clear_escalation_caches()
        try:
            resp = client.get('/api/v2/dashboard/escalation-analytics')
        finally:
            _clear_escalation_caches()
        assert resp.status_code == 200
        (entry,) = resp.json()['ESCALATION_ANALYTICS']['per_project']
        sources = entry['origin']['sources']

        served_classes = {cls for source in sources for cls, n in source['classes'].items() if n}
        assert served_classes == {r.cls for r in arch.terminal()} == set(RESOLUTION_CLASSES), (
            'every resolution class the archive holds must be served — none folded away'
        )
        _assert_origin_matches(sources, arch)

        terminal = entry['terminal']
        assert terminal == len(arch.terminal())
        assert sum(sum(source['classes'].values()) for source in sources) == terminal
        assert sum(entry['workflow']['action_mix'].values()) == terminal
