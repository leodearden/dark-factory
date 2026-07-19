"""Tests for the stale priority-override / pin-queue Graphiti edge sweep (task 2781).

Reconciliation Stage 1 (MemoryConsolidator) has no deterministic sweep that
keeps VALID (invalid_at IS NULL) priority-override / pin-queue Graphiti
temporal_facts edges (boost tier, TTL, pin order) in sync with live
scheduler-override state. When a task's override is consumed (task dispatched
-> row cleared by clear_terminal) or expires (clear_expired), the task drops
out of the live override table but its "task N has boost/TTL/pin override"
edge silently persists as a stale valid_at edge until a human catches it
(this required TWO one-time manual backfills — run 2d59c7de, finding
3852bd07). This module adds a small deterministic post-processor mirroring
task 2613's stale_status_snapshot_edge_sweep: enumerate valid edges,
lexically extract each edge's single subject task_id, read LIVE override
state directly from scheduler_overrides.db (the full table, not
get_pin_queue's pinned=1 projection), and invalidate any edge whose task is
absent from the live table OR whose (TTL) edge's live ttl_until has elapsed.

Covers:
- extract_priority_override_task_id: pure lexical extractor — returns the
  single subject task_id a priority-override edge asserts, gated on the
  "priority override" phrase; None when the gate fails or the fact has no
  single distinct task-id subject (multi/no subject).
- is_ttl_override_fact: pure classifier — True iff BOTH the "priority
  override" phrase AND a "TTL" token are present (either order,
  separator-tolerant).
- select_stale_priority_override_edges: pure decision core — selects edges
  whose task is absent from the live override map, or whose TTL edge's live
  ttl_until has elapsed.
- read_live_override_state: async reader over scheduler_overrides.db (full
  table read).
- sweep_stale_priority_override_edges: async orchestrator — enumerates,
  reads live state, and invalidates, best-effort throughout.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.stale_priority_override_edge_sweep import (
    extract_priority_override_task_id,
    is_ttl_override_fact,
    read_live_override_state,
    select_stale_priority_override_edges,
    sweep_stale_priority_override_edges,
)


def _make_memory_service() -> MagicMock:
    """MagicMock memory_service with an AsyncMock .graphiti.get_all_valid_edges
    and .update_edge (mirrors test_stale_status_snapshot_edge_sweep.py's
    _make_memory_service)."""
    memory_service = MagicMock()
    memory_service.graphiti = MagicMock()
    memory_service.graphiti.get_all_valid_edges = AsyncMock(return_value={})
    memory_service.update_edge = AsyncMock()
    return memory_service

# Full scheduler_overrides.db schema (mirrors server/tools.py::_OVERRIDE_SCHEMA)
# so the real-DB reader test exercises read_live_override_state against a
# faithful table, not a trimmed stand-in.
_OVERRIDE_SCHEMA_DDL = """
CREATE TABLE overrides (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    boost_tier    TEXT,
    pinned        INTEGER NOT NULL DEFAULT 0,
    pin_order     INTEGER,
    reserve_now   INTEGER NOT NULL DEFAULT 0,
    ttl_until     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);
"""


def _seed_overrides_db(project_root: Path, rows: list[dict]) -> None:
    """Create data/orchestrator/scheduler_overrides.db under *project_root*
    and INSERT *rows* via raw sqlite (each dict supplies the override
    columns; project_root/created_at/updated_at are filled in)."""
    db_path = project_root / 'data' / 'orchestrator' / 'scheduler_overrides.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_OVERRIDE_SCHEMA_DDL)
        for r in rows:
            conn.execute(
                'INSERT INTO overrides (project_root, task_id, boost_tier, pinned, '
                'pin_order, reserve_now, ttl_until, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    str(project_root),
                    r['task_id'],
                    r.get('boost_tier'),
                    r.get('pinned', 0),
                    r.get('pin_order'),
                    r.get('reserve_now', 0),
                    r.get('ttl_until'),
                    '2026-07-19T00:00:00+00:00',
                    '2026-07-19T00:00:00+00:00',
                ),
            )
        conn.commit()
    finally:
        conn.close()

_NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
_PAST = _NOW - timedelta(hours=1)
_FUTURE = _NOW + timedelta(hours=1)


def _boost_edge(uuid: str = 'edge-boost-5166') -> dict:
    return {
        'uuid': uuid,
        'fact': "Set priority override for task 5166: {'boost_tier': 'high'}",
        'name': '',
    }


def _ttl_edge(uuid: str = 'edge-ttl-4940') -> dict:
    return {
        'uuid': uuid,
        'fact': 'Task 4940 priority override with a TTL of 3600s',
        'name': '',
    }


class TestExtractPriorityOverrideTaskId:
    """extract_priority_override_task_id(fact) returns the single subject
    task_id a priority-override edge asserts, or None.

    Gated on the "priority override" phrase: a fact without it is never a
    priority-override edge this sweep concerns itself with. Returns None when
    the fact has no single distinct extractable subject (a multi-subject
    event record or a count-only/no-subject fact), mirroring 2613's
    count-only exclusion.
    """

    def test_set_override_audit_form(self):
        """"Set priority override for task 5166: {...}" -> 5166."""
        assert (
            extract_priority_override_task_id(
                "Set priority override for task 5166: {'boost_tier': 'high'}"
            )
            == 5166
        )

    def test_pin_order_form(self):
        """"Task 4079 priority override pin order 3" -> 4079 (the '3' is not
        a task reference)."""
        assert (
            extract_priority_override_task_id('Task 4079 priority override pin order 3')
            == 4079
        )

    def test_ttl_form(self):
        """"Task 4940 priority override with a TTL of 3600s" -> 4940 (the
        '3600' is not a task reference)."""
        assert (
            extract_priority_override_task_id(
                'Task 4940 priority override with a TTL of 3600s'
            )
            == 4940
        )

    def test_non_override_fact_gated_out(self):
        """"Task 5 is done" -> None: no "priority override" phrase, so the
        gate fails before any id is extracted."""
        assert extract_priority_override_task_id('Task 5 is done') is None

    def test_count_only_no_subject_returns_none(self):
        """"There are 8 tasks in progress" -> None (no gate phrase, no
        single subject)."""
        assert extract_priority_override_task_id('There are 8 tasks in progress') is None

    def test_multi_subject_event_record_returns_none(self):
        """"Reordered pin queue: [1, 2, 3]" -> None: a multi-subject event
        record with no single "priority override" subject."""
        assert extract_priority_override_task_id('Reordered pin queue: [1, 2, 3]') is None

    def test_multiple_distinct_subjects_with_phrase_returns_none(self):
        """Even WITH the gate phrase, two distinct task ids -> None: the
        extractor returns a subject only when exactly one distinct id is
        present."""
        assert (
            extract_priority_override_task_id(
                'priority override reordered for task 1, task 2 and task 3'
            )
            is None
        )

    def test_incidental_date_digits_not_swept_in(self):
        """"Task 5166 priority override boost set 2026-07-19" -> 5166: the
        date's digits are not task references and must not defeat the
        single-subject rule."""
        assert (
            extract_priority_override_task_id(
                'Task 5166 priority override boost set 2026-07-19'
            )
            == 5166
        )


# --------------------------------------------------------------------------- #
# is_ttl_override_fact
# --------------------------------------------------------------------------- #


class TestIsTtlOverrideFact:
    """is_ttl_override_fact(fact) is True iff BOTH the "priority override"
    phrase AND a "TTL" token are present (either order, separator-tolerant),
    mirroring memory_service._PRIORITY_OVERRIDE_TTL_FACT_RE. Requiring both
    tokens keeps this classifier scoped to the genuinely single-valued TTL
    scalar (never a boost/pin edge, never a bare-TTL non-override fact).
    """

    def test_ttl_form_forward_order(self):
        """"Task 4940 priority override with a TTL of 3600s" -> True."""
        assert is_ttl_override_fact('Task 4940 priority override with a TTL of 3600s') is True

    def test_ttl_reversed_order_with_double_space_and_newline_separators(self):
        """TTL appearing BEFORE the phrase, with a double-space separator
        inside the phrase and a newline between the tokens -> True (either
        order, separator-tolerant across newlines via re.S)."""
        assert (
            is_ttl_override_fact('TTL expiry recorded;\npriority  override still active')
            is True
        )

    def test_boost_fact_without_ttl_is_false(self):
        """"priority override boost tier high" -> False (no TTL token)."""
        assert is_ttl_override_fact('priority override boost tier high') is False

    def test_pin_fact_without_ttl_is_false(self):
        """"priority override pin order 3" -> False (no TTL token)."""
        assert is_ttl_override_fact('priority override pin order 3') is False

    def test_non_override_fact_is_false(self):
        """"Task 5 is done" -> False (neither token)."""
        assert is_ttl_override_fact('Task 5 is done') is False

    def test_ttl_token_without_phrase_is_false(self):
        """A "TTL" token WITHOUT the "priority override" phrase -> False:
        both tokens are required, so an unrelated TTL mention never
        classifies as a priority-override TTL edge."""
        assert is_ttl_override_fact('Task 7 cache entry has a TTL of 60s') is False


# --------------------------------------------------------------------------- #
# select_stale_priority_override_edges — pure decision core
# --------------------------------------------------------------------------- #


class TestSelectStalePriorityOverrideEdges:
    """select_stale_priority_override_edges(edges, live_overrides, *, now) is
    the pure decision core. An edge is selected (stale) iff its extracted
    subject task is ABSENT from the live override map, OR the edge is a TTL
    edge whose live ttl_until has elapsed (now >= ttl_until). A task present
    with any live override (and a non-elapsed / null ttl) is never selected —
    positively-determinable-only, conservative under-invalidation.
    """

    def test_boost_edge_absent_from_live_selected(self):
        """Boost edge for 5166, absent from a live map holding other tasks ->
        selected (the override was consumed/cleared)."""
        edge = _boost_edge()
        result = select_stale_priority_override_edges(
            [edge], {'999': {'ttl_until': None}}, now=_NOW,
        )
        assert result == [edge]

    def test_boost_edge_present_in_live_not_selected(self):
        """Boost edge for 5166 present in the live map -> NOT selected."""
        edge = _boost_edge()
        result = select_stale_priority_override_edges(
            [edge], {'5166': {'ttl_until': None}}, now=_NOW,
        )
        assert result == []

    def test_ttl_edge_elapsed_selected(self):
        """TTL edge for 4940 present with ttl_until in the past -> selected."""
        edge = _ttl_edge()
        result = select_stale_priority_override_edges(
            [edge], {'4940': {'ttl_until': _PAST}}, now=_NOW,
        )
        assert result == [edge]

    def test_ttl_edge_future_not_selected(self):
        """TTL edge present with ttl_until in the future -> NOT selected."""
        edge = _ttl_edge()
        result = select_stale_priority_override_edges(
            [edge], {'4940': {'ttl_until': _FUTURE}}, now=_NOW,
        )
        assert result == []

    def test_ttl_edge_ttl_until_none_not_selected(self):
        """TTL edge present but the live row's ttl_until is None -> NOT
        selected (no absolute expiry to compare against)."""
        edge = _ttl_edge()
        result = select_stale_priority_override_edges(
            [edge], {'4940': {'ttl_until': None}}, now=_NOW,
        )
        assert result == []

    def test_non_override_edge_never_selected(self):
        """A non-override edge (extractor returns None) is never selected,
        regardless of the live map contents."""
        edge = {'uuid': 'edge-nonoverride', 'fact': 'Task 5 is done', 'name': ''}
        result = select_stale_priority_override_edges(
            [edge], {'5': {'ttl_until': _PAST}}, now=_NOW,
        )
        assert result == []

    def test_empty_live_map_candidate_selected(self):
        """Empty live map + one candidate edge -> selected (a legitimate
        no-overrides-at-all state — every override has been consumed)."""
        edge = _boost_edge()
        result = select_stale_priority_override_edges([edge], {}, now=_NOW)
        assert result == [edge]

    def test_selected_entries_carry_edge_uuid(self):
        """Selected entries are the original edge dicts, carrying their uuid."""
        edge = _boost_edge(uuid='edge-carries-uuid')
        result = select_stale_priority_override_edges([edge], {}, now=_NOW)
        assert len(result) == 1
        assert result[0]['uuid'] == 'edge-carries-uuid'


# --------------------------------------------------------------------------- #
# read_live_override_state — real DB reader
# --------------------------------------------------------------------------- #


class TestReadLiveOverrideState:
    """read_live_override_state(project_root) reads the FULL overrides table
    (not get_pin_queue's pinned=1 projection) from scheduler_overrides.db,
    returning {str(task_id): {..., 'ttl_until': datetime | None}}.
    """

    @pytest.mark.asyncio
    async def test_full_table_read_includes_non_pinned_and_parses_ttl(self, tmp_path):
        """A pinned row (with ttl_until ISO), a non-pinned boost-only row
        (ttl_until NULL), and a pinned row with no ttl are ALL returned
        (full-table read), keyed by str(task_id); ttl_until is parsed to a
        tz-aware datetime, or None when NULL."""
        _seed_overrides_db(
            tmp_path,
            [
                {
                    'task_id': '5166',
                    'boost_tier': 'high',
                    'pinned': 1,
                    'pin_order': 1,
                    'ttl_until': '2026-07-19T13:00:00+00:00',
                },
                {
                    'task_id': '4079',
                    'boost_tier': 'medium',
                    'pinned': 0,
                    'pin_order': None,
                    'ttl_until': None,
                },
                {
                    'task_id': '4940',
                    'boost_tier': None,
                    'pinned': 1,
                    'pin_order': 2,
                    'ttl_until': None,
                },
            ],
        )

        result = await read_live_override_state(str(tmp_path))

        # Full-table read: the non-pinned row (4079) must be present — a
        # pinned=1-only projection would wrongly omit it.
        assert set(result.keys()) == {'5166', '4079', '4940'}, (
            f'Expected all three rows keyed by str(task_id) (full-table read, '
            f'incl. the non-pinned 4079), got {result!r}'
        )

        ttl = result['5166']['ttl_until']
        assert isinstance(ttl, datetime)
        assert ttl.tzinfo is not None, 'ttl_until must be parsed tz-aware'
        assert ttl == datetime(2026, 7, 19, 13, 0, tzinfo=UTC)

        assert result['4079']['ttl_until'] is None
        assert result['4940']['ttl_until'] is None

    @pytest.mark.asyncio
    async def test_absent_db_file_returns_empty_map(self, tmp_path):
        """No scheduler_overrides.db under project_root -> {} (no overrides
        recorded yet); no crash."""
        result = await read_live_override_state(str(tmp_path))
        assert result == {}


# --------------------------------------------------------------------------- #
# sweep_stale_priority_override_edges — core behavior
# --------------------------------------------------------------------------- #


class TestSweepStalePriorityOverrideEdgesCore:
    """sweep_stale_priority_override_edges enumerates valid edges via
    memory_service.graphiti.get_all_valid_edges, reads live override state via
    the injected read_live, and invalidates only the edges
    select_stale_priority_override_edges identifies as stale.
    """

    @pytest.mark.asyncio
    async def test_happy_path_invalidates_only_the_stale_edge(self):
        """One stale edge (boost for task 5166, absent from the live map) +
        one healthy edge (boost for task 999, present in the live map).
        update_edge must be awaited exactly once, for the stale edge's uuid
        only, with a datetime invalid_at, project_id threaded, the stable
        agent_id, and run_id as causation_id."""
        memory_service = _make_memory_service()

        stale_edge = {
            'uuid': 'edge-stale',
            'fact': "Set priority override for task 5166: {'boost_tier': 'high'}",
            'name': '',
        }
        healthy_edge = {
            'uuid': 'edge-healthy',
            'fact': "Set priority override for task 999: {'boost_tier': 'high'}",
            'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge, healthy_edge]},
        )
        read_live = AsyncMock(return_value={'999': {'ttl_until': None}})

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '/tmp/reify',
            run_id='run-1', read_live=read_live,
        )

        memory_service.update_edge.assert_awaited_once()
        update_call = memory_service.update_edge.await_args
        assert update_call.args[0] == 'edge-stale', (
            f'Expected update_edge awaited for the stale edge uuid only, got {update_call!r}'
        )
        assert isinstance(update_call.kwargs.get('invalid_at'), datetime), (
            'Expected update_edge called with a datetime invalid_at'
        )
        assert update_call.kwargs.get('project_id') == 'test_project'
        assert update_call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator', (
            "Expected update_edge stamped with the sweep's stable agent_id, "
            f'got {update_call!r}'
        )
        assert update_call.kwargs.get('causation_id') == 'run-1', (
            'Expected update_edge threaded the reconciliation run_id as causation_id, '
            f'got {update_call!r}'
        )

        read_live.assert_awaited_once_with('/tmp/reify')

        assert stats == {'scanned': 2, 'candidate_edges': 1, 'invalidated': 1, 'errors': 0}, (
            f'Expected exactly the stale edge counted+invalidated, got stats={stats!r}'
        )


def _boost_fact_edge(uuid: str, task_id: int) -> dict:
    return {
        'uuid': uuid,
        'fact': f"Set priority override for task {task_id}: {{'boost_tier': 'high'}}",
        'name': '',
    }


# --------------------------------------------------------------------------- #
# sweep_stale_priority_override_edges — guards
# --------------------------------------------------------------------------- #


class TestSweepStalePriorityOverrideEdgesGuards:
    """Guard behavior: a falsy project_root, an enumeration failure, or an
    enumeration with no candidate edges must short-circuit without
    unnecessary backend calls."""

    @pytest.mark.asyncio
    async def test_empty_project_root_yields_all_zero_stats_with_no_calls(self):
        """project_root='' -> all-zero stats; get_all_valid_edges and
        read_live never awaited."""
        memory_service = _make_memory_service()
        read_live = AsyncMock()

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '', run_id='run-1', read_live=read_live,
        )

        assert stats == {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 0}
        memory_service.graphiti.get_all_valid_edges.assert_not_awaited()
        read_live.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_all_valid_edges_failure_yields_all_zero_stats_without_raising(self):
        """get_all_valid_edges raises -> all-zero stats except errors=1, no
        raise, read_live/update_edge not awaited."""
        memory_service = _make_memory_service()
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            side_effect=RuntimeError('transient read timeout'),
        )
        read_live = AsyncMock()

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '/tmp/reify', run_id='run-1', read_live=read_live,
        )

        assert stats == {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 1}, (
            f'Expected all-zero stats with the failure tallied as an error, got {stats!r}'
        )
        read_live.assert_not_awaited()
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_candidate_edges_skips_read_live(self):
        """Enumeration returns only non-override edges (no candidate ids) ->
        read_live never awaited and invalidated == 0."""
        memory_service = _make_memory_service()
        non_candidate = {'uuid': 'edge-x', 'fact': 'Task 5 is done', 'name': ''}
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [non_candidate]},
        )
        read_live = AsyncMock()

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '/tmp/reify', run_id='run-1', read_live=read_live,
        )

        read_live.assert_not_awaited()
        assert stats['invalidated'] == 0


# --------------------------------------------------------------------------- #
# sweep_stale_priority_override_edges — best-effort resilience
# --------------------------------------------------------------------------- #


class TestSweepStalePriorityOverrideEdgesBestEffort:
    """A transient failure reading live state, or invalidating one edge, must
    not abort the sweep for the remaining edges — it is tallied into
    stats['errors']. asyncio.CancelledError is never swallowed."""

    @pytest.mark.asyncio
    async def test_read_live_failure_is_best_effort_no_op(self):
        """read_live raises -> best-effort no-op: errors tallied, no
        update_edge attempted, no raise."""
        memory_service = _make_memory_service()
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [_boost_fact_edge('edge-c', 5166)]},
        )
        read_live = AsyncMock(side_effect=RuntimeError('db locked'))

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '/tmp/reify', run_id='run-1', read_live=read_live,
        )

        assert stats == {'scanned': 1, 'candidate_edges': 0, 'invalidated': 0, 'errors': 1}, (
            f'Expected the read_live failure tallied as an error with no invalidation, got {stats!r}'
        )
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_edge_failure_is_swallowed_and_second_still_attempted(self):
        """update_edge raises for the first of two stale edges; the second is
        still attempted. Only the successful one counts as invalidated, the
        failure counts as an error."""
        memory_service = _make_memory_service()
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={
                'entity-a': [
                    _boost_fact_edge('edge-stale-1', 5166),
                    _boost_fact_edge('edge-stale-2', 4079),
                ],
            },
        )
        # Empty live map -> both absent -> both stale.
        read_live = AsyncMock(return_value={})
        memory_service.update_edge = AsyncMock(side_effect=[RuntimeError('lock contention'), None])

        stats = await sweep_stale_priority_override_edges(
            memory_service, 'test_project', '/tmp/reify', run_id='run-1', read_live=read_live,
        )

        assert memory_service.update_edge.await_count == 2, (
            'The second stale edge must still be attempted after the first update fails'
        )
        assert stats == {'scanned': 2, 'candidate_edges': 2, 'invalidated': 1, 'errors': 1}, (
            f'Expected the failed update tallied without blocking the second, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_from_update_edge_propagates(self):
        """asyncio.CancelledError raised from update_edge must propagate — it
        is never swallowed as a best-effort error."""
        memory_service = _make_memory_service()
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [_boost_fact_edge('edge-stale', 5166)]},
        )
        read_live = AsyncMock(return_value={})
        memory_service.update_edge = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await sweep_stale_priority_override_edges(
                memory_service, 'test_project', '/tmp/reify', run_id='run-1', read_live=read_live,
            )
