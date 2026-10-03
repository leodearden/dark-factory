"""Tests for cleanup_pin_queue_edges.py.

Loaded by path via ``_fm_helpers.load_script_module`` (the script is not on
PYTHONPATH), mirroring ``test_cleanup_count_snapshots.py``.

``main`` is deliberately not driven: per the fused-memory ops-script
convention it is thin live wiring, and every behaviour worth pinning sits in
``scan`` and ``run``.
"""
from __future__ import annotations

import argparse
import logging
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import load_script_module
from _store_mutation_preflight_contract import (
    SENTINEL,
    deny,
    fail_closed_records,
    neutralise_fixture,
)

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'cleanup_pin_queue_edges.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='cleanup_pin_queue_edges')


_neutralise = neutralise_fixture(
    _mod,
    note="""``run(args, ..., apply=True)`` runs the preflight before it connects
    to FalkorDB or reads ``scheduler_overrides.db`` (task 3834's guard site).
    Substrate: a ``_FakeGraph`` injected through ``run``'s ``graph`` keyword,
    and a MagicMock memory with AsyncMock mutators.
    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.""",
)


# ===========================================================================
# Helpers
# ===========================================================================

Edge = tuple[str, str, list[str], str, str]


class _FakeGraph:
    """A falkordb graph handle answering the three query shapes ``scan`` issues.

    Every call is recorded in ``queries``, so a test can assert the graph was
    (or was never) read without mixing mock-assertion styles.
    """

    def __init__(
        self,
        episodes: dict[str, str],
        edges: list[Edge],
        reported_total: int | None = None,
    ) -> None:
        self.episodes = episodes
        self.edges = edges
        self.reported_total = len(edges) if reported_total is None else reported_total
        self.queries: list[tuple[str, dict[str, Any] | None]] = []
        self._edge_pages_served = 0

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> types.SimpleNamespace:
        self.queries.append((cypher, params))
        if params and 'p' in params:
            rows: list = [
                (uuid, content)
                for uuid, content in self.episodes.items()
                if content.startswith(params['p'])
            ]
        elif 'count(' in cypher:
            rows = [[self.reported_total]]
        elif 'SKIP' in cypher:
            rows = list(self.edges) if self._edge_pages_served == 0 else []
            self._edge_pages_served += 1
        else:
            raise AssertionError(f'_FakeGraph does not recognise this query shape: {cypher!r}')
        return types.SimpleNamespace(result_set=rows)


def _audit_line(task_id: str, pin_order: int | None, boost_tier: str = 'high') -> str:
    """The deterministic audit-episode line ``_emit_override_audit`` writes."""
    payload = {'boost_tier': boost_tier, 'pinned': pin_order is not None, 'pin_order': pin_order}
    return f'Set priority override for task {task_id}: {payload}'


def _live(**pin_orders: int) -> dict[str, dict]:
    """A live-overrides map ``{task_id: row}`` keyed by ``t<id>=pin_order``."""
    return {
        task_id.removeprefix('t'): {
            'boost_tier': 'high', 'pinned': True, 'pin_order': pin_order, 'ttl_until': None,
        }
        for task_id, pin_order in pin_orders.items()
    }


def _scan(episodes: dict[str, str], edges: list[Edge], live: dict[str, dict]) -> list[dict]:
    return _mod.scan(_FakeGraph(episodes, edges), live)


def _fields(target: dict, *keys: str) -> dict:
    return {key: target[key] for key in keys}


_SELECTION_KEYS = ('edge_uuid', 'reason', 'subject', 'attribution')

_LEXICAL_FACT = 'Task 3541 is pinned with pin order of 2.'
_LEXICAL_EPISODES = {'ep-3541': _audit_line('3541', 2)}
_LEXICAL_EDGE: Edge = ('edge-lex', _LEXICAL_FACT, ['ep-3541'], 'ent-src', 'ent-tgt')

_EPISODE_FACT = 'Pin order is set to 10.'
_EPISODE_EPISODES = {'ep-4000': _audit_line('4000', 7)}
_EPISODE_EDGE: Edge = ('edge-ep', _EPISODE_FACT, ['ep-4000'], 'ent-a', 'ent-b')


# ===========================================================================
# Tests: scan
# ===========================================================================

class TestScan:
    """Characterisation of ``scan(graph, live)``'s target selection.

    Asserts on the structured target fields only, never on prose.
    """

    def test_a_lexically_attributed_edge_with_no_live_row_is_stale_absent(self):
        targets = _scan(_LEXICAL_EPISODES, [_LEXICAL_EDGE], live={})

        assert [_fields(t, *_SELECTION_KEYS) for t in targets] == [{
            'edge_uuid': 'edge-lex', 'reason': 'stale-absent',
            'subject': '3541', 'attribution': 'lexical',
        }]

    def test_a_lexically_attributed_edge_disagreeing_with_its_live_row_is_value_drift(self):
        targets = _scan(_LEXICAL_EPISODES, [_LEXICAL_EDGE], live=_live(t3541=5))

        assert [_fields(t, *_SELECTION_KEYS, 'asserted', 'live') for t in targets] == [{
            'edge_uuid': 'edge-lex', 'reason': 'value-drift',
            'subject': '3541', 'attribution': 'lexical',
            'asserted': 2, 'live': 5,
        }]

    def test_an_edge_matching_its_live_row_is_never_selected(self):
        assert _scan(_LEXICAL_EPISODES, [_LEXICAL_EDGE], live=_live(t3541=2)) == []

    def test_a_fact_naming_no_task_is_attributed_through_its_episode(self):
        targets = _scan(_EPISODE_EPISODES, [_EPISODE_EDGE], live={})

        assert [_fields(t, *_SELECTION_KEYS) for t in targets] == [{
            'edge_uuid': 'edge-ep', 'reason': 'stale-absent',
            'subject': '4000', 'attribution': 'episode',
        }]

    def test_an_episode_attributed_edge_takes_its_asserted_value_from_the_episode(self):
        targets = _scan(_EPISODE_EPISODES, [_EPISODE_EDGE], live=_live(t4000=10))

        assert [_fields(t, *_SELECTION_KEYS, 'asserted', 'live') for t in targets] == [{
            'edge_uuid': 'edge-ep', 'reason': 'value-drift',
            'subject': '4000', 'attribution': 'episode',
            'asserted': 7, 'live': 10,
        }]

    def test_a_terminal_event_assertion_is_never_selected(self):
        episodes = {'ep-clear': 'Cleared all priority override(s) for task 4880'}
        edge: Edge = (
            'edge-cleared', 'All priority overrides for task 4880 have been cleared.',
            ['ep-clear'], 'ent-a', 'ent-b',
        )

        assert _scan(episodes, [edge], live={}) == []

    def test_a_pairwise_pin_queue_ordering_fact_is_reorder_noise(self):
        edge: Edge = (
            'edge-reorder', 'Task 101 is reordered with task 102 in the pin queue.',
            ['ep-reaped'], 'ent-101', 'ent-102',
        )

        targets = _scan({}, [edge], live={})

        assert [_fields(t, 'edge_uuid', 'reason', 'subject') for t in targets] == [{
            'edge_uuid': 'edge-reorder', 'reason': 'reorder-noise', 'subject': None,
        }]

    def test_ordinary_prose_about_pinning_is_not_reorder_noise(self):
        edge: Edge = (
            'edge-prose', 'The fix task must be pinned before the blocked task can be resolved.',
            ['ep-unrelated'], 'ent-a', 'ent-b',
        )

        assert _scan({}, [edge], live={}) == []

    @pytest.mark.parametrize('live_task', ['4880', '5166'])
    def test_a_multi_subject_edge_is_skipped_while_any_subject_is_live(self, live_task):
        episodes, edge = self._multi_subject_edge()

        assert _scan(episodes, [edge], live=_live(**{f't{live_task}': 1})) == []

    def test_a_multi_subject_edge_is_selected_once_every_subject_is_gone(self):
        episodes, edge = self._multi_subject_edge()

        targets = _scan(episodes, [edge], live={})

        assert [_fields(t, *_SELECTION_KEYS) for t in targets] == [{
            'edge_uuid': 'edge-multi', 'reason': 'stale-absent',
            'subject': '4880+5166', 'attribution': 'episode-multi',
        }]

    def test_a_short_edge_enumeration_refuses_to_compute_targets(self):
        graph = _FakeGraph(_LEXICAL_EPISODES, [_LEXICAL_EDGE], reported_total=2)

        with pytest.raises(RuntimeError, match='incomplete'):
            _mod.scan(graph, {})

    @staticmethod
    def _multi_subject_edge() -> tuple[dict[str, str], Edge]:
        episodes = {
            'ep-4880': _audit_line('4880', None),
            'ep-5166': _audit_line('5166', None),
        }
        edge: Edge = (
            'edge-multi', 'The boost tier is set to high.',
            ['ep-4880', 'ep-5166'], 'ent-a', 'ent-b',
        )
        return episodes, edge


# ===========================================================================
# Tests: run() store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Mirrors ``test_cleanup_count_snapshots.TestRunApplyStoreMutationPreflight``.
    ``apply_cleanup`` invalidates each edge BEFORE writing its rollback-audit
    memory, and each mutation sits behind its own best-effort
    ``except Exception``. ``StoreMutationUnavailable`` subclasses
    ``RuntimeError``, so only a run-wide probe ahead of the first read keeps a
    refusal from stranding invalidated edges with no record of what they said.
    """

    _ENTITY_UUIDS = ('ent-src', 'ent-tgt')

    @staticmethod
    def _memory() -> MagicMock:
        memory = MagicMock()
        memory.update_edge = AsyncMock(return_value=None)
        memory.add_memory = AsyncMock(return_value=None)
        memory.refresh_entity_summary = AsyncMock(return_value=None)
        return memory

    @staticmethod
    def _graph() -> _FakeGraph:
        """One lexically-attributed stale edge spanning two entities."""
        return _FakeGraph(_LEXICAL_EPISODES, [_LEXICAL_EDGE])

    @staticmethod
    def _args(tmp_path: Path, *, apply: bool) -> argparse.Namespace:
        """``tmp_path`` holds no scheduler_overrides.db, so nothing is live."""
        return argparse.Namespace(
            apply=apply,
            project_id='dark_factory',
            project_root=str(tmp_path),
            falkor_host='localhost',
            falkor_port=6379,
        )

    @staticmethod
    def _record_backend_reads(monkeypatch, graph: _FakeGraph) -> list[str]:
        """Route ``run``'s two backend opens through recorders.

        ``graph`` is served by the recorded ``connect_graph``, so a caller that
        omits ``run``'s ``graph`` keyword observes the connect, the
        scheduler_overrides.db read, and every graph query.
        """
        reads: list[str] = []

        def _connect(_args: argparse.Namespace) -> _FakeGraph:
            reads.append('connect_graph')
            return graph

        def _read_live(_project_root: str) -> dict[str, dict]:
            reads.append('read_live_overrides')
            return {}

        monkeypatch.setattr(_mod, 'connect_graph', _connect)
        monkeypatch.setattr(_mod, 'read_live_overrides', _read_live)
        return reads

    @staticmethod
    def _assert_no_mutation(memory: MagicMock) -> None:
        memory.update_edge.assert_not_awaited()
        memory.add_memory.assert_not_awaited()
        memory.refresh_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_performs_zero_mutations_when_the_store_is_unwritable(
        self, monkeypatch, tmp_path
    ):
        """All three mutation entry points are asserted un-awaited: each sits
        behind its own swallowing ``except Exception``, so a zero-mutation claim
        covering only one of them would be vacuous."""
        deny(_mod, monkeypatch)
        memory = self._memory()

        with pytest.raises(_mod.StoreMutationUnavailable, match=SENTINEL):
            await _mod.run(self._args(tmp_path, apply=True), memory=memory, graph=self._graph())

        self._assert_no_mutation(memory)

    @pytest.mark.asyncio
    async def test_the_guard_sits_before_every_backend_read(self, monkeypatch, tmp_path):
        deny(_mod, monkeypatch)
        graph = self._graph()
        reads = self._record_backend_reads(monkeypatch, graph)

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(self._args(tmp_path, apply=True), memory=self._memory())

        assert reads == []
        assert graph.queries == []

    @pytest.mark.asyncio
    async def test_the_refusal_logs_its_fail_closed_diagnosis(self, monkeypatch, tmp_path, caplog):
        """``main`` has no handler, so a refusal exits as an uncaught traceback
        and this ERROR record is the operator's only statement of what was
        refused and where to route the mutation instead."""
        deny(_mod, monkeypatch)

        with (
            caplog.at_level(logging.ERROR, logger='cleanup_pin_queue_edges'),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            await _mod.run(
                self._args(tmp_path, apply=True), memory=self._memory(), graph=self._graph()
            )

        assert fail_closed_records(caplog, 'cleanup_pin_queue_edges')

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch, tmp_path):
        """Also the non-vacuity check for the guard-ordering test: the same
        recorders observe every backend read once the guard is not in the way."""
        deny(_mod, monkeypatch)
        memory = self._memory()
        graph = self._graph()
        reads = self._record_backend_reads(monkeypatch, graph)

        report = await _mod.run(self._args(tmp_path, apply=False), memory=memory)

        assert report['dry_run'] is True
        assert report['target_count'] == 1
        assert reads == ['connect_graph', 'read_live_overrides']
        assert graph.queries
        self._assert_no_mutation(memory)

    @pytest.mark.asyncio
    async def test_apply_is_unchanged_when_the_preflight_passes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)
        memory = self._memory()

        report = await _mod.run(
            self._args(tmp_path, apply=True), memory=memory, graph=self._graph()
        )

        assert report['dry_run'] is False
        assert report['applied_count'] == 1
        memory.update_edge.assert_awaited_once()
        assert memory.update_edge.await_args.kwargs['edge_uuid'] == 'edge-lex'
        memory.add_memory.assert_awaited_once()
        refreshed = {
            call.kwargs['entity_uuid'] for call in memory.refresh_entity_summary.await_args_list
        }
        assert memory.refresh_entity_summary.await_count == len(self._ENTITY_UUIDS)
        assert refreshed == set(self._ENTITY_UUIDS)

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_being_gated(self, monkeypatch, tmp_path):
        """Probed once for the RUN, not once per edge, and attributable in a log."""
        calls: list[dict] = []
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw))

        await _mod.run(
            self._args(tmp_path, apply=True), memory=self._memory(), graph=self._graph()
        )

        assert len(calls) == 1, 'probed ONCE per run, not once per edge'
        assert 'cleanup_pin_queue_edges' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']
