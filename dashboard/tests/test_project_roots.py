"""Tests for ``DashboardConfig.all_project_roots``, the one walk over every
configured project root.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx

from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus
from dashboard.data.active_tasks import project_roots_for_label
from dashboard.data.db import DbPool
from dashboard.data.escalation_corpus import QueueKind, corpus_queues
from dashboard.data.metrics import fan_out_list_tickets
from dashboard.project_dbs import _project_scoped_dbs_labeled

_LOGGER_NAME = 'dashboard.config'
_NO_URL_MARKER = 'No escalation URL discovered for project root'


class _ReversedRoots(DashboardConfig):
    """A config whose root policy differs from the stock one: a change to root resolution."""

    @property
    def all_project_roots(self) -> tuple[Path, ...]:
        return tuple(reversed(super().all_project_roots))


class _WithoutLastRoot(DashboardConfig):
    """A config whose roster leaves out the last configured root."""

    @property
    def all_project_roots(self) -> tuple[Path, ...]:
        return super().all_project_roots[:-1]


def _dirs(tmp_path, *names):
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.mkdir(parents=True)
    return paths


def _record_one_completion(root: Path) -> None:
    """Give *root* a runs.db holding one completed task of a project named after the root."""
    runs_db = root / 'data' / 'orchestrator' / 'runs.db'
    runs_db.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(runs_db)) as conn:
        conn.execute(
            'CREATE TABLE task_results (task_id TEXT, project_id TEXT, outcome TEXT, '
            'completed_at TEXT, duration_ms INTEGER, review_cycles INTEGER, '
            'verify_attempts INTEGER, steward_invocations INTEGER)'
        )
        conn.execute(
            "INSERT INTO task_results VALUES ('1', ?, 'done', ?, 0, 0, 0, 0)",
            (root.name, datetime.now(UTC).isoformat()),
        )
        conn.commit()


class TestAllProjectRoots:
    def test_primary_first_then_known_roots_in_configured_order(self, tmp_path):
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        config = DashboardConfig(project_root=a, known_project_roots=[b, c])
        assert config.all_project_roots == (a.resolve(), b.resolve(), c.resolve())

    def test_a_repeated_root_keeps_its_first_position(self, tmp_path):
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        config = DashboardConfig(project_root=a, known_project_roots=[b, a, c, b])
        assert config.all_project_roots == (a.resolve(), b.resolve(), c.resolve())

    def test_a_symlinked_alias_is_the_same_root(self, tmp_path):
        (real,) = _dirs(tmp_path, 'real')
        alias = tmp_path / 'alias'
        alias.symlink_to(real, target_is_directory=True)
        config = DashboardConfig(project_root=alias, known_project_roots=[real])
        assert config.all_project_roots == (real.resolve(),)

    def test_a_primary_only_config_names_just_the_primary(self, tmp_path):
        (a,) = _dirs(tmp_path, 'a')
        config = DashboardConfig(project_root=a)
        assert config.all_project_roots == (a.resolve(),)

    def test_escalation_url_discovery_probes_each_distinct_root_once(self, tmp_path, caplog):
        a, b = _dirs(tmp_path, 'a', 'b')
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            DashboardConfig(project_root=a, known_project_roots=[a, b, b])
        no_url = [
            r.getMessage()
            for r in caplog.records
            if r.name == _LOGGER_NAME and _NO_URL_MARKER in r.getMessage()
        ]
        assert len(no_url) == 2, no_url
        assert sum(str(a.resolve()) in m for m in no_url) == 1
        assert sum(str(b.resolve()) in m for m in no_url) == 1


class TestEveryConsumerWalksTheOneEnumerator:
    """Each consumer follows whatever ``all_project_roots`` answers, through its config."""

    def _reversed(self, tmp_path) -> _ReversedRoots:
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        return _ReversedRoots(project_root=a, known_project_roots=[b, c])

    async def test_project_scoped_dbs_labeled(self, tmp_path):
        config = self._reversed(tmp_path)
        pool = DbPool()
        try:
            labeled = await _project_scoped_dbs_labeled(
                config, pool, Path('data/orchestrator/runs.db'),
            )
        finally:
            await pool.close_all()
        assert [label for label, _ in labeled] == [str(r) for r in config.all_project_roots], (
            'project_dbs._project_scoped_dbs_labeled walks its own roster'
        )

    def test_performance_route(self, client, tmp_path):
        roots = _dirs(tmp_path, 'a', 'b', 'c')
        for root in roots:
            _record_one_completion(root)
        config = _WithoutLastRoot(project_root=roots[0], known_project_roots=roots[1:])
        client.app.state.config = config
        resp = client.get('/api/v2/dashboard/performance')
        assert resp.status_code == 200, resp.text
        assert sorted(resp.json()['PERFORMANCE']) == [r.name for r in config.all_project_roots], (
            'GET /api/v2/dashboard/performance walks its own roster'
        )

    def test_corpus_queues(self, tmp_path):
        config = self._reversed(tmp_path)
        queues = corpus_queues(config)
        orchestrator_ids = [q.id for q in queues if q.kind is QueueKind.ORCHESTRATOR]
        assert orchestrator_ids == [str(r) for r in config.all_project_roots], (
            'escalation_corpus.corpus_queues walks its own roster'
        )
        assert queues[-1].kind is QueueKind.RECONCILIATION

    def test_project_roots_for_label(self, tmp_path):
        primary, x_proj, y_proj = _dirs(tmp_path, 'primary', 'x/proj', 'y/proj')
        config = _ReversedRoots(project_root=primary, known_project_roots=[x_proj, y_proj])
        expected = [r for r in config.all_project_roots if r.name == 'proj']
        assert project_roots_for_label(config, 'proj') == expected, (
            'active_tasks.project_roots_for_label walks its own roster'
        )

    async def test_fan_out_list_tickets(self, tmp_path):
        a, b, c = _dirs(tmp_path, 'a', 'b', 'c')
        config = _WithoutLastRoot(
            project_root=a,
            known_project_roots=[b, c],
            fused_memory_urls=['http://fused-memory.invalid'],
        )
        queried: list[str] = []

        async def _record(http_client, url, tool, args, **kwargs):
            queried.append(args['project_root'])
            return {'count': 0, 'tickets': [], 'project_id': 'p'}

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        with patch('dashboard.data.metrics.mcp_tool_call', new=_record):
            async with httpx.AsyncClient(transport=transport) as http_client:
                await fan_out_list_tickets(http_client, config)

        assert sorted(queried) == sorted(str(r) for r in config.all_project_roots), (
            'metrics.fan_out_list_tickets walks its own roster'
        )
        assert len(queried) == len(set(queried)), f'a root was queried twice: {queried}'

    def test_invariants_route(self, client, tmp_path):
        config = self._reversed(tmp_path)
        escalation_corpus._corpus_cache_clear()
        try:
            client.app.state.config = config
            resp = client.get('/api/v2/dashboard/invariants')
        finally:
            escalation_corpus._corpus_cache_clear()
        assert resp.status_code == 200, resp.text
        roots = [record['root'] for record in resp.json()['INVARIANTS']]
        assert roots == [str(r) for r in config.all_project_roots], (
            'GET /api/v2/dashboard/invariants walks its own roster'
        )
