"""Tests for Harness._maybe_write_digest's landing-count wiring (task 8563).

The digest's window figure is the landing count, read off the event loop
(INV-8: landings() runs a blocking git rev-list); the EWA breaker still
divides by the former count, task_completed done events.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _landing_history import landing_repo
from shared.testing_landing_history import LandingRepo, record_merge_done, record_train
from test_harness_park_stop import _make_harness_with_mocks

import orchestrator.digest as digest
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness

ALPHA = 0.5
SUBMISSIONS = 2


def _ago(minutes: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()


def _seed(event_store: EventStore | None, repo: LandingRepo | None) -> None:
    assert event_store is not None
    conn = sqlite3.connect(event_store.db_path)
    try:
        if repo is not None:
            record_merge_done(conn, 'ff', repo.fast_forward('ff'), _ago(30))
            record_train(conn, 't2', ['t1', 't2'], repo.merge('task/t1'), _ago(20))
        conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
            "VALUES (?, 'run-test-0001', 'ff', 'task_completed', ?)",
            (_ago(25), json.dumps({'outcome': 'done'})),
        )
        conn.commit()
    finally:
        conn.close()


async def _harness(
    tmp_path: Path, project_root: Path, cost_store_factory, *, main_branch: str = 'main',
) -> Harness:
    harness, _, _ = _make_harness_with_mocks(tmp_path)
    harness.config = OrchestratorConfig(
        project_root=project_root,
        git=GitConfig(main_branch=main_branch),
        digest_dir=str(tmp_path / 'digests'),
        digest_every_n_escalations=1,
        digest_ewa_alpha=ALPHA,
        digest_ewa_threshold=999.0,
    )
    harness.cost_store = await cost_store_factory()
    harness._escalation_event_count = SUBMISSIONS
    harness._last_digest_event_count = 0
    harness._escalation_submit_count = SUBMISSIONS
    harness._last_digest_submit_count = 0
    return harness


def _written_digest(tmp_path: Path) -> str:
    files = sorted((tmp_path / 'digests').glob('digest-*.md'))
    assert len(files) == 1, files
    return files[0].read_text()


def _section(markdown: str, heading: str) -> str:
    return markdown.split(f'{heading}\n', 1)[1].split('\n## ', 1)[0]


class TestHarnessDigestLandings:
    @pytest.mark.asyncio
    async def test_digest_renders_landings_and_keeps_done_runs_as_the_ewa_denominator(
        self, tmp_path: Path, _cost_store_factory,
    ) -> None:
        repo = landing_repo(tmp_path / 'repo')
        harness = await _harness(tmp_path, repo.root, _cost_store_factory)
        _seed(harness.event_store, repo)
        prev = harness._ewa_value

        await harness._maybe_write_digest()

        content = _written_digest(tmp_path)
        assert _section(content, '## Landings in window').splitlines()[0] == '3'
        assert f'- {digest.DONE_RUNS_LABEL}, the EWA denominator: 1' in (
            _section(content, '## EWA').splitlines()
        )
        assert harness._ewa_value == pytest.approx(
            digest.update_ewa(prev, SUBMISSIONS, 1, ALPHA))

    @pytest.mark.asyncio
    async def test_landings_are_read_off_the_configured_main_branch(
        self, tmp_path: Path, _cost_store_factory,
    ) -> None:
        repo = landing_repo(tmp_path / 'repo', branch='trunk')
        harness = await _harness(tmp_path, repo.root, _cost_store_factory, main_branch='trunk')
        _seed(harness.event_store, repo)

        await harness._maybe_write_digest()

        assert _section(_written_digest(tmp_path), '## Landings in window').splitlines()[0] == '3'

    @pytest.mark.asyncio
    async def test_landings_are_read_off_the_event_loop_thread(
        self, tmp_path: Path, _cost_store_factory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = landing_repo(tmp_path / 'repo')
        harness = await _harness(tmp_path, repo.root, _cost_store_factory)
        _seed(harness.event_store, repo)
        real = digest.landings_in_window
        threads: list[int] = []

        def recording(*args, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        monkeypatch.setattr(digest, 'landings_in_window', recording)

        await harness._maybe_write_digest()

        assert len(threads) == 1
        assert threads[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_a_root_that_is_not_a_repository_still_writes_the_digest(
        self, tmp_path: Path, _cost_store_factory,
    ) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        harness = await _harness(tmp_path, plain, _cost_store_factory)
        _seed(harness.event_store, None)

        await harness._maybe_write_digest()

        section = _section(_written_digest(tmp_path), '## Landings in window')
        assert section.startswith('_unavailable')

    @pytest.mark.asyncio
    async def test_a_harness_without_an_event_store_renders_landings_unavailable(
        self, tmp_path: Path, _cost_store_factory,
    ) -> None:
        repo = landing_repo(tmp_path / 'repo')
        harness = await _harness(tmp_path, repo.root, _cost_store_factory)
        harness.event_store = None

        await harness._maybe_write_digest()

        section = _section(_written_digest(tmp_path), '## Landings in window')
        assert section.startswith('_unavailable')
