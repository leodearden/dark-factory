"""Tests for the periodic WAL-checkpoint loop in ``server/main.py``.

The loop's job is to call ``PRAGMA wal_checkpoint(TRUNCATE)`` on every live
SQLite store on a fixed cadence so the on-disk main DB advances independent
of reader-snapshot pinning — the 2026-05-13 failure mode.

We test ``_run_checkpoint_cycle`` (the single-pass extraction) directly:

  * each registered target's ``checkpoint`` callable is awaited once
  * tuple results are recorded as ``{busy, log, checkpointed}``
  * dict results (SqliteTaskBackend.checkpoint_all multi-project shape) are
    aggregated and recorded
  * a busy>0 row emits a WARN log
  * a target that raises is caught, recorded as busy=-1, and does not abort
    the cycle for subsequent targets
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from shared.async_sqlite_base import CheckpointResult

from fused_memory.server import main as server_main
from fused_memory.server.tools import (
    _checkpoint_overrides_db_if_exists,
    _open_overrides_db,
)


@pytest.fixture(autouse=True)
def _clear_checkpoint_status():
    server_main._CHECKPOINT_STATUS.clear()
    yield
    server_main._CHECKPOINT_STATUS.clear()


@pytest.mark.asyncio
async def test_cycle_records_tuple_results():
    """Tuple ``(busy, log, checkpointed)`` returns land in _CHECKPOINT_STATUS."""
    async def ok_checkpoint():
        return (0, 5, 5)

    await server_main._run_checkpoint_cycle([('store_a', ok_checkpoint)])

    status = server_main._CHECKPOINT_STATUS['store_a']
    assert status['busy'] == 0
    assert status['log'] == 5
    assert status['checkpointed'] == 5
    assert status['detail'] is None
    assert isinstance(status['ts'], str)


@pytest.mark.asyncio
async def test_cycle_aggregates_dict_results():
    """SqliteTaskBackend.checkpoint_all dict shape is aggregated across projects."""
    async def task_backend_checkpoint_all():
        return {
            '/p/a': {'busy': 0, 'log': 3, 'checkpointed': 3},
            '/p/b': {'busy': 0, 'log': 7, 'checkpointed': 7},
        }

    await server_main._run_checkpoint_cycle(
        [('task_backend', task_backend_checkpoint_all)]
    )

    status = server_main._CHECKPOINT_STATUS['task_backend']
    assert status['busy'] == 0
    assert status['log'] == 10  # 3 + 7
    assert status['checkpointed'] == 10
    assert status['detail'] == '2 project(s)'


@pytest.mark.asyncio
async def test_cycle_warns_on_busy(caplog):
    """A busy>0 row should produce a WARN log so stalls are visible."""
    async def busy_checkpoint():
        return (1, 50, 0)

    with caplog.at_level(logging.WARNING):
        await server_main._run_checkpoint_cycle([('busy_store', busy_checkpoint)])

    assert any(
        'checkpoint busy' in rec.getMessage() and 'busy_store' in rec.getMessage()
        for rec in caplog.records
    )
    assert server_main._CHECKPOINT_STATUS['busy_store']['busy'] == 1


@pytest.mark.asyncio
async def test_cycle_continues_past_raising_target(caplog):
    """One failing target must not abort the cycle for the others."""
    async def raising_checkpoint():
        raise RuntimeError('boom')

    async def ok_checkpoint():
        return (0, 2, 2)

    with caplog.at_level(logging.ERROR):
        await server_main._run_checkpoint_cycle(
            [('broken', raising_checkpoint), ('ok', ok_checkpoint)],
        )

    # Broken one is recorded with busy=-1 and a detail mentioning the error.
    broken_status = server_main._CHECKPOINT_STATUS['broken']
    assert broken_status['busy'] == -1
    detail = broken_status['detail']
    assert isinstance(detail, str) and 'boom' in detail

    # The subsequent target still ran and was recorded.
    assert server_main._CHECKPOINT_STATUS['ok']['busy'] == 0


@pytest.mark.asyncio
async def test_collect_targets_filters_to_those_with_checkpoint(monkeypatch):
    """_collect_checkpoint_targets must skip objects that don't expose checkpoint()."""
    class HasCheckpoint:
        async def checkpoint(self):
            return (0, 0, 0)

    class NoCheckpoint:
        pass

    class MemSvc:
        durable_queue = HasCheckpoint()
        planned_episode_registry = NoCheckpoint()

    targets = server_main._collect_checkpoint_targets(
        taskmaster=HasCheckpoint(),  # has 'checkpoint' but not 'checkpoint_all' → skipped
        recon_journal=HasCheckpoint(),
        event_buffer=None,
        ticket_store=NoCheckpoint(),
        write_journal=HasCheckpoint(),
        memory_service=MemSvc(),
    )
    names = {name for name, _ in targets}
    # taskmaster only matches if it has checkpoint_all (the per-project shape) —
    # bare 'checkpoint' is NOT collected as 'task_backend'.
    assert 'task_backend' not in names
    assert 'recon_journal' in names
    assert 'write_journal' in names
    assert 'durable_queue' in names
    assert 'ticket_store' not in names
    assert 'planned_episode_registry' not in names


# ---------------------------------------------------------------------------
# Step 1 — existence-guarded override-DB checkpoint helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 3 — override-target collection + cycle integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_targets_includes_override_dbs(tmp_path: Path) -> None:
    """_collect_checkpoint_targets appends one overrides:{pid} target per known project."""

    class _NoAttr:
        """Fake object with no durable_queue / planned_episode_registry."""

    targets = server_main._collect_checkpoint_targets(
        taskmaster=None,
        recon_journal=None,
        event_buffer=None,
        ticket_store=None,
        write_journal=None,
        memory_service=_NoAttr(),
        known_projects={
            'proj_a': str(tmp_path / 'a'),
            'proj_b': str(tmp_path / 'b'),
        },
    )
    names = {name for name, _ in targets}
    assert {'overrides:proj_a', 'overrides:proj_b'} <= names


@pytest.mark.asyncio
async def test_collect_targets_cycle_integration(tmp_path: Path) -> None:
    """Collected override targets flow through _run_checkpoint_cycle correctly."""

    class _NoAttr:
        pass

    # Create proj_a's DB; leave proj_b's absent.
    proj_a_root = str(tmp_path / 'a')
    proj_b_root = str(tmp_path / 'b')
    db = await _open_overrides_db(proj_a_root)
    await db.close()

    targets = server_main._collect_checkpoint_targets(
        taskmaster=None,
        recon_journal=None,
        event_buffer=None,
        ticket_store=None,
        write_journal=None,
        memory_service=_NoAttr(),
        known_projects={
            'proj_a': proj_a_root,
            'proj_b': proj_b_root,
        },
    )

    # Run the cycle over both targets.
    await server_main._run_checkpoint_cycle(targets)

    # proj_a: DB exists → real checkpoint → busy==0, detail==None (tuple path).
    a_status = server_main._CHECKPOINT_STATUS['overrides:proj_a']
    assert a_status['busy'] == 0
    assert a_status['detail'] is None

    # proj_b: DB absent → CheckpointResult(0,0,0) → busy==0, detail==None.
    b_status = server_main._CHECKPOINT_STATUS['overrides:proj_b']
    assert b_status['busy'] == 0
    assert b_status['detail'] is None

    # Guard must not have created proj_b's DB.
    proj_b_db = Path(proj_b_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'
    assert not proj_b_db.exists()


def test_collect_targets_omits_overrides_when_no_known_projects() -> None:
    """With known_projects=None (default), no overrides: targets are added."""

    class _NoAttr:
        pass

    targets = server_main._collect_checkpoint_targets(
        taskmaster=None,
        recon_journal=None,
        event_buffer=None,
        ticket_store=None,
        write_journal=None,
        memory_service=_NoAttr(),
    )
    names = {name for name, _ in targets}
    assert not any(n.startswith('overrides:') for n in names)


# ---------------------------------------------------------------------------
# Step 1 — existence-guarded override-DB checkpoint helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_overrides_db_if_exists_noop_when_absent(
    tmp_path: Path,
) -> None:
    """Returns CheckpointResult(0,0,0) and does NOT create the DB when absent."""
    project_root = str(tmp_path / 'proj')

    result = await _checkpoint_overrides_db_if_exists(project_root)

    assert result == CheckpointResult(0, 0, 0)
    db_file = Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'
    assert not db_file.exists(), 'guard must not create the DB file'


@pytest.mark.asyncio
async def test_checkpoint_overrides_db_if_exists_delegates_when_present(
    tmp_path: Path,
) -> None:
    """Delegates to _checkpoint_overrides_db when the DB file exists."""
    project_root = str(tmp_path / 'proj')

    # Create the DB via the canonical opener.
    db = await _open_overrides_db(project_root)
    await db.close()

    result = await _checkpoint_overrides_db_if_exists(project_root)

    assert isinstance(result, CheckpointResult)
    assert result.busy == 0
    db_file = Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'
    assert db_file.exists(), 'DB file must still exist after checkpoint'
