"""Contract tests for :class:`SqliteTaskBackend`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic import ValidationError

from fused_memory.backends.sqlite_task_backend import (
    SqliteTaskBackend,
    _classify_residual_group,
    _format_task_id,
    _merge_metadata,
    _parse_qualified_dep,
    _parse_task_id,
    _resolve_metadata_mode,
)
from fused_memory.backends.task_backend_errors import (
    DoneProvenanceWriteAuthorityError,
    StatusWriteAuthorityError,
    TaskmasterError,
    done_provenance_via_update_task_error,
    status_via_update_task_error,
)
from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.candidate_key import compute_candidate_key


@pytest_asyncio.fixture
async def backend(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


@pytest.fixture(autouse=True)
def _clear_malformed_metadata_warning_dedup():
    """Reset the module-level dedup set so each test sees a clean WARN gate."""
    from fused_memory.backends import sqlite_task_backend as _sb
    if hasattr(_sb, '_warned_malformed_task_ids'):
        _sb._warned_malformed_task_ids.clear()
    yield
    if hasattr(_sb, '_warned_malformed_task_ids'):
        _sb._warned_malformed_task_ids.clear()


# ── ID parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'raw,expected_id',
    [
        ('5', 5),
        ('  10 ', 10),
        (7, 7),
    ],
)
def test_parse_task_id_bare_only(raw, expected_id):
    """_parse_task_id returns a bare int; dotted ids raise after DF-D step-6."""
    result = _parse_task_id(raw)
    assert result == expected_id


@pytest.mark.parametrize('raw', ['', 'abc', '1.2.3', '5.x', 'x.5', '292.1', '1.1'])
def test_parse_task_id_rejects_malformed(raw):
    with pytest.raises(TaskmasterError) as exc:
        _parse_task_id(raw)
    assert exc.value.code == 'INVALID_TASK_ID'


def test_format_task_id_round_trips():
    assert _format_task_id(7) == '7'
    assert _format_task_id(2) == '2'


# ── _parse_qualified_dep ───────────────────────────────────────────


@pytest.mark.parametrize(
    'raw,expected_pid,expected_id',
    [
        ('dark_factory:13', 'dark_factory', 13),
        ('dark-factory:13', 'dark_factory', 13),   # hyphen normalized
        (' dark_factory : 13 ', 'dark_factory', 13),  # whitespace stripped
        ('DARK_FACTORY:13', 'dark_factory', 13),    # uppercase lowercased
        ('Dark-Factory:13', 'dark_factory', 13),    # mixed case + hyphen both normalized
    ],
)
def test_parse_qualified_dep_accepts_valid(raw, expected_pid, expected_id):
    pid, dep_id = _parse_qualified_dep(raw)
    assert pid == expected_pid
    assert dep_id == expected_id


@pytest.mark.parametrize(
    'raw',
    [
        ':13',               # empty project_id
        'dark_factory:',     # empty task_id
        'dark_factory:abc',  # non-numeric task_id
        'a:b:c',             # extra colon
        'dark_factory:5.1',  # dotted/subtask id
        'dark_factory:0',    # non-positive (zero)
        'dark_factory:-1',   # non-positive (negative)
    ],
)
def test_parse_qualified_dep_rejects_malformed(raw):
    with pytest.raises(TaskmasterError) as exc:
        _parse_qualified_dep(raw)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'


# ── Lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_close_idempotent(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    await b.start()  # idempotent
    assert b.connected is True
    assert b.restart_count == 1
    await b.close()
    await b.close()  # idempotent
    assert b.connected is False


@pytest.mark.asyncio
async def test_is_alive_reports_state(backend, project_root):
    alive, err = await backend.is_alive()
    assert alive is True
    assert err is None
    await backend.close()
    alive, err = await backend.is_alive()
    assert alive is False


# ── add_task / get_task / get_tasks ────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_then_get_returns_dto(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root, title='First', description='desc',
        details='details', priority='high',
    )
    assert dto['id'] == '1'
    assert 'Successfully added' in dto['message']

    one = await backend.get_task('1', project_root=project_root)
    assert one['id'] == 1  # singular get returns int per Taskmaster wire
    assert one['title'] == 'First'
    assert one['priority'] == 'high'
    assert one['status'] == 'pending'
    assert one['subtasks'] == []
    assert 'parentTaskId' not in one
    assert 'parentId' not in one

    listing = await backend.get_tasks(project_root=project_root)
    assert isinstance(listing['tasks'], list)
    assert listing['tasks'][0]['id'] == '1'  # plural get_tasks returns string
    assert all(t['subtasks'] == [] for t in listing['tasks'])


@pytest.mark.asyncio
async def test_add_task_status_param_creates_row_in_given_status(backend, project_root):
    """add_task(status='deferred') lands the row directly in deferred — one INSERT."""
    dto = await backend.add_task(
        project_root=project_root, title='Deferred task', status='deferred',
    )
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['status'] == 'deferred'


@pytest.mark.asyncio
async def test_add_task_status_defaults_to_pending(backend, project_root):
    """Omitting status preserves the historical default of 'pending'."""
    dto = await backend.add_task(project_root=project_root, title='Default task')
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['status'] == 'pending'


@pytest.mark.asyncio
async def test_add_task_increments_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    await backend.add_task(project_root=project_root, title='two')
    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_add_task_promotes_prompt_to_title(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root,
        prompt='Build a frobinator that does X\n\nDetails here',
    )
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['title'].startswith('Build a frobinator')
    assert 'Details here' in one['description']


@pytest.mark.asyncio
async def test_add_task_without_title_or_prompt_raises(backend, project_root):
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_task(project_root=project_root)
    assert 'prompt' in exc.value.message


# ── candidate_key (task 2186 — fm-task-dedup W8 task A1) ───────────


@pytest.mark.asyncio
async def test_add_task_computes_and_exposes_candidate_key(backend, project_root):
    """add_task computes candidate_key from title + metadata['files'] and
    stores it; get_task exposes it (store-level dedup key, fm-task-dedup A1)."""
    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix parser', ['a.py', 'b.py'])
    assert one['candidate_key'] is not None


@pytest.mark.asyncio
async def test_add_task_candidate_key_falls_back_to_files_to_modify(backend, project_root):
    """metadata['files_to_modify'] is accepted when 'files' is absent (Open Q #5)."""
    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files_to_modify': ['x.py']}),
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix parser', ['x.py'])


@pytest.mark.asyncio
async def test_add_task_candidate_key_computed_with_no_metadata(backend, project_root):
    """metadata=None still yields a title-only (non-None) candidate_key."""
    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=None,
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix parser', [])
    assert one['candidate_key'] is not None


@pytest.mark.asyncio
async def test_add_task_candidate_key_computed_with_empty_metadata_dict(backend, project_root):
    """metadata='{}' (no files key at all) also yields a title-only candidate_key."""
    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({}),
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix parser', [])
    assert one['candidate_key'] is not None


# ── write-boundary validation (task 2162, warn-mode census) ──────────


@pytest.mark.asyncio
async def test_add_task_warn_mode_emits_schema_warning_and_proceeds(
    backend, project_root, caplog,
):
    """Warn-mode add_task: an invariant-violating metadata write still lands.

    ``{"task_kind": "deterministic"}`` violates TaskMetadata's cross-field
    invariant (I3) — a deterministic task requires ``before_done`` or
    ``always_escalates``. The default backend is warn-mode
    (``task_metadata_enforce=False``), so exactly one
    ``task_metadata.schema_warning`` census line is emitted (carrying the new
    task's id, the offending field, and the error), and the write proceeds:
    the task is created and its metadata is stored raw/unchanged.
    """
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({'task_kind': 'deterministic'}),
        )

    census_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and 'task_metadata.schema_warning' in r.message
    ]
    assert len(census_msgs) == 1, (
        f'Expected exactly one task_metadata.schema_warning census line; got '
        f'{len(census_msgs)}: {census_msgs}'
    )
    combined = census_msgs[0]
    assert f'task_id={dto["id"]}' in combined, (
        f'Expected labeled task_id={dto["id"]!r} token in census line; got: {combined!r}'
    )
    assert '<metadata>' in combined, (
        f'Expected the whole-metadata sentinel field in census line; got: {combined!r}'
    )
    assert 'before_done' in combined, (
        f'Expected the invariant error text in census line; got: {combined!r}'
    )

    # The write proceeded: the task exists and its metadata is preserved raw
    # (original bytes — no repair, no schema_version stamp).
    task = await backend.get_task(dto['id'], project_root=project_root)
    assert task['metadata'] == {'task_kind': 'deterministic'}


@pytest.mark.asyncio
async def test_add_task_valid_metadata_emits_no_schema_warning(
    backend, project_root, caplog,
):
    """A schema-clean metadata write emits zero task_metadata.schema_warning lines."""
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({'files': ['a.py']}),
        )

    census_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and 'task_metadata.schema_warning' in r.message
    ]
    assert census_msgs == [], (
        f'Expected no task_metadata.schema_warning lines for valid metadata; got: {census_msgs}'
    )


@pytest.mark.asyncio
async def test_add_task_enforce_mode_rejects_invariant_violation(tmp_path, project_root):
    """Enforce-mode add_task: an invariant-violating write raises and rolls back.

    ``task_metadata_enforce=True`` flips parse_metadata's write-boundary
    failure policy from warn-and-proceed to raise: the malformed write is
    rejected with pydantic.ValidationError, no row is persisted, and the
    allocated id is not consumed by the rolled-back txn — a subsequent valid
    add still gets id '1'.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        with pytest.raises(ValidationError):
            await backend.add_task(
                project_root=project_root, title='t',
                metadata=json.dumps({'task_kind': 'deterministic'}),
            )

        listing = await backend.get_tasks(project_root=project_root)
        assert listing['tasks'] == [], (
            f'Expected no rows persisted after a rolled-back txn; got: {listing["tasks"]}'
        )

        dto = await backend.add_task(project_root=project_root, title='valid')
        assert dto['id'] == '1', (
            f'Expected the rolled-back id to not be consumed; got id={dto["id"]!r}'
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_add_task_enforce_mode_rejects_unparseable_json(tmp_path, project_root):
    """Enforce-mode add_task: unparseable metadata JSON raises (json.JSONDecodeError)."""
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        with pytest.raises(ValueError):
            await backend.add_task(
                project_root=project_root, title='t',
                metadata='{not valid json',
            )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_get_task_not_found_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    with pytest.raises(TaskmasterError) as exc:
        await backend.get_task('999', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found' in exc.value.message


@pytest.mark.asyncio
async def test_get_task_surfaces_claimant_columns_default_none(backend, project_root):
    """A freshly added task exposes claimant_run_id/heartbeat_at, both None."""
    await backend.add_task(project_root=project_root, title='one')
    one = await backend.get_task('1', project_root=project_root)
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_get_tasks_surfaces_claimant_columns_default_none(backend, project_root):
    """get_tasks (plural) also exposes claimant_run_id/heartbeat_at, both None."""
    await backend.add_task(project_root=project_root, title='one')
    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'][0]['claimant_run_id'] is None
    assert listing['tasks'][0]['heartbeat_at'] is None


# ── set_task_status ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_returns_per_id_payload(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    result = await backend.set_task_status(
        '1', 'done', project_root=project_root,
    )
    assert 'done' in result['message']
    assert result['tasks'] == [{
        'taskId': '1',
        'oldStatus': 'pending',
        'newStatus': 'done',
    }]


@pytest.mark.asyncio
async def test_set_task_status_unknown_id_raises(backend, project_root):
    with pytest.raises(TaskmasterError):
        await backend.set_task_status('99', 'done', project_root=project_root)


# ── set_task_claimant ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_claimant_persists_without_changing_status(backend, project_root):
    """set_task_claimant stamps both columns and leaves status untouched."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.set_task_status('1', 'in-progress', project_root=project_root)

    await backend.set_task_claimant(
        '1', project_root=project_root,
        claimant_run_id='run-abc',
        heartbeat_at='2026-07-07T00:00:00+00:00',
    )

    one = await backend.get_task('1', project_root=project_root)
    assert one['claimant_run_id'] == 'run-abc'
    assert one['heartbeat_at'] == '2026-07-07T00:00:00+00:00'
    assert one['status'] == 'in-progress'


@pytest.mark.asyncio
async def test_set_task_claimant_none_clears_both_columns(backend, project_root):
    """A follow-up set_task_claimant(..., None, None) clears both to NULL."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.set_task_claimant(
        '1', project_root=project_root,
        claimant_run_id='run-abc',
        heartbeat_at='2026-07-07T00:00:00+00:00',
    )

    await backend.set_task_claimant(
        '1', project_root=project_root,
        claimant_run_id=None,
        heartbeat_at=None,
    )

    one = await backend.get_task('1', project_root=project_root)
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_set_task_claimant_no_kwargs_is_a_noop(backend, project_root):
    """No claimant_run_id/heartbeat_at supplied -> early-return, no write, no error."""
    await backend.add_task(project_root=project_root, title='x')

    result = await backend.set_task_claimant('1', project_root=project_root)

    assert 'No claimant changes supplied' in result['message']
    one = await backend.get_task('1', project_root=project_root)
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None


# ── set_task_status claimant extension (task 2182 step-5/6) ────────


@pytest.mark.asyncio
async def test_set_task_status_claimant_kwargs_persist(backend, project_root):
    """set_task_status(..., claimant_run_id=..., heartbeat_at=...) stamps both columns."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id='run-x', heartbeat_at='2026-07-07T00:00:00+00:00',
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['status'] == 'in-progress'
    assert one['claimant_run_id'] == 'run-x'
    assert one['heartbeat_at'] == '2026-07-07T00:00:00+00:00'


@pytest.mark.asyncio
async def test_set_task_status_claimant_none_releases_on_status_change(backend, project_root):
    """Release: set_task_status(..., 'done', claimant_run_id=None, heartbeat_at=None) clears both."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id='run-x', heartbeat_at='2026-07-07T00:00:00+00:00',
    )

    await backend.set_task_status(
        '1', 'done', project_root=project_root,
        claimant_run_id=None, heartbeat_at=None,
    )

    one = await backend.get_task('1', project_root=project_root)
    assert one['status'] == 'done'
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_set_task_status_without_claimant_kwargs_leaves_claimant_intact(backend, project_root):
    """_UNSET semantics: a plain status change must not wipe a live claimant."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.set_task_status(
        '1', 'in-progress', project_root=project_root,
        claimant_run_id='run-x', heartbeat_at='2026-07-07T00:00:00+00:00',
    )

    await backend.set_task_status('1', 'review', project_root=project_root)

    one = await backend.get_task('1', project_root=project_root)
    assert one['status'] == 'review'
    assert one['claimant_run_id'] == 'run-x'
    assert one['heartbeat_at'] == '2026-07-07T00:00:00+00:00'


def _make_v2_stamped_db_without_claimant_columns(db_path: Path) -> None:
    """Create a tasks.db in the v1 shape but stamped ``user_version = 2`` (columns absent).

    Simulates a connection whose claimant columns never got ALTERed in —
    e.g. a routine orchestrator restart racing ahead of the fused-memory
    deploy that ships this migration. Opening it runs only the v2->v3
    candidate_key step (the v1->v2 claimant ALTER is gated on ``version < 2``
    and is skipped for an already-v2 DB), so the claimant columns stay
    absent, exercising set_task_status's fail-safe (WARNING, no error) path.
    """
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag           TEXT NOT NULL DEFAULT 'master',
            id            INTEGER NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            details       TEXT,
            test_strategy TEXT,
            status        TEXT NOT NULL,
            priority      TEXT,
            metadata      TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    conn.execute(
        "INSERT INTO tasks (tag, id, title, status, updated_at) "
        "VALUES ('master', 1, 'stranded-shape task', 'pending', '2026-01-01T00:00:00.000Z')",
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_set_task_status_claimant_fails_safe_when_columns_absent(tmp_path, caplog):
    """A connection whose columns never got ALTERed must not error on a claimant write."""
    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v2_stamped_db_without_claimant_columns(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
            result = await b.set_task_status(
                '1', 'in-progress', project_root=project_root,
                claimant_run_id='run-x', heartbeat_at='2026-07-07T00:00:00+00:00',
            )
        one = await b.get_task('1', project_root=project_root)
    finally:
        await b.close()

    assert 'in-progress' in result['message']
    assert one['status'] == 'in-progress'
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None

    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, 'Expected a WARNING when claimant columns are absent'


@pytest.mark.asyncio
async def test_set_task_claimant_fails_safe_when_columns_absent(tmp_path, caplog):
    """set_task_claimant on a not-yet-migrated connection must not error either."""
    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v2_stamped_db_without_claimant_columns(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
            result = await b.set_task_claimant(
                '1', project_root=project_root,
                claimant_run_id='run-x', heartbeat_at='2026-07-07T00:00:00+00:00',
            )
        one = await b.get_task('1', project_root=project_root)
    finally:
        await b.close()

    assert 'Claimant columns unavailable' in result['message']
    # status untouched, and no claimant leaked onto the wire dict either.
    assert one['status'] == 'pending'
    assert one['claimant_run_id'] is None
    assert one['heartbeat_at'] is None

    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, 'Expected a WARNING when claimant columns are absent'


# ── status vocabulary guard ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_rejects_unknown_status(backend, project_root):
    """Store floor: set_task_status must reject a status outside the shared vocabulary.

    'in_progress' (underscore) is a plausible typo of the real 'in-progress'
    (hyphen) status — exactly the silent-stranding failure this guard closes.
    The write must be blocked: re-reading the task shows the status unchanged.
    """
    await backend.add_task(project_root=project_root, title='x')
    with pytest.raises(TaskmasterError) as exc:
        await backend.set_task_status('1', 'in_progress', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'Invalid status' in exc.value.message
    task = await backend.get_task('1', project_root=project_root)
    assert task['status'] == 'pending'


@pytest.mark.asyncio
async def test_set_task_status_unknown_status_rejection_precedes_existence_check(
    backend, project_root,
):
    """Vocabulary guard runs BEFORE the task SELECT, so it beats 'No tasks found'."""
    with pytest.raises(TaskmasterError) as exc:
        await backend.set_task_status('999', 'in_progress', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'Invalid status' in exc.value.message
    assert 'No tasks found' not in exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status',
    ['done', 'in-progress', 'review', 'merge-deferred', 'infra-hold'],
)
async def test_set_task_status_accepts_valid_statuses(backend, project_root, status):
    """The guard must not over-reject — every shared-vocabulary status is accepted."""
    await backend.add_task(project_root=project_root, title='x')
    result = await backend.set_task_status('1', status, project_root=project_root)
    assert result['tasks'][0]['newStatus'] == status
    task = await backend.get_task('1', project_root=project_root)
    assert task['status'] == status


@pytest.mark.asyncio
async def test_add_task_rejects_unknown_status(backend, project_root):
    """Store floor: add_task must reject a status outside the shared vocabulary.

    No row is created and the id counter is not consumed — a subsequent
    valid add_task still gets id '1'.
    """
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_task(project_root=project_root, title='x', status='in_progress')
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'Invalid status' in exc.value.message
    assert await backend.get_statuses(project_root) == {}
    dto = await backend.add_task(project_root=project_root, title='y')
    assert dto['id'] == '1'


@pytest.mark.asyncio
async def test_add_task_accepts_infra_hold(backend, project_root):
    dto = await backend.add_task(project_root=project_root, title='held', status='infra-hold')
    task = await backend.get_task(dto['id'], project_root=project_root)
    assert task['status'] == 'infra-hold'


# ── update_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'pending', 'cancelled', 'in-progress', 'blocked', ''])
async def test_update_task_rejects_non_none_status(backend, project_root, status):
    """Backend floor: update_task must raise StatusWriteAuthorityError for any non-None status.

    (a) Seeded-task rejection — the write is blocked and the task stays 'pending'.
    (b) Empty-string '' pins is-not-None semantics over truthiness.
    (c) The subclass IS-A TaskmasterError, so the code/message assertions stay valid.
    """
    await backend.add_task(project_root=project_root, title='x')
    with pytest.raises(StatusWriteAuthorityError) as exc:
        await backend.update_task('1', project_root=project_root, status=status)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'set_task_status' in exc.value.message
    assert exc.value.to_error_dict() == status_via_update_task_error('1', status)
    # Confirm the write was blocked — status must still be 'pending'
    task = await backend.get_task('1', project_root=project_root)
    assert task['status'] == 'pending'


@pytest.mark.asyncio
async def test_update_task_status_rejection_precedes_existence_check(backend, project_root):
    """Status guard runs BEFORE the task SELECT, so rejection beats 'No tasks found'."""
    with pytest.raises(StatusWriteAuthorityError) as exc:
        await backend.update_task('999', project_root=project_root, status='done')
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'set_task_status' in exc.value.message
    assert exc.value.to_error_dict() == status_via_update_task_error('999', 'done')
    assert 'No tasks found' not in exc.value.message


@pytest.mark.asyncio
async def test_update_task_status_rejection_precedes_connection_error(tmp_path, project_root):
    """Status guard runs BEFORE ensure_connected(), so rejection beats a connection error.

    Uses a closed backend (ensure_connected() would raise RuntimeError) to prove
    the ordering comment in the guard is accurate — not just implied by the code
    position.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    closed_backend = SqliteTaskBackend(cfg)
    await closed_backend.start()
    await closed_backend.close()  # ensure_connected() now raises RuntimeError

    with pytest.raises(StatusWriteAuthorityError) as exc:
        await closed_backend.update_task('1', project_root=project_root, status='done')
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'set_task_status' in exc.value.message
    assert exc.value.to_error_dict() == status_via_update_task_error('1', 'done')


# ── update_task: done_provenance write-authority floor ─────────────


@pytest.mark.asyncio
async def test_update_task_rejects_done_provenance_in_metadata(backend, project_root):
    """Backend floor: update_task must raise DoneProvenanceWriteAuthorityError
    when metadata parses to a dict containing 'done_provenance', and the
    write must be blocked entirely."""
    await backend.add_task(project_root=project_root, title='x')
    with pytest.raises(DoneProvenanceWriteAuthorityError) as exc:
        await backend.update_task(
            '1', project_root=project_root,
            metadata=json.dumps({'done_provenance': {'kind': 'merged', 'commit': 'abc'}}),
        )
    assert exc.value.to_error_dict() == done_provenance_via_update_task_error('1')
    task = await backend.get_task('1', project_root=project_root)
    assert 'done_provenance' not in task['metadata']


@pytest.mark.asyncio
async def test_update_task_rejects_done_provenance_in_dict_metadata(backend, project_root):
    """Defense-in-depth: a caller that bypasses the documented ``str | None``
    signature and passes ``metadata`` as an already-parsed dict must still
    trip the done_provenance floor. Mirrors the interceptor's
    ``_reject_done_provenance_in_update_metadata``, which special-cases
    ``isinstance(metadata, dict)`` before falling back to ``json.loads`` —
    without that, ``json.loads(dict)`` raises ``TypeError``, is swallowed,
    and the floor would silently permit the write."""
    await backend.add_task(project_root=project_root, title='x')
    with pytest.raises(DoneProvenanceWriteAuthorityError) as exc:
        await backend.update_task(
            '1', project_root=project_root,
            metadata={'done_provenance': {'kind': 'merged', 'commit': 'abc'}},
        )
    assert exc.value.to_error_dict() == done_provenance_via_update_task_error('1')
    task = await backend.get_task('1', project_root=project_root)
    assert 'done_provenance' not in task['metadata']


@pytest.mark.asyncio
async def test_update_task_done_provenance_rejection_precedes_existence_check(backend, project_root):
    """done_provenance guard runs BEFORE the task SELECT, so rejection beats 'No tasks found'."""
    with pytest.raises(DoneProvenanceWriteAuthorityError) as exc:
        await backend.update_task(
            '999', project_root=project_root,
            metadata=json.dumps({'done_provenance': {'kind': 'merged', 'commit': 'abc'}}),
        )
    assert exc.value.to_error_dict() == done_provenance_via_update_task_error('999')
    assert 'No tasks found' not in exc.value.message


@pytest.mark.asyncio
async def test_update_task_allows_metadata_without_done_provenance(backend, project_root):
    """Regression guard: metadata lacking the 'done_provenance' key merges normally."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['src']}),
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['files'] == ['src']


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_metadata', ['not json{', json.dumps(['a', 'list'])])
async def test_update_task_done_provenance_floor_skips_unparseable_metadata(
    backend, project_root, bad_metadata,
):
    """Parse-safety guard: unparseable or non-dict metadata does not trip the
    done_provenance floor — mirrors the interceptor's
    _reject_done_provenance_in_update_metadata, which returns None (no
    reject) on a JSON error or non-dict payload."""
    await backend.add_task(project_root=project_root, title='x')
    # Must not raise DoneProvenanceWriteAuthorityError.
    await backend.update_task('1', project_root=project_root, metadata=bad_metadata)


@pytest.mark.asyncio
async def test_update_task_appends_metadata(backend, project_root):
    await backend.add_task(
        project_root=project_root,
        title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    dto = await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['src']}),
        append=True,
    )
    assert dto['updated'] is True
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata']['prd'] == 'old.md'
    assert one['metadata']['files'] == ['src']


@pytest.mark.asyncio
async def test_update_task_overwrites_metadata_without_append(backend, project_root):
    await backend.add_task(
        project_root=project_root, title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    # Default is now shallow-merge; use explicit replace to overwrite wholesale.
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'prd': 'new.md'}),
        metadata_mode='replace',
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata'] == {'prd': 'new.md'}


# ── update_task: candidate_key stays in sync (task 2186 amendment) ──


@pytest.mark.asyncio
async def test_update_task_recomputes_candidate_key_on_title_change(backend, project_root):
    """candidate_key must track the row's CURRENT title. Without this, an
    update that changes title would leave a stale key that no longer
    describes the row — silently undermining the future A2 dedup index."""
    await backend.add_task(
        project_root=project_root, title='Fix parser',
        metadata=json.dumps({'files': ['a.py']}),
    )
    await backend.update_task('1', project_root=project_root, title='Fix the lexer')
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix the lexer', ['a.py'])


@pytest.mark.asyncio
async def test_update_task_recomputes_candidate_key_on_metadata_files_change(backend, project_root):
    """Changing metadata['files'] via update_task must also refresh candidate_key."""
    await backend.add_task(
        project_root=project_root, title='Fix parser',
        metadata=json.dumps({'files': ['a.py']}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
        metadata_mode='replace',
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['candidate_key'] == compute_candidate_key('Fix parser', ['a.py', 'b.py'])


@pytest.mark.asyncio
async def test_update_task_candidate_key_unchanged_when_neither_title_nor_metadata_touched(
    backend, project_root,
):
    """An update touching only e.g. priority leaves candidate_key exactly as
    computed at insert time — it is not blanked or recomputed from nothing."""
    await backend.add_task(
        project_root=project_root, title='Fix parser',
        metadata=json.dumps({'files': ['a.py']}),
    )
    before = await backend.get_task('1', project_root=project_root)
    await backend.update_task('1', project_root=project_root, priority='high')
    after = await backend.get_task('1', project_root=project_root)
    assert after['candidate_key'] == before['candidate_key']
    assert after['candidate_key'] == compute_candidate_key('Fix parser', ['a.py'])


# ── stamp_audit_metadata: privileged, non-protocol audit-field seam ─


@pytest.mark.asyncio
async def test_stamp_audit_metadata_persists_and_preserves_sibling_keys(backend, project_root):
    """Direct call (NOT via set_task_status — the interceptor rewire is task C2)
    persists done_provenance and leaves untouched sibling metadata keys intact —
    the read-modify-write 'preserve omitted keys' contract."""
    await backend.add_task(
        project_root=project_root, title='x',
        metadata=json.dumps({
            'memory_hints': {'entities': ['A'], 'queries': ['q1']},
            'files': ['src/a.py'],
            'external_deps': ['dark_factory:42'],
        }),
    )
    await backend.stamp_audit_metadata(
        '1', project_root,
        {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['done_provenance'] == {'kind': 'merged', 'commit': 'abc123'}
    assert task['metadata']['memory_hints'] == {'entities': ['A'], 'queries': ['q1']}
    assert task['metadata']['files'] == ['src/a.py']
    assert task['metadata']['external_deps'] == ['dark_factory:42']


@pytest.mark.asyncio
async def test_stamp_audit_metadata_second_stamp_merges_without_dropping_earlier(backend, project_root):
    """A second stamp (e.g. reopen_* fields) merges in without erasing the
    done_provenance a prior stamp wrote."""
    await backend.add_task(project_root=project_root, title='x')
    await backend.stamp_audit_metadata(
        '1', project_root,
        {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
    )
    await backend.stamp_audit_metadata(
        '1', project_root,
        {
            'reopen_reason': 'regression found',
            'reopen_from': 'done',
            'reopen_at': '2026-07-09T00:00:00+00:00',
        },
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['done_provenance'] == {'kind': 'merged', 'commit': 'abc123'}
    assert task['metadata']['reopen_reason'] == 'regression found'
    assert task['metadata']['reopen_from'] == 'done'
    assert task['metadata']['reopen_at'] == '2026-07-09T00:00:00+00:00'


@pytest.mark.asyncio
async def test_stamp_audit_metadata_missing_task_id_raises(backend, project_root):
    """A missing task_id raises the same 'No tasks found for ID(s): …' shape
    used by update_task/set_task_claimant."""
    with pytest.raises(TaskmasterError) as exc:
        await backend.stamp_audit_metadata(
            '999', project_root,
            {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
        )
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found for ID(s): 999' in exc.value.message


def test_stamp_audit_metadata_is_privileged_non_protocol_seam():
    """API-surface contract: stamp_audit_metadata is a privileged seam on
    SqliteTaskBackend that must NOT be declared on TaskBackendProtocol — only
    TaskInterceptor may hold a reference to it (PRD C-C)."""
    assert hasattr(SqliteTaskBackend, 'stamp_audit_metadata'), (
        'SqliteTaskBackend.stamp_audit_metadata must exist.'
    )
    assert not hasattr(TaskBackendProtocol, 'stamp_audit_metadata'), (
        'stamp_audit_metadata must NOT be part of TaskBackendProtocol — '
        'it is a privileged seam reachable only from TaskInterceptor.'
    )


# ── _merge_metadata: new additive-merge semantics ─────────────────


def test_merge_metadata_list_collision_appends():
    """(a) Top-level list collision under append=True concatenates."""
    result = json.loads(_merge_metadata('{"tags":["a"]}', '{"tags":["b"]}', mode='additive'))
    assert result == {"tags": ["a", "b"]}


def test_merge_metadata_list_collision_dedupes_stable_order():
    """(b) Duplicate items are deduped in stable old-then-new order."""
    result = json.loads(
        _merge_metadata('{"tags":["a","b"]}', '{"tags":["b","c"]}', mode='additive')
    )
    assert result == {"tags": ["a", "b", "c"]}


def test_merge_metadata_scalar_collision_old_wins_under_append():
    """(c) Regression: scalar collision still resolves OLD-wins under append=True."""
    result = json.loads(
        _merge_metadata('{"prd":"old.md"}', '{"prd":"new.md"}', mode='additive')
    )
    assert result == {"prd": "old.md"}


def test_merge_metadata_append_false_replaces_verbatim():
    """(d) Regression: append=False replaces the metadata verbatim."""
    result = json.loads(
        _merge_metadata('{"prd":"old.md"}', '{"prd":"new.md"}', mode='replace')
    )
    assert result == {"prd": "new.md"}


# ── _merge_metadata: recursive dict-merge (memory_hints shape) ────


def test_merge_metadata_nested_dict_lists_union():
    """(a) memory_hints dict shape: inner list values union additively."""
    old_raw = '{"memory_hints":{"entities":["A"],"queries":["q1"]}}'
    new_raw = '{"memory_hints":{"entities":["B"],"queries":["q2"]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    assert result == {"memory_hints": {"entities": ["A", "B"], "queries": ["q1", "q2"]}}


def test_merge_metadata_nested_dict_lists_dedup():
    """(b) Overlap within inner lists is deduped in stable order."""
    old_raw = '{"memory_hints":{"entities":["A","B"],"queries":[]}}'
    new_raw = '{"memory_hints":{"entities":["B","C"],"queries":[]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    assert result == {"memory_hints": {"entities": ["A", "B", "C"], "queries": []}}


def test_merge_metadata_nested_scalar_collision_old_wins():
    """(c) Nested scalar collision resolves OLD-wins."""
    old_raw = '{"audit":{"created_by":"x"}}'
    new_raw = '{"audit":{"created_by":"y","updated_by":"z"}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    assert result == {"audit": {"created_by": "x", "updated_by": "z"}}


@pytest.mark.asyncio
async def test_update_task_memory_hints_union(backend, project_root):
    """(d) End-to-end through update_task: memory_hints union via append=True."""
    await backend.add_task(
        project_root=project_root,
        title='hinted',
        metadata=json.dumps({'memory_hints': {'entities': ['A'], 'queries': ['q1']}}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['B'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    hints = task['metadata']['memory_hints']
    assert hints == {'entities': ['A', 'B'], 'queries': ['q1', 'q2']}


@pytest.mark.asyncio
async def test_update_task_preserves_sibling_keys_during_memory_hints_append(backend, project_root):
    """Regression: stage2 prompt promises siblings (`files`, `spawned_from`, audit dicts)
    survive an additive merge whose incoming payload supplies only `memory_hints`. Lock
    that promise end-to-end through `update_task`."""
    await backend.add_task(
        project_root=project_root,
        title='audit-row',
        metadata=json.dumps({
            'files': ['src/a.py', 'src/b.py'],
            'spawned_from': 'task-100',
            'audit': {'created_by': 'x'},
        }),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E1'], 'queries': ['q1']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {
        'files': ['src/a.py', 'src/b.py'],
        'spawned_from': 'task-100',
        'audit': {'created_by': 'x'},
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
    }


def test_merge_metadata_list_of_dicts_concatenates_without_dedup():
    """Unhashable list items (dicts) fall back to plain concatenation — no dedup."""
    old_raw = '{"x":[{"k":1}]}'
    new_raw = '{"x":[{"k":1}]}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    # Both dicts present; unhashable items are NOT deduped (plain concat).
    assert result == {"x": [{"k": 1}, {"k": 1}]}


def test_merge_metadata_type_mismatch_old_wins_for_non_hint_keys():
    """Type mismatch (old=list, new=dict) resolves to OLD wins for arbitrary keys.

    This audit-field-protection rule is intentional for generic keys (e.g. ``x``,
    ``done_provenance``) where a malformed/unexpected write should not be allowed
    to overwrite a structured value.

    Note: ``memory_hints`` is the only key that receives special treatment — it is
    normalised from legacy list-of-dicts shape to canonical dict shape before
    _merge_values runs, so the dict-vs-dict recursive union path handles the merge
    instead.  See test_merge_metadata_legacy_list_hints_coerce_and_union_with_new_dict.
    """
    old_raw = '{"x":[1,2]}'
    new_raw = '{"x":{"a":1}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    assert result["x"] == [1, 2]


# ── _merge_metadata: legacy memory_hints migration ───────────────────


def test_merge_metadata_legacy_list_hints_coerce_and_union_with_new_dict():
    """Legacy list-of-dicts memory_hints is coerced to dict shape and union-merged.

    When an existing row carries the legacy memory_hints shape
    ``[{"entity": ..., "query": ...}, ...]`` and the incoming payload carries the
    canonical shape ``{"entities": [...], "queries": [...]}`` with append=True, the
    legacy list must be coerced to dict shape BEFORE _merge_values runs — so the
    merge falls into the dict-vs-dict recursive path (which unions the inner lists)
    rather than the type-mismatch OLD-wins path (which silently discards the new dict).

    Old-then-new stable order is preserved (same policy as the dict-vs-dict union).

    Also covers symmetric cases to ensure normalization is applied to both sides:
    * old=canonical dict, new=legacy list → union (new side is also normalised)
    * old=legacy list,     new=legacy list → both coerced then unioned
    """
    # Primary case: old=legacy list, new=canonical dict
    old_raw = '{"memory_hints":[{"entity":"E1","query":"q1"},{"entity":"E2","query":"q2"}]}'
    new_raw = '{"memory_hints":{"entities":["E3"],"queries":["q3"]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    assert result == {"memory_hints": {"entities": ["E1", "E2", "E3"], "queries": ["q1", "q2", "q3"]}}

    # Symmetric case 1: old=canonical dict, new=legacy list → union
    old_raw_sym = '{"memory_hints":{"entities":["E1"],"queries":["q1"]}}'
    new_raw_sym = '{"memory_hints":[{"entity":"E2","query":"q2"}]}'
    result_sym = json.loads(_merge_metadata(old_raw_sym, new_raw_sym, mode='additive'))
    assert result_sym == {"memory_hints": {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}}

    # Symmetric case 2: old=legacy list, new=legacy list → both coerced, then unioned
    old_raw_ll = '{"memory_hints":[{"entity":"E1","query":"q1"}]}'
    new_raw_ll = '{"memory_hints":[{"entity":"E2","query":"q2"}]}'
    result_ll = json.loads(_merge_metadata(old_raw_ll, new_raw_ll, mode='additive'))
    assert result_ll == {"memory_hints": {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}}


def test_merge_metadata_legacy_hints_not_normalized_on_one_sided_write():
    """Normalization is scoped to the collision path: one-sided writes do not migrate.

    When only the *old* side carries ``memory_hints`` (and the incoming write
    does not touch that key), the stored legacy list shape is left unchanged.
    Normalization only fires when BOTH sides carry ``memory_hints``, keeping
    the special case strictly scoped to the merge-collision path and avoiding
    any implicit side-effect on unrelated writes.
    """
    old_raw = '{"tag":"old","memory_hints":[{"entity":"E1","query":"q1"}]}'
    new_raw = '{"tag":"new"}'  # does not carry memory_hints
    result = json.loads(_merge_metadata(old_raw, new_raw, mode='additive'))
    # scalar collision on "tag" → OLD wins
    assert result["tag"] == "old"
    # memory_hints was NOT in the incoming write, so normalization does not fire;
    # the legacy list shape is preserved verbatim in the merged result.
    assert result["memory_hints"] == [{"entity": "E1", "query": "q1"}]


@pytest.mark.asyncio
async def test_update_task_legacy_list_hints_coerce_under_append_true(backend, project_root):
    """End-to-end: legacy list-shape memory_hints row + append=True dict write → union.

    Locks the Stage-2 LLM call path: update_task(append=True) with a canonical-dict
    memory_hints payload now correctly merges with a row that was seeded in legacy
    list-of-dicts shape, rather than silently discarding the incoming dict.
    """
    await backend.add_task(
        project_root=project_root,
        title='legacy-row',
        metadata=json.dumps({'memory_hints': [{'entity': 'E1', 'query': 'q1'}]}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E2'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['memory_hints'] == {
        'entities': ['E1', 'E2'],
        'queries': ['q1', 'q2'],
    }


@pytest.mark.asyncio
async def test_update_task_legacy_hints_migration_preserves_sibling_metadata(backend, project_root):
    """Sibling metadata keys are untouched when a legacy-list hints row is migrated.

    Mirrors test_update_task_preserves_sibling_keys_during_memory_hints_append but
    proves the no-collateral-damage promise still holds when the row starts in
    legacy list-of-dicts shape rather than canonical dict shape.
    """
    await backend.add_task(
        project_root=project_root,
        title='sibling-row',
        metadata=json.dumps({
            'files': ['src/a.py'],
            'spawned_from': 'task-100',
            'audit': {'created_by': 'x'},
            'memory_hints': [{'entity': 'E1', 'query': 'q1'}],
        }),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E2'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {
        'files': ['src/a.py'],
        'spawned_from': 'task-100',
        'audit': {'created_by': 'x'},
        'memory_hints': {'entities': ['E1', 'E2'], 'queries': ['q1', 'q2']},
    }


@pytest.mark.asyncio
async def test_sqlite_task_backend_has_no_add_subtask_method():
    """SqliteTaskBackend must NOT have an add_subtask method after DF-D (task 1543).

    RED assertion: fails while add_subtask is still present, passes once step-4
    removes it.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    assert not hasattr(SqliteTaskBackend, 'add_subtask'), (
        'SqliteTaskBackend.add_subtask still exists; '
        'DF-D (task 1543) step-4 must delete it.'
    )


@pytest.mark.asyncio
async def test_row_to_task_returns_empty_dict_for_malformed_metadata(backend, project_root):
    """_row_to_task coerces malformed metadata JSON to {} for top-level rows.

    Regression guard: if a legacy row holds a non-JSON string in the metadata
    column, the except branch in _row_to_task must surface {} rather than the
    raw string, so downstream `(task.get('metadata') or {}).get(...)` callers
    never receive a str and raise AttributeError.
    """
    # Set up a top-level task.
    await backend.add_task(project_root=project_root, title='parent')

    # Directly corrupt the row's metadata column with a non-JSON string.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON' WHERE id = 1"
    )
    await conn.commit()

    # Top-level task: malformed metadata must surface as {}, not 'NOT_JSON'.
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['metadata'] == {}


@pytest.mark.asyncio
async def test_row_to_task_warns_on_malformed_metadata(backend, project_root, caplog):
    """_row_to_task emits a WARNING when it coerces malformed metadata JSON to {}.

    The warning must include the row's tag, id, and a truncated preview of the
    bad metadata_raw value so an operator can locate and repair the offending row.
    The {}-coercion contract must also hold.
    """
    # Create a top-level task.
    await backend.add_task(project_root=project_root, title='parent')

    # Directly corrupt the row's metadata column with a non-JSON string.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_GARBAGE_xyz' WHERE id = 1"
    )
    await conn.commit()

    # Capture WARNING-level records.
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        task = await backend.get_task('1', project_root=project_root)

    # The {}-coercion contract holds.
    assert task['metadata'] == {}

    # At least one WARNING record must mention tag, id, and the payload preview.
    # Use labeled tokens (e.g. 'id=1') rather than bare digits to prevent false positives.
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, 'Expected at least one WARNING log record; got none'
    combined = ' '.join(warning_msgs)

    # Top-level row: tag=master, id=1.
    assert 'master' in combined, f'Expected tag "master" in warning; got: {combined!r}'
    assert re.search(r'\bid=1\b', combined), (
        f'Expected word-bounded labeled token "id=1" in warning; got: {combined!r}'
    )
    assert 'NOT_JSON_GARBAGE' in combined, (
        f'Expected metadata_raw preview in warning; got: {combined!r}'
    )

    # The warning must carry a labeled project_root= token so an operator can
    # identify which DB is corrupt (added by task 1263).
    assert 'project_root=' in combined, (
        f'Expected labeled token "project_root=" in warning; got: {combined!r}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_deduplicated_per_id_per_process(
    backend, project_root, caplog,
):
    """Repeated reads of the same malformed-metadata row emit at most one WARNING.

    `_get_tasks_internal` invokes `_row_to_task` on every row of every `get_tasks`
    call. A project DB with many corrupted rows would otherwise flood the log
    with one WARNING per row per call. The dedup gate caches `(project_root, tag,
    id)` triples already warned about and skips subsequent emissions for the
    lifetime of the process.
    """
    await backend.add_task(project_root=project_root, title='parent')
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_DEDUP' WHERE id = 1"
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        first = await backend.get_task('1', project_root=project_root)
        second = await backend.get_task('1', project_root=project_root)
        listing = await backend.get_tasks(project_root=project_root)

    # Coercion contract still holds for every read.
    assert first['metadata'] == {}
    assert second['metadata'] == {}
    assert listing['tasks'][0]['metadata'] == {}

    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 1, (
        f'Expected exactly one malformed-metadata WARNING across three reads '
        f'of the same row; got {len(malformed_msgs)}: {malformed_msgs}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_dedup_key_distinguishes_distinct_ids(
    backend, project_root, caplog,
):
    """Two distinct top-level task ids (id=1 and id=2) dedup independently.

    The WARNING gate must key on the full (project_root, tag, id) triple so
    both rows surface their own WARNING once (not collapsed into one).
    """
    await backend.add_task(project_root=project_root, title='task_one')
    await backend.add_task(project_root=project_root, title='task_two')
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_KEYS' WHERE id = 1"
    )
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_KEYS' WHERE id = 2"
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        await backend.get_task('1', project_root=project_root)
        await backend.get_task('2', project_root=project_root)

    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 2, (
        f'Expected two distinct dedup keys (id=1 vs id=2); got '
        f'{len(malformed_msgs)}: {malformed_msgs}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_dedup_distinguishes_project_roots(
    backend, tmp_path, caplog,
):
    """Two project_roots sharing the same (tag, id) row emit distinct WARNs.

    A single SqliteTaskBackend instance services all project_roots.  Before the
    fix, the dedup key was (tag, parent_id, id), so both project_roots' corrupted
    (master, 0, 1) rows collided on the same key — the second WARN was silently
    swallowed.  The fix prepends project_root to the tuple, making each project
    DB's WARNING independent.  Each WARNING must also carry a ``project_root=``
    labeled token so an operator can pin the WARN to its DB.
    """
    proj_a = str(tmp_path / 'proj_a')
    proj_b = str(tmp_path / 'proj_b')

    # Each project_root gets a canonical (tag=master, id=1) row.
    await backend.add_task(project_root=proj_a, title='parent_a')
    await backend.add_task(project_root=proj_b, title='parent_b')

    # Corrupt both DBs' metadata column with a non-JSON string.
    conn_a = await backend._get_connection(proj_a)
    await conn_a.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_PROJ' WHERE id = 1"
    )
    await conn_a.commit()

    conn_b = await backend._get_connection(proj_b)
    await conn_b.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_PROJ' WHERE id = 1"
    )
    await conn_b.commit()

    # Read from both project_roots and capture warnings.
    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        task_a = await backend.get_task('1', project_root=proj_a)
        task_b = await backend.get_task('1', project_root=proj_b)

    # The {}-coercion contract holds for both.
    assert task_a['metadata'] == {}
    assert task_b['metadata'] == {}

    # Both project_roots must produce their own WARNING — the dedup tuple
    # now distinguishes (proj_a, master, 1) from (proj_b, master, 1).
    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 2, (
        f'Expected exactly two malformed-metadata WARNs (one per project_root); '
        f'got {len(malformed_msgs)}: {malformed_msgs}'
    )

    # Each individual warning must contain its respective project_root path.
    msgs_containing_proj_a = [m for m in malformed_msgs if proj_a in m]
    msgs_containing_proj_b = [m for m in malformed_msgs if proj_b in m]
    assert msgs_containing_proj_a, (
        f'Expected a WARNING containing {proj_a!r}; messages: {malformed_msgs}'
    )
    assert msgs_containing_proj_b, (
        f'Expected a WARNING containing {proj_b!r}; messages: {malformed_msgs}'
    )

    # Every WARNING must carry the labeled project_root= token.
    for msg in malformed_msgs:
        assert 'project_root=' in msg, (
            f'Expected "project_root=" token in WARNING message; got: {msg!r}'
        )


# ── remove_tasks with cascade ──────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_tasks_unknown_id_returns_failure_dto(backend, project_root):
    dto = await backend.remove_tasks(['99'], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 1
    assert dto['removed_ids'] == []


@pytest.mark.asyncio
async def test_remove_tasks_batch_mixed_existing_missing(backend, project_root):
    # Two top-levels exist (1, 2); 3 and 99 do not.
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    dto = await backend.remove_tasks(
        ['1', '2', '3', '99'], project_root=project_root,
    )

    assert dto['successful'] == 2
    assert dto['failed'] == 2
    assert sorted(dto['removed_ids']) == ['1', '2']
    assert '3' in dto['message']
    assert '99' in dto['message']

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'] == []


@pytest.mark.asyncio
async def test_remove_tasks_atomicity_on_malformed_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    with pytest.raises(TaskmasterError):
        # 'oops' is not a parseable id — the whole batch fails before any
        # delete runs. Verify state is unchanged afterwards.
        await backend.remove_tasks(
            ['1', 'oops', '2'], project_root=project_root,
        )

    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_rejects_nested_subtask_id_atomically(backend, project_root):
    """remove_tasks raises INVALID_TASK_ID for any dotted id and rolls back.

    After DF-D step-6, _parse_task_id rejects ALL dotted ids — not only
    3+-level nested ones. The whole batch must fail before any delete runs.
    """
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    with pytest.raises(TaskmasterError) as exc_info:
        # '1.1' is a single-level dotted id — all dotted ids are now invalid.
        await backend.remove_tasks(
            ['1', '1.1', '2'], project_root=project_root,
        )

    assert exc_info.value.code == 'INVALID_TASK_ID'
    # Key off the offending id repr rather than pinning the prose.
    assert "'1.1'" in exc_info.value.message

    # State must be unchanged — both tasks still present.
    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_empty_list_is_noop(backend, project_root):
    dto = await backend.remove_tasks([], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 0
    assert dto['removed_ids'] == []


# ── Dependencies ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_remove_dependency_round_trip(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')

    add = await backend.add_dependency('2', '1', project_root=project_root)
    assert add['id'] == '2' and add['dependency_id'] == '1'

    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == [1]

    remove = await backend.remove_dependency(
        '2', '1', project_root=project_root,
    )
    assert remove['id'] == '2'
    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == []


@pytest.mark.asyncio
async def test_add_dependency_self_loop_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    with pytest.raises(TaskmasterError):
        await backend.add_dependency('1', '1', project_root=project_root)


# ── add_dependency — qualified (cross-project) happy path ──────────


@pytest.mark.asyncio
async def test_qualified_dep_stored_in_external_deps(backend, project_root):
    """add_dependency with a qualified dep stores it in metadata.external_deps."""
    await backend.add_task(project_root=project_root, title='a')
    result = await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)
    assert result['id'] == '1'
    assert result['dependency_id'] == 'dark_factory:13'

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['dark_factory:13']
    assert task['dependencies'] == []


@pytest.mark.asyncio
async def test_qualified_dep_idempotent_no_duplicate(backend, project_root):
    """Adding the same qualified dep twice does not produce duplicates."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['dark_factory:13']


@pytest.mark.asyncio
async def test_qualified_dep_accumulates_multiple(backend, project_root):
    """Two distinct qualified deps accumulate in external_deps."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)
    await backend.add_dependency('1', 'reify:7', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['dark_factory:13', 'reify:7']


@pytest.mark.asyncio
async def test_qualified_dep_hyphen_normalized(backend, project_root):
    """'dark-factory:13' stores canonical 'dark_factory:13'."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'dark-factory:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['dark_factory:13']


@pytest.mark.asyncio
async def test_qualified_dep_preserves_sibling_metadata(backend, project_root):
    """Qualified add_dependency preserves other metadata keys (e.g. memory_hints)."""
    import json as _json
    await backend.add_task(project_root=project_root, title='a')
    await backend.update_task('1', project_root, metadata=_json.dumps({'sibling_key': 'preserved'}))
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['dark_factory:13']
    assert task['metadata']['sibling_key'] == 'preserved'


@pytest.mark.asyncio
async def test_qualified_dep_lenient_foreign_target_missing(backend, project_root):
    """Qualified dep succeeds even when the foreign target does not exist."""
    await backend.add_task(project_root=project_root, title='a')
    # 'other_project:999' — foreign target never created; should NOT raise.
    await backend.add_dependency(
        '1', 'other_project:999', project_root=project_root,
    )
    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['other_project:999']


@pytest.mark.asyncio
async def test_qualified_and_bare_dep_coexist(backend, project_root):
    """A task can have both an integer dep (dependencies table) and a qualified dep."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')

    await backend.add_dependency('2', '1', project_root=project_root)
    await backend.add_dependency('2', 'dark_factory:13', project_root=project_root)

    task = await backend.get_task('2', project_root)
    assert task['dependencies'] == [1]
    assert task['metadata']['external_deps'] == ['dark_factory:13']


# ── add_dependency — qualified rejection tests ─────────────────────


@pytest.mark.asyncio
async def test_qualified_dep_self_raises(backend, project_root):
    """Qualified dep that points to itself (same project + same id) raises TaskmasterError."""
    from fused_memory.models.scope import resolve_project_id
    await backend.add_task(project_root=project_root, title='a')
    self_dep = f'{resolve_project_id(project_root)}:1'
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_dependency('1', self_dep, project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'cannot depend on itself' in str(exc.value)


@pytest.mark.asyncio
async def test_qualified_dep_nonexistent_dependent_raises(backend, project_root):
    """Qualified dep where the dependent task (the 'id') does not exist raises TaskmasterError."""
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_dependency('999', 'dark_factory:13', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found' in str(exc.value)


@pytest.mark.asyncio
async def test_qualified_dep_self_raises_mixed_case(backend, project_root):
    """Self-loop detection is case-insensitive: DARK_FACTORY:1 still rejected for task 1."""
    from fused_memory.models.scope import resolve_project_id
    await backend.add_task(project_root=project_root, title='a')
    # Upper-cased project_id canonicalizes to same as resolve_project_id output.
    self_dep = f'{resolve_project_id(project_root).upper()}:1'
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_dependency('1', self_dep, project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'cannot depend on itself' in str(exc.value)


@pytest.mark.asyncio
async def test_add_dependency_rejects_dotted_dependent_id(backend, project_root):
    """add_dependency raises INVALID_TASK_ID when the dependent task id is dotted.

    After DF-D step-6, _parse_task_id rejects all dotted ids, so '1.1' as the
    dependent (first) arg must raise — not silently route to a subtask row.
    """
    await backend.add_task(project_root=project_root, title='a')
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_dependency('1.1', 'dark_factory:5', project_root=project_root)
    assert exc.value.code == 'INVALID_TASK_ID'


@pytest.mark.asyncio
async def test_remove_dependency_rejects_dotted_dependent_id(backend, project_root):
    """remove_dependency raises INVALID_TASK_ID when the dependent task id is dotted."""
    await backend.add_task(project_root=project_root, title='a')
    with pytest.raises(TaskmasterError) as exc:
        await backend.remove_dependency('1.1', 'dark_factory:5', project_root=project_root)
    assert exc.value.code == 'INVALID_TASK_ID'


# ── remove_dependency — qualified (cross-project) tests ────────────


@pytest.mark.asyncio
async def test_qualified_remove_dep_removes_one_leaves_other(backend, project_root):
    """remove_dependency with a qualified dep removes only that entry."""
    import json as _json
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)
    await backend.add_dependency('1', 'reify:7', project_root=project_root)
    # Also set a sibling key to verify it survives.
    sibling_meta = _json.dumps({'extra': 'keep'})
    await backend.update_task('1', project_root, metadata=sibling_meta, append=True)

    await backend.remove_dependency('1', 'dark_factory:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['reify:7']
    assert task['metadata']['extra'] == 'keep'


@pytest.mark.asyncio
async def test_qualified_remove_dep_hyphen_normalized(backend, project_root):
    """Hyphen form 'dark-factory:13' removes the canonical 'dark_factory:13'."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'dark_factory:13', project_root=project_root)

    await backend.remove_dependency('1', 'dark-factory:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata'].get('external_deps', []) == []


@pytest.mark.asyncio
async def test_qualified_remove_dep_idempotent_absent(backend, project_root):
    """Removing an absent qualified dep is a no-op (no error)."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_dependency('1', 'reify:7', project_root=project_root)

    # 'nope:1' was never added — should not raise.
    await backend.remove_dependency('1', 'nope:1', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == ['reify:7']


@pytest.mark.asyncio
async def test_qualified_remove_dep_integer_table_unaffected(backend, project_root):
    """Qualified remove_dependency does not touch the integer dependencies table."""
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')

    await backend.add_dependency('2', '1', project_root=project_root)
    await backend.add_dependency('2', 'dark_factory:13', project_root=project_root)

    await backend.remove_dependency('2', 'dark_factory:13', project_root=project_root)

    task = await backend.get_task('2', project_root)
    assert task['dependencies'] == [1]
    assert task['metadata'].get('external_deps', []) == []


@pytest.mark.asyncio
async def test_validate_dependencies_reports_dangling(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')
    await backend.add_dependency('2', '1', project_root=project_root)
    # Remove the target so the dependency on it dangles.
    await backend.remove_tasks(['1'], project_root=project_root)
    res = await backend.validate_dependencies(project_root=project_root)
    assert 'Dangling dependencies' in res['message']
    assert '2 -> 1' in res['message']


@pytest.mark.asyncio
async def test_validate_dependencies_clean_returns_success(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    res = await backend.validate_dependencies(project_root=project_root)
    assert res['message'] == 'Dependencies validated successfully'


# ── Persistence on disk ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_file_lives_at_taskmaster_tasks_dir(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    expected = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    assert expected.exists()


@pytest.mark.asyncio
async def test_state_survives_close_and_reopen(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    project_root = str(tmp_path / 'proj')
    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.add_task(project_root=project_root, title='persisted')
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    listing = await b2.get_tasks(project_root=project_root)
    assert [t['title'] for t in listing['tasks']] == ['persisted']
    await b2.close()


@pytest.mark.asyncio
async def test_checkpoint_all_reports_per_project_result(tmp_path):
    """``checkpoint_all`` returns ``{root: {busy, log, checkpointed}}`` for
    every open project, and an empty dict when no project has been touched."""
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()

    # No projects opened yet → empty result.
    assert await backend.checkpoint_all() == {}

    root_a = str(tmp_path / 'a')
    root_b = str(tmp_path / 'b')
    await backend.add_task(project_root=root_a, title='a')
    await backend.add_task(project_root=root_b, title='b')

    results = await backend.checkpoint_all()
    assert set(results.keys()) == {root_a, root_b}
    for root, r in results.items():
        # busy=0 with no concurrent readers; log/checkpointed are non-negative.
        assert r['busy'] == 0, f'{root}: unexpected busy {r}'
        assert r['log'] >= 0
        assert r['checkpointed'] >= 0
    await backend.close()


@pytest.mark.asyncio
async def test_close_runs_final_truncate_checkpoint(tmp_path):
    """``close()`` should run a final TRUNCATE checkpoint so the next open
    sees an empty WAL and the main DB file is fully up-to-date — minimises
    recovery work on the next start."""
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    project_root = str(tmp_path / 'proj')
    await backend.add_task(project_root=project_root, title='one')
    await backend.close()

    wal_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db-wal'
    # After clean close, the WAL file either does not exist or has been
    # truncated to its 32-byte header (= 0-frame state). Either is acceptable.
    if wal_path.exists():
        # 32 bytes is the WAL header with zero frames.
        assert wal_path.stat().st_size <= 32, (
            f'WAL not truncated on close: {wal_path.stat().st_size} bytes'
        )


# ── Schema migration (DF-D step-8) ────────────────────────────────


_OLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    parent_id     INTEGER NOT NULL DEFAULT 0,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    priority      TEXT,
    metadata      TEXT,
    updated_at    TEXT,
    PRIMARY KEY (tag, parent_id, id)
);
CREATE INDEX IF NOT EXISTS ix_tasks_parent ON tasks (tag, parent_id);
CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
CREATE TABLE IF NOT EXISTS dependencies (
    tag        TEXT NOT NULL DEFAULT 'master',
    task_id    INTEGER NOT NULL,
    parent_id  INTEGER NOT NULL DEFAULT 0,
    depends_on INTEGER NOT NULL,
    PRIMARY KEY (tag, parent_id, task_id, depends_on)
);
CREATE TABLE IF NOT EXISTS id_counters (
    tag       TEXT NOT NULL DEFAULT 'master',
    parent_id INTEGER NOT NULL DEFAULT 0,
    max_id    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tag, parent_id)
);
"""


def _make_old_schema_db(db_path: Path) -> None:
    """Create a tasks.db with the old (parent_id-inclusive) schema."""
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_OLD_SCHEMA_SQL)
    # Top-level row (parent_id=0, id=1)
    conn.execute(
        "INSERT INTO tasks (tag, id, parent_id, title, status) VALUES ('master', 1, 0, 'top-level', 'pending')",
    )
    # Straggler subtask row (parent_id=1, id=1)
    conn.execute(
        "INSERT INTO tasks (tag, id, parent_id, title, status) VALUES ('master', 1, 1, 'straggler-subtask', 'pending')",
    )
    conn.execute(
        "INSERT INTO id_counters (tag, parent_id, max_id) VALUES ('master', 0, 1)",
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_migration_drops_parent_id_column_and_straggler(tmp_path):
    """Opening a legacy DB triggers the migration: parent_id columns gone, subtask dropped.

    RED until step-8 adds the _migrate() routine.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_old_schema_db(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        await b.get_tasks(project_root=project_root)  # triggers connection-open + migration
    finally:
        await b.close()

    conn = sqlite3.connect(str(db_path))
    try:
        tasks_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        deps_cols = {row[1] for row in conn.execute("PRAGMA table_info(dependencies)")}
        counters_cols = {row[1] for row in conn.execute("PRAGMA table_info(id_counters)")}
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(tasks)")}
        surviving_rows = conn.execute("SELECT title FROM tasks WHERE tag='master'").fetchall()
    finally:
        conn.close()

    assert 'parent_id' not in tasks_cols, f'tasks still has parent_id column: {tasks_cols}'
    assert 'parent_id' not in deps_cols, f'dependencies still has parent_id column: {deps_cols}'
    assert 'parent_id' not in counters_cols, f'id_counters still has parent_id column: {counters_cols}'
    assert user_version == 4, f'Expected user_version=4 after migration; got {user_version}'
    assert {'claimant_run_id', 'heartbeat_at'} <= tasks_cols, (
        f'Expected claimant_run_id/heartbeat_at columns after full-rebuild migration; got {tasks_cols}'
    )
    # v0->v3 chained path also ALTERs in candidate_key (review S3): the
    # rebuilt-then-ALTERed table must carry the candidate_key column so the
    # v2->v3 backfill lands on this path too.
    assert 'candidate_key' in tasks_cols, (
        f'Expected candidate_key column after v0->v3 chained migration; got {tasks_cols}'
    )
    assert 'ix_tasks_parent' not in indexes, f'ix_tasks_parent should be gone: {indexes}'
    assert any('ix_tasks_status' in idx for idx in indexes), f'ix_tasks_status missing: {indexes}'
    titles = [r[0] for r in surviving_rows]
    assert titles == ['top-level'], f'Expected only top-level row; got {titles}'


@pytest.mark.asyncio
async def test_migration_idempotent_second_open(tmp_path):
    """Opening an already-migrated DB a second time is a no-op: user_version stays 4."""
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_old_schema_db(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.get_tasks(project_root=project_root)
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    try:
        await b2.get_tasks(project_root=project_root)
    finally:
        await b2.close()

    conn = sqlite3.connect(str(db_path))
    try:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        tasks_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    finally:
        conn.close()

    assert user_version == 4
    assert 'parent_id' not in tasks_cols
    assert {'claimant_run_id', 'heartbeat_at'} <= tasks_cols


@pytest.mark.asyncio
async def test_fresh_db_has_no_parent_id_and_user_version_4(tmp_path):
    """A brand-new DB is created with the post-migration schema from the start."""
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        await b.add_task(project_root=project_root, title='fresh task')
        listing = await b.get_tasks(project_root=project_root)
    finally:
        await b.close()

    assert listing['tasks'][0]['title'] == 'fresh task'

    conn = sqlite3.connect(str(db_path))
    try:
        tasks_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    assert 'parent_id' not in tasks_cols, f'New DB should not have parent_id; got {tasks_cols}'
    assert user_version == 4, f'Fresh DB should have user_version=4; got {user_version}'
    assert {'claimant_run_id', 'heartbeat_at'} <= tasks_cols, (
        f'Expected claimant_run_id/heartbeat_at columns in fresh schema; got {tasks_cols}'
    )
    assert 'candidate_key' in tasks_cols, (
        f'Expected candidate_key column in fresh schema; got {tasks_cols}'
    )


def _make_v1_schema_db(db_path: Path) -> None:
    """Create a tasks.db with the v1 schema: no parent_id, NO claimant/candidate_key columns.

    Mirrors ``_SCHEMA_SQL`` as it existed before the claimant / candidate_key
    columns were added, stamped ``user_version = 1``. This is the common
    production shape (parent_id already dropped by a prior deploy, later
    columns not yet added) that the v1->v2 and v2->v3 ALTER-TABLE migration
    steps must handle.
    """
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag           TEXT NOT NULL DEFAULT 'master',
            id            INTEGER NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            details       TEXT,
            test_strategy TEXT,
            status        TEXT NOT NULL,
            priority      TEXT,
            metadata      TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    conn.execute(
        "INSERT INTO tasks (tag, id, title, status, updated_at) "
        "VALUES ('master', 1, 'v1 task', 'pending', '2026-01-01T00:00:00.000Z')",
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_migration_v1_to_v2_adds_claimant_columns(tmp_path):
    """Opening an already-migrated v1 DB ALTERs in the claimant columns and
    then chains the v2->v3 candidate_key step and the v3->v4 index step,
    landing at v4.

    This is the common production case: parent_id is already gone (a prior
    deploy ran the v0->v1 rebuild), but claimant_run_id/heartbeat_at don't
    exist yet. Distinct from test_migration_drops_parent_id_column_and_straggler
    above, which exercises the v0->v3 full-rebuild-then-ALTER path for DBs that
    still have parent_id. The single seed row is duplicate-free, so the
    v3->v4 residual audit is clean and the migration reaches v4 uninterrupted.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v1_schema_db(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        listing = await b.get_tasks(project_root=project_root)  # triggers connection-open + migration
    finally:
        await b.close()

    # Pre-existing row survives the ALTER untouched.
    assert listing['tasks'][0]['title'] == 'v1 task'

    conn = sqlite3.connect(str(db_path))
    try:
        tasks_cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    assert {'claimant_run_id', 'heartbeat_at'} <= tasks_cols, (
        f'Expected ALTER TABLE to add claimant_run_id/heartbeat_at; got {tasks_cols}'
    )
    assert user_version == 4, f'Expected user_version=4 after v1->v4 migration; got {user_version}'


def _make_v1_schema_db_no_candidate_key(db_path: Path) -> None:
    """Create a v1 (flat, post-parent_id) tasks.db WITHOUT a candidate_key
    column — simulates a real pre-task-2186 production DB that already went
    through the parent_id->flat migration but predates candidate_key.

    Rows:
      id=1: non-cancelled (pending), title='Fix the bug', files=[a.py, b.py]
      id=2: non-cancelled (``done``), title='fix   the  bug' (extra internal
            whitespace + different case), files=[b.py, a.py] (swapped
            order) — normalizes to the SAME candidate_key as id=1 (one
            duplicate group of size 2). Status is deliberately ``done``
            (fm-task-dedup self-heal amendment) so ``_classify_residual_group``
            flags this group as ``mixed_status`` rather than auto-healing
            it — the tests seeding this fixture assert the pre-self-heal
            skip+escalate outcome (user_version stays at 3, no index).
      id=3: CANCELLED, same title+files as id=1 — must NOT backfill/count.
      id=4: non-cancelled, unique title+files — no duplicate.
    """
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag           TEXT NOT NULL DEFAULT 'master',
            id            INTEGER NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            details       TEXT,
            test_strategy TEXT,
            status        TEXT NOT NULL,
            priority      TEXT,
            metadata      TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    rows = [
        (1, 'Fix the bug', 'pending', {'files': ['a.py', 'b.py']}),
        (2, 'fix   the  bug', 'done', {'files': ['b.py', 'a.py']}),
        (3, 'Fix the bug', 'cancelled', {'files': ['a.py', 'b.py']}),
        (4, 'Totally different task', 'pending', {'files': ['z.py']}),
    ]
    for task_id, title, status, metadata in rows:
        conn.execute(
            "INSERT INTO tasks (tag, id, title, status, metadata, updated_at) "
            "VALUES ('master', ?, ?, ?, ?, '2026-01-01T00:00:00.000Z')",
            (task_id, title, status, json.dumps(metadata)),
        )
    conn.execute("INSERT INTO id_counters (tag, max_id) VALUES ('master', 4)")
    conn.execute('PRAGMA user_version = 1')
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_v2_to_v3_migration_backfills_candidate_key_and_audits_duplicates(
    tmp_path, caplog,
):
    """Opening a legacy v1 DB (no candidate_key column) chains v1->v2->v3: the
    v2->v3 step backfills candidate_key for non-cancelled rows, leaves
    cancelled rows' candidate_key NULL, emits a report-only audit log naming
    the duplicate-group count, bumps user_version to 3, and creates NO index —
    nothing is ever deleted.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v1_schema_db_no_candidate_key(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        with caplog.at_level(
            logging.INFO, logger='fused_memory.backends.sqlite_task_backend',
        ):
            # Triggers connection-open (_SCHEMA_SQL + _migrate).
            await b.get_tasks(project_root=project_root)
        # (h) get_task exposes candidate_key through the normal read path.
        one = await b.get_task('1', project_root=project_root)
    finally:
        await b.close()

    expected_key = compute_candidate_key('Fix the bug', ['a.py', 'b.py'])
    assert one['candidate_key'] == expected_key

    conn = sqlite3.connect(str(db_path))
    try:
        tasks_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        indexes = {r[1] for r in conn.execute('PRAGMA index_list(tasks)')}
        rows = conn.execute(
            "SELECT id, status, candidate_key FROM tasks WHERE tag='master' ORDER BY id",
        ).fetchall()
    finally:
        conn.close()
    by_id = {r[0]: (r[1], r[2]) for r in rows}

    # (a) candidate_key column now exists.
    assert 'candidate_key' in tasks_cols, f'candidate_key column missing: {tasks_cols}'

    # (g) report-only — all 4 original rows survive untouched, nothing deleted.
    assert set(by_id) == {1, 2, 3, 4}, f'Expected all 4 rows to survive; got ids={set(by_id)}'

    # (b) both non-cancelled dup rows (id=1, id=2) backfilled to the SAME key
    # — case/whitespace-insensitive title match, order-insensitive files.
    assert by_id[1][1] == expected_key, by_id[1]
    assert by_id[2][1] == expected_key, by_id[2]

    # (c) the cancelled row's (id=3) candidate_key IS NULL — cancelled rows
    # are excluded from backfill (a cancelled task's work may be re-filed).
    assert by_id[3][0] == 'cancelled'
    assert by_id[3][1] is None

    # id=4 is unique — gets its own, different key.
    unique_key = compute_candidate_key('Totally different task', ['z.py'])
    assert by_id[4][1] == unique_key
    assert by_id[4][1] != expected_key

    # (e) user_version bumped to 3.
    assert user_version == 3, f'Expected user_version=3 after migration; got {user_version}'

    # (f) no index references candidate_key yet — the UNIQUE index is task
    # A2's job, gated on a clean audit; this task must not create one.
    assert not any('candidate_key' in idx for idx in indexes), (
        f'No index should reference candidate_key yet (A2 builds it); got: {indexes}'
    )

    # (d) exactly one audit log record naming duplicate_groups=1 (id=1/id=2
    # form the one duplicate group; the cancelled id=3 does not count).
    audit_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.INFO and 'duplicate_groups=' in r.message
    ]
    assert len(audit_msgs) == 1, (
        f'Expected exactly one candidate_key audit log record; got '
        f'{len(audit_msgs)}: {audit_msgs}'
    )
    assert re.search(r'\bduplicate_groups=1\b', audit_msgs[0]), (
        f'Expected duplicate_groups=1 in audit log; got: {audit_msgs[0]!r}'
    )


def _make_v1_schema_db_no_candidate_key_all_unique(db_path: Path) -> None:
    """Create a v1 (flat, post-parent_id) tasks.db WITHOUT a candidate_key
    column, where every non-cancelled row has a distinct (title, files) —
    the CLEAN audit case (duplicate_groups == 0). Companion to
    ``_make_v1_schema_db_no_candidate_key`` (which pins the >0 / WARNING
    branch); this fixture pins the ==0 / INFO branch.

    Rows (all non-cancelled, all unique):
      id=1: title='Fix the bug',        files=[a.py]
      id=2: title='Add the feature',    files=[c.py]
      id=3: title='Refactor the thing', files=[] (no metadata key at all)
    """
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag           TEXT NOT NULL DEFAULT 'master',
            id            INTEGER NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            details       TEXT,
            test_strategy TEXT,
            status        TEXT NOT NULL,
            priority      TEXT,
            metadata      TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    rows = [
        (1, 'Fix the bug', 'pending', {'files': ['a.py']}),
        (2, 'Add the feature', 'in-progress', {'files': ['c.py']}),
        (3, 'Refactor the thing', 'done', None),
    ]
    for task_id, title, status, metadata in rows:
        conn.execute(
            "INSERT INTO tasks (tag, id, title, status, metadata, updated_at) "
            "VALUES ('master', ?, ?, ?, ?, '2026-01-01T00:00:00.000Z')",
            (task_id, title, status, json.dumps(metadata) if metadata is not None else None),
        )
    conn.execute("INSERT INTO id_counters (tag, max_id) VALUES ('master', 3)")
    conn.execute('PRAGMA user_version = 1')
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_v2_to_v3_migration_clean_audit_logs_info_with_zero_duplicates(
    tmp_path, caplog,
):
    """Companion to the duplicate-group migration test above: when every
    non-cancelled row's candidate_key is unique, the migration still
    backfills all of them, but the one-shot audit line logs at INFO (not
    WARNING) naming duplicate_groups=0 — the expected steady-state outcome
    that gates task A2's UNIQUE index.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v1_schema_db_no_candidate_key_all_unique(db_path)

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        with caplog.at_level(
            logging.INFO, logger='fused_memory.backends.sqlite_task_backend',
        ):
            # Triggers connection-open (_SCHEMA_SQL + _migrate).
            await b.get_tasks(project_root=project_root)
    finally:
        await b.close()

    conn = sqlite3.connect(str(db_path))
    try:
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        rows = conn.execute(
            "SELECT id, candidate_key FROM tasks WHERE tag='master' ORDER BY id",
        ).fetchall()
    finally:
        conn.close()
    by_id = {r[0]: r[1] for r in rows}

    # All three (unique, non-cancelled) rows are backfilled with distinct,
    # non-NULL keys.
    assert by_id[1] is not None and by_id[2] is not None and by_id[3] is not None, by_id
    assert len({by_id[1], by_id[2], by_id[3]}) == 3, f'Expected 3 distinct keys; got {by_id}'
    assert by_id[1] == compute_candidate_key('Fix the bug', ['a.py'])
    assert by_id[2] == compute_candidate_key('Add the feature', ['c.py'])
    assert by_id[3] == compute_candidate_key('Refactor the thing', [])
    # The v2->v3 backfill's own audit is clean (3/3 unique), and the v3->v4
    # residual audit over the same rows is trivially clean too, so the chain
    # reaches v4 (index built) rather than stopping at v3.
    assert user_version == 4, f'Expected user_version=4 after migration; got {user_version}'

    # Exactly one audit record, at INFO (not WARNING), naming duplicate_groups=0.
    audit_records = [r for r in caplog.records if 'duplicate_groups=' in r.message]
    assert len(audit_records) == 1, (
        f'Expected exactly one candidate_key audit log record; got '
        f'{len(audit_records)}: {[r.message for r in audit_records]}'
    )
    assert audit_records[0].levelno == logging.INFO, (
        f'Expected INFO level for a clean (0-duplicate) audit; got '
        f'{logging.getLevelName(audit_records[0].levelno)}'
    )
    assert re.search(r'\bduplicate_groups=0\b', audit_records[0].message), (
        f'Expected duplicate_groups=0 in audit log; got: {audit_records[0].message!r}'
    )


def _dup_row(
    task_id: int, title: str, status: str, files: list[str], candidate_key: str | None = None,
) -> dict[str, Any]:
    """Build one pure dict-like residual-group row for
    ``_classify_residual_group`` tests (no DB involved).

    ``candidate_key`` defaults to the REAL recompute of ``(title, files)`` —
    the common case where the stored key still accurately reflects the row's
    current content. Pass an explicit value to simulate a stale/
    title-divergent stored key (mirrors the 5-tuple override accepted by
    ``_make_v3_db_with_dup_groups``).
    """
    return {
        'id': task_id,
        'title': title,
        'status': status,
        'metadata': json.dumps({'files': files}) if files else None,
        'candidate_key': (
            candidate_key if candidate_key is not None
            else compute_candidate_key(title, files)
        ),
    }


class TestClassifyResidualGroup:
    """Pure (no DB) tests for the v3->v4 self-heal classifier.

    ``_classify_residual_group`` consumes the non-cancelled rows of ONE
    residual (tag, candidate_key) duplicate group and decides whether it is
    safe to auto-heal (cancel all but a canonical survivor) or must be
    flagged for human review.
    """

    def test_all_active_identical_title_heals_preferring_lowest_in_progress(self):
        """canonical prefers the lowest-id IN-PROGRESS row over pending rows,
        even when a lower pending id exists."""
        rows = [
            _dup_row(5, 'Fix the bug', 'pending', ['a.py']),
            _dup_row(3, 'Fix the bug', 'in-progress', ['a.py']),
            _dup_row(9, 'Fix the bug', 'pending', ['a.py']),
        ]
        assert _classify_residual_group(rows) == ('heal', 3, [5, 9])

    def test_all_active_no_in_progress_heals_preferring_lowest_id(self):
        """No in-progress row in the group -> canonical falls back to the
        lowest id overall."""
        rows = [
            _dup_row(7, 'Fix the bug', 'pending', ['a.py']),
            _dup_row(4, 'Fix the bug', 'blocked', ['a.py']),
        ]
        assert _classify_residual_group(rows) == ('heal', 4, [7])

    def test_group_containing_done_row_is_flagged_mixed_status(self):
        """A `done` row anywhere in the group blocks auto-heal — cancelling
        completed work needs a human, even though the candidate_key matches."""
        rows = [
            _dup_row(2, 'Fix the bug', 'done', ['a.py']),
            _dup_row(6, 'Fix the bug', 'pending', ['a.py']),
        ]
        assert _classify_residual_group(rows) == ('flag', 'mixed_status')

    def test_title_divergent_stale_key_group_is_flagged(self):
        """Rows sharing a STORED candidate_key that no longer matches a fresh
        recompute of (title, files) are NOT a genuine content-duplicate —
        flag rather than heal."""
        shared_stale_key = 'stale1234567890a'
        rows = [
            _dup_row(1, 'Fix the bug', 'pending', ['a.py'], candidate_key=shared_stale_key),
            _dup_row(
                2, 'Totally different task', 'pending', ['z.py'],
                candidate_key=shared_stale_key,
            ),
        ]
        assert _classify_residual_group(rows) == ('flag', 'title_divergent')


def _make_v3_db_with_dup_groups(
    db_path: Path,
    rows: list[tuple[int, str, str, list[str]] | tuple[int, str, str, list[str], str]],
) -> None:
    """Create a tasks.db seeded directly at schema v3: ``candidate_key``
    column present and backfilled, id_counters seeded, NO
    ``ux_tasks_candidate_key`` index — the exact shape the v3->v4 self-heal
    migration step (and the live ``reaudit_candidate_key_index`` re-run)
    classifies and acts on.

    Mirrors ``_make_v1_schema_db_no_candidate_key`` but seeds straight at
    ``user_version = 3`` with the full v3 column set (including
    ``claimant_run_id``/``heartbeat_at``, present on any real v3 DB since the
    v1->v2 step necessarily ran first in the cumulative chain) so these tests
    drive the v3->v4 step in isolation without re-exercising v1->v2->v3.

    ``rows`` entries are ``(id, title, status, files)`` — the stored
    ``candidate_key`` is backfilled via ``compute_candidate_key(title,
    files)``, exactly like a real v2->v3 backfill would — or the 5-tuple
    ``(id, title, status, files, explicit_candidate_key)``, which stores
    ``explicit_candidate_key`` verbatim regardless of what ``(title, files)``
    would recompute to. The 5-tuple form is how a test seeds a title-divergent
    / stale-key group: rows whose STORED candidate_key collides (same
    explicit value) but whose current (title, files) recompute to DIFFERENT
    keys.
    """
    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag             TEXT NOT NULL DEFAULT 'master',
            id              INTEGER NOT NULL,
            title           TEXT NOT NULL,
            description     TEXT,
            details         TEXT,
            test_strategy   TEXT,
            status          TEXT NOT NULL,
            priority        TEXT,
            metadata        TEXT,
            updated_at      TEXT NOT NULL,
            claimant_run_id TEXT,
            heartbeat_at    TEXT,
            candidate_key   TEXT,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    max_id = 0
    for row in rows:
        if len(row) == 5:
            task_id, title, status, files, candidate_key = row
        else:
            task_id, title, status, files = row
            candidate_key = compute_candidate_key(title, files)
        metadata = json.dumps({'files': files}) if files else None
        conn.execute(
            "INSERT INTO tasks (tag, id, title, status, metadata, updated_at, candidate_key) "
            "VALUES ('master', ?, ?, ?, ?, '2026-01-01T00:00:00.000Z', ?)",
            (task_id, title, status, metadata, candidate_key),
        )
        max_id = max(max_id, task_id)
    conn.execute("INSERT INTO id_counters (tag, max_id) VALUES ('master', ?)", (max_id,))
    conn.execute('PRAGMA user_version = 3')
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_v3_to_v4_migration_clean_audit_builds_partial_unique_index(
    backend, project_root,
):
    """A fresh DB chains through v0->v4: the v3->v4 step's residual-duplicate
    audit is trivially clean on an empty table, so it builds the partial
    UNIQUE index over (tag, candidate_key) and stamps user_version=4.
    """
    import sqlite3

    await backend.add_task(project_root=project_root, title='fresh task')

    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    conn = sqlite3.connect(str(db_path))
    try:
        index_rows = {row[1]: row for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='ux_tasks_candidate_key'",
        ).fetchone()
    finally:
        conn.close()

    # (a) the partial UNIQUE index is present.
    assert 'ux_tasks_candidate_key' in index_rows, (
        f'Expected ux_tasks_candidate_key index; got: {sorted(index_rows)}'
    )
    # (b) it is UNIQUE — index_list column 2 is the `unique` flag.
    assert index_rows['ux_tasks_candidate_key'][2] == 1, (
        f'Expected ux_tasks_candidate_key to be UNIQUE; got {index_rows["ux_tasks_candidate_key"]}'
    )
    # (c) the stored SQL carries the partial predicate over (tag, candidate_key).
    assert sql_row is not None, 'Expected ux_tasks_candidate_key SQL in sqlite_master'
    index_sql = sql_row[0]
    assert 'candidate_key IS NOT NULL' in index_sql, index_sql
    assert "status != 'cancelled'" in index_sql, index_sql
    assert 'tag' in index_sql and 'candidate_key' in index_sql, index_sql
    # (d) user_version advances to 4 on a clean build.
    assert user_version == 4, (
        f'Expected user_version=4 after clean v3->v4 migration; got {user_version}'
    )


@pytest.mark.asyncio
async def test_v3_to_v4_self_gating_skips_index_and_escalates_on_residual_duplicates(
    tmp_path, caplog,
):
    """When a residual non-cancelled duplicate candidate_key group is still
    present at connection-open, the v3->v4 step must SKIP the index build,
    leave user_version at 3, log a loud ERROR naming the group, and invoke
    the injectable ``residual_dup_escalation_cb`` seam — all without raising
    (connection-open migrations must be fail-safe).
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v1_schema_db_no_candidate_key(db_path)

    recorded: list[tuple[str, list[dict]]] = []

    def recording_stub(project_root_arg, residual_groups):
        recorded.append((project_root_arg, residual_groups))

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=recording_stub)
    await b.start()
    try:
        with caplog.at_level(
            logging.ERROR, logger='fused_memory.backends.sqlite_task_backend',
        ):
            # Triggers connection-open (_SCHEMA_SQL + _migrate, chaining
            # v1->v2->v3->v4).
            await b.get_tasks(project_root=project_root)  # (a) must not raise
    finally:
        await b.close()

    conn = sqlite3.connect(str(db_path))
    try:
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
    finally:
        conn.close()

    # (b) no index references candidate_key — the build was skipped.
    assert not any('candidate_key' in idx for idx in indexes), (
        f'No index should reference candidate_key on a residual-dup skip; got: {indexes}'
    )
    # (c) user_version stays at 3 — the skip does not stamp 4.
    assert user_version == 3, (
        f'Expected user_version to stay at 3 on residual-dup skip; got {user_version}'
    )

    expected_key = compute_candidate_key('Fix the bug', ['a.py', 'b.py'])

    # (d) exactly one ERROR record naming the residual group via
    # residual_group_count=, and it must NOT contain the v2->v3 audit's
    # duplicate_groups= token — the two audits' log-scraping assertions must
    # never collide.
    error_records = [
        r for r in caplog.records
        if r.levelno == logging.ERROR and 'residual_group_count=' in r.message
    ]
    assert len(error_records) == 1, (
        f'Expected exactly one residual_group_count= ERROR record; got '
        f'{len(error_records)}: {[r.message for r in error_records]}'
    )
    msg = error_records[0].message
    assert re.search(r'\bresidual_group_count=1\b', msg), msg
    assert 'duplicate_groups=' not in msg, (
        f'v3->v4 ERROR message must not reuse the v2->v3 duplicate_groups= token; got: {msg!r}'
    )
    assert expected_key in msg, (
        f'Expected the shared candidate_key in the ERROR message; got: {msg!r}'
    )
    assert 'ids=[1,2]' in msg, (
        f'Expected the offending ids named in the ERROR message; got: {msg!r}'
    )

    # (e) the escalation stub was invoked once, naming the same group.
    assert len(recorded) == 1, (
        f'Expected exactly one escalation callback invocation; got {recorded}'
    )
    called_project_root, residual_groups = recorded[0]
    assert called_project_root == project_root
    assert residual_groups == [
        {
            'tag': 'master', 'candidate_key': expected_key,
            'task_ids': ['1', '2'], 'count': 2, 'reason': 'mixed_status',
        },
    ], residual_groups


@pytest.mark.asyncio
async def test_v3_to_v4_raising_escalation_cb_is_caught_and_logged(tmp_path, caplog):
    """A misbehaving ``residual_dup_escalation_cb`` (raises instead of just
    recording) must be caught and logged, never propagated -- connection-open
    stays fail-safe even when the injected escalation seam itself is broken,
    and the residual-dup skip outcome (no index, user_version stays at 3) is
    unaffected by the callback's failure.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v1_schema_db_no_candidate_key(db_path)

    def raising_cb(project_root_arg, residual_groups):
        raise RuntimeError('escalation backend unreachable')

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=raising_cb)
    await b.start()
    try:
        with caplog.at_level(
            logging.ERROR, logger='fused_memory.backends.sqlite_task_backend',
        ):
            # Must return normally despite the callback raising.
            await b.get_tasks(project_root=project_root)
    finally:
        await b.close()

    conn = sqlite3.connect(str(db_path))
    try:
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
    finally:
        conn.close()

    # The residual-dup skip outcome is unchanged by the callback failure.
    assert not any('candidate_key' in idx for idx in indexes), (
        f'No index should reference candidate_key when the residual audit '
        f'found duplicates, regardless of the escalation callback outcome; '
        f'got: {indexes}'
    )
    assert user_version == 3, (
        f'Expected user_version to stay at 3 on residual-dup skip even when '
        f'the escalation callback raises; got {user_version}'
    )

    # The callback's own exception was caught and logged (not swallowed
    # silently, not propagated out of connection-open).
    cb_failure_records = [
        r for r in caplog.records
        if r.levelno == logging.ERROR
        and 'residual_dup_escalation_cb' in r.message
        and 'raised while escalating' in r.message
    ]
    assert len(cb_failure_records) == 1, (
        f'Expected exactly one ERROR record logging the raising callback; got '
        f'{len(cb_failure_records)}: {[r.message for r in caplog.records]}'
    )
    assert cb_failure_records[0].exc_info is not None, (
        'Expected the callback failure to be logged with exc_info (via '
        'logger.exception) so the traceback is captured'
    )


@pytest.mark.asyncio
async def test_v3_to_v4_self_heal_cancels_non_canonical_and_builds_index(
    tmp_path, caplog,
):
    """A genuine content-duplicate residual group (all-active, identical
    normalized title+files, no ``done`` row) is auto-healed at
    connection-open: every non-canonical row is cancelled with a durable
    ``auto_cancelled_by_self_heal`` metadata provenance stamp, the canonical
    (lowest-id in-progress) row survives untouched, the escalation callback
    is never invoked (nothing ambiguous to flag), and — since no residual
    remains after healing — the partial UNIQUE index is built and
    user_version advances to 4 in the SAME connection-open, with no restart.
    This is the fix for reify incident esc-candidate-key-migration-2 (37 dup
    groups / 58 rows previously required a manual set_task_status cancel
    plus a server restart).
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            (1, 'Fix the bug', 'pending', ['a.py', 'b.py']),
            (2, 'fix   the  bug', 'in-progress', ['a.py', 'b.py']),
            (3, 'Totally different task', 'pending', ['z.py']),
        ],
    )

    recorded: list[tuple[str, list[dict]]] = []

    def recording_stub(project_root_arg, residual_groups):
        recorded.append((project_root_arg, residual_groups))

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=recording_stub)
    await b.start()
    try:
        with caplog.at_level(
            logging.INFO, logger='fused_memory.backends.sqlite_task_backend',
        ):
            # Triggers connection-open (_migrate); must not raise.
            await b.get_tasks(project_root=project_root)
    finally:
        await b.close()

    expected_key = compute_candidate_key('Fix the bug', ['a.py', 'b.py'])

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        rows = {
            row['id']: row
            for row in conn.execute(
                "SELECT id, status, metadata FROM tasks WHERE tag='master'",
            )
        }
    finally:
        conn.close()

    # (a) the non-canonical row (id=1, pending) is auto-cancelled ...
    assert rows[1]['status'] == 'cancelled', rows[1]['status']
    # (b) ... while the canonical (lowest-id in-progress) row survives
    # non-cancelled, untouched otherwise.
    assert rows[2]['status'] == 'in-progress', rows[2]['status']
    # ... and the unrelated unique row is unaffected.
    assert rows[3]['status'] == 'pending', rows[3]['status']

    # (c) the cancelled row's metadata carries a durable provenance stamp
    # naming the canonical survivor and the shared candidate_key — the
    # human sign-off this self-heal replaces.
    cancelled_metadata = json.loads(rows[1]['metadata'])
    stamp = cancelled_metadata.get('auto_cancelled_by_self_heal')
    assert stamp is not None, cancelled_metadata
    assert stamp['canonical_id'] == 2, stamp
    assert stamp['candidate_key'] == expected_key, stamp
    # Original metadata content survives the merge.
    assert cancelled_metadata['files'] == ['a.py', 'b.py'], cancelled_metadata

    # (d) no residual remains after healing, so the partial UNIQUE index IS
    # built and user_version advances to 4 in this SAME connection-open —
    # no restart required.
    assert 'ux_tasks_candidate_key' in indexes, (
        f'Expected the index to be built once the residual is self-healed; got {indexes}'
    )
    assert user_version == 4, (
        f'Expected user_version=4 after a fully self-healed residual; got {user_version}'
    )

    # (e) nothing ambiguous here — the escalation callback must NOT fire.
    assert recorded == [], (
        f'Expected no escalation for a genuine auto-healed group; got {recorded}'
    )

    # (f) a loud log names the healed group (canonical + cancelled ids).
    heal_records = [
        r for r in caplog.records
        if r.levelno >= logging.INFO
        and 'self-heal' in r.message
        and expected_key in r.message
    ]
    assert len(heal_records) == 1, (
        f'Expected exactly one self-heal log record; got {len(heal_records)}: '
        f'{[r.message for r in caplog.records]}'
    )
    assert 'canonical' in heal_records[0].message.lower(), heal_records[0].message


@pytest.mark.asyncio
async def test_v3_to_v4_mixed_status_group_is_flagged_with_reason(tmp_path):
    """A duplicate group containing a ``done`` row is NOT auto-healed —
    cancelling completed work needs a human even though the content
    genuinely matches. The group is left untouched, the index build is
    skipped, and the escalation callback receives the group tagged
    ``reason == 'mixed_status'``.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            (1, 'Fix the bug', 'done', ['a.py']),
            (2, 'Fix the bug', 'pending', ['a.py']),
        ],
    )

    recorded: list[tuple[str, list[dict]]] = []

    def recording_stub(project_root_arg, residual_groups):
        recorded.append((project_root_arg, residual_groups))

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=recording_stub)
    await b.start()
    try:
        await b.get_tasks(project_root=project_root)  # must not raise
    finally:
        await b.close()

    expected_key = compute_candidate_key('Fix the bug', ['a.py'])

    conn = sqlite3.connect(str(db_path))
    try:
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        statuses = dict(conn.execute("SELECT id, status FROM tasks WHERE tag='master'"))
    finally:
        conn.close()

    # Neither row is cancelled — mixed-status groups are left for a human.
    assert statuses == {1: 'done', 2: 'pending'}, statuses
    assert not any('candidate_key' in idx for idx in indexes), indexes
    assert user_version == 3, user_version

    assert len(recorded) == 1, recorded
    _, residual_groups = recorded[0]
    assert residual_groups == [
        {
            'tag': 'master', 'candidate_key': expected_key,
            'task_ids': ['1', '2'], 'count': 2, 'reason': 'mixed_status',
        },
    ], residual_groups


@pytest.mark.asyncio
async def test_v3_to_v4_title_divergent_group_is_flagged_with_reason(tmp_path):
    """Rows sharing a STORED candidate_key that no longer matches a fresh
    recompute of (title, files) are not a genuine content-duplicate — the
    group is flagged (``reason == 'title_divergent'``), not healed.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    stale_key = 'stale1234567890a'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            (1, 'Fix the bug', 'pending', ['a.py'], stale_key),
            (2, 'Totally different task', 'pending', ['z.py'], stale_key),
        ],
    )

    recorded: list[tuple[str, list[dict]]] = []

    def recording_stub(project_root_arg, residual_groups):
        recorded.append((project_root_arg, residual_groups))

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=recording_stub)
    await b.start()
    try:
        await b.get_tasks(project_root=project_root)  # must not raise
    finally:
        await b.close()

    conn = sqlite3.connect(str(db_path))
    try:
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        statuses = dict(conn.execute("SELECT id, status FROM tasks WHERE tag='master'"))
    finally:
        conn.close()

    assert statuses == {1: 'pending', 2: 'pending'}, statuses
    assert not any('candidate_key' in idx for idx in indexes), indexes
    assert user_version == 3, user_version

    assert len(recorded) == 1, recorded
    _, residual_groups = recorded[0]
    assert residual_groups == [
        {
            'tag': 'master', 'candidate_key': stale_key,
            'task_ids': ['1', '2'], 'count': 2, 'reason': 'title_divergent',
        },
    ], residual_groups


@pytest.mark.asyncio
async def test_v3_to_v4_mixed_db_heals_genuine_while_flagging_ambiguous(tmp_path):
    """A DB holding BOTH a genuine all-active duplicate group AND a flagged
    (mixed-status) group: the genuine group's non-canonical row IS
    cancelled, while the flagged group is left untouched and escalated —
    and since a residual (the flagged group) remains, the index build is
    STILL skipped and user_version stays at 3.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            (1, 'Fix the bug', 'pending', ['a.py', 'b.py']),
            (2, 'fix   the  bug', 'in-progress', ['a.py', 'b.py']),
            (3, 'Add feature', 'done', ['c.py']),
            (4, 'Add feature', 'pending', ['c.py']),
        ],
    )

    recorded: list[tuple[str, list[dict]]] = []

    def recording_stub(project_root_arg, residual_groups):
        recorded.append((project_root_arg, residual_groups))

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg, residual_dup_escalation_cb=recording_stub)
    await b.start()
    try:
        await b.get_tasks(project_root=project_root)  # must not raise
    finally:
        await b.close()

    flagged_key = compute_candidate_key('Add feature', ['c.py'])

    conn = sqlite3.connect(str(db_path))
    try:
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
        user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        statuses = dict(conn.execute("SELECT id, status FROM tasks WHERE tag='master'"))
    finally:
        conn.close()

    # The genuine group healed: id=1 (pending, non-canonical) cancelled,
    # id=2 (in-progress, canonical) survives.
    assert statuses[1] == 'cancelled', statuses
    assert statuses[2] == 'in-progress', statuses
    # The flagged (mixed-status) group is untouched.
    assert statuses[3] == 'done', statuses
    assert statuses[4] == 'pending', statuses

    # A residual (the flagged group) remains, so the index is STILL skipped.
    assert not any('candidate_key' in idx for idx in indexes), indexes
    assert user_version == 3, user_version

    # Escalation carries ONLY the flagged group, not the healed one.
    assert len(recorded) == 1, recorded
    _, residual_groups = recorded[0]
    assert residual_groups == [
        {
            'tag': 'master', 'candidate_key': flagged_key,
            'task_ids': ['3', '4'], 'count': 2, 'reason': 'mixed_status',
        },
    ], residual_groups


# ── rebuild-without-restart: live re-audit (task 2402) ──────────────────


@pytest.mark.asyncio
async def test_reaudit_candidate_key_index_builds_on_live_connection_without_restart(
    tmp_path,
):
    """A running server holding a pre-audit cached connection never re-runs
    ``_migrate_v3_to_v4`` on its own (it only fires at connection-open) --
    reproducing the incident's second failure mode. ``reaudit_candidate_key_index``
    closes that gap: called on the SAME live backend after an operator
    resolves the residual (no reopen/restart), it re-runs the self-heal audit
    and lands the index. A second call is an idempotent no-op.
    """
    import sqlite3

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            # Mixed-status flagged group -- blocks the index build at
            # connection-open.
            (1, 'Ambiguous task', 'done', ['x.py']),
            (2, 'Ambiguous task', 'pending', ['x.py']),
        ],
    )

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        # Connection-open migration: flags the group, skips the index build.
        await b.get_tasks(project_root=project_root)

        conn = sqlite3.connect(str(db_path))
        try:
            indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
            user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        finally:
            conn.close()
        assert not any('candidate_key' in idx for idx in indexes), (
            f'Precondition: the index must be ABSENT before the re-audit; got {indexes}'
        )
        assert user_version == 3, user_version

        # Operator resolves the residual on the SAME live backend -- no
        # reopen/restart anywhere in this test.
        await b.set_task_status('2', 'cancelled', project_root=project_root)

        result = await b.reaudit_candidate_key_index(project_root)
        assert result['index_built'] is True, result

        conn = sqlite3.connect(str(db_path))
        try:
            indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
            user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        finally:
            conn.close()
        assert any('candidate_key' in idx for idx in indexes), (
            f'Expected ux_tasks_candidate_key to be built by the live re-audit; got {indexes}'
        )
        assert user_version == 4, user_version

        # Idempotent second call -- already at v4, no-op.
        result2 = await b.reaudit_candidate_key_index(project_root)
        assert result2['index_built'] is True, result2
        assert result2.get('already_at_v4') is True, result2
    finally:
        await b.close()


# ── index-independent write-path dedup (fm-task-dedup self-heal amendment) ──


@pytest.mark.asyncio
async def test_add_task_dedup_guard_rejects_duplicate_when_index_absent(tmp_path):
    """Reproduces the incident's exact condition: a flagged residual group
    blocks the v3->v4 partial UNIQUE index build (``ux_tasks_candidate_key``
    stays absent, user_version stays at 3), yet a SEPARATE, unrelated
    non-cancelled row R still must not be duplicated — the index-independent
    pre-INSERT SELECT guard in ``add_task`` catches the collision even with
    no DB-level UNIQUE backstop.
    """
    import sqlite3

    from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            # Flagged (mixed-status) group -- unrelated to R, keeps the
            # v3->v4 index build skipped so this test exercises the
            # index-ABSENT window.
            (1, 'Ambiguous task', 'done', ['x.py']),
            (2, 'Ambiguous task', 'pending', ['x.py']),
            # R: a normal, unique, non-cancelled row.
            (3, 'Existing unique task', 'pending', ['r.py']),
        ],
    )

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        # Triggers connection-open migration: flags the group, skips the
        # index build.
        await b.get_tasks(project_root=project_root)

        conn = sqlite3.connect(str(db_path))
        try:
            indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
            user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        finally:
            conn.close()
        assert not any('candidate_key' in idx for idx in indexes), (
            f'Precondition: the index must be ABSENT for this test to '
            f'exercise the index-independent guard; got {indexes}'
        )
        assert user_version == 3, user_version

        with pytest.raises(DuplicateCandidateKeyError) as exc_info:
            await b.add_task(
                project_root=project_root,
                title='Existing unique task',
                metadata=json.dumps({'files': ['r.py']}),
            )
        exc = exc_info.value
        assert exc.existing_id == 3, f'Expected the collision to name survivor id=3; got {exc.existing_id!r}'
        assert exc.existing_status == 'pending', (
            f'Expected the survivor status to be pending; got {exc.existing_status!r}'
        )

        r_key = compute_candidate_key('Existing unique task', ['r.py'])
        listing = await b.get_tasks(project_root=project_root)
        matching = [
            t for t in listing['tasks']
            if t['status'] != 'cancelled' and t.get('candidate_key') == r_key
        ]
        assert len(matching) == 1, (
            f'Expected exactly one non-cancelled row with the colliding key '
            f'(no orphan from the rejected insert); got {len(matching)}: {matching}'
        )
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_update_task_dedup_guard_rejects_recompute_collision_when_index_absent(
    tmp_path,
):
    """Same incident condition as the add_task guard above, exercised on the
    edit path: with the v3->v4 index build blocked by an unrelated flagged
    residual group, an ``update_task`` recompute (title and/or metadata
    touched) that lands on ANOTHER non-cancelled row's candidate_key must be
    rejected before the UPDATE lands — no DB-level UNIQUE backstop exists
    while the index is absent, so the pre-UPDATE SELECT guard is the only
    thing preventing a silent duplicate reactivation.
    """
    import sqlite3

    from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError

    project_root = str(tmp_path / 'proj')
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    _make_v3_db_with_dup_groups(
        db_path,
        [
            # Flagged (mixed-status) group -- unrelated to A/B, keeps the
            # v3->v4 index build skipped so this test exercises the
            # index-ABSENT window.
            (1, 'Ambiguous task', 'done', ['x.py']),
            (2, 'Ambiguous task', 'pending', ['x.py']),
            # A and B: two normal, distinct, non-cancelled rows.
            (3, 'Task A', 'pending', ['a.py']),
            (4, 'Task B', 'pending', ['b.py']),
        ],
    )

    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    try:
        # Triggers connection-open migration: flags the group, skips the
        # index build.
        await b.get_tasks(project_root=project_root)

        conn = sqlite3.connect(str(db_path))
        try:
            indexes = {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}
            user_version = conn.execute('PRAGMA user_version').fetchone()[0]
        finally:
            conn.close()
        assert not any('candidate_key' in idx for idx in indexes), (
            f'Precondition: the index must be ABSENT for this test to '
            f'exercise the index-independent guard; got {indexes}'
        )
        assert user_version == 3, user_version

        # Recompute B (id=4) onto A's (title, files) -- same candidate_key.
        with pytest.raises(DuplicateCandidateKeyError) as exc_info:
            await b.update_task(
                task_id='4',
                project_root=project_root,
                title='Task A',
                metadata=json.dumps({'files': ['a.py']}),
            )
        exc = exc_info.value
        assert exc.existing_id == 3, f'Expected the collision to name survivor id=3; got {exc.existing_id!r}'
        assert exc.existing_status == 'pending', (
            f'Expected the survivor status to be pending; got {exc.existing_status!r}'
        )

        # B is unchanged -- the guard fired before the UPDATE landed.
        b_task = await b.get_task(task_id='4', project_root=project_root)
        assert b_task['title'] == 'Task B', (
            f"Expected B's title untouched by the rejected update; got {b_task['title']!r}"
        )
        assert b_task['candidate_key'] == compute_candidate_key('Task B', ['b.py']), (
            f"Expected B's candidate_key untouched by the rejected update; "
            f"got {b_task['candidate_key']!r}"
        )
    finally:
        await b.close()


# ── candidate_key collision (fm-task-dedup W8 task A2) ───────────────


@pytest.mark.asyncio
async def test_add_task_duplicate_candidate_key_raises_and_no_orphan(backend, project_root):
    """A second add_task whose normalized (title, files) collides with an
    existing non-cancelled row raises DuplicateCandidateKeyError naming the
    survivor, and creates NO orphan row — the partial UNIQUE index rejects
    the INSERT, ``_txn`` rolls it back, and get_tasks still shows exactly
    one non-cancelled row.
    """
    from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError

    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )

    # Same normalized title (case + extra internal whitespace) and the same
    # files (order swapped) — compute_candidate_key is case/whitespace
    # insensitive on title and order-insensitive on files, so this collides.
    with pytest.raises(DuplicateCandidateKeyError) as exc_info:
        await backend.add_task(
            project_root=project_root,
            title='fix  parser',
            metadata=json.dumps({'files': ['b.py', 'a.py']}),
        )

    exc = exc_info.value
    assert exc.existing_id == 1, f'Expected the collision to name survivor id=1; got {exc.existing_id!r}'
    assert exc.existing_status == 'pending', (
        f'Expected the survivor status to be pending; got {exc.existing_status!r}'
    )

    listing = await backend.get_tasks(project_root=project_root)
    non_cancelled = [t for t in listing['tasks'] if t['status'] != 'cancelled']
    assert len(non_cancelled) == 1, (
        f'Expected exactly one non-cancelled row (no orphan from the '
        f'rejected insert); got {len(non_cancelled)}: {listing["tasks"]}'
    )


@pytest.mark.asyncio
async def test_add_task_cancelled_row_allows_refile(backend, project_root):
    """BT-A5: cancelling the surviving row then re-filing the identical
    (title, files) SUCCEEDS — the partial UNIQUE index excludes cancelled
    rows (``WHERE ... status != 'cancelled'``), so a legitimate refile after
    cancellation is never falsely blocked as a collision.
    """
    first = await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    await backend.set_task_status(first['id'], 'cancelled', project_root=project_root)

    second = await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    assert second['id'] == '2', f'Expected the refile to land as a new id=2; got {second["id"]!r}'

    listing = await backend.get_tasks(project_root=project_root)
    statuses_by_id = {t['id']: t['status'] for t in listing['tasks']}
    assert statuses_by_id == {'1': 'cancelled', '2': 'pending'}, statuses_by_id


@pytest.mark.asyncio
async def test_set_task_status_uncancel_collision_raises_duplicate_candidate_key_error(
    backend, project_root,
):
    """Review amendment (fm-task-dedup W8 task A2): un-cancelling a row whose
    candidate_key collides with an existing non-cancelled row must raise
    DuplicateCandidateKeyError, not a raw sqlite3.IntegrityError.
    Un-cancelling moves the row back into the partial UNIQUE index's
    predicate (``status != 'cancelled'``); if the refiled duplicate (BT-A5)
    already occupies that (tag, candidate_key), reactivating the original is
    rejected. The rejected UPDATE must roll back — task 1 stays cancelled.
    """
    from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError

    first = await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    await backend.set_task_status(first['id'], 'cancelled', project_root=project_root)
    # BT-A5 refile: succeeds because the partial index excludes cancelled rows.
    second = await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    assert second['id'] == '2'

    with pytest.raises(DuplicateCandidateKeyError) as exc_info:
        await backend.set_task_status(first['id'], 'pending', project_root=project_root)

    exc = exc_info.value
    assert exc.existing_id == 2, (
        f'Expected the collision to name survivor id=2; got {exc.existing_id!r}'
    )
    assert exc.existing_status == 'pending', exc.existing_status

    # The rejected UPDATE rolled back — task 1 is still cancelled.
    listing = await backend.get_tasks(project_root=project_root)
    statuses_by_id = {t['id']: t['status'] for t in listing['tasks']}
    assert statuses_by_id == {'1': 'cancelled', '2': 'pending'}, statuses_by_id


@pytest.mark.asyncio
async def test_update_task_recompute_collision_raises_duplicate_candidate_key_error(
    backend, project_root,
):
    """Review amendment (fm-task-dedup W8 task A2): update_task recomputes
    candidate_key whenever title/metadata is touched; if the recomputed key
    collides with another non-cancelled row, this must raise
    DuplicateCandidateKeyError rather than a raw sqlite3.IntegrityError. The
    rejected UPDATE must roll back — the row's title stays unchanged.
    """
    from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError

    await backend.add_task(
        project_root=project_root,
        title='Fix parser',
        metadata=json.dumps({'files': ['a.py', 'b.py']}),
    )
    other = await backend.add_task(
        project_root=project_root,
        title='Unrelated task',
        metadata=json.dumps({'files': ['z.py']}),
    )

    with pytest.raises(DuplicateCandidateKeyError) as exc_info:
        await backend.update_task(
            other['id'],
            project_root=project_root,
            title='fix  parser',
            metadata=json.dumps({'files': ['b.py', 'a.py']}),
        )

    exc = exc_info.value
    assert exc.existing_id == 1, (
        f'Expected the collision to name survivor id=1; got {exc.existing_id!r}'
    )
    assert exc.existing_status == 'pending', exc.existing_status

    # The rejected UPDATE rolled back — task 2's title/candidate_key are unchanged.
    two = await backend.get_task(other['id'], project_root=project_root)
    assert two['title'] == 'Unrelated task', two['title']
    assert two['candidate_key'] == compute_candidate_key('Unrelated task', ['z.py'])


# ── Concurrency ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_add_task_yields_unique_ids(backend, project_root):
    """The per-project write lock must serialise id allocation."""
    coros = [
        backend.add_task(project_root=project_root, title=f't{i}')
        for i in range(20)
    ]
    results = await asyncio.gather(*coros)
    ids = sorted(int(r['id']) for r in results)
    assert ids == list(range(1, 21))


# ── Monotonic id allocation (id-recycling regression) ───────────────
#
# ``add_task`` allocates ``max(MAX(tasks.id), id_counters.max_id) + 1`` so a
# deleted id is NEVER reissued.  Without this, deleting the top task frees its
# id, the next ``add_task`` re-mints it, and an orphaned worktree keyed on that
# numeric id gets misadopted for unrelated work (reify task 3770).


@pytest.mark.asyncio
async def test_top_level_id_not_reused_after_delete(backend, project_root):
    """Core regression: deleting the top task must NOT free its id for reuse."""
    one = await backend.add_task(project_root=project_root, title='first')
    assert one['id'] == '1'
    await backend.remove_tasks(['1'], project_root=project_root)
    two = await backend.add_task(project_root=project_root, title='second')
    assert two['id'] == '2'  # NOT '1'


@pytest.mark.asyncio
async def test_id_monotonic_across_delete_add_cycles(backend, project_root):
    """Repeated create+delete of the trailing task keeps bumping the id."""
    ids = []
    for _ in range(5):
        dto = await backend.add_task(project_root=project_root, title='cycle')
        ids.append(int(dto['id']))
        await backend.remove_tasks([dto['id']], project_root=project_root)
    assert ids == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_delete_current_max_still_bumps(backend, project_root):
    """Deleting the current MAX row still advances past it (counter holds)."""
    await backend.add_task(project_root=project_root, title='a')  # 1
    await backend.add_task(project_root=project_root, title='b')  # 2
    await backend.remove_tasks(['2'], project_root=project_root)  # max row gone
    three = await backend.add_task(project_root=project_root, title='c')
    assert three['id'] == '3'  # NOT '2'


@pytest.mark.asyncio
async def test_id_counter_per_tag_isolation(backend, project_root):
    """Counters are scoped per tag — a delete in one tag never affects another."""
    await backend.add_task(project_root=project_root, title='m1', tag='master')
    await backend.add_task(project_root=project_root, title='f1', tag='feature')
    await backend.remove_tasks(['1'], project_root=project_root, tag='master')
    m2 = await backend.add_task(project_root=project_root, title='m2', tag='master')
    f2 = await backend.add_task(project_root=project_root, title='f2', tag='feature')
    assert m2['id'] == '2'  # master counter held past the delete
    assert f2['id'] == '2'  # feature sequence independent, unaffected


@pytest.mark.asyncio
async def test_id_counter_survives_close_reopen(tmp_path):
    """The counter persists across a connection close/reopen.

    Mirrors a ``systemctl restart fused-memory`` cycle: the high-water mark
    must outlive the process so a delete-then-restart-then-add can't recycle.
    """
    proot = str(tmp_path / 'proj')
    cfg = TaskmasterConfig(project_root=str(tmp_path))

    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.add_task(project_root=proot, title='one')   # id 1
    await b1.remove_tasks(['1'], project_root=proot)
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    try:
        two = await b2.add_task(project_root=proot, title='two')
        assert two['id'] == '2'  # counter survived the reopen — NOT '1'
    finally:
        await b2.close()


@pytest.mark.asyncio
async def test_id_counter_self_heals_when_empty_but_tasks_present(backend, project_root):
    """A legacy DB (tasks present, id_counters empty) honours the row high-water.

    Simulates an upgrade onto a DB that predates the counter: the first
    post-upgrade alloc must be ``MAX(tasks.id) + 1``, then the counter is
    seeded so it holds the line on subsequent deletes.
    """
    await backend.add_task(project_root=project_root, title='a')  # 1
    await backend.add_task(project_root=project_root, title='b')  # 2

    # Wipe the counter to mimic a pre-Fix-A DB.
    conn = await backend._get_connection(project_root)
    await conn.execute('DELETE FROM id_counters')
    await conn.commit()

    three = await backend.add_task(project_root=project_root, title='c')
    assert three['id'] == '3'  # self-healed from MAX(tasks.id)

    # And the counter now holds across a delete of the current max.
    await backend.remove_tasks(['3'], project_root=project_root)
    four = await backend.add_task(project_root=project_root, title='d')
    assert four['id'] == '4'


# ── Cancellation hardening ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_status_cancellation_leaves_connection_clean(
    backend, project_root,
):
    """A cancellation arriving while ``set_task_status`` is queued behind
    the write_lock must not leave the connection mid-transaction.

    Reproduces the soak-cancel signature: hold the per-project write lock,
    queue a ``set_task_status`` against it, cancel the awaiter via
    ``wait_for(timeout=0.001)``, then assert the next ``set_task_status``
    applies cleanly. Pre-fix (Exception-only suppress + unshielded
    rollback) the connection could end up holding an open BEGIN, which
    surfaces as ``cannot start a transaction within a transaction``
    on the next mutation.
    """
    # Seed: one task to flip.
    await backend.add_task(project_root=project_root, title='t0')
    assert (await backend.get_task('1', project_root))['status'] == 'pending'

    # Acquire the per-project write lock so the next set_task_status blocks.
    lock = backend._write_lock(project_root)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                backend.set_task_status('1', 'in-progress', project_root),
                timeout=0.001,
            )
    finally:
        lock.release()

    # Connection state must be clean: the next mutation succeeds.
    res = await backend.set_task_status('1', 'done', project_root)
    assert res['tasks'][0]['newStatus'] == 'done'
    assert (await backend.get_task('1', project_root))['status'] == 'done'


# ── get_statuses_raw ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_statuses_raw_returns_all_and_skips_decode(backend, project_root, monkeypatch):
    """get_statuses_raw(ids=None) returns all {str(id): status} without calling _row_to_task.

    Proves:
    - str-keyed, verbatim status passthrough (incl. 'merge-deferred' holding status)
    - _row_to_task (the sole json.loads gateway) is NEVER called on this path
    - result matches the reference from the existing full-tree get_tasks path
    """
    from unittest.mock import MagicMock

    import fused_memory.backends.sqlite_task_backend as _sb

    # Seed 3 tasks with distinct statuses; give one non-trivial metadata to
    # represent the amplification scenario (the decode we must avoid).
    await backend.add_task(project_root=project_root, title='T1')  # id=1, status=pending
    await backend.add_task(
        project_root=project_root, title='T2', status='done',
        metadata=json.dumps({'memory_hints': ['search(project context)'], 'files': ['a.py']}),
    )
    await backend.add_task(
        project_root=project_root, title='T3', status='merge-deferred',
    )

    # Spy on _row_to_task to confirm it is NOT called on the get_statuses_raw path.
    spy = MagicMock(wraps=_sb._row_to_task)
    monkeypatch.setattr(_sb, '_row_to_task', spy)

    mapping = await backend.get_statuses_raw(project_root)

    # Contract: str-keyed, verbatim status (including 'merge-deferred').
    assert mapping == {'1': 'pending', '2': 'done', '3': 'merge-deferred'}

    # Oracle: no metadata decode on this path.
    spy.assert_not_called()

    # Cross-check against the full-tree reference path.
    # (We restore _row_to_task first so get_tasks works normally.)
    monkeypatch.undo()
    ref = await backend.get_tasks(project_root)
    ref_mapping = {str(t['id']): t['status'] for t in ref['tasks']}
    assert mapping == ref_mapping


@pytest.mark.asyncio
async def test_get_statuses_raw_filters_by_ids(backend, project_root):
    """get_statuses_raw(ids=...) filters to the requested subset.

    (a) ids=['1','3'] -> only those two; id 2 absent.
    (b) unknown id: ids=['1','9999'] -> {'1':<s1>} and '9999' absent.
    (c) empty: ids=[] -> {} (NOT the full tree).
    """
    await backend.add_task(project_root=project_root, title='T1')  # id=1 pending
    await backend.add_task(project_root=project_root, title='T2', status='done')
    await backend.add_task(project_root=project_root, title='T3', status='in-progress')

    # (a) subset filter
    result_a = await backend.get_statuses_raw(project_root, ids=['1', '3'])
    assert result_a == {'1': 'pending', '3': 'in-progress'}, (
        f'Expected subset {{1,3}}, got: {result_a}'
    )
    assert '2' not in result_a

    # (b) unknown id silently omitted
    result_b = await backend.get_statuses_raw(project_root, ids=['1', '9999'])
    assert result_b == {'1': 'pending'}, f'Expected only id 1, got: {result_b}'
    assert '9999' not in result_b

    # (c) empty ids -> {} (must NOT return all 3 tasks)
    result_c = await backend.get_statuses_raw(project_root, ids=[])
    assert result_c == {}, f'Expected empty dict, got: {result_c}'


@pytest.mark.asyncio
async def test_get_tasks_status_filter_pushed_into_sql(backend, project_root, monkeypatch):
    """get_tasks(statuses=...) pushes the filter into SQL and returns only matching tasks.

    Four sub-assertions (mirroring test_get_statuses_raw_filters_by_ids):
    (a) statuses=['pending','in-progress'] → only those two tasks returned as full dicts,
        ordered by id, and the issued SQL carries a 'status IN (' predicate.
    (b) statuses omitted (None) → full unfiltered tree returned AND the SQL does NOT
        contain a 'status IN (' predicate (byte-identical to the current path).
    (c) statuses=[] → {'tasks': []} (early return, NOT the full tree).
    """
    # Seed 4 tasks with distinct statuses
    await backend.add_task(project_root=project_root, title='T-pending')       # id=1
    await backend.add_task(project_root=project_root, title='T-done', status='done')  # id=2
    await backend.add_task(project_root=project_root, title='T-inprog', status='in-progress')  # id=3
    await backend.add_task(project_root=project_root, title='T-cancelled', status='cancelled')  # id=4

    # --- Set up spy on conn.execute ---
    conn = await backend._get_connection(project_root)
    recorded_sql: list[str] = []
    _orig_execute = conn.execute

    async def _spy_execute(sql: str, *args, **kwargs):
        recorded_sql.append(sql)
        return await _orig_execute(sql, *args, **kwargs)

    monkeypatch.setattr(conn, 'execute', _spy_execute)

    # (a) Filtered: statuses=['pending', 'in-progress']
    recorded_sql.clear()
    result_a = await backend.get_tasks(project_root, statuses=['pending', 'in-progress'])

    assert 'tasks' in result_a, f'Expected tasks key in result: {result_a}'
    returned_statuses = {t['status'] for t in result_a['tasks']}
    assert returned_statuses == {'pending', 'in-progress'}, (
        f'Expected only pending+in-progress tasks, got statuses: {returned_statuses}'
    )
    returned_ids = [t['id'] for t in result_a['tasks']]
    assert returned_ids == sorted(returned_ids), (
        f'Tasks not in id order: {returned_ids}'
    )
    assert len(result_a['tasks']) == 2, f'Expected 2 tasks, got: {len(result_a["tasks"])}'
    def _norm(s):
        return ' '.join(s.split()).lower()

    assert any('status in (' in _norm(sql) for sql in recorded_sql), (
        f'Expected "status IN (" in issued SQL, got: {recorded_sql}'
    )

    # (b) Unfiltered: statuses omitted (None) → full tree, no IN predicate
    recorded_sql.clear()
    result_b = await backend.get_tasks(project_root)

    assert len(result_b['tasks']) == 4, (
        f'Expected all 4 tasks without filter, got: {len(result_b["tasks"])}'
    )
    assert not any('status in (' in _norm(sql) for sql in recorded_sql), (
        f'Full-tree path must NOT emit "status IN (": {recorded_sql}'
    )

    # (c) Empty statuses list → {'tasks': []} early return
    recorded_sql.clear()
    result_c = await backend.get_tasks(project_root, statuses=[])

    assert result_c == {'tasks': []}, (
        f'Expected empty tasks list for statuses=[], got: {result_c}'
    )


# ── get_statuses_fresh (task 2388) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_statuses_fresh_sees_committed_write_despite_pinned_cached_snapshot(
    backend, project_root,
):
    """get_statuses_fresh reads a live WAL snapshot even when the cached
    WRITE connection has a read transaction pinned open.

    Reproduces the task 2388 root cause: ``_get_connection`` opens its
    cached WRITE connection in legacy deferred-transaction mode, so a read
    transaction left open on it pins a stale WAL snapshot. Originally
    ``get_statuses``/``get_statuses_raw`` shared that cached connection and
    went stale with it; task 2455 hardened the hot ``get_statuses`` path to
    read via a dedicated cached AUTOCOMMIT connection instead (see
    ``_get_read_connection``), so it is now fresh here too — this test
    pins that. ``get_statuses_fresh`` remains the dedicated
    short-lived-autocommit-connection census read for callers (like
    ``cross_verify_task_counts``) that need a guaranteed-uncached fresh
    read regardless of what the hot path's cached read connection is
    doing.
    """
    from shared.async_sqlite_base import apply_wal_pragmas, connect_daemon

    # Seed two tasks and mark both 'done'.
    await backend.add_task(project_root=project_root, title='T1')  # id=1
    await backend.add_task(project_root=project_root, title='T2')  # id=2
    await backend.set_task_status('1', 'done', project_root)
    await backend.set_task_status('2', 'done', project_root)

    # Pin the cached WRITE connection's WAL read-snapshot by leaving a read
    # transaction open on it (materialize the snapshot via fetchall()).
    conn = await backend._get_connection(project_root)
    await conn.execute('BEGIN')
    cur = await conn.execute('SELECT id, status FROM tasks')
    await cur.fetchall()

    # Simulate a separate process committing a status change out-of-band,
    # via a fresh autocommit connection to the same DB file on disk.
    db_path = SqliteTaskBackend._db_path(project_root)
    writer = await connect_daemon(str(db_path), isolation_level=None)
    try:
        await apply_wal_pragmas(writer, busy_timeout_ms=5000)
        await writer.execute("UPDATE tasks SET status='cancelled' WHERE id=1")
        await writer.commit()
    finally:
        await writer.close()

    # The hot get_statuses path is now FRESH (task 2455) despite the pinned
    # write connection — it reads via a separate cached autocommit
    # connection that the write-side pin can't touch.
    fresh_hot = await backend.get_statuses(project_root)
    assert fresh_hot.get('1') == 'cancelled', (
        f"Expected the hardened hot get_statuses read to see 'cancelled', got: {fresh_hot}"
    )

    # The dedicated census read also reflects LIVE committed state.
    fresh = await backend.get_statuses_fresh(project_root)
    assert fresh['1'] == 'cancelled', (
        f"Expected the fresh read to see 'cancelled', got: {fresh}"
    )

    # Release the pin so the fixture's backend.close() isn't left mid-txn.
    await conn.rollback()


@pytest.mark.asyncio
async def test_get_statuses_fresh_returns_empty_when_db_file_missing(backend, project_root):
    """get_statuses_fresh fails open to {} when the project's tasks.db file
    has never been created — e.g. reconciliation runs for a project before
    any task has ever been written for it. Must not raise; this pins the
    ``if not db_path.exists(): return {}`` fast path documented on
    get_statuses_fresh ("{} if the DB file does not exist yet...").
    """
    assert not SqliteTaskBackend._db_path(project_root).exists()

    result = await backend.get_statuses_fresh(project_root)

    assert result == {}


@pytest.mark.asyncio
async def test_get_statuses_fresh_returns_empty_when_connection_open_raises(
    backend, project_root, monkeypatch,
):
    """get_statuses_fresh fails open to {} when opening its dedicated
    short-lived connection raises for any reason (permission error, disk
    I/O failure, corrupt file, ...) — not just when a read on an
    already-open connection fails. Pins the ``except Exception: return {}``
    branch that the reconciliation cycle relies on never raising through.
    """
    from fused_memory.backends import sqlite_task_backend as _sb

    # Seed via the real connect_daemon first so the DB file exists and we
    # get past the "missing file" fast path exercised above.
    await backend.add_task(project_root=project_root, title='T1')

    def _boom(*_args, **_kwargs):
        raise OSError('simulated connection-open failure')

    monkeypatch.setattr(_sb, 'connect_daemon', _boom)

    result = await backend.get_statuses_fresh(project_root)

    assert result == {}


# ── get_statuses hot-path freshness (task 2455) ─────────────────────────


@pytest.mark.asyncio
async def test_get_statuses_hot_path_fresh_despite_pinned_write_connection(
    backend, project_root,
):
    """get_statuses (and get_statuses_raw, bulk and scoped) must observe the
    latest committed WAL state even when the cached WRITE connection
    (``_get_connection``) has a read transaction pinned open.

    Reproduces the task 2388 pinned-snapshot harness, but flips the
    expectation: prior to the task 2455 fix, get_statuses shared the
    pinnable cached write connection and went stale together with it
    (see the historical assertion this superseded, just below in this
    file). The fix routes get_statuses_raw through a dedicated per-project
    cached AUTOCOMMIT connection that can never hold a read transaction
    open across statements, so it can never be pinned.
    """
    from shared.async_sqlite_base import apply_wal_pragmas, connect_daemon

    # Seed two tasks and mark both 'done'.
    await backend.add_task(project_root=project_root, title='T1')  # id=1
    await backend.add_task(project_root=project_root, title='T2')  # id=2
    await backend.set_task_status('1', 'done', project_root)
    await backend.set_task_status('2', 'done', project_root)

    # Pin the cached WRITE connection's WAL read-snapshot by leaving a read
    # transaction open on it (materialize the snapshot via fetchall()).
    conn = await backend._get_connection(project_root)
    await conn.execute('BEGIN')
    cur = await conn.execute('SELECT id, status FROM tasks')
    await cur.fetchall()

    # Simulate a separate process committing a status change out-of-band,
    # via a fresh autocommit connection to the same DB file on disk.
    db_path = SqliteTaskBackend._db_path(project_root)
    writer = await connect_daemon(str(db_path), isolation_level=None)
    try:
        await apply_wal_pragmas(writer, busy_timeout_ms=5000)
        await writer.execute("UPDATE tasks SET status='cancelled' WHERE id=1")
        await writer.commit()
    finally:
        await writer.close()

    # The hot path must be FRESH despite the write connection's open pin —
    # both bulk and scoped (ids=[...]) reads.
    bulk = await backend.get_statuses(project_root)
    assert bulk['1'] == 'cancelled', (
        f"Expected the hot bulk get_statuses to see fresh 'cancelled', got: {bulk}"
    )

    scoped = await backend.get_statuses(project_root, ids=['1'])
    assert scoped['1'] == 'cancelled', (
        f"Expected the hot scoped get_statuses to see fresh 'cancelled', got: {scoped}"
    )

    # Release the pin so the fixture's backend.close() isn't left mid-txn.
    await conn.rollback()


@pytest.mark.asyncio
async def test_close_drains_cached_read_connections(backend, project_root):
    """close() must drain the cached read connections opened by
    :meth:`~SqliteTaskBackend._get_read_connection` (task 2455), not just
    the write connections in ``self._connections`` — otherwise the
    autocommit read connection is leaked open (a stray file handle / WAL
    reader) past shutdown.
    """
    await backend.add_task(project_root=project_root, title='T1')
    # Lazily opens a cached read connection for project_root.
    await backend.get_statuses(project_root)

    assert project_root in backend._read_connections, (
        'Expected a cached read connection to have been opened for project_root'
    )
    read_conn = backend._read_connections[project_root]

    await backend.close()

    assert backend._read_connections == {}, (
        f'Expected _read_connections to be drained/cleared by close(), got: {backend._read_connections}'
    )
    with pytest.raises(Exception):
        await read_conn.execute('SELECT 1')


# ── Corrupt-blob refusal tests (task 1813) ──────────────────────────────


# Shared corrupt blob constant: invalid JSON that still contains the
# external_deps substring so tests can assert it survives unchanged.
_CORRUPT_BLOB = '{"external_deps": ["dark_factory:42"], BROKEN'


def test_merge_metadata_refuses_to_clobber_corrupt_existing_blob():
    """_merge_metadata must raise TaskmasterError when the EXISTING blob is
    corrupt JSON and append=True — it must NOT return incoming and clobber.

    Also locks the two escape hatches:
    - append=False always returns incoming (last-write-wins; the sanctioned
      repair path — caller explicitly chose to replace the corrupt row).
    - existing_raw=None + append=True returns incoming (no existing data to
      preserve).

    RED: current code's ``except (TypeError, ValueError): return incoming``
    returns incoming on the append path instead of raising TaskmasterError.
    After step-2 the kwargs ``project_root/tag/task_id`` are added to
    ``_merge_metadata`` and the corrupt-existing branch raises.
    """
    incoming = json.dumps({"note": "x"})

    # (a) Main assertion: corrupt existing + append=True must raise TaskmasterError.
    with pytest.raises(TaskmasterError) as exc:
        _merge_metadata(
            _CORRUPT_BLOB, incoming, mode='additive',
            project_root='/p', tag='master', task_id=1,
        )
    assert exc.value.raw == _CORRUPT_BLOB, (
        f'Expected exc.value.raw == _CORRUPT_BLOB (original bytes preserved as '
        f'distinguishable signal), got: {exc.value.raw!r}'
    )

    # (b) Escape hatch: append=False must return incoming (explicit replace —
    #     the sanctioned path to repair a corrupt row).
    result_replace = _merge_metadata(_CORRUPT_BLOB, incoming, mode='replace')
    assert result_replace == incoming

    # (c) No-op escape hatch: existing_raw=None + append=True returns incoming.
    result_none = _merge_metadata(None, incoming, mode='additive')
    assert result_none == incoming


@pytest.mark.asyncio
async def test_update_task_refuses_to_clobber_corrupt_metadata(
    backend, project_root, caplog,
):
    """update_task raises TaskmasterError and emits exactly one deduped
    malformed-metadata WARNING when the stored blob is corrupt and append=True.
    The stored metadata column must be byte-for-byte unchanged after the
    refused write (external_deps substring survives).

    Setup deliberately avoids a read-path access before the write so the
    dedup slot is not pre-consumed by _row_to_task.

    RED after step-2: _merge_metadata raises (so raises+bytes-preserved
    assertions pass) but update_task passes no context kwargs to _merge_metadata
    → _warn_malformed_metadata_once is not called → zero WARNINGs → the
    warn-count assertion fails.
    """
    await backend.add_task(project_root=project_root, title='t')

    # Directly inject a corrupt blob WITHOUT reading the row first.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,),
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ), pytest.raises(TaskmasterError):
        await backend.update_task(
            '1', project_root=project_root,
            metadata=json.dumps({'note': 'incoming'}),
            append=True,
        )

    # Exactly one deduped malformed-metadata WARNING must have been emitted.
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(warn_records) == 1, (
        f'Expected exactly one malformed-metadata WARNING; got {len(warn_records)}: '
        f'{[r.message for r in warn_records]}'
    )

    # The original corrupt blob must be byte-for-byte unchanged (no overwrite).
    # Use raw SQL — NOT get_task — so _row_to_task does not pollute the
    # WARNING count or hide the raw bytes via its own coercion path.
    raw_conn = await backend._get_connection(project_root)
    cursor = await raw_conn.execute(
        'SELECT metadata FROM tasks WHERE id = 1',
    )
    row = await cursor.fetchone()
    assert row['metadata'] == _CORRUPT_BLOB, (
        f'Expected metadata unchanged (corrupt blob preserved); got: {row["metadata"]!r}'
    )
    assert 'dark_factory:42' in row['metadata'], (
        f'Expected external_deps substring in preserved metadata; got: {row["metadata"]!r}'
    )


@pytest.mark.asyncio
async def test_stamp_audit_metadata_refuses_to_clobber_corrupt_metadata(
    backend, project_root, caplog,
):
    """stamp_audit_metadata is the sole done_provenance writer gating the
    done-flow — its corrupt-existing-blob contract must match update_task's:
    it reuses _merge_metadata(mode='merge'), which raises TaskmasterError
    and refuses to overwrite a corrupt existing blob rather than silently
    succeeding or best-effort stamping the audit fields. Exactly one deduped
    malformed-metadata WARNING is emitted and the blob survives byte-for-byte.

    Setup deliberately avoids a read-path access before the write so the
    dedup slot is not pre-consumed by _row_to_task.
    """
    await backend.add_task(project_root=project_root, title='t')

    # Directly inject a corrupt blob WITHOUT reading the row first.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,),
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ), pytest.raises(TaskmasterError) as exc:
        await backend.stamp_audit_metadata(
            '1', project_root,
            {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
        )
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'corrupt metadata blob' in exc.value.message

    # Exactly one deduped malformed-metadata WARNING must have been emitted.
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(warn_records) == 1, (
        f'Expected exactly one malformed-metadata WARNING; got {len(warn_records)}: '
        f'{[r.message for r in warn_records]}'
    )

    # The original corrupt blob must be byte-for-byte unchanged — the audit
    # stamp did NOT silently succeed nor clobber the row.
    raw_conn = await backend._get_connection(project_root)
    cursor = await raw_conn.execute(
        'SELECT metadata FROM tasks WHERE id = 1',
    )
    row = await cursor.fetchone()
    assert row['metadata'] == _CORRUPT_BLOB, (
        f'Expected metadata unchanged (corrupt blob preserved); got: {row["metadata"]!r}'
    )
    assert 'dark_factory:42' in row['metadata'], (
        f'Expected external_deps substring in preserved metadata; got: {row["metadata"]!r}'
    )


@pytest.mark.asyncio
async def test_add_dependency_qualified_refuses_corrupt_metadata(
    backend, project_root, caplog,
):
    """add_dependency raises TaskmasterError and emits exactly one deduped
    malformed-metadata WARNING when the stored metadata blob is corrupt,
    leaving the blob byte-for-byte unchanged (the new dep was NOT added).

    RED after step-4: _merge_metadata raises (raises+bytes-preserved pass)
    but add_dependency's qualified path passes no context kwargs to
    _merge_metadata → zero WARNINGs → warn-count assertion fails.
    """
    await backend.add_task(project_root=project_root, title='t')

    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,),
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ), pytest.raises(TaskmasterError):
        await backend.add_dependency(
            '1', 'dark_factory:13', project_root=project_root,
        )

    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(warn_records) == 1, (
        f'Expected exactly one malformed-metadata WARNING; got {len(warn_records)}: '
        f'{[r.message for r in warn_records]}'
    )

    # The new external dep must NOT have been added; original blob unchanged.
    raw_conn = await backend._get_connection(project_root)
    cursor = await raw_conn.execute('SELECT metadata FROM tasks WHERE id = 1')
    row = await cursor.fetchone()
    assert row['metadata'] == _CORRUPT_BLOB, (
        f'Expected metadata unchanged; got: {row["metadata"]!r}'
    )


@pytest.mark.asyncio
async def test_remove_dependency_qualified_warns_and_does_not_falsely_claim_removal(
    backend, project_root, caplog,
):
    """remove_dependency warns once and returns an accurate DependencyResult
    (NOT claiming clean removal) when the stored metadata blob is corrupt.
    The blob must remain byte-for-byte unchanged (no write occurs).

    RED: current except sets meta={}, finds nothing to remove, then returns
    the false 'Removed external dependency: …' message with no WARNING.
    After step-8 the except block warns via the shared gate and returns an
    accurate message that does NOT start with 'Removed external dependency'.
    """
    await backend.add_task(project_root=project_root, title='t')

    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,),
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        result = await backend.remove_dependency(
            '1', 'dark_factory:13', project_root=project_root,
        )

    # Exactly one deduped malformed-metadata WARNING.
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(warn_records) == 1, (
        f'Expected exactly one malformed-metadata WARNING; got {len(warn_records)}: '
        f'{[r.message for r in warn_records]}'
    )

    # Result message must NOT falsely claim the dep was removed.
    assert 'Removed external dependency' not in result['message'], (
        f'Result message falsely claims removal: {result["message"]!r}'
    )
    # Message should convey the blob was left intact / could not remove.
    msg_lower = result['message'].lower()
    assert 'corrupt' in msg_lower or 'intact' in msg_lower, (
        f'Result message should convey blob left intact; got: {result["message"]!r}'
    )

    # Blob is byte-for-byte unchanged.
    raw_conn = await backend._get_connection(project_root)
    cursor = await raw_conn.execute('SELECT metadata FROM tasks WHERE id = 1')
    row = await cursor.fetchone()
    assert row['metadata'] == _CORRUPT_BLOB, (
        f'Expected metadata unchanged; got: {row["metadata"]!r}'
    )


@pytest.mark.asyncio
async def test_malformed_metadata_warn_dedup_shared_across_read_and_write(
    backend, project_root, caplog,
):
    """The shared _warned_malformed_task_ids set deduplicates across the read
    and write paths: a read via get_task followed by a write via update_task
    on the SAME (project_root, tag='master', id=1) produces exactly ONE
    malformed-metadata WARNING total.

    Locks the 'unify onto the extracted handler, don't duplicate' directive:
    a regression that gave the write path a separate dedup set would yield
    2 WARNING records and fail this assertion.

    Behavior delivered by steps 2+4; this step adds no production code.
    """
    await backend.add_task(project_root=project_root, title='t')

    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,),
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        # Read path warns once via _row_to_task → _warn_malformed_metadata_once.
        await backend.get_task('1', project_root=project_root)

        # Write path on the same (project_root, master, 1) — the dedup key
        # was already added above, so the shared gate skips the second WARN.
        with pytest.raises(TaskmasterError):
            await backend.update_task(
                '1', project_root=project_root,
                metadata=json.dumps({'x': 1}),
                append=True,
            )

    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(warn_records) == 1, (
        f'Expected exactly 1 malformed-metadata WARNING (deduped across '
        f'read+write paths on same key); got {len(warn_records)}: '
        f'{[r.message for r in warn_records]}'
    )


# ── _resolve_metadata_mode (step-1 RED / step-2 GREEN) ──────────────────────


@pytest.mark.parametrize(
    'metadata_mode, append, expected',
    [
        # (a) explicit metadata_mode returned verbatim regardless of append
        ('merge',    None,  'merge'),
        ('additive', None,  'additive'),
        ('replace',  None,  'replace'),
        # (b) metadata_mode=None + append=True -> 'additive'
        (None, True,  'additive'),
        # (c) metadata_mode=None + append=False -> 'replace'
        (None, False, 'replace'),
        # (d) metadata_mode=None + append=None -> 'merge' (new default)
        (None, None,  'merge'),
        # (f) explicit metadata_mode wins over conflicting append (distinct combos)
        ('merge',    True,  'merge'),
        ('additive', False, 'additive'),
        ('replace',  True,  'replace'),
    ],
)
def test_resolve_metadata_mode_mapping(metadata_mode, append, expected):
    """_resolve_metadata_mode returns the correct mode for all precedence cases."""
    result = _resolve_metadata_mode(metadata_mode, append)
    assert result == expected, (
        f'_resolve_metadata_mode({metadata_mode!r}, {append!r}) -> '
        f'{result!r}, expected {expected!r}'
    )


def test_resolve_metadata_mode_invalid_raises():
    """An invalid metadata_mode raises TaskmasterError(TASKMASTER_TOOL_ERROR)."""
    with pytest.raises(TaskmasterError) as exc:
        _resolve_metadata_mode('bogus', None)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR', (
        f'Expected TASKMASTER_TOOL_ERROR; got {exc.value.code!r}'
    )


# ── _merge_metadata mode= API (step-3 RED / step-4 GREEN) ───────────────────


def test_merge_metadata_mode_merge_shallow_last_write_wins():
    """mode='merge': sibling key preserved, scalar collision overwrites (not OLD-wins),
    list collision overwrites wholesale (can shrink)."""
    existing = json.dumps({'_causation_id': 'abc', 'branch': 'old', 'files': [1, 2, 3]})
    incoming = json.dumps({'branch': 'new', 'files': [4]})
    result = json.loads(_merge_metadata(existing, incoming, mode='merge'))
    # Sibling preserved
    assert result['_causation_id'] == 'abc', f'sibling clobbered: {result}'
    # Scalar overwritten (not old-wins)
    assert result['branch'] == 'new', f'scalar not updated: {result}'
    # List overwritten wholesale (can shrink)
    assert result['files'] == [4], f'list not overwritten: {result}'


def test_merge_metadata_mode_merge_corrupt_existing_raises():
    """mode='merge' with a corrupt existing blob raises TaskmasterError and
    exc.value.raw == the corrupt bytes."""
    incoming = json.dumps({'note': 'x'})
    with pytest.raises(TaskmasterError) as exc:
        _merge_metadata(_CORRUPT_BLOB, incoming, mode='merge',
                        project_root='/p', tag='master', task_id=1)
    assert exc.value.raw == _CORRUPT_BLOB, (
        f'Expected raw == corrupt bytes; got {exc.value.raw!r}'
    )


def test_merge_metadata_mode_replace_returns_incoming_verbatim():
    """mode='replace' returns incoming verbatim and bypasses the corrupt guard."""
    incoming = json.dumps({'new': 'val'})
    # Non-corrupt existing: returns incoming
    result = _merge_metadata(json.dumps({'old': 1}), incoming, mode='replace')
    assert result == incoming
    # Corrupt existing: replace bypasses the guard (no raise)
    result_corrupt = _merge_metadata(_CORRUPT_BLOB, incoming, mode='replace')
    assert result_corrupt == incoming


def test_merge_metadata_mode_additive_unions_lists_and_old_wins_scalars():
    """mode='additive': list union+dedup, scalar collisions OLD-wins."""
    existing = json.dumps({'items': [1, 2], 'flag': 'original'})
    incoming = json.dumps({'items': [2, 3], 'flag': 'updated'})
    result = json.loads(_merge_metadata(existing, incoming, mode='additive'))
    # List union+dedup stable order
    assert result['items'] == [1, 2, 3], f'additive list wrong: {result}'
    # Scalar OLD-wins
    assert result['flag'] == 'original', f'additive scalar should be old-wins: {result}'


def test_merge_metadata_mode_merge_existing_none_returns_incoming():
    """mode='merge' with existing_raw=None returns incoming (no existing data)."""
    incoming = json.dumps({'k': 'v'})
    assert _merge_metadata(None, incoming, mode='merge') == incoming


# ── update_task end-to-end: default-merge + metadata_mode (step-5 RED / step-6 GREEN) ──


@pytest.mark.asyncio
async def test_update_task_default_merge_regression_4271(backend, project_root):
    """Regression #4271: default update_task merge preserves _causation_id+memory_hints
    while adding files — no metadata_mode or append arg passed."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({
            '_causation_id': 'caus-abc',
            'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
        }),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['src/a.py']}),
    )
    task = await backend.get_task('1', project_root=project_root)
    meta = task['metadata']
    assert meta.get('_causation_id') == 'caus-abc', f'_causation_id clobbered: {meta}'
    assert meta.get('memory_hints') == {'entities': ['E1'], 'queries': ['q1']}, (
        f'memory_hints clobbered: {meta}'
    )
    assert meta.get('files') == ['src/a.py'], f'files not written: {meta}'


@pytest.mark.asyncio
async def test_update_task_default_merge_updates_existing_scalar(backend, project_root):
    """Default merge updates an existing scalar key (not OLD-wins)."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'branch_base_sha': 'aaa', '_causation_id': 'caus'}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'branch_base_sha': 'bbb'}),
    )
    task = await backend.get_task('1', project_root=project_root)
    meta = task['metadata']
    assert meta.get('branch_base_sha') == 'bbb', f'scalar not updated: {meta}'
    assert meta.get('_causation_id') == 'caus', f'sibling clobbered: {meta}'


@pytest.mark.asyncio
async def test_update_task_metadata_mode_additive_unions_list(backend, project_root):
    """metadata_mode='additive' unions a list (dry_run_proposals-style append)."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'dry_run_proposals': ['prop-1']}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'dry_run_proposals': ['prop-2']}),
        metadata_mode='additive',
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['dry_run_proposals'] == ['prop-1', 'prop-2'], (
        f'additive union failed: {task["metadata"]}'
    )


@pytest.mark.asyncio
async def test_update_task_metadata_mode_replace_overwrites(backend, project_root):
    """metadata_mode='replace' overwrites the whole blob (siblings dropped)."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'_causation_id': 'keep', 'extra': 'sibling'}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['f.py']}),
        metadata_mode='replace',
    )
    task = await backend.get_task('1', project_root=project_root)
    meta = task['metadata']
    assert list(meta.keys()) == ['files'], f'replace should drop siblings: {meta}'
    assert meta['files'] == ['f.py']


@pytest.mark.asyncio
async def test_update_task_legacy_append_true_additive(backend, project_root):
    """Legacy append=True -> additive union (shim end-to-end)."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'items': [1]}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'items': [2]}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['items'] == [1, 2], (
        f'legacy append=True should additive-union: {task["metadata"]}'
    )


@pytest.mark.asyncio
async def test_update_task_legacy_append_false_replace(backend, project_root):
    """Legacy append=False -> replace overwrite (shim end-to-end)."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'_causation_id': 'keep', 'extra': 'gone'}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['f.py']}),
        append=False,
    )
    task = await backend.get_task('1', project_root=project_root)
    meta = task['metadata']
    assert 'extra' not in meta, f'legacy append=False should replace: {meta}'
    assert meta['files'] == ['f.py']


@pytest.mark.asyncio
async def test_update_task_default_corrupt_blob_refused(backend, project_root, caplog):
    """Default (no-arg) merge refuses a corrupt existing blob — raises TaskmasterError
    and leaves stored bytes unchanged."""
    await backend.add_task(project_root=project_root, title='t')

    conn = await backend._get_connection(project_root)
    await conn.execute('UPDATE tasks SET metadata = ? WHERE id = 1', (_CORRUPT_BLOB,))
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ), pytest.raises(TaskmasterError):
        await backend.update_task(
            '1', project_root=project_root,
            metadata=json.dumps({'note': 'x'}),
        )

    raw_conn = await backend._get_connection(project_root)
    cursor = await raw_conn.execute('SELECT metadata FROM tasks WHERE id = 1')
    row = await cursor.fetchone()
    assert row['metadata'] == _CORRUPT_BLOB, (
        f'corrupt blob must be byte-for-byte unchanged; got: {row["metadata"]!r}'
    )


@pytest.mark.asyncio
async def test_update_task_invalid_metadata_mode_always_raises(
    backend, project_root,
):
    """Invalid metadata_mode must raise immediately regardless of whether
    ``metadata`` is supplied (regression guard for the unconditional validation
    introduced in the amendment pass — previously the check only fired inside
    the ``if metadata is not None:`` block)."""
    await backend.add_task(project_root=project_root, title='t')

    # Without metadata — the original bug: bad mode was silently ignored.
    with pytest.raises(TaskmasterError) as exc:
        await backend.update_task(
            '1', project_root=project_root,
            metadata_mode='bogus',
        )
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR', (
        f'Expected TASKMASTER_TOOL_ERROR; got {exc.value.code!r}'
    )

    # With metadata — should also raise (sanity check, same code path).
    with pytest.raises(TaskmasterError) as exc2:
        await backend.update_task(
            '1', project_root=project_root,
            metadata=json.dumps({'k': 'v'}),
            metadata_mode='bogus',
        )
    assert exc2.value.code == 'TASKMASTER_TOOL_ERROR'


# ── write-boundary validation (task 2162, update_task post-merge I3) ──


@pytest.mark.asyncio
async def test_update_task_warn_mode_emits_schema_warning_and_proceeds(
    backend, project_root, caplog,
):
    """Warn-mode update_task: a post-merge I3 violation still lands.

    Seeding a normal task with ``{"foo": "bar"}`` and then updating with
    ``{"task_kind": "deterministic"}`` (default additive merge — no
    ``append``/``metadata_mode`` passed) produces the post-merge blob
    ``{"foo": "bar", "task_kind": "deterministic"}``, which violates
    TaskMetadata's cross-field invariant (I3: a deterministic task requires
    ``before_done`` or ``always_escalates``). update_task validates the
    POST-MERGE blob, so this is caught on update, not only on submit.

    The default backend is warn-mode (``task_metadata_enforce=False``), so
    the whole-metadata invariant violation emits exactly one census line
    carrying the ``<metadata>`` sentinel field, and the write proceeds: the
    merged metadata — including the pre-existing ``foo`` sibling, which is
    itself unrecognised by TaskMetadata's schema and so contributes its own
    independent ``unknown_key`` census line — is stored raw/unchanged.
    """
    dto = await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'foo': 'bar'}),
    )
    tid = dto['id']
    # The seed add_task above emits its own census line for the "foo" unknown
    # key (unrelated to what this test verifies) — caplog.records accumulates
    # for the whole test regardless of the at_level() scope below, so drop it
    # to keep the assertions focused on the update_task call under test.
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        result = await backend.update_task(
            tid, project_root=project_root,
            metadata=json.dumps({'task_kind': 'deterministic'}),
        )
    assert result['updated'] is True

    census_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and 'task_metadata.schema_warning' in r.message
    ]
    whole_metadata_msgs = [m for m in census_msgs if '<metadata>' in m]
    assert len(whole_metadata_msgs) == 1, (
        f'Expected exactly one whole-metadata task_metadata.schema_warning line; got '
        f'{len(whole_metadata_msgs)}: {whole_metadata_msgs} (all census lines: {census_msgs})'
    )
    combined = whole_metadata_msgs[0]
    assert f'task_id={tid}' in combined, (
        f'Expected labeled task_id={tid!r} token in census line; got: {combined!r}'
    )
    assert 'before_done' in combined, (
        f'Expected the invariant error text in census line; got: {combined!r}'
    )

    # The write proceeded: the merged metadata (including the "foo" sibling)
    # is stored raw/unchanged — no repair, no schema_version stamp.
    task = await backend.get_task(tid, project_root=project_root)
    assert task['metadata'] == {'foo': 'bar', 'task_kind': 'deterministic'}


@pytest.mark.asyncio
async def test_update_task_valid_metadata_emits_no_schema_warning(
    backend, project_root, caplog,
):
    """A schema-clean update_task merge emits zero task_metadata.schema_warning lines."""
    await backend.add_task(
        project_root=project_root, title='t',
        metadata=json.dumps({'files': ['a.py']}),
    )

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        await backend.update_task(
            '1', project_root=project_root,
            metadata=json.dumps({'files': ['b.py']}),
        )

    census_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and 'task_metadata.schema_warning' in r.message
    ]
    assert census_msgs == [], (
        f'Expected no task_metadata.schema_warning lines for a valid merge; got: {census_msgs}'
    )


@pytest.mark.asyncio
async def test_update_task_enforce_mode_rejects_invariant_violation(tmp_path, project_root):
    """Enforce-mode update_task: a post-merge I3 violation raises and rolls back.

    ``task_metadata_enforce=True`` flips the write-boundary failure policy
    from warn-and-proceed to raise: the merged blob's invariant violation is
    rejected with ``pydantic.ValidationError`` and the stored metadata is
    left byte-for-byte unchanged — the update's txn rolled back.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        seed_metadata = json.dumps({'foo': 'bar'})
        dto = await backend.add_task(
            project_root=project_root, title='t', metadata=seed_metadata,
        )
        tid = dto['id']

        with pytest.raises(ValidationError):
            await backend.update_task(
                tid, project_root=project_root,
                metadata=json.dumps({'task_kind': 'deterministic'}),
            )

        task = await backend.get_task(tid, project_root=project_root)
        assert task['metadata'] == {'foo': 'bar'}, (
            f'Expected the rolled-back txn to leave metadata unchanged; got: {task["metadata"]}'
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_update_task_tolerates_untouched_invalid_done_provenance(tmp_path, project_root):
    """Enforce-mode update_task scopes its raise to the fields the write touches (task 2401).

    Many ``done`` tasks carry a legacy ``metadata.done_provenance`` written
    before ``kind`` became a required field (e.g. ``{"commit": "abc123"}``).
    Under enforce mode, ``update_task`` validates the POST-MERGE blob — so
    without scoping, a legacy row would permanently reject *every* future
    metadata patch, even ones that never touch ``done_provenance``. This test
    proves the write-boundary gate only blocks writes that are themselves
    responsible for the invalid field (or a whole-blob invariant); an
    untouched legacy ``done_provenance`` is tolerated and preserved as-is,
    while a patch that itself TOUCHES ``done_provenance`` is still rejected.

    Note (task 2201): update_task now carries an UNCONDITIONAL
    write-authority floor that rejects any incoming metadata containing a
    ``done_provenance`` key with :class:`DoneProvenanceWriteAuthorityError`,
    raised BEFORE schema validation. So a patch that touches
    ``done_provenance`` — valid or invalid — is rejected by the stricter
    write-authority floor rather than the schema ``ValidationError``. The
    untouched-legacy tolerance in (a) is unaffected: the floor inspects only
    the incoming patch, not the post-merge blob.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        # Seed a pre-migration row: temporarily disable enforcement so the
        # legacy (missing 'kind') done_provenance blob can be planted at all —
        # enforce-mode add_task would otherwise reject it outright. This
        # faithfully reproduces a blob written before the schema tightened.
        backend._task_metadata_enforce = False
        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({'done_provenance': {'commit': 'abc123'}}),
        )
        backend._task_metadata_enforce = True

        # (a) An UNRELATED patch is tolerated: the untouched legacy
        # done_provenance survives, and the new field lands.
        await backend.update_task(
            dto['id'], project_root=project_root,
            metadata=json.dumps({'files': ['b.py']}),
        )
        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata']['files'] == ['b.py']
        assert task['metadata']['done_provenance'] == {'commit': 'abc123'}, (
            f'Expected the untouched legacy done_provenance to be preserved; '
            f'got: {task["metadata"].get("done_provenance")!r}'
        )

        # (b) Guard rail: a patch that itself TOUCHES done_provenance is
        # rejected. Under task 2201's unconditional write-authority floor this
        # is a DoneProvenanceWriteAuthorityError raised before schema
        # validation ever runs (previously a ValidationError on the invalid
        # blob).
        with pytest.raises(DoneProvenanceWriteAuthorityError):
            await backend.update_task(
                dto['id'], project_root=project_root,
                metadata=json.dumps({'done_provenance': {'commit': 'x'}}),
            )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_update_task_unknown_key_patch_tolerates_untouched_invalid_done_provenance(
    tmp_path, project_root,
):
    """Enforce-mode update_task tolerates an untouched legacy done_provenance
    even when the patch's OWN keys are themselves unrecognised (task 2405).

    Task 2401 scoped the enforce-mode raise to ``incoming_keys`` (the current
    write's own top-level keys), but ``should_reraise`` treated ANY warning
    on an incoming key as fatal — including a merely-informational
    ``unknown_key`` warning. A reconciliation-sidecar patch (e.g.
    ``{"540_status": ..., "duplicate_check_required": ...}``, the exact
    shape external projects such as autopilot-video attach to task
    metadata) is made entirely of such unrecognised keys: TaskMetadata's
    ``extra='allow'`` means ``unknown_key`` is NEVER fatal under
    ``enforce=True``. This test proves such a patch no longer trips the
    whole-blob re-validation pass onto an untouched legacy
    ``done_provenance`` (missing the now-required ``kind``), while a patch
    that itself touches ``done_provenance`` with a still-invalid value is
    still rejected.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        # Seed a pre-migration row carrying BOTH the legacy (missing 'kind')
        # done_provenance AND an unknown sidecar key, mirroring the real
        # autopilot_video:544 shape. Enforcement must be disabled to plant it.
        backend._task_metadata_enforce = False
        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({
                'done_provenance': {'commit': 'abc123'},
                'duplicate_check_required': True,
            }),
        )
        backend._task_metadata_enforce = True

        # PRIMARY (currently RED): a patch whose OWN keys are themselves
        # unknown/extra keys (the autopilot_video:544 shape) must not raise —
        # even though the row's untouched done_provenance is still missing
        # 'kind'.
        await backend.update_task(
            dto['id'], project_root=project_root,
            metadata=json.dumps({
                'duplicate_check_required': False,
                '540_status': 'cancelled',
            }),
            metadata_mode='merge',
        )
        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata']['done_provenance'] == {'commit': 'abc123'}, (
            f'Expected the untouched legacy done_provenance to be preserved; '
            f'got: {task["metadata"].get("done_provenance")!r}'
        )
        assert task['metadata']['duplicate_check_required'] is False, (
            f'Expected the unknown-key patch to land; got: '
            f'{task["metadata"].get("duplicate_check_required")!r}'
        )
        assert task['metadata']['540_status'] == 'cancelled', (
            f'Expected the unknown-key patch to land; got: '
            f'{task["metadata"].get("540_status")!r}'
        )

        # SECONDARY guard rail (green before and after — regression guard,
        # not the RED signal): a patch that ITSELF touches done_provenance
        # with a still-invalid value, alongside an unknown key, is still
        # rejected. Per task 2201's unconditional write-authority floor
        # (see test_update_task_tolerates_untouched_invalid_done_provenance
        # part (b)), ANY incoming metadata containing a done_provenance key
        # is rejected with DoneProvenanceWriteAuthorityError before schema
        # validation ever runs — so this is not a ValidationError.
        with pytest.raises(DoneProvenanceWriteAuthorityError):
            await backend.update_task(
                dto['id'], project_root=project_root,
                metadata=json.dumps({
                    'done_provenance': {'commit': 'x'},
                    'duplicate_check_required': True,
                }),
            )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_update_task_unknown_key_patch_still_rejects_invalid_known_field(
    tmp_path, project_root,
):
    """An unknown key in the SAME patch must not mask a genuinely-fatal
    warning on a different incoming KNOWN field (task 2405 regression guard).

    Task 2405's fix narrows ``should_reraise`` to ignore ``unknown_key``
    warnings on incoming keys — but ``unknown_key`` must be the ONLY code
    suppressed. This test pins that down: the patch carries an out-of-enum
    ``task_kind`` (a KNOWN ``TaskMetadata`` field, producing an
    ``invalid_field`` warning — a genuinely fatal code) alongside an
    unrelated unknown key (``duplicate_check_required``, producing a
    non-fatal ``unknown_key`` warning). If a future change widened
    ``_NON_FATAL_WRITE_WARNING_CODES`` too far, or the ``and w.code not in
    ...`` conjunct in ``_validate_metadata_on_write`` regressed to an ``or``,
    this would silently disable per-field enforcement — this test fails
    first.
    """
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg, task_metadata_enforce=True)
    await backend.start()
    try:
        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({'foo': 'bar'}),
        )

        with pytest.raises(ValidationError):
            await backend.update_task(
                dto['id'], project_root=project_root,
                metadata=json.dumps({
                    'task_kind': 'bogus_kind',
                    'duplicate_check_required': True,
                }),
                metadata_mode='merge',
            )

        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata'] == {'foo': 'bar'}, (
            f'Expected the rejected txn to leave metadata unchanged; got: {task["metadata"]}'
        )
    finally:
        await backend.close()


# ── read-path tolerance + collapse (task 2162, step-9/10) ─────────────


@pytest.mark.asyncio
async def test_row_to_task_coerces_valid_non_object_json_to_empty_dict(backend, project_root):
    """A valid-JSON-but-non-object metadata blob (e.g. '[1,2,3]') reads as {}.

    Net new behavior (task 2162): previously a bare JSON array round-tripped
    verbatim through the hand-rolled json.loads/try-except (no exception is
    raised by json.loads for a well-formed array), so it surfaced as a raw
    list. The shared read policy's 'not_an_object' code now flags this as
    malformed, coercing it — more correct for downstream dict-consumers
    (``(task.get('metadata') or {}).get(...)``-style callers). Verified via
    both get_task and get_tasks.
    """
    await backend.add_task(project_root=project_root, title='t')
    conn = await backend._get_connection(project_root)
    await conn.execute('UPDATE tasks SET metadata = ? WHERE id = 1', ('[1,2,3]',))
    await conn.commit()

    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {}

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'][0]['metadata'] == {}


@pytest.mark.asyncio
async def test_get_tasks_coerces_corrupt_json_to_empty_dict_with_one_warning(
    backend, project_root, caplog,
):
    """A corrupt (unparseable) metadata blob reads as {} via get_tasks, warning once.

    Regression safety net for the parse_metadata collapse: an unparseable
    blob must never raise out of get_tasks, must coerce to {}, and must still
    emit exactly one deduped 'malformed metadata' WARNING (the pre-existing
    _warn_malformed_metadata_once contract, reused unchanged on the collapsed
    read path).
    """
    await backend.add_task(project_root=project_root, title='t')
    conn = await backend._get_connection(project_root)
    await conn.execute('UPDATE tasks SET metadata = ? WHERE id = 1', ('{not json',))
    await conn.commit()

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        listing = await backend.get_tasks(project_root=project_root)

    assert listing['tasks'][0]['metadata'] == {}
    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 1, (
        f'Expected exactly one malformed-metadata WARNING; got '
        f'{len(malformed_msgs)}: {malformed_msgs}'
    )


@pytest.mark.asyncio
async def test_row_to_task_preserves_unknown_key_without_typed_defaults(backend, project_root):
    """A valid object with an unrecognised key round-trips exactly, untouched.

    No schema_version stamp or typed-field defaults (task_kind,
    always_escalates, before_done=None, external_deps=[], files=[]) are
    injected into the read — the read path surfaces the raw json.loads dict,
    never a parse_metadata(...).model_dump().
    """
    await backend.add_task(project_root=project_root, title='t')
    conn = await backend._get_connection(project_root)
    await conn.execute(
        'UPDATE tasks SET metadata = ? WHERE id = 1',
        ('{"prd": "x", "unknown_key": 1}',),
    )
    await conn.commit()

    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {'prd': 'x', 'unknown_key': 1}
