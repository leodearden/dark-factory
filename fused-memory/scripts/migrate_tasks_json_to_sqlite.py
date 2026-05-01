#!/usr/bin/env python3
"""One-shot migration: ``<project_root>/.taskmaster/tasks/tasks.json`` → sibling ``tasks.db``.

Idempotent — re-running re-creates the DB from the JSON in place.
Validates by reading the freshly-written DB back through
:class:`SqliteTaskBackend` and deep-comparing the resulting
``get_tasks`` payload against the source JSON (after normalising the
volatile fields the two layers handle differently).

Usage::

    uv run python -m scripts.migrate_tasks_json_to_sqlite \\
        /home/leo/src/dark-factory --replace

Multi-project::

    uv run python -m scripts.migrate_tasks_json_to_sqlite \\
        /path/to/proj-a /path/to/proj-b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow ``python migrate_tasks_json_to_sqlite.py`` from anywhere.
_ROOT = Path(__file__).resolve().parent.parent / 'src'
if _ROOT.exists() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import aiosqlite  # noqa: E402

from fused_memory.backends.sqlite_task_backend import (  # noqa: E402
    DEFAULT_TAG,
    SqliteTaskBackend,
    _SCHEMA_SQL,
    _TOP_LEVEL_SENTINEL,
)

logger = logging.getLogger('migrate_tasks_json_to_sqlite')


def _tasks_json_path(project_root: Path) -> Path:
    return project_root / '.taskmaster' / 'tasks' / 'tasks.json'


def _tasks_db_path(project_root: Path) -> Path:
    return project_root / '.taskmaster' / 'tasks' / 'tasks.db'


def _coerce_int(value: Any) -> int | None:
    """Coerce Taskmaster's mixed-int/string id fields to int (or None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or not s.lstrip('-').isdigit():
            return None
        return int(s)
    return None


def _coerce_dependencies(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        n = _coerce_int(item)
        if n is not None:
            out.append(n)
    return sorted(set(out))


async def _migrate_one(project_root: Path, *, replace: bool) -> bool:
    """Migrate a single project. Returns True on success."""
    json_path = _tasks_json_path(project_root)
    db_path = _tasks_db_path(project_root)
    if not json_path.exists():
        logger.warning('SKIP %s — no tasks.json', project_root)
        return False
    if db_path.exists() and not replace:
        logger.error(
            'REFUSE %s — tasks.db already exists; pass --replace to overwrite',
            project_root,
        )
        return False
    if db_path.exists() and replace:
        logger.info('REPLACE %s — removing existing tasks.db', db_path)
        db_path.unlink()
        # Also remove WAL/SHM siblings so the new connection sees a clean state.
        for sidecar in (db_path.with_suffix('.db-wal'), db_path.with_suffix('.db-shm')):
            if sidecar.exists():
                sidecar.unlink()

    try:
        with json_path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        logger.error('FAIL %s — could not read tasks.json: %s', project_root, exc)
        return False

    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute('PRAGMA journal_mode=WAL')
        await conn.execute('PRAGMA busy_timeout=5000')
        await conn.execute('PRAGMA synchronous=NORMAL')
        await conn.executescript(_SCHEMA_SQL)
        await conn.commit()

        task_count = 0
        subtask_count = 0
        dep_count = 0
        for tag_name, tag_obj in raw.items():
            if not isinstance(tag_obj, dict):
                continue
            tag = tag_name or DEFAULT_TAG
            tasks = tag_obj.get('tasks', [])
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                t_id = _coerce_int(task.get('id'))
                if t_id is None:
                    logger.warning(
                        'SKIP %s — task without numeric id: %r', project_root, task,
                    )
                    continue
                metadata_raw = task.get('metadata')
                metadata_blob = (
                    json.dumps(metadata_raw)
                    if isinstance(metadata_raw, (dict, list))
                    else (metadata_raw if isinstance(metadata_raw, str) else None)
                )
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks (
                        tag, id, parent_id, title, description, details,
                        test_strategy, status, priority, metadata, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tag, t_id, _TOP_LEVEL_SENTINEL,
                        str(task.get('title', '')),
                        str(task.get('description', '') or ''),
                        str(task.get('details', '') or ''),
                        str(task.get('testStrategy', '') or ''),
                        str(task.get('status', 'pending')),
                        str(task.get('priority', 'medium')),
                        metadata_blob,
                        str(task.get('updatedAt') or ''),
                    ),
                )
                task_count += 1

                for dep in _coerce_dependencies(task.get('dependencies')):
                    await conn.execute(
                        'INSERT OR IGNORE INTO dependencies '
                        '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                        (tag, t_id, _TOP_LEVEL_SENTINEL, dep),
                    )
                    dep_count += 1

                for sub in task.get('subtasks') or []:
                    if not isinstance(sub, dict):
                        continue
                    s_id = _coerce_int(sub.get('id'))
                    if s_id is None:
                        continue
                    await conn.execute(
                        """
                        INSERT OR REPLACE INTO tasks (
                            tag, id, parent_id, title, description, details,
                            test_strategy, status, priority, metadata, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, '', ?, NULL, NULL, ?)
                        """,
                        (
                            tag, s_id, t_id,
                            str(sub.get('title', '')),
                            str(sub.get('description', '') or ''),
                            str(sub.get('details', '') or ''),
                            str(sub.get('status', 'pending')),
                            str(sub.get('updatedAt') or ''),
                        ),
                    )
                    subtask_count += 1
                    for dep in _coerce_dependencies(sub.get('dependencies')):
                        await conn.execute(
                            'INSERT OR IGNORE INTO dependencies '
                            '(tag, task_id, parent_id, depends_on) VALUES (?, ?, ?, ?)',
                            (tag, s_id, t_id, dep),
                        )
                        dep_count += 1
        await conn.commit()

    logger.info(
        'OK %s — tasks=%d subtasks=%d deps=%d → %s',
        project_root, task_count, subtask_count, dep_count, db_path,
    )
    return await _validate(project_root, raw)


async def _validate(project_root: Path, source_json: dict[str, Any]) -> bool:
    """Read the migrated DB back and deep-compare against the source JSON.

    Volatile fields (``updatedAt``, ``metadata`` JSON encoding, missing
    optional fields like ``priority``) are normalised on both sides so the
    comparison only fails on actual data loss / shape drift.
    """
    backend = SqliteTaskBackend(config=None)
    await backend.start()
    try:
        ok = True
        for tag_name, tag_obj in source_json.items():
            if not isinstance(tag_obj, dict):
                continue
            tag = tag_name or DEFAULT_TAG
            source_tasks = tag_obj.get('tasks', [])
            if not isinstance(source_tasks, list):
                continue
            result = await backend.get_tasks(str(project_root), tag=tag)
            db_tasks = result['tasks']
            if not _compare_task_lists(source_tasks, db_tasks):
                logger.error(
                    'VALIDATE FAILED for project=%s tag=%s', project_root, tag,
                )
                ok = False
        if ok:
            logger.info('VALIDATE OK %s', project_root)
        return ok
    finally:
        await backend.close()


def _norm_top(t: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': str(_coerce_int(t.get('id')) or 0),
        'title': str(t.get('title', '')),
        'description': str(t.get('description', '') or ''),
        'details': str(t.get('details', '') or ''),
        'status': str(t.get('status', 'pending')),
        'priority': str(t.get('priority', 'medium') or 'medium'),
        'dependencies': _coerce_dependencies(t.get('dependencies')),
        'subtasks': [_norm_sub(s) for s in (t.get('subtasks') or [])],
    }


def _norm_sub(s: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': _coerce_int(s.get('id')) or 0,
        'title': str(s.get('title', '')),
        'description': str(s.get('description', '') or ''),
        'details': str(s.get('details', '') or ''),
        'status': str(s.get('status', 'pending')),
        'dependencies': _coerce_dependencies(s.get('dependencies')),
    }


def _compare_task_lists(
    source: list[dict[str, Any]], db: list[dict[str, Any]],
) -> bool:
    src_norm = sorted((_norm_top(t) for t in source), key=lambda x: x['id'])
    db_norm = sorted((_norm_top(t) for t in db), key=lambda x: x['id'])
    if src_norm == db_norm:
        return True
    src_ids = {t['id'] for t in src_norm}
    db_ids = {t['id'] for t in db_norm}
    if src_ids != db_ids:
        logger.error(
            'task-id mismatch — source-only=%s db-only=%s',
            sorted(src_ids - db_ids), sorted(db_ids - src_ids),
        )
    else:
        for src_t, db_t in zip(src_norm, db_norm):
            if src_t != db_t:
                logger.error('first-mismatch id=%s src=%s db=%s', src_t['id'], src_t, db_t)
                break
    return False


async def main_async(project_roots: list[Path], *, replace: bool) -> int:
    failures = 0
    for project_root in project_roots:
        try:
            ok = await _migrate_one(project_root.resolve(), replace=replace)
        except Exception:
            logger.exception('FAIL %s — unhandled exception', project_root)
            failures += 1
            continue
        if not ok:
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'project_roots', nargs='+', type=Path,
        help='One or more project roots containing .taskmaster/tasks/tasks.json',
    )
    parser.add_argument(
        '--replace', action='store_true',
        help='Overwrite an existing tasks.db (otherwise the script refuses)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show debug logs',
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    failures = asyncio.run(main_async(args.project_roots, replace=args.replace))
    if failures:
        logger.error('migration failed for %d project(s)', failures)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
