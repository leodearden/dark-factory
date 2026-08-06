#!/usr/bin/env python3
"""Retire ad-hoc task-metadata keys by renaming them into the `x_` namespace.

One-off migration for task 3697. Task 3083 accumulated a set of bespoke,
timestamped metadata keys (an evidence log parked under
``markup_tripwire_rejections_<date>`` spellings, plus several
``origin_*``/``related_*`` annotations). None of them is a typed
``TaskMetadata`` field or a Tier-A blessed key, so every write touching that
task emits a ``code=unknown_key`` schema warning per key — measured at SEVEN
on 3083's live blob (2026-08-06).

Per the Tier-C rule in ``docs/task-authoring.md`` §8, an ad-hoc key with no
code reader is retired by renaming it under ``x_``, which the schema treats
as an explicitly-namespaced escape hatch and does not warn about. Verified
by grep across ``orchestrator/``, ``fused-memory/src``, ``shared/src``,
``escalation/``, ``dashboard/``, ``scripts/`` and ``docs/``: the six default
targets have no reader anywhere. (The seventh warning on 3083,
``last_blocked_at``, is NOT migrated — the orchestrator both writes and reads
it, so it was promoted to Tier-A ``_BLESSED_METADATA_KEYS`` instead.)

WHY ``metadata_mode='replace'`` AND NOT ``'merge'``
---------------------------------------------------
Merge cannot retire a key. ``SqliteTaskBackend._merge_metadata`` implements
mode='merge' as a shallow ``{**old, **new}`` with NO deletion sentinel, so it
can ADD ``x_<key>`` but can never REMOVE the old spelling — both would coexist
and the ``unknown_key`` warnings this migration exists to eliminate would
persist. ``update_task``'s own docstring names ``metadata_mode='replace'``
"the sanctioned repair path" for exactly this reason, and
``scripts/strip_leaked_control_keys.py`` (whose structure this script copies)
reached the same conclusion for the same reason: merge "does NOT retroactively
self-heal existing rows".

The thing merge was protecting — sibling metadata keys must survive — is
achieved here by construction (the full blob that was just read is written
back, so siblings survive) and then PROVEN by a mandatory read-back diff of
every key and value. That is evidence, where merge would have given trust.

SAFETY
------
The write payload carries ONLY ``{id, project_root, metadata, metadata_mode}``.
``update_task`` rewrites exactly the columns it is handed, and task 3083's
``description`` (6269 B) and ``details`` (13965 B) hold a one-of-a-kind
hand-curated evidence log. ``build_update_payload`` is the single place that
payload is constructed, and
``tests/test_migrate_task_metadata_to_x_namespace.py`` asserts each dangerous
column absent by name. The post-write read-back additionally verifies the
sha256 of both columns is unchanged.

Idempotent — safe to re-run; an already-migrated task is a no-op. Defaults to
a dry run: a real write requires an explicit ``--apply``.

Usage:
    # rehearse (default)
    python fused-memory/scripts/migrate_task_metadata_to_x_namespace.py --task-id 3083
    # apply
    python fused-memory/scripts/migrate_task_metadata_to_x_namespace.py --task-id 3083 --apply
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Sequence
from pathlib import Path

DEFAULT_SERVER = 'http://127.0.0.1:8002'
DEFAULT_PROJECT_ROOT = '/home/leo/src/dark-factory'

# The six ad-hoc keys measured on task 3083's live blob (2026-08-06) that have
# no code reader anywhere in the repo. `last_blocked_at` is deliberately NOT
# here: it is machine-written and machine-read, and was promoted to Tier-A.
DEFAULT_KEYS: tuple[str, ...] = (
    'markup_tripwire_rejections_20260730',
    'markup_tripwire_rejections_20260730_burst3',
    'related_reify_memories',
    'related_reify_tasks',
    'origin_escalation',
    'origin_reify_task',
)


class MigrationCollisionError(RuntimeError):
    """Both ``k`` and ``x_k`` exist with DIFFERENT values.

    Refusing is the only safe move: silently merging would fabricate a
    record that never existed, and discarding either side would destroy
    evidence. A human decides.
    """


def plan_x_namespace_migration(
    meta: dict, keys: Sequence[str],
) -> tuple[dict, list[str]]:
    """Pure transform: rename each present ``k`` in *keys* to ``x_k``.

    Returns ``(migrated_meta, sorted(applied_keys))``. Non-target keys are
    preserved with identical values, key order is otherwise stable, and the
    caller's dict is never mutated (the caller keeps it as the pre-write
    snapshot to diff the read-back against).

    An absent target key is a silent no-op, so a partially-migrated row is
    recoverable. A target whose ``x_`` form already holds an EQUAL value
    collapses to just the ``x_`` form; an unequal one raises
    :class:`MigrationCollisionError`.
    """
    targets = [k for k in keys if k in meta]

    for key in targets:
        new_key = f'x_{key}'
        if new_key in meta and meta[new_key] != meta[key]:
            raise MigrationCollisionError(
                f'refusing to migrate {key!r}: {new_key!r} already exists with a '
                f'different value ({meta[new_key]!r} != {meta[key]!r}). '
                f'Resolve by hand — never silently merge two evidence records.'
            )

    target_set = set(targets)
    migrated: dict = {}
    for key, value in meta.items():
        if key in target_set:
            migrated[f'x_{key}'] = value
        elif key.startswith('x_') and key[2:] in target_set:
            # The equal-valued collision case: the x_ form is already correct
            # and is (re)written by the branch above, so skip the duplicate.
            continue
        else:
            migrated[key] = value

    return migrated, sorted(targets)


def build_update_payload(task_id: str, project_root: str, meta: dict) -> dict:
    """Build the ``update_task`` arguments — ONLY the metadata columns.

    The single place this payload is constructed, deliberately. ``update_task``
    rewrites exactly the columns it is given, so a stray ``description`` or
    ``details`` key here would overwrite task 3083's one-of-a-kind curated
    evidence log. Never add a field to this dict.

    ``metadata`` is serialized with ``json.dumps``, matching the replace-mode
    write in ``strip_leaked_control_keys.py``.
    """
    return {
        'id': task_id,
        'project_root': project_root,
        'metadata': json.dumps(meta),
        'metadata_mode': 'replace',
    }


def _sha256(text: str | None) -> str:
    return hashlib.sha256((text or '').encode()).hexdigest()


def _load_sibling_client() -> type:
    """Reuse ``FusedMemoryClient`` from ``strip_leaked_control_keys.py``.

    Loaded lazily and by path: the sibling script imports the fused_memory
    backend package at module scope, and keeping that off this module's import
    path lets the pure transform, the payload builder and the CLI parser be
    imported (and unit-tested) with no heavy dependencies. Deliberately NOT a
    second protocol client — there is one JSON-RPC handshake in this repo.
    """
    sibling = Path(__file__).parent / 'strip_leaked_control_keys.py'
    mod_name = 'strip_leaked_control_keys'
    existing = sys.modules.get(mod_name)
    if existing is not None and hasattr(existing, 'FusedMemoryClient'):
        return existing.FusedMemoryClient
    spec = importlib.util.spec_from_file_location(mod_name, sibling)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the JSON-RPC client from {sibling}')
    module: types.ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module.FusedMemoryClient


async def _fetch_task(client, task_id: str, project_root: str) -> dict:
    task = await client.call_tool(
        'get_task', {'id': task_id, 'project_root': project_root},
    )
    if not isinstance(task, dict) or 'id' not in task:
        raise RuntimeError(f'get_task returned an unexpected payload: {task!r}')
    return task


def _coerce_metadata(task: dict) -> dict:
    meta = task.get('metadata')
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not isinstance(meta, dict):
        raise RuntimeError(f'task {task.get("id")!r} has no dict metadata: {meta!r}')
    return meta


def _verify_read_back(
    *,
    before_meta: dict,
    expected_meta: dict,
    applied: list[str],
    before_task: dict,
    after_task: dict,
) -> list[str]:
    """Prove the write did exactly what was intended and nothing else.

    Returns a list of human-readable mismatches — empty means verified.
    Checks, in order: (a) every ``x_<key>`` present with the pre-write value,
    (b) every old spelling gone, (c) every untouched key byte-identical,
    (d) description/details sha256 unchanged, (e) status unchanged.
    """
    problems: list[str] = []
    after_meta = _coerce_metadata(after_task)

    for key in applied:
        new_key = f'x_{key}'
        if new_key not in after_meta:
            problems.append(f'(a) {new_key} missing from the stored blob')
        elif after_meta[new_key] != before_meta[key]:
            problems.append(
                f'(a) {new_key} value changed: {after_meta[new_key]!r} != {before_meta[key]!r}'
            )
        if key in after_meta:
            problems.append(f'(b) old spelling {key} still present')

    if set(after_meta) != set(expected_meta):
        missing = sorted(set(expected_meta) - set(after_meta))
        extra = sorted(set(after_meta) - set(expected_meta))
        problems.append(f'(c) key-set drift: missing={missing} unexpected={extra}')
    for key in sorted(set(after_meta) & set(expected_meta)):
        if after_meta[key] != expected_meta[key]:
            problems.append(f'(c) value drift on {key}')

    for column in ('description', 'details'):
        before_sha = _sha256(before_task.get(column))
        after_sha = _sha256(after_task.get(column))
        if before_sha != after_sha:
            problems.append(
                f'(d) {column} CHANGED: {before_sha} -> {after_sha} '
                f'({len(before_task.get(column) or "")} -> '
                f'{len(after_task.get(column) or "")} bytes)'
            )

    if before_task.get('status') != after_task.get('status'):
        problems.append(
            f'(e) status changed: {before_task.get("status")!r} -> {after_task.get("status")!r}'
        )

    return problems


async def main_async(args: argparse.Namespace) -> int:
    project_root = args.project_root or os.environ.get(
        'DARK_FACTORY_PROJECT_ROOT', DEFAULT_PROJECT_ROOT,
    )
    keys = list(args.keys)

    print(f'Server:       {args.server_url}')
    print(f'Project root: {project_root}')
    print(f'Task ID:      {args.task_id}')
    print(f'Keys:         {keys}')
    print(f'Mode:         {"APPLY" if args.apply else "dry-run"}')
    print()

    client_cls = _load_sibling_client()
    async with client_cls(args.server_url) as client:
        before_task = await _fetch_task(client, args.task_id, project_root)
        before_meta = _coerce_metadata(before_task)

        print(f'  before: {len(before_meta)} metadata keys, status={before_task.get("status")!r}')
        print(
            f'          description={len(before_task.get("description") or "")}B '
            f'sha={_sha256(before_task.get("description"))[:12]} '
            f'details={len(before_task.get("details") or "")}B '
            f'sha={_sha256(before_task.get("details"))[:12]}'
        )

        try:
            migrated, applied = plan_x_namespace_migration(before_meta, keys)
        except MigrationCollisionError as exc:
            print(f'  [error] {exc}', file=sys.stderr)
            return 1

        if not applied:
            print('  no-op (already migrated, or no target key present)')
            return 0

        print(f'  would rename ({len(applied)}): {applied}')
        untouched = sorted(k for k in migrated if not k.startswith('x_'))
        print(f'  untouched ({len(untouched)}): {untouched}')

        if not args.apply:
            print()
            print('  DRY RUN — nothing written. Re-run with --apply to write.')
            return 0

        await client.call_tool(
            'update_task', build_update_payload(args.task_id, project_root, migrated),
        )
        print('  write submitted; verifying read-back...')

        after_task = await _fetch_task(client, args.task_id, project_root)
        problems = _verify_read_back(
            before_meta=before_meta,
            expected_meta=migrated,
            applied=applied,
            before_task=before_task,
            after_task=after_task,
        )
        if problems:
            print('  [error] READ-BACK VERIFICATION FAILED:', file=sys.stderr)
            for problem in problems:
                print(f'    - {problem}', file=sys.stderr)
            return 1

        print('  read-back verified: x_ keys present, old spellings gone, '
              'siblings identical, description/details sha256 unchanged, status unchanged.')
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # REQUIRED, no default: these spellings live on other tasks too
    # (origin_escalation on 19, related_reify_tasks on 8, origin_reify_task on
    # 4, as of 2026-08-06), so a defaulted id would invite a corpus-wide sweep.
    parser.add_argument(
        '--task-id', dest='task_id', required=True,
        help='Task id to migrate (REQUIRED — no corpus-wide default).',
    )
    parser.add_argument(
        '--keys', dest='keys', nargs='+', default=list(DEFAULT_KEYS),
        help='Metadata keys to rename under x_ (default: the six task-3083 targets).',
    )
    parser.add_argument(
        '--project-root', dest='project_root', default=None,
        help=f'Project root (default: $DARK_FACTORY_PROJECT_ROOT or {DEFAULT_PROJECT_ROOT}).',
    )
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER})',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually write. Without it the script only rehearses (dry run).',
    )
    return parser


def main() -> None:
    sys.exit(asyncio.run(main_async(build_parser().parse_args())))


if __name__ == '__main__':
    main()
