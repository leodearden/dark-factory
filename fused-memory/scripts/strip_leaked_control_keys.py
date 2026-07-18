#!/usr/bin/env python3
"""Clean up leaked update_task control-flag keys from live task metadata.

One-off cleanup for task 2735: the Stage-2 recon LLM conflated update_task's
CALL-FLAGS (``append``, ``metadata_mode``) with actual metadata fields,
calling ``update_task(metadata={..., "append": True}, append=True)``.
``_merge_metadata`` faithfully persists every key in the incoming payload,
so the literal ``append`` key leaked into the stored blob (task 2682).

The task 2735 backend fix (``SqliteTaskBackend.update_task`` ->
``_strip_reserved_control_keys``) prevents this going forward by stripping
reserved control-flag keys from the INCOMING payload before the merge runs
— but it does NOT retroactively self-heal existing rows (merge preserves
untouched keys, and silently mutating an untouched key on an unrelated
write would violate the merge's omitted-keys-preserved contract). This
script explicitly corrects already-dirty rows via a replace-mode write of
the corrected full blob.

Idempotent — safe to re-run; a task with no reserved keys in its metadata
is a no-op.

Talks to the running fused-memory MCP server over JSON-RPC, mirroring
scripts/migrate_metadata_modules_to_files.py. Discovers project roots from
``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated), falling back to the
dark-factory root.

Usage:
    python fused-memory/scripts/strip_leaked_control_keys.py [--dry-run] \\
        [--server-url http://127.0.0.1:8002] [--task-ids 2682,2683]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

import httpx

from fused_memory.backends.sqlite_task_backend import _drop_reserved_control_keys

DEFAULT_SERVER = 'http://127.0.0.1:8002'
DEFAULT_ROOTS = ['/home/leo/src/dark-factory']
DEFAULT_TASK_IDS = '2682'


def correct_task_metadata(meta: dict) -> tuple[dict, list[str]]:
    """Pure transform: drop reserved update_task control-flag keys.

    Reuses the backend's reserved-key set — single source of truth is
    :func:`fused_memory.backends.sqlite_task_backend._drop_reserved_control_keys`
    — rather than redefining it here. Returns ``(cleaned_meta,
    sorted(dropped_keys))``.
    """
    cleaned, dropped = _drop_reserved_control_keys(meta)
    return cleaned, sorted(dropped)


class FusedMemoryClient:
    """Minimal HTTP/JSON-RPC client for the fused-memory MCP server."""

    def __init__(self, server_url: str):
        self._url = server_url.rstrip('/')
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None

    async def __aenter__(self) -> FusedMemoryClient:
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._session_id = uuid.uuid4().hex
        await self._initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        assert self._client is not None
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'mcp-session-id': self._session_id or '',
        }
        resp = await self._client.post(f'{self._url}/mcp/', json=payload, headers=headers)
        resp.raise_for_status()
        # 202 Accepted (notifications) returns no body.
        if resp.status_code == 202 or not resp.content:
            return {}
        ctype = resp.headers.get('content-type', '')
        if 'text/event-stream' in ctype:
            for line in resp.text.splitlines():
                if line.startswith('data:'):
                    return json.loads(line[5:].strip())
            raise RuntimeError(f'no SSE data line in response: {resp.text[:200]}')
        return resp.json()

    async def _initialize(self) -> None:
        await self._post({
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {'name': 'strip-leaked-control-keys', 'version': '1.0'},
                'capabilities': {},
            },
        })
        await self._post({
            'jsonrpc': '2.0',
            'method': 'notifications/initialized',
            'params': {},
        })

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._post({
            'jsonrpc': '2.0',
            'id': uuid.uuid4().hex,
            'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })
        if 'error' in result:
            raise RuntimeError(f'{name} failed: {result["error"]}')
        # MCP tools/call returns content list; structuredContent is the parsed dict
        content = result.get('result', {})
        if 'structuredContent' in content:
            return content['structuredContent']
        # Fall back to first text content
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {'_raw': entry['text']}
        return content


async def _clean_one_project(
    client: FusedMemoryClient, project_root: str, task_ids: list[str], *, dry_run: bool,
) -> tuple[int, int, int]:
    """Visit each requested task id in one project. Returns (visited, cleaned, errors)."""
    visited = 0
    cleaned_count = 0
    error_count = 0

    for task_id in task_ids:
        try:
            task = await client.call_tool(
                'get_task', {'id': task_id, 'project_root': project_root},
            )
        except Exception as exc:
            print(f'  [skip] get_task failed for {project_root}#{task_id}: {exc}', file=sys.stderr)
            error_count += 1
            continue

        visited += 1
        meta = task.get('metadata')
        if not isinstance(meta, dict):
            print(f'  [{project_root}] task={task_id} no-op (no dict metadata)')
            continue

        cleaned, dropped = correct_task_metadata(meta)
        if not dropped:
            print(f'  [{project_root}] task={task_id} no-op (clean)')
            continue

        # Safety: correct_task_metadata must differ from the fetched blob
        # ONLY by removing the reported dropped keys — never touch or drop
        # any other key. Refuse the replace-mode write (rather than
        # silently persist a truncated blob) if that invariant doesn't
        # hold, e.g. because get_task's returned metadata was already
        # unexpectedly partial.
        expected = {k: v for k, v in meta.items() if k not in dropped}
        if cleaned != expected:
            print(
                f'  [error][{project_root}] task={task_id} refusing to write: '
                f'cleaned metadata differs from source by more than the '
                f'dropped keys {dropped} (before={meta!r} cleaned={cleaned!r})',
                file=sys.stderr,
            )
            error_count += 1
            continue

        if dry_run:
            print(
                f'  [dry-run][{project_root}] task={task_id} would drop {dropped} '
                f'before={meta!r} after={cleaned!r}'
            )
        else:
            try:
                await client.call_tool('update_task', {
                    'id': task_id,
                    'project_root': project_root,
                    'metadata': json.dumps(cleaned),
                    'metadata_mode': 'replace',
                })
                print(
                    f'  [{project_root}] task={task_id} dropped {dropped} '
                    f'before={meta!r} after={cleaned!r}'
                )
            except Exception as exc:
                print(
                    f'  [error][{project_root}] task={task_id} update_task '
                    f'failed: {exc}',
                    file=sys.stderr,
                )
                error_count += 1
                continue
        cleaned_count += 1

    return (visited, cleaned_count, error_count)


async def main_async(args: argparse.Namespace) -> int:
    roots_env = os.environ.get('DASHBOARD_KNOWN_PROJECT_ROOTS', '').strip()
    if args.project_roots:
        roots = list(args.project_roots)
    elif roots_env:
        roots = [p.strip() for p in roots_env.split(',') if p.strip()]
    else:
        roots = DEFAULT_ROOTS

    task_ids = [t.strip() for t in args.task_ids.split(',') if t.strip()]

    print(f'Server:   {args.server_url}')
    print(f'Roots:    {roots}')
    print(f'Task IDs: {task_ids}')
    print(f'Dry-run:  {args.dry_run}')
    print()

    totals = {'visited': 0, 'cleaned': 0, 'errors': 0}
    async with FusedMemoryClient(args.server_url) as client:
        for root in roots:
            print(f'Project: {root}')
            visited, cleaned_count, error_count = await _clean_one_project(
                client, root, task_ids, dry_run=args.dry_run,
            )
            totals['visited'] += visited
            totals['cleaned'] += cleaned_count
            totals['errors'] += error_count
            print(f'  visited={visited} cleaned={cleaned_count} errors={error_count}')
            print()

    print('---- summary ----')
    for k, v in totals.items():
        print(f'  {k}: {v}')
    # Non-zero exit on any per-task failure (get_task/update_task errors,
    # or the cleaned-vs-source safety check above) so an automated caller
    # can detect partial failure instead of reading a bare 0 as success.
    return 1 if totals['errors'] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Report changes without applying.')
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER})',
    )
    parser.add_argument(
        '--project-root', dest='project_roots', action='append',
        help='Project root to clean. May be repeated. Falls back to '
        '$DASHBOARD_KNOWN_PROJECT_ROOTS or the dark-factory root.',
    )
    parser.add_argument(
        '--task-ids', dest='task_ids', default=DEFAULT_TASK_IDS,
        help=f'Comma-separated task ids to clean (default: {DEFAULT_TASK_IDS!r}).',
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == '__main__':
    main()
