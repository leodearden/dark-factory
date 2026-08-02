#!/usr/bin/env python3
"""Migrate live tasks: ``metadata.modules`` → ``metadata.files``.

Run this BEFORE landing the principled-files-rename code change. Idempotent —
safe to re-run.

Behavior per task carrying ``metadata.modules``:

- If ``metadata.files`` is missing or empty: copy ``modules`` into ``files``,
  then drop ``modules``.
- If ``metadata.files`` is non-empty: keep ``files`` as authoritative, drop
  ``modules``.

Talks to the running fused-memory MCP server over JSON-RPC. Discovers project
roots from ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated). Falls back to
the dark-factory root.

Usage:
    python scripts/migrate_metadata_modules_to_files.py [--dry-run] \\
        [--server-url http://127.0.0.1:8002]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any

import httpx

DEFAULT_SERVER = 'http://127.0.0.1:8002'
DEFAULT_ROOTS = ['/home/leo/src/dark-factory']


def _flatten_tasks(tasks_payload: Any) -> list[dict]:
    out: list[dict] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for t in items:
            if isinstance(t, dict):
                out.append(t)
                _walk(t.get('subtasks') or [])

    if isinstance(tasks_payload, dict):
        _walk(tasks_payload.get('tasks'))
    return out


class FusedMemoryClient:
    """Minimal HTTP/JSON-RPC client for the fused-memory MCP server."""

    #: clientInfo.name sent at handshake. LOAD-BEARING, not cosmetic:
    #: fused_memory/server/tools.py `_resolve_identity` derives a write's
    #: agent_id from ctx.session.client_params.clientInfo.name, so a subclass
    #: that does not set this files its writes under THIS migration's identity.
    #: Override it in a subclass — do NOT restate `_initialize` to change it.
    _client_name: str = 'migrate-metadata'

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
                'clientInfo': {'name': self._client_name, 'version': '1.0'},
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


async def _migrate_one_project(
    client: FusedMemoryClient, project_root: str, *, dry_run: bool,
) -> tuple[int, int, int]:
    """Walk one project's tasks and migrate metadata. Returns (visited, copied, dropped_only)."""
    try:
        tasks_result = await client.call_tool(
            'get_tasks', {'project_root': project_root, 'with_subtasks': True},
        )
    except Exception as exc:
        print(f'  [skip] get_tasks failed for {project_root}: {exc}', file=sys.stderr)
        return (0, 0, 0)

    tasks = _flatten_tasks(tasks_result)
    # Done/cancelled tasks are immutable to update_task — skip them so the
    # migration log only reports tasks the server will actually touch.
    skip_statuses = {'done', 'cancelled', 'deferred'}
    visited = 0
    copied = 0
    dropped_only = 0

    for task in tasks:
        if str(task.get('status', '')) in skip_statuses:
            continue
        visited += 1
        meta = task.get('metadata')
        if not isinstance(meta, dict):
            continue
        modules = meta.get('modules')
        if not modules:
            continue
        files = meta.get('files') or []
        task_id = str(task.get('id', ''))

        # Build a full new metadata dict (None-deletion via merge doesn't
        # work — a merge/append=True keeps existing keys). We copy the existing
        # metadata, mutate, and write the COMPLETE result under the explicit
        # metadata_mode='replace' co-signal — a bare append=False is now
        # rejected by the task-2180 metadata-wipe guard in
        # _resolve_metadata_mode.
        # Strip done_provenance — update_task rejects metadata writes that
        # carry it (set_task_status is the only sanctioned writer); we
        # already skip done/cancelled, but be defensive about orphan stamps.
        new_meta = {
            k: v for k, v in meta.items()
            if k not in ('modules', 'done_provenance')
        }
        if not files:
            new_meta['files'] = modules
            action = 'copy'
        else:
            action = 'drop'

        if dry_run:
            print(
                f'  [dry-run][{project_root}] task={task_id} action={action} '
                f'modules={modules!r} files={files!r}'
            )
        else:
            try:
                await client.call_tool('update_task', {
                    'id': task_id,
                    'project_root': project_root,
                    'metadata': json.dumps(new_meta),
                    'metadata_mode': 'replace',
                })
                print(
                    f'  [{project_root}] task={task_id} action={action}',
                )
            except Exception as exc:
                print(
                    f'  [error][{project_root}] task={task_id} update_task '
                    f'failed: {exc}',
                    file=sys.stderr,
                )
                continue

        if action == 'copy':
            copied += 1
        else:
            dropped_only += 1

    return (visited, copied, dropped_only)


async def main_async(args: argparse.Namespace) -> int:
    roots_env = os.environ.get('DASHBOARD_KNOWN_PROJECT_ROOTS', '').strip()
    if args.project_roots:
        roots = list(args.project_roots)
    elif roots_env:
        roots = [p.strip() for p in roots_env.split(',') if p.strip()]
    else:
        roots = DEFAULT_ROOTS

    print(f'Server:   {args.server_url}')
    print(f'Roots:    {roots}')
    print(f'Dry-run:  {args.dry_run}')
    print()

    totals = {'visited': 0, 'copied': 0, 'dropped': 0}
    async with FusedMemoryClient(args.server_url) as client:
        for root in roots:
            print(f'Project: {root}')
            visited, copied, dropped = await _migrate_one_project(
                client, root, dry_run=args.dry_run,
            )
            totals['visited'] += visited
            totals['copied'] += copied
            totals['dropped'] += dropped
            print(
                f'  visited={visited} copied_modules→files={copied} '
                f'dropped_modules_only={dropped}'
            )
            print()

    print('---- summary ----')
    for k, v in totals.items():
        print(f'  {k}: {v}')
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Report changes without applying.')
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER})',
    )
    parser.add_argument(
        '--project-root', dest='project_roots', action='append',
        help='Project root to migrate. May be repeated. Falls back to '
        '$DASHBOARD_KNOWN_PROJECT_ROOTS or the dark-factory root.',
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == '__main__':
    main()
