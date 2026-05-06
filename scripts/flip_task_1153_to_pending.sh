#!/usr/bin/env bash
# flip_task_1153_to_pending.sh — Post-cutover cleanup task auto-flip.
#
# Fired by flip-task-1153.timer at 2026-05-10 10:04 BST. Flips task 1153
# (post-cutover Taskmaster + dual_compare retirement) from `deferred` to
# `pending` so the orchestrator can pick it up. The 8-day deferral
# represents the stability soak between SQLite cutover (2026-05-02) and
# dead-code removal — see task 1153's description.
#
# Idempotent: if the task is no longer `deferred` (already flipped, done,
# or cancelled), exits 0 without modifying anything.
#
# This file was reconstructed on 2026-05-06 after the original was
# deleted in a housekeeping pass; the timer (transient systemd-run unit)
# pointed at this exact path, so re-creating the file was the cheapest
# fix vs. cancelling and re-scheduling the timer.
set -euo pipefail

exec /home/leo/src/dark-factory/.venv/bin/python - <<'PY'
"""Flip task 1153 deferred → pending via the fused-memory MCP."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid

import httpx

URL = 'http://127.0.0.1:8002'
PROJECT_ROOT = '/home/leo/src/dark-factory'
TASK_ID = '1153'


async def main() -> int:
    session_id = uuid.uuid4().hex
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'mcp-session-id': session_id,
    }

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:

        async def post(payload: dict) -> dict:
            resp = await client.post(f'{URL}/mcp/', json=payload, headers=headers)
            resp.raise_for_status()
            if resp.status_code == 202 or not resp.content:
                return {}
            if 'text/event-stream' in resp.headers.get('content-type', ''):
                for line in resp.text.splitlines():
                    if line.startswith('data:'):
                        return json.loads(line[5:].strip())
                raise RuntimeError(f'no SSE data line: {resp.text[:200]}')
            return resp.json()

        await post({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {'name': 'flip-task-1153', 'version': '1.0'},
                'capabilities': {},
            },
        })
        await post({
            'jsonrpc': '2.0', 'method': 'notifications/initialized',
            'params': {},
        })

        async def call(name: str, args: dict) -> dict:
            res = await post({
                'jsonrpc': '2.0', 'id': uuid.uuid4().hex,
                'method': 'tools/call',
                'params': {'name': name, 'arguments': args},
            })
            if 'error' in res:
                raise RuntimeError(f'{name} failed: {res["error"]}')
            content = res.get('result', {})
            if 'structuredContent' in content:
                return content['structuredContent']
            for entry in content.get('content', []) or []:
                if entry.get('type') == 'text':
                    try:
                        return json.loads(entry['text'])
                    except json.JSONDecodeError:
                        return {'_raw': entry['text']}
            return content

        task = await call(
            'get_task',
            {'id': TASK_ID, 'project_root': PROJECT_ROOT},
        )
        status = task.get('status')
        print(f'[flip-task-1153] task {TASK_ID} current status: {status!r}')

        if status != 'deferred':
            print(
                f'[flip-task-1153] task {TASK_ID} is not deferred '
                f'(status={status!r}) — nothing to do; exiting 0'
            )
            return 0

        await call(
            'set_task_status',
            {'id': TASK_ID, 'status': 'pending', 'project_root': PROJECT_ROOT},
        )
        print(f'[flip-task-1153] task {TASK_ID} flipped: deferred → pending')
        return 0


sys.exit(asyncio.run(main()))
PY
