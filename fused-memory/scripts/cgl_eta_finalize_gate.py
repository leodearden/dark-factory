#!/usr/bin/env python3
"""CGL-η gate finalizer — auto-close esc-2273-1 after a CLEAN bulk apply.

Called by the wrapper ONLY when the apply impl exited 0 (post-verify clean:
0 foreign :Entity residual). Marks the Phase-1 gate task 2273 `done` and then
dismisses its now-satisfied born-at-L2 escalation esc-2273-1.

Order matters and is deliberate:
  1. set_task_status(2273, done)  — terminal state FIRST, so there is never a
     blocked-without-open-escalation window for the stranded-blocked reconciler
     to reclaim + re-dispatch 2273 (which, as a pure gate, would just re-escalate).
  2. resolve_issue(esc-2273-1, close_only) — dismiss the stale escalation on the
     now-done task (`close_only` = no status effect; the done in step 1 is what
     matters). NB: no resolve *action* yields `done` (ACTION_EFFECTS maps
     resume/restart->pending, close_only->none, abandon->cancelled), which is
     exactly why step 1 goes through fused-memory set_task_status instead.

Best-effort by design: every call is guarded and this script ALWAYS exits 0.
A finalize miss must never flip a successful, verified migration into a
predicate escalation — if it fails, esc-2273-1 simply stays pending for the
escalation-watcher / operator (a done-task L2 is the benign stale-escalation
case a watcher self-heals). Idempotent: re-running on a predicate resume after
2273 is already done / esc already closed is a harmless no-op.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgl_eta_scheduler_gate import McpClient  # noqa: E402  (self-contained MCP/JSON-RPC client)

REPO = '/home/leo/src/dark-factory'
FUSED_URL = os.environ.get('CGL_FUSED_URL', 'http://127.0.0.1:8002')
ESC_URL = os.environ.get('CGL_ESC_URL', 'http://127.0.0.1:8102')
GATE_TASK = os.environ.get('CGL_GATE_TASK', '2273')
GATE_ESC = os.environ.get('CGL_GATE_ESC', 'esc-2273-1')
STAMP = os.environ.get('CGL_RUN_STAMP', 'autorun')


def _log(msg: str) -> None:
    print(f'[cgl-finalize] {msg}', flush=True)


async def main_async() -> int:
    note = (
        f'CGL-η Phase-1 bulk cross-graph migration auto-applied and post-verified '
        f'clean (0 foreign :Entity residual). See fused-memory/data/cgl-eta/'
        f'apply-report-{STAMP}.json. Gate satisfied by the deterministic auto-apply task.'
    )

    # 1. Mark the gate task done (terminal FIRST — avoids stranded-blocked reclaim).
    try:
        async with McpClient(FUSED_URL) as fm:
            res = await fm.call_tool('set_task_status', {
                'id': GATE_TASK, 'status': 'done', 'project_root': REPO,
                'done_provenance': {'note': note},
            })
        _log(f'set_task_status({GATE_TASK}, done): {res}')
    except Exception as exc:
        _log(f'set_task_status({GATE_TASK}, done) FAILED (non-fatal): {exc!r} '
             f'-- esc-{GATE_ESC} left for the watcher/operator.')
        return 0  # do not attempt to close the esc if the task is not terminal

    # 2. Dismiss the now-stale escalation on the done task.
    try:
        async with McpClient(ESC_URL) as esc:
            res = await esc.call_tool('resolve_issue', {
                'escalation_id': GATE_ESC,
                'resolution': note,
                'action': 'close_only',
                'resolved_by': 'cgl-eta-auto-apply',
            })
        _log(f'resolve_issue({GATE_ESC}, close_only): {res}')
    except Exception as exc:
        _log(f'resolve_issue({GATE_ESC}) FAILED (non-fatal): {exc!r} '
             f'-- task {GATE_TASK} is done; a watcher self-heals the stale L2.')
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == '__main__':
    sys.exit(main())
