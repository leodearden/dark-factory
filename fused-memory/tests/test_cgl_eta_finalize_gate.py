"""Tests for the CGL-η gate finalizer script (cgl_eta_finalize_gate.py).

Covers the fix for the CGL-η 2273 stranding class: the finalize script must
send `done_provenance` with a server-accepted `kind` (`'deterministic-gate'`,
the pure-gate-resolved kind added by task 2334), not a bare `{'note': ...}`
blob — the shape that was unconditionally rejected as `done_provenance_invalid`
and left task 2273 blocked with a stale born-at-L2 escalation.

Step 1: RED test for `_gate_done_provenance` (fails until step-2 adds the
helper). Step 3 (added later): RED test proving the helper is actually wired
into `main_async`'s `set_task_status` call, not just declared.

Amendment pass: added an ordering assertion (set_task_status must precede
resolve_issue — the script's documented "order matters" invariant), a test
for the best-effort failure path (set_task_status raises -> exit 0, no
resolve_issue call), and a `finalize_module` fixture that centralizes the
sys.path dance previously duplicated in each test.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from shared.task_metadata import DoneProvenance

_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))


@pytest.fixture
def finalize_module():
    """Import cgl_eta_finalize_gate with its scripts dir on sys.path; clean up after.

    cgl_eta_finalize_gate.py is a standalone script (not an installed
    package), so importing it requires putting fused-memory/scripts on
    sys.path for the duration of the test. Centralized here so individual
    tests don't each repeat the insert/remove dance.
    """
    sys.path.insert(0, _SCRIPTS_DIR)
    try:
        import cgl_eta_finalize_gate  # pyright: ignore[reportMissingImports]
        yield cgl_eta_finalize_gate
    finally:
        sys.path.remove(_SCRIPTS_DIR)


def test_gate_done_provenance_has_accepted_kind(finalize_module):
    """`_gate_done_provenance` produces a shape the real DoneProvenance model accepts.

    Ties the helper's output directly to the server acceptance contract that
    stranded task 2273: kind='deterministic-gate' is a value already
    recognized by shared.task_metadata.DoneProvenance (task 2334), so a
    script emitting this shape will not be rejected as done_provenance_invalid.
    """
    result = finalize_module._gate_done_provenance('clean migration note')

    assert result == {'kind': 'deterministic-gate', 'note': 'clean migration note'}
    assert DoneProvenance(**result).kind == 'deterministic-gate'


def _make_recording_mcp_client(calls: list, *, raise_on: str | None = None) -> type:
    """Build a fake MCP client class recording call_tool(name, arguments) into *calls*.

    Mirrors the real McpClient's shape (cgl_eta_scheduler_gate.McpClient):
    an async context manager constructed with a single `url` positional arg,
    exposing async call_tool(). No network/server involved — every call is
    appended to the shared *calls* list (preserving order) and answered with
    {}, UNLESS its tool name matches *raise_on*, in which case it raises a
    RuntimeError instead of recording anything — used to exercise the
    script's best-effort failure path.
    """

    class _RecordingMcpClient:
        def __init__(self, url):
            self._url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def call_tool(self, name, arguments):
            if name == raise_on:
                raise RuntimeError(f'simulated failure calling {name!r}')
            calls.append((name, arguments))
            return {}

    return _RecordingMcpClient


def test_finalize_sends_deterministic_gate_provenance(monkeypatch, finalize_module):
    """main_async's set_task_status call carries done_provenance with kind='deterministic-gate'.

    Proves _gate_done_provenance is actually wired into the real finalize
    call (not just declared) — the fix for the CGL-η 2273 stranding class.
    cgl_eta_finalize_gate.McpClient is monkeypatched with a recording fake so
    the whole flow runs with no network/server; asyncio.run drives
    main_async synchronously, same as main() does.
    """
    calls: list = []
    monkeypatch.setattr(finalize_module, 'McpClient', _make_recording_mcp_client(calls))

    exit_code = asyncio.run(finalize_module.main_async())

    assert exit_code == 0

    # The module docstring's "order matters" invariant: set_task_status(done)
    # must run before resolve_issue(close_only), so there is never a
    # blocked-without-open-escalation window a stranded-blocked reconciler
    # could reclaim. Pin call order, not just call counts, so a reordering
    # regression fails this test.
    assert [c[0] for c in calls] == ['set_task_status', 'resolve_issue']

    _, set_status_args = calls[0]
    _, resolve_args = calls[1]

    # resolve_issue's `resolution` carries the same `note` local variable
    # main_async passes into done_provenance — comparing against it proves
    # the note text is unchanged by the fix, without hardcoding the (long,
    # STAMP-dependent) note text in this test.
    expected_note = resolve_args['resolution']
    assert set_status_args['done_provenance'] == {
        'kind': 'deterministic-gate',
        'note': expected_note,
    }
    assert DoneProvenance(**set_status_args['done_provenance']).kind == 'deterministic-gate'


def test_finalize_exits_zero_and_skips_resolve_when_set_task_status_fails(monkeypatch, finalize_module):
    """Best-effort contract: a raising set_task_status still exits 0 and skips resolve_issue.

    Pins the script's central robustness claim (module docstring: "Best-effort
    by design ... ALWAYS exits 0" and "do not attempt to close the esc if the
    task is not terminal"): if set_task_status(2273, done) raises, main_async
    must swallow it, return 0, and never call resolve_issue on the
    still-open escalation.
    """
    calls: list = []
    monkeypatch.setattr(
        finalize_module,
        'McpClient',
        _make_recording_mcp_client(calls, raise_on='set_task_status'),
    )

    exit_code = asyncio.run(finalize_module.main_async())

    assert exit_code == 0
    assert calls == []


def test_finalize_exits_zero_and_leaves_task_done_when_resolve_issue_fails(monkeypatch, finalize_module):
    """Best-effort contract: a raising resolve_issue still exits 0 and the done write stands.

    Covers the other failure leg (mirrors
    test_finalize_exits_zero_and_skips_resolve_when_set_task_status_fails above,
    which pins the set_task_status-fails leg). Module docstring: if
    resolve_issue fails, "task {GATE_TASK} is done; a watcher self-heals the
    stale L2" — i.e. set_task_status must have already succeeded and recorded
    (the task is done) *before* resolve_issue's exception is swallowed, and
    main_async must still return 0 rather than let the exception propagate
    into a spurious non-zero exit for an otherwise-successful migration.
    """
    calls: list = []
    monkeypatch.setattr(
        finalize_module,
        'McpClient',
        _make_recording_mcp_client(calls, raise_on='resolve_issue'),
    )

    exit_code = asyncio.run(finalize_module.main_async())

    assert exit_code == 0
    # set_task_status recorded (the task is done); resolve_issue was
    # attempted and its exception swallowed rather than propagated.
    assert [c[0] for c in calls] == ['set_task_status']
    assert calls[0][1]['status'] == 'done'
