"""Tests for the CGL-η gate finalizer script (cgl_eta_finalize_gate.py).

Covers the fix for the CGL-η 2273 stranding class: the finalize script must
send `done_provenance` with a server-accepted `kind` (`'deterministic-gate'`,
the pure-gate-resolved kind added by task 2334), not a bare `{'note': ...}`
blob — the shape that was unconditionally rejected as `done_provenance_invalid`
and left task 2273 blocked with a stale born-at-L2 escalation.

Step 1: RED test for `_gate_done_provenance` (fails until step-2 adds the
helper). Step 3 (added later): RED test proving the helper is actually wired
into `main_async`'s `set_task_status` call, not just declared.
"""

from __future__ import annotations


def test_gate_done_provenance_has_accepted_kind():
    """`_gate_done_provenance` produces a shape the real DoneProvenance model accepts.

    Ties the helper's output directly to the server acceptance contract that
    stranded task 2273: kind='deterministic-gate' is a value already
    recognized by shared.task_metadata.DoneProvenance (task 2334), so a
    script emitting this shape will not be rejected as done_provenance_invalid.
    """
    import os
    import sys
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    sys.path.insert(0, os.path.abspath(scripts_dir))
    try:
        import cgl_eta_finalize_gate
        from shared.task_metadata import DoneProvenance

        result = cgl_eta_finalize_gate._gate_done_provenance('clean migration note')

        assert result == {'kind': 'deterministic-gate', 'note': 'clean migration note'}
        assert DoneProvenance(**result).kind == 'deterministic-gate'
    finally:
        sys.path.remove(os.path.abspath(scripts_dir))


def _make_recording_mcp_client(calls: list) -> type:
    """Build a fake MCP client class recording call_tool(name, arguments) into *calls*.

    Mirrors the real McpClient's shape (cgl_eta_scheduler_gate.McpClient):
    an async context manager constructed with a single `url` positional arg,
    exposing async call_tool(). No network/server involved — every call is
    just appended to the shared *calls* list and answered with {}.
    """

    class _RecordingMcpClient:
        def __init__(self, url):
            self._url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {}

    return _RecordingMcpClient


def test_finalize_sends_deterministic_gate_provenance(monkeypatch):
    """main_async's set_task_status call carries done_provenance with kind='deterministic-gate'.

    Proves _gate_done_provenance is actually wired into the real finalize
    call (not just declared) — the fix for the CGL-η 2273 stranding class.
    cgl_eta_finalize_gate.McpClient is monkeypatched with a recording fake so
    the whole flow runs with no network/server; asyncio.run drives
    main_async synchronously, same as main() does.
    """
    import asyncio
    import os
    import sys
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    sys.path.insert(0, os.path.abspath(scripts_dir))
    try:
        import cgl_eta_finalize_gate
        from shared.task_metadata import DoneProvenance

        calls: list = []
        monkeypatch.setattr(cgl_eta_finalize_gate, 'McpClient', _make_recording_mcp_client(calls))

        exit_code = asyncio.run(cgl_eta_finalize_gate.main_async())

        assert exit_code == 0
        set_status_calls = [c for c in calls if c[0] == 'set_task_status']
        resolve_calls = [c for c in calls if c[0] == 'resolve_issue']
        assert len(set_status_calls) == 1, f'expected exactly 1 set_task_status call, got {calls}'
        assert len(resolve_calls) == 1, f'expected exactly 1 resolve_issue call, got {calls}'

        _, set_status_args = set_status_calls[0]
        _, resolve_args = resolve_calls[0]

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
    finally:
        sys.path.remove(os.path.abspath(scripts_dir))
