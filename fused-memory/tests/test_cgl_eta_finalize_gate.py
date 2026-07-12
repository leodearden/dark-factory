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
