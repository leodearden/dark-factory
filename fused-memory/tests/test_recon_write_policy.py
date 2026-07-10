"""Tests for the server-side ReconWritePolicy gate (W5-ζ, task 2224).

Rejects recon-stage task writes at the interceptor boundary so the post-hoc
reconciliation guards (μ's job to delete) become redundant. Three independent
early-return gates in :func:`recon_write_policy.check`:

1. ``op == 'update_task'`` AND ``live_status`` is terminal ->
   ``ReconTerminalWriteRejected``.
2. ``op == 'set_task_status'`` AND a live workflow is detected for the task ->
   ``ReconLiveWorkflowWriteRejected``.
3. ``snapshot_token is not None`` AND it disagrees with ``live_status`` (any
   op) -> ``ReconStaleSnapshotRejected``.

This module unit-tests the pure ``Verdict``/``check()``/``extract_snapshot_token``
building blocks in isolation. Interceptor boundary tests (P1/P2/P3) reuse the
fixtures from ``test_task_write_agent_id.py`` and live further down this file.
"""

from __future__ import annotations

from fused_memory.middleware import recon_write_policy


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_ok_verdict_is_not_rejection(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.is_rejection is False

    def test_ok_verdict_to_error_dict_is_empty(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.to_error_dict() == {}

    def test_rejection_verdict_is_rejection(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        assert verdict.is_rejection is True

    def test_rejection_verdict_to_error_dict_shape(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        error_dict = verdict.to_error_dict()
        assert isinstance(error_dict['error'], str)
        assert error_dict['error_type'] == 'ReconTerminalWriteRejected'
        assert error_dict['agent_id'] == 'recon-stage-x'
        assert error_dict['task_id'] == '1'
        assert error_dict['op'] == 'update_task'
        assert error_dict['live_status'] == 'done'
