"""Tests for emit_residual_candidate_key_escalation (fm-task-dedup W8 task A2).

This is the injectable escalation seam invoked by the sqlite backend's
v3->v4 self-gating migration when residual non-cancelled duplicate
candidate_key groups are found at connection-open. Mirrors the defensive
HAS_ESCALATION / EscalationQueue never-raise pattern established by
``middleware.scope_violation_escalator``.
"""

from __future__ import annotations

import json

from fused_memory.middleware import candidate_key_escalation as cke_mod
from fused_memory.middleware.candidate_key_escalation import (
    emit_residual_candidate_key_escalation,
)


def test_emit_residual_candidate_key_escalation_never_raises_and_returns_id_or_none(tmp_path):
    """Never raises; returns an escalation id str (escalation package
    importable -- a file lands under {project_root}/data/escalations) or
    None (HAS_ESCALATION is False)."""
    residual_groups = [
        {'tag': 'master', 'candidate_key': 'abc123', 'task_ids': ['1', '2'], 'count': 2},
    ]
    result = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path),
        residual_groups=residual_groups,
    )
    if cke_mod.HAS_ESCALATION:
        assert isinstance(result, str)
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'expected one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == result
    else:
        assert result is None


def test_emit_residual_candidate_key_escalation_dedupes_against_existing_pending(tmp_path):
    """Review amendment: a second call while the first escalation is still
    pending must reuse its id rather than filing a duplicate — repeated
    process restarts while residuals persist (a connection, and therefore
    this migration step, runs once per project_root per process) must not
    flood the operator queue with near-identical escalations. Once the
    original is resolved, a later call is free to file a fresh one.
    """
    residual_groups = [
        {'tag': 'master', 'candidate_key': 'abc123', 'task_ids': ['1', '2'], 'count': 2},
    ]
    first_id = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path), residual_groups=residual_groups,
    )
    second_id = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path), residual_groups=residual_groups,
    )

    if not cke_mod.HAS_ESCALATION:
        assert first_id is None
        assert second_id is None
        return

    assert first_id is not None
    assert second_id == first_id, (
        f'Expected the second call (first still pending) to reuse the open '
        f'escalation; got first={first_id!r} second={second_id!r}'
    )
    queue_dir = tmp_path / 'data' / 'escalations'
    assert len(list(queue_dir.glob('esc-*.json'))) == 1, (
        'Expected exactly one escalation file on disk after two calls with '
        'the first still pending'
    )

    # Once resolved, the condition is no longer "open" — a later call must
    # be free to file a fresh escalation rather than being dedup-blocked
    # forever.
    from escalation.queue import EscalationQueue

    EscalationQueue(queue_dir).resolve(first_id, 'residuals cleaned up')

    third_id = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path), residual_groups=residual_groups,
    )
    assert third_id is not None
    assert third_id != first_id, (
        f'Expected a fresh escalation once the prior one was resolved; got '
        f'the same id {third_id!r} again'
    )
