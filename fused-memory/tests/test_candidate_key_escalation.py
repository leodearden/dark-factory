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
