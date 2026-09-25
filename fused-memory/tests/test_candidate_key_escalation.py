"""Tests for emit_residual_candidate_key_escalation (fm-task-dedup W8 task A2).

This is the injectable escalation seam invoked by the sqlite backend's
v3->v4 self-gating migration when residual non-cancelled duplicate
candidate_key groups are found at connection-open.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.middleware import candidate_key_escalation as cke_mod
from fused_memory.middleware.candidate_key_escalation import (
    emit_residual_candidate_key_escalation,
)


def test_emit_residual_candidate_key_escalation_files_into_the_projects_queue(tmp_path):
    """Returns the escalation id, and a file lands under
    {project_root}/data/escalations."""
    pytest.importorskip('escalation')
    residual_groups = [
        {'tag': 'master', 'candidate_key': 'abc123', 'task_ids': ['1', '2'], 'count': 2},
    ]
    result = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path),
        residual_groups=residual_groups,
    )
    assert isinstance(result, str)
    queue_dir = tmp_path / 'data' / 'escalations'
    files = list(queue_dir.glob('esc-*.json'))
    assert len(files) == 1, f'expected one escalation file, found: {files}'
    payload = json.loads(files[0].read_text())
    assert payload['id'] == result


def test_emit_residual_candidate_key_escalation_dedupes_against_existing_pending(tmp_path):
    """Review amendment: a second call while the first escalation is still
    pending must reuse its id rather than filing a duplicate — repeated
    process restarts while residuals persist (a connection, and therefore
    this migration step, runs once per project_root per process) must not
    flood the operator queue with near-identical escalations. Once the
    original is resolved, a later call is free to file a fresh one.
    """
    pytest.importorskip('escalation')
    residual_groups = [
        {'tag': 'master', 'candidate_key': 'abc123', 'task_ids': ['1', '2'], 'count': 2},
    ]
    first_id = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path), residual_groups=residual_groups,
    )
    second_id = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path), residual_groups=residual_groups,
    )

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


def test_emit_residual_candidate_key_escalation_detail_surfaces_group_reason(tmp_path):
    """fm-task-dedup self-heal amendment: the v3->v4 migration now only
    escalates AMBIGUOUS groups, each carrying a per-group ``reason``
    (``'mixed_status'`` | ``'title_divergent'``). The filed escalation's
    detail text must surface each group's reason — and explain that
    genuine content-duplicates were already auto-healed — so an operator
    understands why THESE groups still need a human without
    cross-referencing the migration source.
    """
    pytest.importorskip('escalation')
    residual_groups = [
        {
            'tag': 'master', 'candidate_key': 'abc123', 'task_ids': ['1', '2'],
            'count': 2, 'reason': 'mixed_status',
        },
        {
            'tag': 'master', 'candidate_key': 'def456', 'task_ids': ['3', '4'],
            'count': 2, 'reason': 'title_divergent',
        },
    ]
    result = emit_residual_candidate_key_escalation(
        project_root=str(tmp_path),
        residual_groups=residual_groups,
    )
    assert result is not None

    queue_dir = tmp_path / 'data' / 'escalations'
    files = list(queue_dir.glob('esc-*.json'))
    assert len(files) == 1, f'expected one escalation file, found: {files}'
    payload = json.loads(files[0].read_text())
    detail = payload['detail']
    assert 'mixed_status' in detail, detail
    assert 'title_divergent' in detail, detail
    assert 'auto-heal' in detail.lower(), detail


class TestDelegatesToTheSharedHelper:
    """What this filer forwards to `file_folded_escalation`."""

    def test_forwards_this_modules_own_anchor_role_and_category(
        self, tmp_path, monkeypatch,
    ):
        seen: dict = {}

        def _spy(project_root, **kwargs):
            seen['project_root'] = project_root
            seen.update(kwargs)
            return 'esc-candidate-key-migration-1'

        monkeypatch.setattr(cke_mod, 'file_folded_escalation', _spy)

        result = emit_residual_candidate_key_escalation(
            str(tmp_path),
            [{'tag': 't', 'candidate_key': 'k', 'task_ids': ['1', '2'],
              'count': 2, 'reason': 'mixed_status'}],
        )

        assert result == 'esc-candidate-key-migration-1'
        assert seen['anchor_task_id'] == 'candidate-key-migration'
        assert seen['agent_role'] == 'fused-memory/candidate-key-migration'
        assert seen['category'] == 'candidate_key_residual_duplicates'
        assert seen['severity'] == 'blocking'
        assert seen['level'] == 1
        assert seen['project_root'] == str(tmp_path)

    def test_forwards_the_group_detail_it_builds(
        self, tmp_path, monkeypatch,
    ):
        seen: dict = {}

        def _spy(_project_root, **kwargs):
            seen.update(kwargs)
            return 'esc-candidate-key-migration-1'

        monkeypatch.setattr(cke_mod, 'file_folded_escalation', _spy)

        emit_residual_candidate_key_escalation(
            str(tmp_path),
            [{'tag': 'bug', 'candidate_key': 'ck-9', 'task_ids': ['7'],
              'count': 1, 'reason': 'title_divergent'}],
        )

        assert "reason='title_divergent'" in seen['detail']
        assert 'ux_tasks_candidate_key' in seen['detail']
        assert 'residual duplicate candidate_key' in seen['summary']

