"""Tests for emit_residual_candidate_key_escalation (fm-task-dedup W8 task A2).

This is the injectable escalation seam invoked by the sqlite backend's
v3->v4 self-gating migration when residual non-cancelled duplicate
candidate_key groups are found at connection-open. Mirrors the defensive
HAS_ESCALATION / EscalationQueue never-raise pattern established by
``middleware.scope_violation_escalator``.
"""

from __future__ import annotations

import json

from fused_memory.middleware import _folded_escalation
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


def test_emit_residual_candidate_key_escalation_detail_surfaces_group_reason(tmp_path):
    """fm-task-dedup self-heal amendment: the v3->v4 migration now only
    escalates AMBIGUOUS groups, each carrying a per-group ``reason``
    (``'mixed_status'`` | ``'title_divergent'``). The filed escalation's
    detail text must surface each group's reason — and explain that
    genuine content-duplicates were already auto-healed — so an operator
    understands why THESE groups still need a human without
    cross-referencing the migration source.
    """
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
    if not cke_mod.HAS_ESCALATION:
        assert result is None
        return

    queue_dir = tmp_path / 'data' / 'escalations'
    files = list(queue_dir.glob('esc-*.json'))
    assert len(files) == 1, f'expected one escalation file, found: {files}'
    payload = json.loads(files[0].read_text())
    detail = payload['detail']
    assert 'mixed_status' in detail, detail
    assert 'title_divergent' in detail, detail
    assert 'auto-heal' in detail.lower(), detail


class TestDelegatesToTheSharedHelper:
    """The filer BODY now lives in `middleware/_folded_escalation`."""

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

    def test_the_anchor_is_still_a_module_attribute_of_THIS_module(self):
        """Cross-imported by tests/server/test_write_triage.py, and read from
        its own home by the pairwise anchor-collision regression."""
        assert cke_mod._ANCHOR_TASK_ID == 'candidate-key-migration'
        assert cke_mod._AGENT_ROLE == 'fused-memory/candidate-key-migration'
        assert cke_mod._CATEGORY == 'candidate_key_residual_duplicates'

    def test_the_group_detail_construction_stays_in_THIS_module(
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


def test_a_queue_construction_failure_returns_none(tmp_path, monkeypatch):
    """BEHAVIOUR CHANGE, pinned deliberately (task 4854).

    Six of the seven copies of this filer skeleton guarded the queue
    constructor; this one did NOT — even though its own docstring promises it
    "NEVER raises: this is called from connection-open migration code, and a
    raise here would defeat the self-gating step's own fail-safe guarantee".
    Constructing an `EscalationQueue` creates its directory, so a read-only or
    missing `project_root` turned a connection-open migration into a crash:
    the exact outcome that docstring rules out.

    Consolidating to one home forces a single answer, and the correct answer is
    the one six siblings already implement and the seventh already documents.
    """
    def _explode(*_a, **_kw):
        raise OSError('cannot create queue dir')

    monkeypatch.setattr(_folded_escalation, 'EscalationQueue', _explode)

    assert emit_residual_candidate_key_escalation(
        str(tmp_path),
        [{'tag': 't', 'candidate_key': 'k', 'task_ids': ['1'], 'count': 1,
          'reason': 'mixed_status'}],
    ) is None
