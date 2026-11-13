"""Public seam of ``fused_memory.reconciliation.reconciler_escalation``.

Driven against a REAL ``EscalationQueue`` on ``tmp_path``, never a fake store:
the contract is what a reconciler-authored L1 looks like on disk, where it
lands, and that failures surface to the caller rather than being contained.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from shared.project_layout import escalation_queue_dir

from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.reconciler_escalation import (
    INSTALLED_ESCALATION_QUEUE_FACTORY,
    file_reconciler_l1,
    pending_escalations,
)

pytest.importorskip('escalation.queue')
from escalation.queue import EscalationQueue  # noqa: E402


def _root(tmp_path: Path) -> ProjectRoot:
    return ProjectRoot(str(tmp_path))


def _file(root: ProjectRoot, *, task_id: str = '42', factory=EscalationQueue) -> str:
    return file_reconciler_l1(
        factory,
        root,
        task_id=task_id,
        severity='info',
        category='risk_identified',
        summary='S',
        detail='D',
        suggested_action='dismiss',
    )


def test_files_one_reconciler_authored_l1_with_the_passed_fields(tmp_path):
    filed_id = _file(_root(tmp_path))

    pending = EscalationQueue(escalation_queue_dir(tmp_path)).get_pending()
    assert len(pending) == 1
    record = pending[0]
    assert record.id == filed_id
    assert record.agent_role == 'reconciler'
    assert record.level == 1
    assert record.task_id == '42'
    assert record.severity == 'info'
    assert record.category == 'risk_identified'
    assert record.summary == 'S'
    assert record.detail == 'D'
    assert record.suggested_action == 'dismiss'


def test_files_into_the_layout_escalation_queue_when_relocated(
    tmp_path, escalation_queue_relocation,
):
    filed_id = _file(_root(tmp_path))

    relocated = tmp_path / escalation_queue_relocation.relocated_relpath
    assert [p.stem for p in sorted(relocated.glob('esc-*.json'))] == [filed_id]
    assert not (tmp_path / escalation_queue_relocation.legacy_relpath).exists()


def test_pending_escalations_lists_every_filed_id(tmp_path):
    root = _root(tmp_path)
    assert pending_escalations(EscalationQueue, root) == []

    filed_ids = {_file(root, task_id='7'), _file(root, task_id='8')}

    assert {esc.id for esc in pending_escalations(EscalationQueue, root)} == filed_ids


def _raise_at_construction(queue_dir: Path) -> EscalationQueue:
    raise OSError('read-only fs')


class _SubmitFailsQueue(EscalationQueue):
    def submit(self, escalation):
        raise RuntimeError('disk full')


class _GetPendingFailsQueue(EscalationQueue):
    def get_pending(self):
        raise RuntimeError('index unreadable')


def test_filing_propagates_a_queue_construction_failure(tmp_path):
    with pytest.raises(OSError, match='read-only fs'):
        _file(_root(tmp_path), factory=_raise_at_construction)


def test_filing_propagates_a_submit_failure(tmp_path):
    with pytest.raises(RuntimeError, match='disk full'):
        _file(_root(tmp_path), factory=_SubmitFailsQueue)


def test_pending_escalations_propagates_a_read_failure(tmp_path):
    with pytest.raises(RuntimeError, match='index unreadable'):
        pending_escalations(_GetPendingFailsQueue, _root(tmp_path))


def test_installed_default_factory_is_the_escalation_queue():
    assert INSTALLED_ESCALATION_QUEUE_FACTORY is EscalationQueue
