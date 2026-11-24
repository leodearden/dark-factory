"""Public seam of ``fused_memory.reconciliation.reconciler_escalation``.

Driven against a REAL ``EscalationQueue`` on ``tmp_path``, never a fake store:
the contract is what a reconciler-authored L1 looks like on disk, where it
lands, and that failures surface to the caller rather than being contained.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.reconciler_escalation import (
    INSTALLED_ESCALATION_QUEUE_FACTORY,
    EscalationQueueFactory,
    file_reconciler_l1,
    pinning_escalation_ids_by_task,
)


def _root(tmp_path: Path) -> ProjectRoot:
    return ProjectRoot(str(tmp_path))


def _file(
    root: ProjectRoot,
    *,
    task_id: str = '42',
    factory: EscalationQueueFactory = EscalationQueue,
) -> str:
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


def _submit(root: ProjectRoot, *, task_id: str, severity: str, level: int) -> str:
    queue = EscalationQueue(escalation_queue_dir(root))
    return queue.submit(Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='harness',
        severity=severity,
        category='risk_identified',
        summary='S',
        level=level,
    ))


@pytest.mark.parametrize('level', [0, 1, 2])
def test_an_info_record_pins_no_task_at_any_level(tmp_path, level):
    root = _root(tmp_path)
    _submit(root, task_id='7', severity='info', level=level)

    assert '7' not in pinning_escalation_ids_by_task(EscalationQueue, root)


def test_a_blocking_l1_pins_its_task(tmp_path):
    root = _root(tmp_path)
    blocking_id = _submit(root, task_id='7', severity='blocking', level=1)

    assert pinning_escalation_ids_by_task(EscalationQueue, root)['7'] == (blocking_id,)


def test_a_blocking_l0_with_no_filing_identity_pins_its_task(tmp_path):
    root = _root(tmp_path)
    blocking_id = _submit(root, task_id='7', severity='blocking', level=0)

    assert pinning_escalation_ids_by_task(EscalationQueue, root)['7'] == (blocking_id,)


def test_a_mixed_set_keeps_only_the_pinning_ids(tmp_path):
    root = _root(tmp_path)
    _submit(root, task_id='7', severity='info', level=2)
    blocking_id = _submit(root, task_id='7', severity='blocking', level=1)

    assert pinning_escalation_ids_by_task(EscalationQueue, root)['7'] == (blocking_id,)


def test_pinning_ids_are_grouped_per_task(tmp_path):
    root = _root(tmp_path)
    blocking_id = _submit(root, task_id='7', severity='blocking', level=1)
    _submit(root, task_id='8', severity='info', level=1)

    assert pinning_escalation_ids_by_task(EscalationQueue, root) == {'7': (blocking_id,)}


def test_an_int_typed_task_id_pins_under_its_str_id(tmp_path):
    root = _root(tmp_path)
    blocking_id = _submit(root, task_id='7', severity='blocking', level=1)
    record_path = Path(escalation_queue_dir(root)) / f'{blocking_id}.json'
    record = json.loads(record_path.read_text())
    record['task_id'] = 7
    record_path.write_text(json.dumps(record))

    assert pinning_escalation_ids_by_task(EscalationQueue, root)['7'] == (blocking_id,)


def test_pinning_ids_propagate_a_read_failure(tmp_path):
    with pytest.raises(RuntimeError, match='index unreadable'):
        pinning_escalation_ids_by_task(_GetPendingFailsQueue, _root(tmp_path))


def test_installed_default_factory_is_the_escalation_queue():
    assert INSTALLED_ESCALATION_QUEUE_FACTORY is EscalationQueue
