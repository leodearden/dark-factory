"""verify_host_trial_stores.LOCK_CONFLICT_REASONS against the workflow that writes them.

The producer passes each reason to TaskWorkflow._requeue_on_lock_conflict as a
bare literal, with no enum to import, and scripts/tests imports no
orchestrator. So this reads the producer's text. A renamed reason would
otherwise make lock_conflict_requeues report no lock conflict, silently.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from verify_host_trial_stores import LOCK_CONFLICT_REASONS

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER = REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'workflow.py'


def test_each_lock_conflict_reason_is_a_literal_the_workflow_writes():
    producer = PRODUCER.read_text(encoding='utf-8')

    assert LOCK_CONFLICT_REASONS
    for reason in LOCK_CONFLICT_REASONS:
        assert f"'{reason}'" in producer or f'"{reason}"' in producer, reason
