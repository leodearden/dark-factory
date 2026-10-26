"""How each workflow_verify reader treats a row's `site` (task 5651).

The reader effects are stated once, at verify_evidence.py::VerifySite; these
tests pin them against a real EventStore.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.completion import merge_completion_eligible
from orchestrator.merge_lane.disposition import _branch_pre_merge_verify_green
from orchestrator.stranded_verified_green import last_verified_green_tip
from orchestrator.verify_checkpoint import green_checkpoint_at_tip
from orchestrator.verify_evidence import VerifySite

_TASK_ID = '5651'
_T1 = 'a' * 40
_T2 = 'b' * 40

#: (site, passed, tip_sha) for one workflow_verify row.
Row = tuple[VerifySite, bool, str]

IMPLEMENT_GREEN_T1: Row = (VerifySite.IMPLEMENT, True, _T1)
PRE_MERGE_GREEN_T2: Row = (VerifySite.PRE_MERGE, True, _T2)
PRE_MERGE_RED_T2: Row = (VerifySite.PRE_MERGE, False, _T2)


def _store_with(tmp_path: Path, rows: list[Row]) -> EventStore:
    store = EventStore(tmp_path / 'runs.db', run_id='r')
    for site, passed, tip_sha in rows:
        store.emit(
            EventType.workflow_verify,
            task_id=_TASK_ID,
            data={
                'passed': passed,
                'tip_sha': tip_sha,
                'base_sha': 'c' * 40,
                'branch': f'task/{_TASK_ID}',
                'site': str(site),
                'duration_ms': 7000,
                'slot_wait_secs': None,
                'plan_hash': 'd' * 64,
                'cache_mode': 'off',
                'units': [],
            },
        )
    return store


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([PRE_MERGE_RED_T2], None, id='pre-merge-red-alone-is-no-branch-evidence'),
    pytest.param([PRE_MERGE_GREEN_T2], True, id='pre-merge-green-counts'),
    pytest.param([IMPLEMENT_GREEN_T1, PRE_MERGE_RED_T2], True, id='implement-green-survives-pre-merge-red'),
])
def test_i5_branch_verdict_ignores_pre_merge_reds(tmp_path, rows, expected):
    assert _branch_pre_merge_verify_green(_store_with(tmp_path, rows), _TASK_ID) is expected


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([PRE_MERGE_GREEN_T2], True, id='pre-merge-green-satisfies-verify-half'),
    pytest.param([PRE_MERGE_RED_T2], False, id='pre-merge-red-does-not'),
])
def test_merge_completion_reads_a_current_run_pre_merge_green(tmp_path, rows, expected):
    store = _store_with(tmp_path, rows)
    store.emit(EventType.phase_enter, task_id=_TASK_ID, phase='merge')

    assert merge_completion_eligible(store, _TASK_ID) is expected


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([IMPLEMENT_GREEN_T1, PRE_MERGE_GREEN_T2], _T2, id='pre-merge-green-is-latest'),
    pytest.param([IMPLEMENT_GREEN_T1, PRE_MERGE_RED_T2], _T1, id='pre-merge-red-is-skipped'),
])
def test_stranded_reaper_latest_green_tip(tmp_path, rows, expected):
    assert last_verified_green_tip(_store_with(tmp_path, rows), _TASK_ID) == expected


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([PRE_MERGE_GREEN_T2], True, id='pre-merge-green-is-a-checkpoint'),
    pytest.param([PRE_MERGE_RED_T2], False, id='pre-merge-red-is-not'),
])
def test_verify_checkpoint_at_the_post_rebase_tip(tmp_path, rows, expected):
    store = _store_with(tmp_path, rows)

    assert green_checkpoint_at_tip(store, EventType.workflow_verify, _TASK_ID, _T2) is expected
