"""How each workflow_verify reader treats a row's `site` (task 5651).

The reader effects are stated once, at verify_evidence.py::VerifySite; these
tests pin them against a real EventStore.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.completion import merge_completion_eligible
from orchestrator.merge_lane.disposition import (
    MergeFailureDisposition,
    classify_merge_failure_disposition,
)
from orchestrator.stranded_verified_green import last_verified_green_tip
from orchestrator.verify import VerifyResult
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


_FAILURE_IN_X = VerifyResult(
    passed=False, test_output='FAILED src/x.py::test_bar - AssertionError', lint_output='',
    type_output='', summary='1 failed', cause_hint='src/x.py::test_bar', category='test_failure',
)


def _commit_x(repo: Path, content: str) -> str:
    (repo / 'src').mkdir(exist_ok=True)
    (repo / 'src' / 'x.py').write_text(content)
    for argv in (['add', '-A'], ['commit', '-q', '-m', content]):
        subprocess.run(['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *argv], cwd=repo, check=True)
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([PRE_MERGE_RED_T2], MergeFailureDisposition.INDETERMINATE, id='pre-merge-red-alone-is-no-green'),
    pytest.param([PRE_MERGE_GREEN_T2], MergeFailureDisposition.INTEGRATION_SKEW, id='pre-merge-green-counts'),
    pytest.param(
        [IMPLEMENT_GREEN_T1, PRE_MERGE_RED_T2], MergeFailureDisposition.INTEGRATION_SKEW,
        id='implement-green-survives-pre-merge-red',
    ),
])
async def test_i5_skew_needs_a_green_that_pre_merge_reds_cannot_erase(tmp_path, rows, expected):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True)
    merge_base_sha = _commit_x(repo, 'v1')
    main_sha = _commit_x(repo, 'v2')

    classification = await classify_merge_failure_disposition(
        verify_result=_FAILURE_IN_X, branch=f'task/{_TASK_ID}',
        merge_base_sha=merge_base_sha, main_sha=main_sha, preexisting=False,
        task_id=_TASK_ID, repo_root=repo, event_store=_store_with(tmp_path, rows),
    )

    assert classification.disposition is expected


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
