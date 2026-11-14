"""How each workflow_verify reader treats a row's `site` (task 5651) and a row with no tip (task 5409).

The reader effects are stated once, at verify_evidence.py::VerifySite; these
tests pin them against a real EventStore, on rows built by the one payload
constructor, verify_evidence.py::workflow_verify_event_data.
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
from orchestrator.verify_evidence import TipUnresolved, VerifySite, workflow_verify_event_data

_TASK_ID = '5651'
_T1 = 'a' * 40
_T2 = 'b' * 40

#: (site, passed, tip_sha) for one workflow_verify row.
Row = tuple[VerifySite, bool, str | TipUnresolved]

IMPLEMENT_GREEN_T1: Row = (VerifySite.IMPLEMENT, True, _T1)
PRE_MERGE_GREEN_T2: Row = (VerifySite.PRE_MERGE, True, _T2)
PRE_MERGE_RED_T2: Row = (VerifySite.PRE_MERGE, False, _T2)
MCP_VOUCH_GREEN_T2: Row = (VerifySite.MCP_MERGE_REQUEST, True, _T2)
UNRESOLVED_GREEN: Row = (VerifySite.IMPLEMENT, True, TipUnresolved.HEAD_UNREADABLE)


def _row_data(row: Row, result: VerifyResult | None = None) -> dict:
    site, passed, tip_sha = row
    return workflow_verify_event_data(
        passed, tip_sha=tip_sha, base_sha='c' * 40, branch=f'task/{_TASK_ID}',
        result=result, site=site, runner='local',
    )


def _store_with(tmp_path: Path, rows: list[Row]) -> EventStore:
    store = EventStore(tmp_path / 'runs.db', run_id='r')
    for row in rows:
        store.emit(EventType.workflow_verify, task_id=_TASK_ID, data=_row_data(row))
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
    pytest.param([MCP_VOUCH_GREEN_T2], MergeFailureDisposition.INTEGRATION_SKEW, id='mcp-vouch-green-counts'),
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
    pytest.param([MCP_VOUCH_GREEN_T2], True, id='mcp-vouch-green-satisfies-verify-half'),
    pytest.param([UNRESOLVED_GREEN], True, id='unresolved-tip-green-still-counts-as-passed'),
])
def test_merge_completion_reads_a_current_run_pre_merge_green(tmp_path, rows, expected):
    store = _store_with(tmp_path, rows)
    store.emit(EventType.phase_enter, task_id=_TASK_ID, phase='merge')

    assert merge_completion_eligible(store, _TASK_ID) is expected


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([IMPLEMENT_GREEN_T1, PRE_MERGE_GREEN_T2], _T2, id='pre-merge-green-is-latest'),
    pytest.param([IMPLEMENT_GREEN_T1, PRE_MERGE_RED_T2], _T1, id='pre-merge-red-is-skipped'),
    pytest.param([IMPLEMENT_GREEN_T1, MCP_VOUCH_GREEN_T2], _T2, id='mcp-vouch-green-is-latest'),
    pytest.param([IMPLEMENT_GREEN_T1, UNRESOLVED_GREEN], _T1, id='unresolved-tip-green-does-not-shadow'),
    pytest.param([UNRESOLVED_GREEN], None, id='unresolved-tip-green-is-no-tip'),
])
def test_stranded_reaper_latest_green_tip(tmp_path, rows, expected):
    assert last_verified_green_tip(_store_with(tmp_path, rows), _TASK_ID) == expected


@pytest.mark.parametrize(('rows', 'expected'), [
    pytest.param([PRE_MERGE_GREEN_T2], True, id='pre-merge-green-is-a-checkpoint'),
    pytest.param([PRE_MERGE_RED_T2], False, id='pre-merge-red-is-not'),
    pytest.param([MCP_VOUCH_GREEN_T2], True, id='mcp-vouch-green-is-a-checkpoint'),
    pytest.param([UNRESOLVED_GREEN], False, id='unresolved-tip-green-is-not'),
])
def test_verify_checkpoint_at_the_post_rebase_tip(tmp_path, rows, expected):
    store = _store_with(tmp_path, rows)

    assert green_checkpoint_at_tip(store, EventType.workflow_verify, _TASK_ID, _T2) is expected


_PASSING = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='ok', duration_secs=7.0,
)


def test_every_row_has_one_shape_that_states_whether_its_tip_resolved():
    resolved = _row_data(IMPLEMENT_GREEN_T1)
    unresolved = _row_data(UNRESOLVED_GREEN)
    with_verdict = _row_data(IMPLEMENT_GREEN_T1, result=_PASSING)

    assert (resolved['tip_sha'], resolved['tip_sha_unresolved'], resolved['tip_sha_unresolved_reason']) == (
        _T1, False, None,
    )
    assert (unresolved['tip_sha'], unresolved['tip_sha_unresolved'], unresolved['tip_sha_unresolved_reason']) == (
        None, True, 'head_unreadable',
    )
    assert resolved.keys() == unresolved.keys() == with_verdict.keys()
    assert _row_data(MCP_VOUCH_GREEN_T2)['site'] == 'mcp:merge_request'


def test_an_empty_tip_is_refused_rather_than_recorded():
    with pytest.raises(ValueError, match='tip_sha') as refused:
        _row_data((VerifySite.IMPLEMENT, True, ''))

    assert "''" in str(refused.value)
