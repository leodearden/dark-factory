"""Tests for the nightly trickle's completion error-streak escalation
(plans/census-incremental-prd.md C7 (a), G7 storm-escape-required).

The census trigger's completion arm runs scripts/check_run_completion.py
once per evaluation. When that check cannot answer on three consecutive
nightly evaluations, the nightly posts ONE escalation: the census can then
fire on the novelty spike alone, and an operator should know. Only a check
that answered (exit 0 or 75) resets the streak; a night with nothing to
check (no previous run id, or the run filed nothing) carries it.

Each night here is one ``nightly.run_nightly`` pass over an empty
projects_root: the quiet path, which reaches the census stage without
digesting, coding or committing anything. The completion check is a
scripted ``completion_runner`` seam, and the census launcher never runs.
"""
from __future__ import annotations

import json
import logging
import subprocess
from collections import deque
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from legibility import nightly, trickle_state
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_PROJECT_ID = 'proj_streak'
_RUN_ID = 'census-proj_streak-20260601'
_FIRST_NIGHT = date(2026, 7, 13)


def _now_after(target_date: date) -> datetime:
    """03:00 UTC the morning after *target_date*, when the timer runs it."""
    day_after = target_date + timedelta(days=1)
    return datetime(day_after.year, day_after.month, day_after.day, 3, tzinfo=UTC)


_ERROR = (2, {'run': _RUN_ID, 'verdict': 'error', 'error': 'store_unreachable'})
_FILED_NOTHING = (2, {'run': _RUN_ID, 'verdict': 'error', 'error': 'empty_run'})
_NOT_YET = (75, {'run': _RUN_ID, 'verdict': 'not_yet', 'share': 0.5, 'vacuous': False})
_PASS = (0, {'run': _RUN_ID, 'verdict': 'pass', 'share': 0.9, 'vacuous': False})


@pytest.fixture(autouse=True)
def _isolate_trickle_state(tmp_path, monkeypatch):
    """Keep every recorded night under tmp_path, never the operator's home."""
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / 'legibility-state'))


def _write_project(root: Path, *, run_id: str | None) -> Path:
    """A minimal project: legibility.yaml, an empty codebook, and a
    census-state whose floor is long clear. Returns the config path."""
    legibility_dir = root / 'docs' / 'legibility'
    legibility_dir.mkdir(parents=True)
    config_path = legibility_dir / 'legibility.yaml'
    config_path.write_text(
        '\n'.join([
            f'project_id: {_PROJECT_ID}',
            f'project_root: {root}',
            'escalation_port: 8199',
            'cwd_prefixes:',
            f'  - {root / "work"}',
        ]) + '\n',
        encoding='utf-8',
    )
    (legibility_dir / 'confusion-codebook.yaml').write_text(
        yaml.safe_dump({'version': 2, 'entries': [], 'candidates': []}), encoding='utf-8',
    )
    month_ago = (_now_after(_FIRST_NIGHT) - timedelta(days=30)).isoformat()
    state = {
        'last_census_at': month_ago,
        'last_census_report': 'plans/confusion-census-prior.md',
        'session_watermark': month_ago,
    }
    if run_id is not None:
        state['last_census_run_id'] = run_id
    (legibility_dir / 'census-state.json').write_text(json.dumps(state), encoding='utf-8')
    return config_path


def _scripted_runner(results: Iterable[tuple[int, dict]]):
    """A ``completion_runner=`` answering each call with the next queued
    ``(returncode, verdict)``, the verdict as the last stdout line."""
    queue = deque(results)

    def _run(args):
        returncode, verdict = queue.popleft()
        return subprocess.CompletedProcess(list(args), returncode, stdout=json.dumps(verdict) + '\n')

    return _run


def _never_invoked(prompt, model):
    raise AssertionError('a quiet night must never reach the coder')


def _no_census_launch(project_root, **kwargs):
    """A FIRE must never spawn the real census.py from a test."""


def _refusing_first_streak_post(posted: list[dict]):
    """A poster whose first completion-streak post is refused, as
    escalate_info's error reply; every other post lands."""
    refused: list[dict] = []

    def _post(url, arguments):
        posted.append(arguments)
        if 'completion check' in arguments['summary'] and not refused:
            refused.append(arguments)
            return {'error': 'synthetic refusal'}
        return {}

    return _post


def _run_nights(
    tmp_path, outcomes, *, run_id: str | None = _RUN_ID, refuse_first_streak_post=False,
):
    """Run one manual night per entry of *outcomes*, on successive target
    dates. Returns ``(results, posted)``: each night's NightlyResult and the
    arguments of every escalation posted."""
    config_path = _write_project(tmp_path, run_id=run_id)
    runner = _scripted_runner(outcomes)
    posted: list[dict] = []
    poster = (
        _refusing_first_streak_post(posted) if refuse_first_streak_post
        else lambda url, arguments: posted.append(arguments)
    )
    results = []
    for night in range(len(outcomes)):
        target_date = _FIRST_NIGHT + timedelta(days=night)
        results.append(nightly.run_nightly(
            config_path=config_path,
            projects_root=tmp_path / 'projects',
            target_date=target_date,
            now=_now_after(target_date),
            invoke=_never_invoked,
            completion_runner=runner,
            poster=poster,
            census_launcher=_no_census_launch,
            pass_kind=nightly.PassKind.MANUAL,
        ))
    return results, posted


def _streak_escalations(posted):
    return [arguments for arguments in posted if 'completion check' in arguments['summary']]


def _recorded_streak() -> int:
    prior = trickle_state.read_prior(_PROJECT_ID)
    assert prior is not None, 'every night must be recorded'
    return prior.completion_error_streak


def test_three_error_nights_post_exactly_one_escalation(tmp_path):
    results, posted = _run_nights(tmp_path, [_ERROR] * 3)

    [escalation] = _streak_escalations(posted)
    assert 'completion check' in escalation['summary']
    assert '3 consecutive evaluations' in escalation['summary']
    assert _PROJECT_ID in escalation['detail']
    assert f'completion: check of {_RUN_ID} failed' in escalation['detail']
    assert 'store_unreachable' in escalation['detail']
    assert results[-1].census_line in escalation['detail']


def test_the_escalation_lands_on_the_third_night_only(tmp_path):
    results, _posted = _run_nights(tmp_path, [_ERROR] * 3)

    kinds = [[alarm.kind for alarm in result.alarms] for result in results]
    assert kinds == [[], [], [nightly.AlarmKind.COMPLETION_ERROR_STREAK]]


def test_a_fourth_error_night_posts_no_second_escalation(tmp_path):
    _results, posted = _run_nights(tmp_path, [_ERROR] * 4)

    assert len(_streak_escalations(posted)) == 1
    assert _recorded_streak() == 4


def test_the_streak_alarm_leaves_the_night_green(tmp_path):
    results, _posted = _run_nights(tmp_path, [_ERROR] * 3)

    assert [result.exit_code for result in results] == [0, 0, 0]
    assert nightly.AlarmKind.COMPLETION_ERROR_STREAK.level == logging.WARNING


def test_no_previous_run_id_never_escalates(tmp_path):
    _results, posted = _run_nights(tmp_path, [_ERROR] * 3, run_id=None)

    assert _streak_escalations(posted) == []
    assert _recorded_streak() == 0


def test_a_run_that_filed_nothing_never_escalates(tmp_path):
    _results, posted = _run_nights(tmp_path, [_FILED_NOTHING] * 3)

    assert _streak_escalations(posted) == []
    assert _recorded_streak() == 0


def test_a_standing_night_carries_the_streak(tmp_path):
    results, posted = _run_nights(tmp_path, [_ERROR, _ERROR, _FILED_NOTHING, _ERROR])

    assert len(_streak_escalations(posted)) == 1
    assert [alarm.kind for alarm in results[-1].alarms] == [
        nightly.AlarmKind.COMPLETION_ERROR_STREAK,
    ]


@pytest.mark.parametrize('concluded', [
    pytest.param(_NOT_YET, id='exit-75'),
    pytest.param(_PASS, id='exit-0'),
])
def test_a_concluded_check_resets_the_streak(tmp_path, concluded):
    _results, posted = _run_nights(tmp_path, [_ERROR, _ERROR, concluded, _ERROR, _ERROR])

    assert _streak_escalations(posted) == []
    assert _recorded_streak() == 2


def test_the_policy_re_arms_after_a_reset(tmp_path):
    _results, posted = _run_nights(tmp_path, [_ERROR] * 3 + [_PASS] + [_ERROR] * 3)

    assert len(_streak_escalations(posted)) == 2


def test_an_escalation_that_did_not_land_is_retried_until_one_does(tmp_path):
    results, posted = _run_nights(tmp_path, [_ERROR] * 5, refuse_first_streak_post=True)

    landed = [
        [alarm.landed for alarm in result.alarms
         if alarm.kind is nightly.AlarmKind.COMPLETION_ERROR_STREAK]
        for result in results
    ]
    assert landed == [[], [], [False], [True], []]
    assert len(_streak_escalations(posted)) == 2
    prior = trickle_state.read_prior(_PROJECT_ID)
    assert prior is not None
    assert prior.completion_escalated is True


def test_a_refused_post_is_retried_on_a_standing_night(tmp_path):
    results, _posted = _run_nights(
        tmp_path, [_ERROR, _ERROR, _ERROR, _FILED_NOTHING], refuse_first_streak_post=True,
    )

    assert [alarm.landed for alarm in results[-1].alarms] == [True]
