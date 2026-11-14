"""Tests for scripts/sitting/trigger_state.py — the sitting trigger evaluator's last-run file (task 8425)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import trigger_state as mod

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAN_AT = '2026-11-06T05:00:00+00:00'
LAUNCHED_AT = '2026-11-06T04:00:00+00:00'


def _state(**overrides) -> mod.TriggerLastRun:
    fields = {'ran_at': RAN_AT, 'outcome': mod.TriggerOutcome.LAUNCHED, 'reason': 'red main on dark_factory',
              'last_launch_at': LAUNCHED_AT, 'lane': 't3'}
    return mod.TriggerLastRun(**{**fields, **overrides})


def test_an_absent_file_is_none(tmp_path):
    assert mod.load(tmp_path / 'trigger-last-run.json') is None


def test_a_saved_state_loads_back_equal(tmp_path):
    path = tmp_path / 'nested' / 'trigger-last-run.json'

    mod.save(path, _state())

    assert mod.load(path) == _state()


def test_a_run_that_never_launched_has_no_launch_instant(tmp_path):
    path = tmp_path / 'trigger-last-run.json'
    never = _state(outcome=mod.TriggerOutcome.NOT_FIRED, reason='nothing fired', last_launch_at=None, lane=None)

    mod.save(path, never)

    assert mod.load(path) == never


def test_the_outcomes_are_the_evaluators_four():
    assert {str(outcome) for outcome in mod.TriggerOutcome} == {'not_fired', 'launched', 'absent', 'suppressed'}


@pytest.mark.parametrize('field', ['ran_at', 'last_launch_at'])
def test_a_stamp_that_is_not_iso_is_refused_on_construction(field):
    with pytest.raises(ValueError):
        _state(**{field: 'yesterday'})


def _valid_body() -> dict:
    return json.loads(_state().to_json())


@pytest.mark.parametrize(('label', 'body'), [
    ('not json', '{not json'),
    ('missing key', json.dumps({k: v for k, v in _valid_body().items() if k != 'lane'})),
    ('unknown key', json.dumps({**_valid_body(), 'extra': 1})),
    ('unknown outcome', json.dumps({**_valid_body(), 'outcome': 'exploded'})),
    ('bad stamp', json.dumps({**_valid_body(), 'ran_at': 'yesterday'})),
    ('not an object', json.dumps([1, 2])),
])
def test_an_unreadable_body_names_the_path_and_reason(tmp_path, label, body):
    path = tmp_path / 'trigger-last-run.json'
    path.write_text(body)

    with pytest.raises(mod.TriggerStateUnreadable) as caught:
        mod.load(path)

    assert caught.value.path == path
    assert caught.value.reason
    assert str(path) in str(caught.value)


def test_the_default_path_is_under_the_repo_sitting_state():
    assert mod.DEFAULT_PATH == REPO_ROOT / 'data' / 'sitting' / 'trigger-last-run.json'
