"""``before_done``'s key set is closed at the submit guard.

PRD plans/completion-driven-triggers-prd.md C2 "Capability report and version
skew". ``deterministic_task_error`` refuses a ``before_done`` key outside
``BeforeDone``'s declared fields with ``error_type='UnsupportedTaskMetadata'``,
naming the key and the running server's capabilities, so a client that meets
an older server learns of the skew instead of having the key silently ignored.
"""

from __future__ import annotations

import os

import pytest
from shared.task_metadata import TASK_METADATA_CAPABILITIES, BeforeDone
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware.deterministic_task_guard import deterministic_task_error

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DATED = {'mode': 'dated', 'at': '2026-11-01T00:00:00+00:00'}


def _install_script(tmp_path) -> None:
    script = tmp_path / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    os.chmod(script, 0o755)


def _predicate(tmp_path, **before_done_fields) -> dict:
    _install_script(tmp_path)
    return {
        'milestone': DATED,
        'before_done': {
            'kind': 'predicate', 'script': 'run.sh', 'timeout_secs': 60, **before_done_fields,
        },
    }


def _deploy(tmp_path, **before_done_fields) -> dict:
    _install_script(tmp_path)
    return {'before_done': {'script': 'run.sh', 'timeout_secs': 60, **before_done_fields}}


def _assert_unsupported(result: dict | None, *, keys: list[str]) -> None:
    assert result is not None
    assert result['error_type'] == 'UnsupportedTaskMetadata'
    assert result['unsupported_keys'] == keys
    for key in keys:
        assert key in result['error'], (key, result['error'])
    assert result['task_metadata_capabilities'] == sorted(TASK_METADATA_CAPABILITIES)
    for capability in TASK_METADATA_CAPABILITIES:
        assert capability in result['error'], (capability, result['error'])
    assert isinstance(result['hint'], str)
    assert result['hint']


class TestUnknownKeysRefused:
    def test_unknown_key_on_a_predicate(self, tmp_path) -> None:
        meta = _predicate(tmp_path, recheck_mins=86400)
        _assert_unsupported(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            keys=['recheck_mins'],
        )

    def test_several_unknown_keys_are_reported_sorted(self, tmp_path) -> None:
        meta = _predicate(tmp_path, zeta=1, alpha=2)
        _assert_unsupported(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            keys=['alpha', 'zeta'],
        )

    def test_unknown_key_is_named_before_a_later_defect(self, tmp_path) -> None:
        meta = _predicate(tmp_path, recheck_mins=86400)
        meta['before_done']['script'] = 'missing.sh'
        _assert_unsupported(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            keys=['recheck_mins'],
        )

    @pytest.mark.parametrize('kind_fields', [{}, {'kind': 'deploy'}], ids=['default', 'explicit'])
    def test_unknown_key_on_a_deploy(self, tmp_path, kind_fields: dict) -> None:
        meta = _deploy(tmp_path, recheck_mins=86400, **kind_fields)
        _assert_unsupported(
            deterministic_task_error('deterministic', meta, str(tmp_path)),
            keys=['recheck_mins'],
        )


_FULL_DEPLOY = {
    'args': ['--dry-run'],
    'env': {'MODE': 'check'},
    'cwd': '.',
    'target_unit': 'orchestrator.service',
    'kind': 'deploy',
    'script_root': 'project',
}
_FULL_PREDICATE = {
    'args': ['--threshold', '0.8'],
    'env': {'MODE': 'check'},
    'cwd': '.',
    'kind': 'predicate',
    'script_root': 'project',
    'recheck_secs': 86400,
}


class TestDeclaredKeysAccepted:
    @pytest.mark.parametrize(
        ('build', 'fields'),
        [(_deploy, _FULL_DEPLOY), (_predicate, _FULL_PREDICATE)],
        ids=['deploy', 'predicate'],
    )
    def test_every_declared_field_is_accepted(self, tmp_path, build, fields: dict) -> None:
        meta = build(tmp_path, **fields)
        assert deterministic_task_error('deterministic', meta, str(tmp_path)) is None

    def test_the_accepted_rows_cover_every_declared_field(self) -> None:
        exercised = {'script', 'timeout_secs'} | set(_FULL_DEPLOY) | set(_FULL_PREDICATE)
        assert exercised == set(BeforeDone.model_fields)


def test_normal_task_keeps_the_deterministic_only_refusal(tmp_path) -> None:
    meta = _deploy(tmp_path, recheck_mins=86400)
    result = deterministic_task_error('normal', meta, str(tmp_path))
    assert result is not None
    assert result['error_type'] == 'ValidationError'
    assert 'only valid on deterministic tasks' in result['error']
