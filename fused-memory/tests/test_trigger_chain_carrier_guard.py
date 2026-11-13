"""``metadata.trigger_chain`` carrier rules at the submit guard.

PRD plans/completion-driven-triggers-prd.md C2. ``deterministic_task_error``
delegates the chain's shape to ``shared.task_metadata.TriggerChain`` and then
checks the carrier: a completion gate is a re-arming predicate on a dated
milestone, a human gate is a pure escalation gate, and neither counts toward
its own run's measure. Every refusal row starts from a valid carrier and breaks
exactly one thing.
"""

from __future__ import annotations

import json
import os
from typing import get_args

import pytest
from shared.task_metadata import TriggerChainRole, TriggerChainSkill

from fused_memory.middleware.deterministic_task_guard import deterministic_task_error

DATED = {'mode': 'dated', 'at': '2026-11-03T00:00:00+00:00'}
DELAYED = {'mode': 'delayed', 'after_secs': 259200}


def _chain(role: str, **fields: object) -> dict:
    return {'role': role, 'skill': 'review', 'run_id': 'review-p-1', **fields}


def completion_gate(tmp_path) -> dict:
    script = tmp_path / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    os.chmod(script, 0o755)
    return {
        'trigger_chain': _chain('completion_gate'),
        'milestone': dict(DATED),
        'before_done': {
            'kind': 'predicate',
            'script': 'run.sh',
            'timeout_secs': 120,
            'recheck_secs': 86400,
            'args': ['--run-id', 'review-p-1'],
        },
    }


def human_gate() -> dict:
    return {'trigger_chain': _chain('human_gate'), 'always_escalates': True}


def _guard(meta: dict | str, tmp_path, task_kind: str = 'deterministic') -> dict | None:
    return deterministic_task_error(task_kind, meta, str(tmp_path))


def _assert_refused(result: dict | None, *, naming: tuple[str, ...] = ()) -> dict:
    assert result is not None
    assert result['error_type'] == 'ValidationError'
    assert 'trigger_chain' in result['error'], result['error']
    for fragment in naming:
        assert fragment in result['error'], (fragment, result['error'])
    assert isinstance(result['hint'], str)
    assert result['hint']
    return result


class TestCarriersAccepted:
    def test_completion_gate(self, tmp_path) -> None:
        assert _guard(completion_gate(tmp_path), tmp_path) is None

    def test_human_gate_without_a_milestone(self, tmp_path) -> None:
        assert _guard(human_gate(), tmp_path) is None

    @pytest.mark.parametrize('milestone', [DELAYED, DATED], ids=['delayed-interim', 'dated'])
    def test_human_gate_with_a_milestone(self, tmp_path, milestone: dict) -> None:
        meta = {**human_gate(), 'milestone': milestone}
        assert _guard(meta, tmp_path) is None

    def test_superseded_chain_task(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        meta['trigger_chain']['superseded_by'] = 'review-p-2'
        assert _guard(meta, tmp_path) is None


class TestShapeRefused:
    def test_slashed_skill_names_the_bare_form(self, tmp_path) -> None:
        meta = human_gate()
        meta['trigger_chain']['skill'] = '/review'
        result = _assert_refused(_guard(meta, tmp_path))
        assert (
            "trigger_chain.skill='/review' must be the bare skill name 'review' "
            '(no leading slash)'
        ) in result['error']
        assert "'review'" in result['hint']
        assert 'no leading slash' in result['hint']

    def test_non_mapping_chain(self, tmp_path) -> None:
        meta = {**human_gate(), 'trigger_chain': 'review'}
        _assert_refused(_guard(meta, tmp_path))

    def test_the_hint_offers_every_role_and_skill_the_model_accepts(self, tmp_path) -> None:
        meta = {**human_gate(), 'trigger_chain': 'review'}
        hint = _assert_refused(_guard(meta, tmp_path))['hint']
        for accepted in (*get_args(TriggerChainRole), *get_args(TriggerChainSkill)):
            assert repr(accepted) in hint, accepted

    def test_run_id_without_the_skill_prefix(self, tmp_path) -> None:
        meta = human_gate()
        meta['trigger_chain']['run_id'] = 'hotspot-survey-p-1'
        _assert_refused(_guard(meta, tmp_path), naming=('run_id',))


class TestCarrierRefused:
    @pytest.mark.parametrize('role', ['completion_gate', 'human_gate'])
    def test_either_role_on_a_normal_task(self, tmp_path, role: str) -> None:
        meta: dict = {'trigger_chain': _chain(role)}
        _assert_refused(_guard(meta, tmp_path, 'normal'), naming=("task_kind='deterministic'",))

    def test_completion_gate_on_a_deploy(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        meta['before_done']['kind'] = 'deploy'
        del meta['before_done']['recheck_secs']
        del meta['before_done']['args']
        _assert_refused(_guard(meta, tmp_path), naming=('predicate',))

    def test_completion_gate_without_recheck_secs(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        del meta['before_done']['recheck_secs']
        _assert_refused(_guard(meta, tmp_path), naming=('recheck_secs',))

    @pytest.mark.parametrize('milestone', [DELAYED, None], ids=['delayed', 'absent'])
    def test_completion_gate_without_a_dated_milestone(
        self, tmp_path, milestone: dict | None
    ) -> None:
        meta = completion_gate(tmp_path)
        if milestone is None:
            del meta['milestone']
        else:
            meta['milestone'] = milestone
        result = _guard(meta, tmp_path)
        assert result is not None
        assert "mode='dated'" in result['error']

    def test_completion_gate_as_a_pure_gate(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        del meta['before_done']
        meta['always_escalates'] = True
        _assert_refused(_guard(meta, tmp_path), naming=('before_done',))

    def test_human_gate_running_a_predicate(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        meta['trigger_chain'] = _chain('human_gate')
        result = _assert_refused(_guard(meta, tmp_path))
        assert 'always_escalates' in result['error'] or 'before_done' in result['error']

    def test_human_gate_with_a_string_always_escalates(self, tmp_path) -> None:
        meta = completion_gate(tmp_path)
        meta['trigger_chain'] = _chain('human_gate')
        meta['always_escalates'] = 'true'
        _assert_refused(_guard(meta, tmp_path), naming=('always_escalates',))

    @pytest.mark.parametrize('role', ['completion_gate', 'human_gate'])
    def test_chain_task_carrying_x_finding_run(self, tmp_path, role: str) -> None:
        meta = completion_gate(tmp_path) if role == 'completion_gate' else human_gate()
        meta['x_finding_run'] = 'review-p-1'
        _assert_refused(_guard(meta, tmp_path), naming=('x_finding_run',))


class TestJsonStringMetadata:
    def test_accepted_as_the_dict(self, tmp_path) -> None:
        assert _guard(json.dumps(completion_gate(tmp_path)), tmp_path) is None

    def test_refused_as_the_dict(self, tmp_path) -> None:
        meta = {**human_gate(), 'x_finding_run': 'review-p-1'}
        assert _guard(json.dumps(meta), tmp_path) == _guard(meta, tmp_path)
        _assert_refused(_guard(json.dumps(meta), tmp_path), naming=('x_finding_run',))
