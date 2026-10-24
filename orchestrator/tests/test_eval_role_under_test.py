"""The role vocabulary an eval cell is stamped with, and EvalConfig's role invariant.

``RoleUnderTest`` values are persisted into every result JSON, so the value set
is a storage contract. ``EvalConfig`` parses its role at construction and admits
only a candidate role: ``end_to_end`` names a methodology stage, not a role a
candidate can vary.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from orchestrator.evals.configs import (
    ARCHITECT_EVAL_CONFIGS,
    EVAL_CONFIGS,
    FINAL_RUN_CONFIGS,
    JUDGE_EVAL_CONFIGS,
    JUDGE_OFAT_IMPLEMENTER_PIN,
    EvalConfig,
    claude_endpoint_candidates,
    codex_pi_candidates,
    ofat_candidates,
)
from orchestrator.evals.metrics import CANDIDATE_ROLES, EvalMetrics, RoleUnderTest


def test_persisted_role_values_are_the_storage_vocabulary():
    assert {r.value for r in RoleUnderTest} == {
        'implementer', 'architect', 'judge', 'end_to_end',
    }


def test_candidate_roles_exclude_the_end_to_end_stage():
    expected = {RoleUnderTest.IMPLEMENTER, RoleUnderTest.ARCHITECT, RoleUnderTest.JUDGE}
    assert expected == CANDIDATE_ROLES


class TestEvalConfigRole:
    def test_defaults_to_implementer(self):
        assert EvalConfig('x', 'claude', 'sonnet', 'high').role is RoleUnderTest.IMPLEMENTER

    def test_end_to_end_is_not_a_candidate_role(self):
        with pytest.raises(ValueError, match='end_to_end') as excinfo:
            EvalConfig('x', 'claude', 'sonnet', 'high', role=RoleUnderTest.END_TO_END)
        assert "'x'" in str(excinfo.value)

    def test_a_str_role_is_parsed_to_the_enum_member(self):
        kwargs: dict[str, Any] = {'role': 'architect'}
        cfg = EvalConfig('x', 'claude', 'sonnet', 'high', **kwargs)
        assert cfg.role is RoleUnderTest.ARCHITECT

    def test_an_unknown_str_role_is_rejected_naming_the_config(self):
        kwargs: dict[str, Any] = {'role': 'reviewer'}
        with pytest.raises(ValueError, match='reviewer') as excinfo:
            EvalConfig('x', 'claude', 'sonnet', 'high', **kwargs)
        assert "'x'" in str(excinfo.value)


def _shipped_candidates() -> list[EvalConfig]:
    return [
        *ofat_candidates(),
        *EVAL_CONFIGS,
        *FINAL_RUN_CONFIGS,
        *ARCHITECT_EVAL_CONFIGS,
        *JUDGE_EVAL_CONFIGS,
        JUDGE_OFAT_IMPLEMENTER_PIN,
        *claude_endpoint_candidates(),
        *codex_pi_candidates(),
    ]


@pytest.mark.parametrize('cfg', _shipped_candidates(), ids=lambda c: c.name)
def test_every_shipped_candidate_carries_a_candidate_role(cfg: EvalConfig):
    assert isinstance(cfg.role, RoleUnderTest)
    assert cfg.role in CANDIDATE_ROLES


class TestEvalMetricsRole:
    def test_role_under_test_defaults_to_none(self):
        assert EvalMetrics().role_under_test is None

    def test_an_enum_role_persists_as_its_string_value(self):
        persisted = json.loads(json.dumps(
            EvalMetrics(role_under_test=RoleUnderTest.JUDGE).to_dict(),
        ))
        assert persisted['role_under_test'] == 'judge'
        assert type(persisted['role_under_test']) is str
