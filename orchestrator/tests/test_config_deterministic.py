"""DeterministicConfig: the re-arming predicate's stall cap and its green-tier reload.

PRD plans/completion-driven-triggers-prd.md C1: after this many consecutive
exit-75 "not yet" verdicts the runner files milestone_check_stalled.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    PREDICATE_RECHECK_STALL_CAP,
    RELOADABLE_FIELDS,
    DeterministicConfig,
    OrchestratorConfig,
    apply_reload,
)


class TestDefaults:
    def test_stall_cap_defaults_to_the_module_constant(self):
        assert DeterministicConfig().predicate_recheck_stall_cap == 30 == PREDICATE_RECHECK_STALL_CAP

    @pytest.mark.usefixtures('code_default_config')
    def test_orchestrator_config_carries_the_default_submodel(self):
        assert OrchestratorConfig().deterministic == DeterministicConfig()


class TestBounds:
    def test_a_cap_below_one_is_rejected(self):
        with pytest.raises(ValidationError, match='predicate_recheck_stall_cap'):
            DeterministicConfig(predicate_recheck_stall_cap=0)

    def test_a_cap_of_one_is_accepted(self):
        assert DeterministicConfig(predicate_recheck_stall_cap=1).predicate_recheck_stall_cap == 1


@pytest.mark.usefixtures('code_default_config')
class TestReloadDisposition:
    @pytest.mark.parametrize('leaf', list(DeterministicConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        assert f'deterministic.{leaf}' in RELOADABLE_FIELDS

    def test_a_retune_applies_in_place_to_held_references(self):
        live = OrchestratorConfig()
        held = live.deterministic
        fresh = OrchestratorConfig(deterministic=DeterministicConfig(predicate_recheck_stall_cap=2))

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        assert 'deterministic.predicate_recheck_stall_cap' in report['applied']
        assert 'deterministic.predicate_recheck_stall_cap' not in report['restart_required']
        assert held.predicate_recheck_stall_cap == 2
