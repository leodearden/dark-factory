"""Where OrchestratorConfig.verify_runners comes from, and which names it may carry."""

import pytest
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig


def _runner(name: str, *, enabled: bool = True) -> dict[str, object]:
    return {
        'name': name,
        'ssh_host': f'{name}-ssh',
        'git_remote': f'{name}-remote',
        'df_checkout_path': '/df',
        'enabled': enabled,
    }


class TestRunnerNames:
    def test_two_runners_with_one_name_are_refused(self):
        with pytest.raises(ValidationError, match=r"(?s)must be unique.*'box'"):
            OrchestratorConfig(verify_runners=[_runner('box'), _runner('box', enabled=False)])

    def test_a_runner_named_after_the_local_slot_is_refused(self):
        with pytest.raises(ValidationError, match=r"(?s)none may be 'local'.*'local'"):
            OrchestratorConfig(verify_runners=[_runner('local')])

    def test_assigning_a_duplicate_list_is_refused_and_keeps_the_old_one(self):
        config = OrchestratorConfig(verify_runners=[_runner('box')])
        original = list(config.verify_runners)

        with pytest.raises(ValidationError, match='must be unique'):
            config.verify_runners = [_runner('dup'), _runner('dup')]

        assert config.verify_runners == original
