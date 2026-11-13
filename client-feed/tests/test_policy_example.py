"""The worked publisher policy for this repository, client-feed/examples/dark-factory.yaml."""

from pathlib import Path

import yaml

from client_feed.policy import Policy

EXAMPLE = Path(__file__).parents[1] / 'examples' / 'dark-factory.yaml'

_MEMBER_CODE = ('client-feed/src/client_feed/policy.py', 'orchestrator/tests/test_verify.py')
_FACTORY_CONFIGURATION = (
    'dark-factory-orchestrator.yaml',
    '.mcp.json',
    'CLAUDE.md',
    'plans/client-feed-prd.md',
    'scripts/setup-host.sh',
    'client-feed/orchestrator.yaml',
)


def _example() -> Policy:
    return Policy.model_validate(yaml.safe_load(EXAMPLE.read_text(encoding='utf-8')))


def test_the_worked_example_is_a_valid_policy() -> None:
    policy = _example()

    assert policy.project_id == 'dark-factory'
    assert policy.out_dir.name == 'out'
    assert policy.in_dir.name == 'in'
    assert policy.out_dir.parent == policy.in_dir.parent


def test_the_worked_example_shows_member_code_and_hides_the_factorys_configuration() -> None:
    policy = _example()

    assert [path for path in _MEMBER_CODE if not policy.is_visible(path)] == []
    assert [path for path in _FACTORY_CONFIGURATION if policy.is_visible(path)] == []
