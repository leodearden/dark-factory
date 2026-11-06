"""The eval config registered under a name, for tests that pick candidates by name."""

from __future__ import annotations

from orchestrator.evals.configs import EvalConfig, get_config_by_name


def named_eval_config(name: str) -> EvalConfig:
    """The config registered under *name*; a missing one fails the test that asked."""
    cfg = get_config_by_name(name)
    assert cfg is not None, f'no eval config named {name!r}'
    return cfg
