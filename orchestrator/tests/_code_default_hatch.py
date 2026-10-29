"""Runtime check that the calling test runs under ``code_default_config``.

A test helper that builds an ``OrchestratorConfig`` meant to hold package
defaults is correct only if every test that reaches it, directly or through
another helper, carries ``@pytest.mark.usefixtures("code_default_config")``.
Neither ratchet in ``test_code_default_hatch_guard`` sees a caller that
forgets: the blanking ratchet looks for writes of ``ORCH_CONFIG_PATH`` and the
default-assertion scan for literal asserts, while the config such a caller
gets silently reads the operational yaml that the autouse
``_isolate_orch_config`` pins.  A helper calls :func:`require_code_default_config`
before it builds its config, so that dependency fails loudly instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def require_code_default_config() -> None:
    """Fail the calling test unless ``ORCH_CONFIG_PATH`` names an absent file.

    ``code_default_config`` (``conftest.py``) leaves exactly that state, and
    it is the only one in which a config reads package defaults wherever the
    process runs: a path to an existing file loads that file, and an empty or
    unset variable falls back to a ``config.yaml`` relative to the cwd.
    """
    named = os.environ.get('ORCH_CONFIG_PATH', '')
    if named and not Path(named).exists():
        return
    pytest.fail(
        f'this helper builds a config that must hold package defaults, but '
        f'ORCH_CONFIG_PATH is {named!r}, so the config would read '
        f'{"that file" if named else "a cwd-relative config.yaml"}. Mark '
        f'every test that reaches the helper with '
        f'@pytest.mark.usefixtures("code_default_config").'
    )
