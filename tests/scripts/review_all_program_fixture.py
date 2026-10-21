"""Shared helpers for the review-all program tooling tests.

``skills/review-all/scripts/program/`` holds stdlib scripts loaded by path, not
an importable package, and tests may not live under ``skills/``
(``test_skills_module_config_decision.py::test_skills_has_no_tests_of_its_own``).
This non-test module is importable because ``tests/scripts/conftest.py``
inserts this directory on ``sys.path``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
PROGRAM_DIR = REPO_ROOT / 'skills' / 'review-all' / 'scripts' / 'program'
FIXTURE_DIR = PROGRAM_DIR / 'fixture'
STATE_PATH = PROGRAM_DIR / 'state.py'
STATE_MODULE_NAME = 'review_all_program_state'


def load_state() -> types.ModuleType:
    """Load ``state.py`` by path, source-authoritative.

    The cached bytecode is unlinked first: the stale-bytecode guard of
    ``test_find_escalation_port.py::_load_find_escalation_port``. The module is
    registered in ``sys.modules`` before it executes, as dataclasses resolving
    their own module expect.
    """
    assert STATE_PATH.is_file(), f'module under test is missing: {STATE_PATH}'
    Path(importlib.util.cache_from_source(str(STATE_PATH))).unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(STATE_MODULE_NAME, STATE_PATH)
    assert spec is not None and spec.loader is not None, f'no spec for {STATE_PATH}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[STATE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
