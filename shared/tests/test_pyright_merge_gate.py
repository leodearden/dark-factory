"""Regression test pinning the pyright merge-gate wiring for the shared subproject.

The merge gate's scoped verification (orchestrator/src/orchestrator/verify.py:
scope_module_config) builds the pyright command from ModuleConfig.type_check_command,
which is loaded from each subproject's orchestrator.yaml via the _OVERRIDABLE_FIELDS
loader (orchestrator/src/orchestrator/config.py).

If shared/orchestrator.yaml loses its type_check_command entry, pyright will
silently stop running on shared's changed files — including test files — during
the merge gate, and type errors in tests (like those fixed in task 1476) can slip
through undetected.

This test fails loudly at normal ``pytest`` time — before the real failure mode
(undetected type errors in shared tests) can bite.

See task 1477 for full context and the gate-wiring fix.
"""

from __future__ import annotations

import re
from pathlib import Path

# shared/orchestrator.yaml — the merge-gate config file for this subproject.
_ORCHESTRATOR_YAML = Path(__file__).parent.parent / 'orchestrator.yaml'

# Regex to extract the type_check_command value from a YAML line.
# Matches: type_check_command: "..." or type_check_command: '...' or unquoted.
_TYPE_CHECK_PATTERN = re.compile(
    r'^type_check_command:\s*["\']?(.+?)["\']?\s*$',
    re.MULTILINE,
)


def _extract_type_check_command(content: str) -> str | None:
    """Extract the type_check_command value from orchestrator.yaml text.

    Returns the stripped command string, or None when the key is absent.
    """
    match = _TYPE_CHECK_PATTERN.search(content)
    return match.group(1).strip() if match else None


def test_shared_orchestrator_yaml_has_type_check_command() -> None:
    """shared/orchestrator.yaml must define type_check_command.

    Without this key, ModuleConfig.type_check_command is None and
    scope_module_config returns None for the pyright command — pyright
    NEVER runs on shared's changed files during the merge gate.
    Type errors in shared tests (like those fixed in task 1476) slip
    through undetected.

    If this test fails: add ``type_check_command`` to
    shared/orchestrator.yaml mirroring orchestrator/orchestrator.yaml.
    See task 1477.
    """
    assert _ORCHESTRATOR_YAML.is_file(), (
        f'shared/orchestrator.yaml not found at {_ORCHESTRATOR_YAML}. '
        'This file is the merge-gate config for the shared subproject. '
        'See task 1477.'
    )
    content = _ORCHESTRATOR_YAML.read_text()
    cmd = _extract_type_check_command(content)
    assert cmd is not None, (
        f'type_check_command is missing from {_ORCHESTRATOR_YAML}. '
        'Without this key pyright never runs on shared changed files '
        'during the merge gate. Add: '
        'type_check_command: "uv run --project shared --directory shared '
        'pyright src/ tests/". See task 1477.'
    )


def test_shared_type_check_command_invokes_pyright() -> None:
    """shared/orchestrator.yaml type_check_command must invoke pyright.

    The merge gate uses type_check_command specifically to run pyright (not
    ruff or another linter). If the command does not contain 'pyright',
    the type-checking gate is absent for shared.

    See task 1477.
    """
    content = _ORCHESTRATOR_YAML.read_text()
    cmd = _extract_type_check_command(content)
    assert cmd is not None, (
        f'type_check_command is missing from {_ORCHESTRATOR_YAML}. '
        'See task 1477.'
    )
    assert 'pyright' in cmd, (
        f'type_check_command in {_ORCHESTRATOR_YAML} does not invoke pyright: '
        f'{cmd!r}. The merge gate needs pyright (not ruff) to catch type errors '
        'in shared test files. See task 1477.'
    )


def test_shared_type_check_command_covers_tests_directory() -> None:
    """shared/orchestrator.yaml type_check_command must cover the tests/ directory.

    The scope_module_config helper in verify.py narrows the command to the task's
    CHANGED .py files, but those files must be in the command's covered paths.
    If 'tests' is absent from type_check_command, changed test files in
    shared/tests/ are excluded from pyright during the merge gate.

    See task 1477.
    """
    content = _ORCHESTRATOR_YAML.read_text()
    cmd = _extract_type_check_command(content)
    assert cmd is not None, (
        f'type_check_command is missing from {_ORCHESTRATOR_YAML}. '
        'See task 1477.'
    )
    assert 'tests' in cmd, (
        f'type_check_command in {_ORCHESTRATOR_YAML} does not cover tests/: '
        f'{cmd!r}. Changed test files in shared/tests/ would be excluded '
        'from pyright during the merge gate. The command must include "tests/" '
        '(or a path containing "tests"). See task 1477.'
    )
