"""The pre-commit hook's module checks are the merge gate's, read from one registry.

Task 5338 found the hook's hand-kept PYRIGHT_PACKAGES list three members short
of the gate; task 8531 retired the list. hooks/project-checks now runs what
scripts/hook_module_checks.py derives from the gate's own registry
(verify_plan.py::effective_merge_module_configs), so this guard drives the REAL
helper over this checkout: one staged probe per registered module, every lint
and type leg the gate configures must be listed, with its command verbatim.
That includes ``scripts``, which the retired list never named.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from hook_checks_fixture import HELPER_TIMEOUT_SECS, parse_listing

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

REPO_ROOT = Path(__file__).parents[2]
HELPER = REPO_ROOT / 'scripts' / 'hook_module_checks.py'


def _probe_listings(prefixes: list[str]) -> dict[str, list[tuple[str, str]]]:
    procs = {
        prefix: subprocess.Popen(
            [sys.executable, str(HELPER)],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for prefix in prefixes
    }
    listings: dict[str, list[tuple[str, str]]] = {}
    try:
        for prefix, proc in procs.items():
            out, err = proc.communicate(f'{prefix}/_drift_probe.py\n', timeout=HELPER_TIMEOUT_SECS)
            assert proc.returncode == 0, f'helper failed for a {prefix} probe: {err}'
            listings[prefix] = parse_listing(out)
    finally:
        for proc in procs.values():
            proc.kill()
            proc.wait()
    return listings


def test_the_hooks_module_legs_are_the_gates(
    root_config: OrchestratorConfig,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    from orchestrator.verify_plan import effective_merge_module_configs

    registry = effective_merge_module_configs(root_config, list(discover_module_configs().values()))
    expected = {
        f'module:{mc.prefix}:{leg}': command
        for mc in registry
        for leg, command in (('lint', mc.lint_command), ('type', mc.type_check_command))
        if command is not None
    }
    listings = _probe_listings([mc.prefix for mc in registry])
    emitted: dict[str, str] = {}
    for units in listings.values():
        for unit, command in units:
            assert emitted.setdefault(unit, command) == command, f'{unit} listed with two commands'

    assert 'module:scripts:type' in expected
    assert emitted == expected
