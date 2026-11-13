"""The pre-commit hook's module checks are the merge gate's, read from one registry.

Task 5338 found the hook's hand-kept PYRIGHT_PACKAGES list three members short
of the gate; task 8531 retired the list. hooks/project-checks now runs what
scripts/hook_module_checks.py derives from the gate's own registry
(verify_plan.py::effective_merge_module_configs), so this guard drives the REAL
helper over this checkout: one staged probe per registered module, every lint
and type leg the gate configures must be listed, with its command verbatim,
``scripts`` included, which the retired list never named. Each probe must
also list its OWN module's legs: the shared and escalation probes fan out to
every type leg, so the union alone would hide a module whose own probe
dropped its type check.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hook_checks_fixture import HELPER_TIMEOUT_SECS, parse_listing
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


def _configured_legs(mc: ModuleConfig) -> dict[str, str]:
    legs = (('lint', mc.lint_command), ('type', mc.type_check_command))
    return {f'module:{mc.prefix}:{leg}': command for leg, command in legs if command is not None}


def test_the_hooks_module_legs_are_the_gates(
    root_config: OrchestratorConfig,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    from orchestrator.verify_plan import effective_merge_module_configs

    registry = effective_merge_module_configs(root_config, list(discover_module_configs().values()))
    own_legs = {mc.prefix: _configured_legs(mc) for mc in registry}
    expected = {unit: command for legs in own_legs.values() for unit, command in legs.items()}
    listings = _probe_listings(list(own_legs))
    emitted: dict[str, str] = {}
    for units in listings.values():
        for unit, command in units:
            assert emitted.setdefault(unit, command) == command, f'{unit} listed with two commands'
    unlisted_own = {
        prefix: sorted(legs.items() - set(listings[prefix]))
        for prefix, legs in own_legs.items()
        if legs.items() - set(listings[prefix])
    }

    assert 'module:scripts:type' in own_legs['scripts']
    assert unlisted_own == {}
    assert emitted == expected
