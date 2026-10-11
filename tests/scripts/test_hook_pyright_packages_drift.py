"""hooks/project-checks' PYRIGHT_PACKAGES must list what the merge gate type-checks.

Task 5338. ``dark-factory-orchestrator.yaml``'s ``type_check_command`` is the
source of truth for which packages get type-checked. The pre-commit hook's
``PYRIGHT_PACKAGES`` array is a second, hand-edited copy; it drifted to three
members while the gate walked seven, so a direct commit to main touching
``shared``, ``escalation``, ``sampler`` or ``cockpit`` never type-checked the
package it changed.

Pins the package DIRECTORIES and their order (the chain walk's semantics), read
live from both artifacts. Placed in ``tests/scripts/`` so it runs under merge
verify, like the neighbouring mirror guards.
"""
from __future__ import annotations

import pathlib
import re

import verify_command_invariants as vci
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
HOOK_PATH = REPO_ROOT / "hooks" / "project-checks"
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

_PYRIGHT_PACKAGES_DECL = re.compile(r"^PYRIGHT_PACKAGES=\(([^)]*)\)", re.MULTILINE)


def _hook_pyright_packages(hook_text: str) -> list[str]:
    match = _PYRIGHT_PACKAGES_DECL.search(hook_text)
    assert match, (
        "no PYRIGHT_PACKAGES=(...) declaration found in hooks/project-checks "
        "(task 5338); this guard would otherwise pass vacuously"
    )
    return match.group(1).split()


def test_hook_pyright_packages_match_the_fleet_type_check_command() -> None:
    gate_cmd = yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["type_check_command"]
    gate_packages = vci.pyright_clause_cwds(gate_cmd, skip_uv_project=False)
    hook_packages = _hook_pyright_packages(HOOK_PATH.read_text(encoding="utf-8"))

    assert gate_packages, f"type_check_command walked to no pyright clause: {gate_cmd!r}"
    assert hook_packages == gate_packages, (
        "hooks/project-checks PYRIGHT_PACKAGES has drifted from "
        "dark-factory-orchestrator.yaml's type_check_command (task 5338).\n"
        f"  MISSING from the hook: {[p for p in gate_packages if p not in hook_packages]}\n"
        f"  EXTRA in the hook: {[p for p in hook_packages if p not in gate_packages]}\n"
        "Widen both together; the yaml chain is the source of truth."
    )


def test_hook_pyright_packages_extractor_refuses_a_missing_declaration() -> None:
    import pytest

    with pytest.raises(AssertionError, match="PYRIGHT_PACKAGES"):
        _hook_pyright_packages("ALL_PACKAGES=(a b)\n")
