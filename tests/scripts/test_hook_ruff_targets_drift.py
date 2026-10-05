"""hooks/project-checks' RUFF_TARGETS must list what the merge gate's ruff leg lints.

Task 5338 (review amendment). The ``ruff check`` leg of
``dark-factory-orchestrator.yaml``'s ``lint_command`` is the source of truth
for which paths get linted. The pre-commit hook's ruff list is a second,
hand-edited copy; it stayed at five packages while the gate walked ten targets,
so a direct commit to main touching ``sampler``, ``cockpit``, ``skills`` or a
root-level non-member file was never ruff-checked by the hook — while the
neighbouring ``PYRIGHT_PACKAGES`` (pinned by
``test_hook_pyright_packages_drift.py``) already type-checked sampler and
cockpit.

Pins the target tokens and their order, read live from both artifacts.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import verify_command_invariants as vci
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
HOOK_PATH = REPO_ROOT / "hooks" / "project-checks"
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

_RUFF_TARGETS_DECL = re.compile(r"^RUFF_TARGETS=\(([^)]*)\)", re.MULTILINE)


def _hook_ruff_targets(hook_text: str) -> list[str]:
    match = _RUFF_TARGETS_DECL.search(hook_text)
    assert match, (
        "no RUFF_TARGETS=(...) declaration found in hooks/project-checks "
        "(task 5338); this guard would otherwise pass vacuously"
    )
    return match.group(1).split()


def _gate_ruff_targets(lint_command: str) -> list[str]:
    label = "dark-factory-orchestrator.yaml lint_command (task 5338)"
    segment = vci.required_segment(lint_command, vci.RUFF, label=label)
    return vci.positional_targets(segment, vci.RUFF, label=label)


def test_hook_ruff_targets_match_the_fleet_lint_command() -> None:
    lint_command = yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]
    gate_targets = _gate_ruff_targets(lint_command)
    hook_targets = _hook_ruff_targets(HOOK_PATH.read_text(encoding="utf-8"))

    assert gate_targets, f"lint_command's ruff leg names no targets: {lint_command!r}"
    assert hook_targets == gate_targets, (
        "hooks/project-checks RUFF_TARGETS has drifted from the `ruff check` leg "
        "of dark-factory-orchestrator.yaml's lint_command (task 5338).\n"
        f"  MISSING from the hook: {[t for t in gate_targets if t not in hook_targets]}\n"
        f"  EXTRA in the hook: {[t for t in hook_targets if t not in gate_targets]}\n"
        "Widen both together; the yaml leg is the source of truth."
    )


def test_hook_ruff_targets_extractor_refuses_a_missing_declaration() -> None:
    with pytest.raises(AssertionError, match="RUFF_TARGETS"):
        _hook_ruff_targets("PYRIGHT_PACKAGES=(a b)\n")


def test_gate_ruff_targets_reads_only_the_ruff_leg() -> None:
    chained = "uv run ruff check alpha beta.py && python3 checker.py alpha/tests"
    assert _gate_ruff_targets(chained) == ["alpha", "beta.py"]
