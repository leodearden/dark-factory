"""hooks/project-checks' MAGICMOCK_TEST_DIRS must list what the merge gate's bare-MagicMock leg scans."""
from __future__ import annotations

import pathlib
import re

import pytest
import verify_command_invariants as vci
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
HOOK_PATH = REPO_ROOT / "hooks" / "project-checks"
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

_MAGICMOCK_DIRS_DECL = re.compile(r"^MAGICMOCK_TEST_DIRS=\(([^)]*)\)", re.MULTILINE)
_MAGICMOCK_CHECKER = "check_bare_magicmock_config.py"


def _hook_magicmock_test_dirs(hook_text: str) -> list[str]:
    match = _MAGICMOCK_DIRS_DECL.search(hook_text)
    assert match, (
        "no MAGICMOCK_TEST_DIRS=(...) declaration found in hooks/project-checks "
        "(task 6308); this guard would otherwise pass vacuously"
    )
    return match.group(1).split()


def _gate_magicmock_test_dirs(lint_command: str) -> list[str]:
    label = "dark-factory-orchestrator.yaml lint_command (task 6308)"
    segment = vci.required_segment(lint_command, _MAGICMOCK_CHECKER, label=label)
    return vci.positional_targets(segment, _MAGICMOCK_CHECKER, path_anchor=True, label=label)


def test_hook_magicmock_test_dirs_match_the_fleet_lint_command() -> None:
    lint_command = yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]
    gate_dirs = _gate_magicmock_test_dirs(lint_command)
    hook_dirs = _hook_magicmock_test_dirs(HOOK_PATH.read_text(encoding="utf-8"))

    assert gate_dirs, f"lint_command's magicmock leg names no directories: {lint_command!r}"
    assert hook_dirs == gate_dirs, (
        "hooks/project-checks MAGICMOCK_TEST_DIRS has drifted from the "
        "`check_bare_magicmock_config.py` leg of dark-factory-orchestrator.yaml's "
        "lint_command (task 6308).\n"
        f"  MISSING from the hook: {[d for d in gate_dirs if d not in hook_dirs]}\n"
        f"  EXTRA in the hook: {[d for d in hook_dirs if d not in gate_dirs]}\n"
        "Widen both together; the yaml leg is the source of truth."
    )


def test_hook_magicmock_test_dirs_extractor_refuses_a_missing_declaration() -> None:
    with pytest.raises(AssertionError, match="MAGICMOCK_TEST_DIRS"):
        _hook_magicmock_test_dirs("RUFF_TARGETS=(a b)\n")


def test_gate_magicmock_test_dirs_reads_only_the_magicmock_leg() -> None:
    chained = (
        "uv run ruff check alpha && "
        "python3 fused-memory/scripts/check_bare_magicmock_config.py alpha/tests beta/tests"
    )
    assert _gate_magicmock_test_dirs(chained) == ["alpha/tests", "beta/tests"]
