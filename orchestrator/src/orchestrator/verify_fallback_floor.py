"""The no-module-config verify fallback's test-slot policy: when it runs the configured test_command whole, and when a slot that ran nothing is the no_tests_ran outcome.

Rationale: docs/fallback-test-floor.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from orchestrator.config import ModuleConfig
from orchestrator.verify_categories import FailureCategory, _assert_sentinels_disjoint

# Source the Python fallback cannot derive a test target for.
NON_PYTHON_SOURCE_EXTENSIONS: tuple[str, ...] = ('.rs',)

BARE_PYTEST_DEFAULT = 'pytest'

NO_TESTS_RAN_CATEGORY = 'no_tests_ran'
_assert_sentinels_disjoint((NO_TESTS_RAN_CATEGORY,), FailureCategory)

# Every reason starts with 'pytest:': verify_plan.planned_run_leg routes a
# PlannedRun to its test slot by that prefix.
CONFIGURED_SUITE_REASON = (
    'pytest: no test target derivable from the touched .py file(s) — configured '
    'test_command runs verbatim as the fallback floor (task 7041)'
)
MERGE_BARE_DEFAULT_REASON = (
    "pytest: no test target derivable and only the bare 'pytest' default is "
    'configured — the merge role runs it whole, so a tree with no tests fails '
    '(rc=5) instead of reading green (task 7041)'
)


def has_configured_suite(test_command: str | None) -> bool:
    return bool(test_command) and test_command != BARE_PYTEST_DEFAULT


def non_python_reason(path: str) -> str:
    return (
        f'pytest: diff carries non-Python source ({path}) that no Python-derived '
        'target covers — configured test_command runs verbatim (task 8714)'
    )


def fallback_test_floor_reason(
    files: Sequence[str],
    test_command: str | None,
    role: Literal['merge', 'task'],
    *,
    derived_target: bool,
) -> str | None:
    if not test_command:
        return None
    non_python = next((f for f in files if f.endswith(NON_PYTHON_SOURCE_EXTENSIONS)), None)
    if non_python is not None:
        return non_python_reason(non_python)
    if derived_target:
        return None
    if has_configured_suite(test_command):
        return CONFIGURED_SUITE_REASON
    if role == 'merge':
        return MERGE_BARE_DEFAULT_REASON
    return None


def apply_fallback_test_floor(
    fallback: ModuleConfig,
    files: Sequence[str],
    test_command: str | None,
    role: Literal['merge', 'task'],
) -> tuple[ModuleConfig, str | None]:
    reason = fallback_test_floor_reason(
        files, test_command, role, derived_target=bool(fallback.test_command),
    )
    if reason is None:
        return fallback, None
    return replace(fallback, test_command=test_command), reason


def fallback_ran_no_tests(
    executed_test_command: str | None, configured_test_command: str | None,
) -> bool:
    """The fallback's test slot executed nothing although the project configures a test command."""
    return executed_test_command is None and bool(configured_test_command)
