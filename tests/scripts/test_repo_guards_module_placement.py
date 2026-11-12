"""tests/repo_guards is a registered, never-cached module, and the repo-wide guards moved into it stay there.

A guard whose subject is the committed tree outside its own module cannot be
keyed soundly by that module's cache key, so it lives in tests/repo_guards,
whose verify result is never cached. This file pins that placement from
OUTSIDE the module: a guard living inside tests/repo_guards would vanish with
the module and pass vacuously. tests/scripts is no workspace member, so its own
key already covers the whole tree and these reads cannot escape it.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PREFIX = 'tests/repo_guards'
MODULE_CONFIG_PATH = REPO_ROOT / MODULE_PREFIX / 'orchestrator.yaml'

#: Below the family ceiling, so a wedged collection fails this test with its
#: captured output rather than being killed by pytest-timeout.
COLLECT_TIMEOUT_SECS = 600

#: Node-id substrings that must be collected from tests/repo_guards.
MOVED_GUARDS = (
    'test_reify_closure_staleness_sweep_retired.py',
    'test_design_invariants_consistency.py',
)

#: Repo-relative homes the whole-file moves vacated.
VACATED_PATHS = (
    'tests/scripts/test_reify_closure_staleness_sweep_retired.py',
    'scripts/tests/test_design_invariants_consistency.py',
)


@functools.cache
def _collected_node_ids() -> tuple[str, ...]:
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            '-p', 'no:cacheprovider', f'{MODULE_PREFIX}/',
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=COLLECT_TIMEOUT_SECS,
    )
    assert result.returncode == 0, (
        f'collecting {MODULE_PREFIX}/ exited {result.returncode}:\n'
        f'{result.stdout}\n{result.stderr}'
    )
    return tuple(line for line in result.stdout.splitlines() if '::' in line)


def test_the_module_is_registered_and_never_cached(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    discovered = discover_module_configs()
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_CONFIG_PATH.relative_to(REPO_ROOT)} is not registered by the '
        f'production module-config walk. Discovered: {sorted(discovered)}'
    )
    assert discovered[MODULE_PREFIX].verify_cache == 'never', (
        f'{MODULE_PREFIX} must declare verify_cache: never, because its guards '
        'read the tree outside any cache key'
    )


def test_every_moved_guard_is_collected_from_the_module() -> None:
    collected = _collected_node_ids()
    missing = [
        guard for guard in MOVED_GUARDS
        if not any(guard in node_id for node_id in collected)
    ]
    assert not missing, (
        f'these moved guards are not collected from {MODULE_PREFIX}/: {missing}'
    )


def test_no_moved_file_remains_at_its_old_path() -> None:
    remaining = [path for path in VACATED_PATHS if (REPO_ROOT / path).exists()]
    assert not remaining, f'moved into {MODULE_PREFIX}/ but still present: {remaining}'
