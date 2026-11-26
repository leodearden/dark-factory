"""A registered module whose whole-tree guard family gives the narrow merge gate a command to run (task 5411)."""
from __future__ import annotations

import yaml

GUARD = 'pkg/tests/test_guard.py'
SOURCE = 'pkg/src/pkg/mod.py'
GUARD_SOURCE = (
    'import pytest\n'
    'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
)
REGISTERED_MODULE: dict[str, str] = {
    'pkg/orchestrator.yaml': yaml.safe_dump({
        'test_command': 'cd pkg && uv run pytest',
        'lint_command': 'cd pkg && uv run ruff check',
        'type_check_command': 'cd pkg && uv run pyright',
    }),
    SOURCE: '"""Module."""\nx = 1\n',
    GUARD: GUARD_SOURCE,
    'pkg/tests/test_plain.py': 'def test_plain():\n    pass\n',
}
