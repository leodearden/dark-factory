"""Tests for check_module_local_testclient.py lint checker.

Tests for the AST-based lint check that flags a pytest FIXTURE constructing its
own ``TestClient`` inside a ``dashboard/tests/test_*.py`` module, where
``conftest.py`` already provides a shared module-scoped ``_client``.

See task 4485.  This lint replaces the AST guard class that task 3571 deleted
from ``dashboard/tests/test_jsx_source_helpers.py`` (commit 9096654196) after
review rejected a source-lint living inside the pytest suite.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

# Load the checker script via importlib to avoid sys.path pollution.
# fused-memory/scripts/ is not on PYTHONPATH per pyproject.toml (pythonpath=['src']).
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_module_local_testclient.py'


def _load_checker() -> types.ModuleType:
    """Load the checker module from its script path."""
    spec = importlib.util.spec_from_file_location('check_module_local_testclient', SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_checker = _load_checker()
find_violations = _checker.find_violations


# A module-scoped fixture building its own TestClient — the exact shape that
# `dashboard/tests/test_tab_tasks_offline_banner.py` carried, byte-identical to
# conftest.py's own `_client`, while a dedup task and its purpose-built guard
# both reported "conftest is the only home".
_MODULE_LOCAL_CLIENT_FIXTURE = '''\
import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c
'''


def _fixture_source(name: str) -> str:
    """A module-local client fixture under an arbitrary *name*."""
    return _MODULE_LOCAL_CLIENT_FIXTURE.replace('def _client()', f'def {name}()')


class TestFindViolationsBasicDetection:
    """Core detection: a pytest fixture that constructs its own TestClient is flagged."""

    def test_flags_module_scoped_fixture_constructing_its_own_testclient(self):
        """The canonical offender: `with TestClient(app) as c: yield c` in a fixture body."""
        violations = find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'test_x.py')

        assert len(violations) == 1
        v = violations[0]
        assert v.filename == 'test_x.py'
        # 1-based lineno of the `TestClient(` call, not of the fixture def.
        assert v.lineno == 10
        assert _MODULE_LOCAL_CLIENT_FIXTURE.splitlines()[v.lineno - 1].lstrip().startswith(
            'with TestClient(app)'
        )
        assert v.col_offset == _MODULE_LOCAL_CLIENT_FIXTURE.splitlines()[v.lineno - 1].index(
            'TestClient('
        )

    def test_violation_message_names_the_remedy_and_the_pragma(self):
        """The message must be self-service: it names conftest's `_client` and the escape hatch."""
        (v,) = find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'test_x.py')

        assert 'conftest' in v.message
        assert '_client' in v.message
        # The suppression pragma is the answer to "but my fixture is legitimate",
        # so it must travel with the rejection rather than live in a wiki.
        assert 'noqa: module-local-testclient' in v.message

    def test_detection_is_fixture_name_agnostic(self):
        """A fixture named `_ro_client` or `_c` is flagged identically to one named `_client`.

        The deleted guard keyed on the fixture NAME, so renaming the copy walked
        straight past it.  The rule is about the CONSTRUCTION, not the binding.
        """
        for name in ('_client', '_ro_client', '_c', 'app_client'):
            violations = find_violations(_fixture_source(name), 'test_x.py')
            assert len(violations) == 1, f'fixture named {name!r} was not flagged'
            assert violations[0].lineno == 10

    def test_two_constructions_in_one_fixture_yield_two_violations(self):
        """One Violation per construction, sorted by (lineno, col_offset)."""
        source = '''\
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def _pair():
    a = TestClient(app)
    b = TestClient(other_app)
    yield a, b
'''
        violations = find_violations(source, 'test_x.py')
        assert [v.lineno for v in violations] == [7, 8]

    def test_bare_fixture_decorator_without_call_is_detected(self):
        """`@pytest.fixture` (no parentheses) marks a fixture just as `@pytest.fixture(...)` does."""
        source = '''\
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def _client():
    with TestClient(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1

    def test_syntax_error_returns_no_violations(self):
        """An unparseable file is ruff's problem, not this checker's."""
        assert find_violations('def broken(:\n', 'test_x.py') == []
