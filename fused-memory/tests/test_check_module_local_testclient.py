"""Tests for check_module_local_testclient.py lint checker.

Tests for the AST-based lint check that flags a pytest FIXTURE constructing its
own ``TestClient`` inside a ``dashboard/tests/test_*.py`` module, where
``conftest.py`` already provides a shared module-scoped ``_client``.

See task 4485.  This lint replaces the AST guard class that task 3571 deleted
from ``dashboard/tests/test_jsx_source_helpers.py`` (commit 9096654196) after
review rejected a source-lint living inside the pytest suite.
"""
from __future__ import annotations

import ast
import importlib.util
import re
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
        assert v.lineno == 9
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
            assert violations[0].lineno == 9

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


class TestFindViolationsResolvesImportAliases:
    """Regression for reviewer objection (c): the trailing-callee-name blind spot.

    The deleted task-3571 guard keyed on the trailing callee NAME, so
    ``from starlette.testclient import TestClient as TC`` followed by ``TC(app)``
    walked straight past it — the exact blind spot its own docstring claimed to
    close.  Matching is now the UNION of an import-alias map and trailing-name
    matching, which is strictly broader than the deleted guard and never narrower.
    """

    def test_aliased_import_is_flagged(self):
        """`from starlette.testclient import TestClient as TC` + `TC(app)` IS flagged."""
        source = '''\
import pytest
from starlette.testclient import TestClient as TC


@pytest.fixture(scope='module')
def _client():
    with TC(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1, 'aliased TestClient import was not resolved'
        assert violations[0].lineno == 7

    def test_fastapi_testclient_import_is_flagged(self):
        """`from fastapi.testclient import TestClient` is the same class from the other package."""
        source = '''\
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    with TestClient(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1
        assert violations[0].lineno == 7

    def test_module_attribute_form_is_flagged(self):
        """`from starlette import testclient` + `testclient.TestClient(app)` IS flagged."""
        source = '''\
import pytest
from starlette import testclient


@pytest.fixture(scope='module')
def _client():
    with testclient.TestClient(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1
        assert violations[0].lineno == 7

    def test_decoy_alias_is_not_flagged(self):
        """`from foo import Bar as TC` + `TC(app)` matches NEITHER arm of the union.

        Broadening must not degenerate into flagging every short uppercase alias:
        the alias map resolves TC to ``foo.Bar`` (not a TestClient), and the
        trailing-name arm sees ``TC``, not ``TestClient``.
        """
        source = '''\
import pytest
from foo import Bar as TC


@pytest.fixture(scope='module')
def _thing():
    with TC(app) as c:
        yield c
'''
        assert find_violations(source, 'test_x.py') == []

    def test_fixture_decorator_is_alias_resolved(self):
        """`from pytest import fixture as fx` + `@fx(scope='module')` still marks a fixture.

        Without this, the same class of hole the alias map closes for TestClient
        would simply reopen one decorator up: rename the decorator import and the
        function stops looking like a fixture.
        """
        source = '''\
from pytest import fixture as fx
from starlette.testclient import TestClient


@fx(scope='module')
def _client():
    with TestClient(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1, 'aliased pytest.fixture decorator was not resolved'
        assert violations[0].lineno == 7


class TestFindViolationsNonGoals:
    """The boundary the deleted guard drew CORRECTLY, carried over verbatim.

    Scoping detection to fixture BODIES is what leaves both of dashboard's
    deliberate non-duplicates clean with no whitelist at all.  They are excluded
    structurally — by not being fixtures — rather than by name.
    """

    def test_plain_test_function_building_a_client_inline_is_not_flagged(self):
        """The `test_scaffold.py:287` idiom: a client built inline in a plain test."""
        source = '''\
from starlette.testclient import TestClient


def test_scaffold_serves_index():
    with TestClient(app) as c:
        assert c.get('/').status_code == 200
'''
        assert find_violations(source, 'test_scaffold.py') == []

    def test_contextmanager_helper_building_a_client_is_not_flagged(self):
        """The `test_api_curator.py:70` `_override_client` idiom: a @contextmanager, not a fixture."""
        source = '''\
from contextlib import contextmanager

from starlette.testclient import TestClient


@contextmanager
def _override_client(config):
    app.state.config = config
    with TestClient(app) as c:
        yield c
'''
        assert find_violations(source, 'test_api_curator.py') == []

    def test_module_level_and_class_body_constructions_are_not_flagged(self):
        """Outside any fixture there is no per-module lifespan duplication to prevent."""
        source = '''\
from starlette.testclient import TestClient

_MODULE_CLIENT = TestClient(app)


class TestThing:
    client = TestClient(app)

    def test_it(self):
        assert self.client
'''
        assert find_violations(source, 'test_x.py') == []

    def test_docstring_mentioning_the_literal_is_not_flagged(self):
        """Matching a real ast.Call, not source text — several dashboard modules say this in prose."""
        source = '''\
"""This module uses `with TestClient(app) as c:` via conftest's shared fixture."""
import pytest


@pytest.fixture(scope='module')
def _thing():
    """Docstring mentioning TestClient(app) — prose, not a call."""
    yield 1
'''
        assert find_violations(source, 'test_x.py') == []


class TestFileSelection:
    """Only ``test_*.py`` is scanned, and ``conftest.py`` is skipped unconditionally.

    The conftest skip is LOAD-BEARING, not incidental: ``hooks/project-checks``
    passes explicit staged file paths, which include ``dashboard/tests/conftest.py``
    whenever it is edited — and conftest is the intended HOME of the shared
    fixture, so scanning it would flag the very thing this rule exists to promote.
    """

    def test_conftest_is_skipped_even_as_a_bare_filename(self):
        assert find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'conftest.py') == []

    def test_conftest_is_skipped_as_an_explicit_path(self):
        """The shape hooks/project-checks actually passes: a repo-relative staged path."""
        assert find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'dashboard/tests/conftest.py') == []

    def test_non_test_module_is_skipped(self):
        for filename in ('helpers.py', '_dashboard_helpers.py', 'app.py', 'contest_test.py'):
            assert find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, filename) == [], (
                f'{filename!r} should not be scanned'
            )

    def test_test_module_is_still_scanned(self):
        """Non-vacuity control: the same source under a test_*.py name IS flagged."""
        assert len(find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'test_x.py')) == 1
        assert len(
            find_violations(_MODULE_LOCAL_CLIENT_FIXTURE, 'dashboard/tests/test_x.py')
        ) == 1


def _fixture_with_prefix(prefix_lines: str, call_line: str = '    with TestClient(app) as c:') -> str:
    """A fixture whose TestClient construction is preceded by *prefix_lines*."""
    return (
        "import pytest\n"
        "from starlette.testclient import TestClient\n"
        "\n"
        "\n"
        "@pytest.fixture(scope='module')\n"
        "def _client():\n"
        f"{prefix_lines}"
        f"{call_line}\n"
        "        yield c\n"
    )


_REASON = 'module-scoped lifespan is the subject under test (task 3503)'


class TestExemptionPragma:
    """Regression for reviewer objections (a) and (b): a site-local, filename-independent escape hatch.

    The deleted task-3571 guard asserted strict equality against a
    ``_CLIENT_FIXTURE_EXEMPT = {'test_fixture_isolation.py'}`` whitelist living
    in a THIRD module, so a module legitimately needing its own client fixture
    went red in a file it never touched — fixable only by editing that whitelist
    (a) — and renaming or splitting the exempt file broke it for a non-defect (b).

    A pragma at the construction site travels with the code under both rename
    and split.  The grammar is inherited verbatim from
    check_bare_magicmock_config.py so the repo keeps ONE suppression grammar.
    """

    def test_pragma_on_preceding_non_blank_line_suppresses(self):
        source = _fixture_with_prefix(f'    # noqa: module-local-testclient — {_REASON}\n')
        assert find_violations(source, 'test_x.py') == []

    def test_ascii_hyphen_separator_is_accepted(self):
        """Both the em-dash and the ASCII hyphen are honored — same contract as bare-magicmock."""
        em = _fixture_with_prefix(f'    # noqa: module-local-testclient — {_REASON}\n')
        ascii_ = _fixture_with_prefix(f'    # noqa: module-local-testclient - {_REASON}\n')
        assert find_violations(em, 'test_x.py') == []
        assert find_violations(ascii_, 'test_x.py') == []

    def test_reasonless_pragma_does_not_suppress(self):
        """A suppression with no stated reason is not informed consent."""
        for prefix in (
            '    # noqa: module-local-testclient\n',
            '    # noqa: module-local-testclient —\n',
            '    # noqa: module-local-testclient -   \n',
        ):
            assert len(find_violations(_fixture_with_prefix(prefix), 'test_x.py')) == 1, (
                f'reasonless pragma {prefix!r} must not suppress'
            )

    def test_inline_trailing_pragma_does_not_suppress(self):
        """Only the nearest PRECEDING non-blank line is inspected — inline is intentionally ignored."""
        source = _fixture_with_prefix(
            '',
            f'    with TestClient(app) as c:  # noqa: module-local-testclient — {_REASON}',
        )
        assert len(find_violations(source, 'test_x.py')) == 1

    def test_intervening_blank_lines_are_tolerated(self):
        source = _fixture_with_prefix(
            f'    # noqa: module-local-testclient — {_REASON}\n\n\n'
        )
        assert find_violations(source, 'test_x.py') == []

    def test_intervening_non_blank_line_breaks_the_exemption(self):
        """Any non-blank, non-matching line between pragma and node breaks it."""
        source = _fixture_with_prefix(
            f'    # noqa: module-local-testclient — {_REASON}\n'
            '    app = build_app()\n'
        )
        assert len(find_violations(source, 'test_x.py')) == 1

    def test_a_different_rules_pragma_does_not_suppress(self):
        """Codes are strictly separate: a bare-magicmock pragma is not consent for this rule."""
        source = _fixture_with_prefix('    # noqa: bare-magicmock — unrelated\n')
        assert len(find_violations(source, 'test_x.py')) == 1

    def test_pragma_is_filename_independent(self):
        """Objections (a) and (b) directly: the same source is exempt under ANY filename.

        The deleted guard's whitelist keyed on ``test_fixture_isolation.py``,
        so renaming or splitting that module broke the guard for a non-defect.
        """
        source = _fixture_with_prefix(f'    # noqa: module-local-testclient — {_REASON}\n')
        for filename in (
            'test_fixture_isolation.py',
            'test_renamed_something_else.py',
            'dashboard/tests/sub/test_split_out_half.py',
        ):
            assert find_violations(source, filename) == [], (
                f'pragma failed to suppress under filename {filename!r}'
            )

    def test_checker_source_contains_no_filename_exemption_list(self):
        """The mechanism this replaces must not survive anywhere in the script.

        Scans the checker's own AST for any string constant that is a concrete
        ``test_*.py`` filename. The rglob glob ``'test_*.py'`` carries a ``*``
        and so is deliberately excluded by the pattern — a real exemption entry
        could not be.
        """
        script_source = SCRIPT_PATH.read_text(encoding='utf-8')
        assert 'test_fixture_isolation' not in script_source, (
            'The checker names test_fixture_isolation.py — the whitelist the pragma replaces'
        )

        concrete_test_filename = re.compile(r'^test_[A-Za-z0-9_]+\.py$')
        offenders = [
            node.value
            for node in ast.walk(ast.parse(script_source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and concrete_test_filename.match(node.value)
        ]
        assert offenders == [], (
            f'The checker contains hardcoded test-module filenames {offenders} — '
            f'exemptions belong at the site, as a # noqa pragma.'
        )
