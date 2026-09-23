"""Tests for check_module_local_testclient.py lint checker.

Tests for the AST-based lint check that flags a pytest FIXTURE constructing its
own ``TestClient`` inside a ``dashboard/tests/test_*.py`` module, where
``conftest.py`` already provides a shared module-scoped ``_client``.

See task 4485.  This lint replaces the AST guard class that task 3571 deleted
from ``dashboard/tests/test_jsx_source_helpers.py`` (commit 9096654196) after
review rejected a source-lint living inside the pytest suite.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _fm_helpers import load_script_module

# Load the checker script by path to avoid sys.path pollution.
# fused-memory/scripts/ is not on PYTHONPATH per pyproject.toml (pythonpath=['src']).
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_module_local_testclient.py'


_checker = load_script_module(SCRIPT_PATH, mod_name='check_module_local_testclient')
find_violations = _checker.find_violations
main = _checker.main


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

    def test_async_fixture_constructing_a_client_is_flagged(self):
        """`@pytest_asyncio.fixture` + `async def` is an ast.AsyncFunctionDef, and counts.

        The rule's cost argument — a second TestClient is a second app lifespan
        per module — does not care whether the fixture is sync or async, and the
        async shape is common enough in this repo that leaving it to the
        `(FunctionDef, AsyncFunctionDef)` tuple untested would let a later edit
        to the walk drop half the rule silently.
        """
        source = '''\
import pytest_asyncio
from starlette.testclient import TestClient


@pytest_asyncio.fixture(scope='module')
async def _client():
    with TestClient(app) as c:
        yield c
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1
        assert violations[0].lineno == 7

    def test_fixture_nested_inside_a_fixture_reports_the_construction_once(self):
        """One construction reachable from two fixture bodies is still ONE violation.

        The inner `def` is walked both as a statement of the outer fixture's
        body and as a fixture in its own right, so without the identity-keyed
        `seen` set the same call would be reported twice and the offender would
        read as two separate defects.
        """
        source = '''\
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def _outer():
    @pytest.fixture
    def _inner():
        with TestClient(app) as c:
            yield c

    yield _inner
'''
        violations = find_violations(source, 'test_x.py')
        assert len(violations) == 1, f'expected one violation, got {violations}'
        assert violations[0].lineno == 9

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


class TestCliExitCodes:
    """main(argv) returns 0 clean / 1 violations / 2 fatal, and prints ruff-style."""

    def test_clean_file_exits_zero_with_empty_stdout(self, tmp_path: Path, capsys):
        clean = tmp_path / 'test_clean.py'
        clean.write_text(
            "import pytest\n\n\n@pytest.fixture\ndef _x():\n    yield 1\n"
        )
        assert main([str(clean)]) == 0
        assert capsys.readouterr().out == ''

    def test_violating_file_exits_one(self, tmp_path: Path, capsys):
        bad = tmp_path / 'test_bad.py'
        bad.write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        assert main([str(bad)]) == 1
        assert capsys.readouterr().out.strip() != ''

    def test_violations_print_in_ruff_style_path_lineno_col_message(self, tmp_path: Path, capsys):
        bad = tmp_path / 'test_bad.py'
        bad.write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        main([str(bad)])
        out = capsys.readouterr().out
        assert out.startswith(f'{bad}:9:9: '), out
        assert 'noqa: module-local-testclient' in out

    def test_missing_explicit_path_exits_two_before_any_scan_work(self, tmp_path: Path, capsys):
        """Fail fast: a missing explicit path returns 2 and NOTHING is scanned.

        The violating file is passed alongside the missing one; an empty stdout
        proves discovery validated up front rather than part-way through the scan.
        """
        bad = tmp_path / 'test_bad.py'
        bad.write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        assert main([str(tmp_path / 'test_absent.py'), str(bad)]) == 2
        captured = capsys.readouterr()
        assert captured.out == '', 'scan ran despite a missing explicit path'
        assert 'test_absent.py' in captured.err


class TestCliDirectoryScan:
    """A directory argument recursively discovers test_*.py — and never conftest.py."""

    def test_directory_scan_finds_nested_test_modules_and_skips_conftest(
        self, tmp_path: Path, capsys
    ):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'test_nested.py').write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        # conftest.py is the intended HOME of the shared fixture — never an offender.
        (tmp_path / 'conftest.py').write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        # Not a test module at all.
        (tmp_path / 'helpers.py').write_text(_MODULE_LOCAL_CLIENT_FIXTURE)

        assert main([str(tmp_path)]) == 1
        out = capsys.readouterr().out
        # Assert on the REPORTED PATH, not on raw output: the violation message
        # itself names conftest.py as the remedy, so a substring check over the
        # whole line could never fail.
        reported = {line.split(':', 1)[0] for line in out.splitlines() if line.strip()}
        assert str(sub / 'test_nested.py') in reported
        assert str(tmp_path / 'conftest.py') not in reported
        assert str(tmp_path / 'helpers.py') not in reported

    def test_clean_directory_exits_zero(self, tmp_path: Path, capsys):
        (tmp_path / 'test_ok.py').write_text('def test_ok():\n    assert True\n')
        assert main([str(tmp_path)]) == 0
        assert capsys.readouterr().out == ''


class TestCliErrorHandling:
    """A transient per-file read error is reported without discarding other files' violations."""

    def test_unreadable_file_reports_on_stderr_and_keeps_other_violations(
        self, tmp_path: Path, capsys
    ):
        """Undecodable bytes on the FIRST file scanned must not abort the run.

        Names are chosen so the bad file sorts first, which is the stronger
        ordering: an implementation that returned early on the read error would
        print no violations at all.
        """
        (tmp_path / 'test_a_unreadable.py').write_bytes(b'\xff\xfe not utf-8 at all\n')
        (tmp_path / 'test_b_violating.py').write_text(_MODULE_LOCAL_CLIENT_FIXTURE)

        assert main([str(tmp_path)]) == 2
        captured = capsys.readouterr()
        assert 'test_a_unreadable.py' in captured.err
        assert 'test_b_violating.py' in captured.out, (
            'a read error discarded violations collected from other files'
        )


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_TESTS = _REPO_ROOT / 'dashboard' / 'tests'


class TestRealDashboardTestsDirectoryIsClean:
    """Regression guard: dashboard/tests/ must produce zero violations.

    This is the assertion the whole task exists to make durable. Task 3571
    deleted an in-suite guard that made the same claim and shipped GREEN over a
    live, byte-identical duplicate of conftest's `_client` in
    test_tab_tasks_offline_banner.py — which sat in the tree for eleven days
    while a dedup task, its purpose-built guard and a human review all reported
    "conftest is the only home".
    """

    def test_real_dashboard_tests_directory_is_clean_under_check(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(_DASHBOARD_TESTS)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f'Unexpected module-local TestClient fixtures in dashboard/tests/:\n'
            f'{result.stdout}\n'
            f'Each offender should either request conftest.py\'s shared `_client`'
            f' fixture, or carry a # noqa: module-local-testclient — <reason> pragma'
            f' on the preceding non-blank line.'
        )
        assert result.stdout == ''

    def test_dashboard_tests_scan_is_non_vacuous(self):
        """The scan must actually discover files, so this cannot pass against a moved tree."""
        assert _DASHBOARD_TESTS.is_dir(), f'{_DASHBOARD_TESTS} is not a directory'
        discovered = list(_DASHBOARD_TESTS.rglob('test_*.py'))
        assert len(discovered) > 0, (
            f'No test_*.py files discovered under {_DASHBOARD_TESTS} — the cleanliness'
            f' assertion above would pass vacuously.'
        )


_HOOKS_PATH = _REPO_ROOT / 'hooks' / 'project-checks'
_DASHBOARD_YAML = _REPO_ROOT / 'dashboard' / 'orchestrator.yaml'


def _live_dashboard_lint_command() -> str:
    """The dashboard lint_command as the orchestrator actually reads it — through YAML."""
    config = yaml.safe_load(_DASHBOARD_YAML.read_text(encoding='utf-8'))
    return config['lint_command']


def _hook_staged_files_pathspec(content: str, var: str) -> str:
    """The pathspec text of the hook's ``<var>="$( ... )"`` staged-files command substitution.

    Slices from the assignment to the closing ``)"`` and strips comments, so a
    caller asserts against ONE gate's own scan target rather than against any
    line in the file.  Every gate in hooks/project-checks follows this shape and
    several name the same directories, so a whole-file scan cannot distinguish
    them and reports green for a pathspec that moved to another package.
    """
    _, marker, after = content.partition(f'{var}="$(')
    assert marker, f'no {var}="$(...)" staged-files assignment in hooks/project-checks'
    block, closer, _ = after.partition(')"')
    assert closer, f'unterminated {var}="$(...)" command substitution'
    return '\n'.join(line.split('#')[0] for line in block.splitlines())


class TestWiring:
    """The gate must actually be wired, in both places this suite owns.

    Mirrors TestHooksIntegration in test_check_asyncmock_assertion_style.py,
    including its comment-stripping and word-boundary technique.

    Scope boundary: these tests assert that the wiring NAMES THIS SCRIPT and
    points at dashboard/tests.  Whether the resulting chain survives verify's
    scoper, and whether it matches the YAML byte for byte, belongs to the
    orchestrator's own verify-config corpus and is asserted there — see
    ``test_dashboard_lint_command_carries_a_leg_for_THIS_script``.
    """

    def test_hook_invokes_check_with_python3_not_uv_run(self):
        """hooks/project-checks must invoke the checker via a python3 token, scoped to dashboard/tests.

        The filter checks `'check_module_local_testclient.py' in line.split('#')[0]`
        so the script name must appear in the NON-COMMENT portion of the line.
        That excludes both full-line bash comments and inline trailing ones,
        either of which would otherwise land in invocation_lines and fail the
        python3/no-uv-run assertions on a benign edit.

        The scoping half reads THIS gate's own pathspec block, not the whole
        file. A whole-file scan for `dashboard/tests` is vacuous here: the
        bare-magicmock gate two blocks up already lists `dashboard/tests` among
        five pathspecs on a non-comment line, so rewriting this gate's pathspec
        to another package left the assertion green — exactly the ships-green
        failure mode this checker exists to close.
        """
        content = _HOOKS_PATH.read_text(encoding='utf-8')
        invocation_lines = [
            line for line in content.splitlines()
            if 'check_module_local_testclient.py' in line.split('#')[0]
        ]
        assert invocation_lines, (
            'No invocation of check_module_local_testclient.py found in hooks/project-checks'
        )
        assert all('staged_tc' in line for line in invocation_lines), (
            f'The checker invocation must consume the staged_tc file list: {invocation_lines}'
        )

        pathspec = _hook_staged_files_pathspec(content, 'staged_tc')
        assert 'dashboard/tests' in pathspec, (
            f'staged_tc must collect dashboard/tests paths, got: {pathspec!r}'
        )
        for foreign in ('shared/tests', 'escalation/tests', 'fused-memory/tests', 'orchestrator/tests'):
            assert foreign not in pathspec, (
                f'This gate is dashboard-only; {foreign!r} must not be in its pathspec: {pathspec!r}'
            )

        for line in invocation_lines:
            assert re.search(r'\bpython3(?:\.\d+)?\b', line), (
                f'Expected a python3 token (plain, versioned, or absolute path), got: {line!r}'
            )
            assert 'uv run' not in line, (
                f'Found uv run in the checker invocation (should use plain python3): {line!r}'
            )

    def test_dashboard_lint_command_carries_a_leg_for_THIS_script(self):
        """dashboard's lint_command must invoke this very script against dashboard/tests.

        Deliberately narrow.  The chain's SHAPE — its ruff head, its exact
        3-segment text, and its survival through verify's scoper with both
        sibling checker legs verbatim — is pinned on the orchestrator side, by
        ``orchestrator/tests/_verify_config_corpus.py::DASHBOARD_LINT_COMMAND``
        (asserted byte-equal to the live YAML) and
        ``orchestrator/tests/test_verify_plan.py::test_dashboard_lint_chain_scopes_ruff_and_keeps_both_checkers``.
        Re-asserting any of that here would be a second copy of one fact in a
        second package, and the copy paying the module-boundary cost.

        What is NOT pinned there, and is this suite's own business, is the link
        between the checker and its wiring: the leg is derived from SCRIPT_PATH,
        so renaming or deleting the script while quietly updating the YAML and
        the orchestrator corpus in lockstep still goes red here.
        """
        cmd = _live_dashboard_lint_command()

        legs = [leg.strip() for leg in cmd.split('&&')]
        checker_legs = [leg for leg in legs if SCRIPT_PATH.name in leg]
        assert len(checker_legs) == 1, (
            f'Expected exactly one {SCRIPT_PATH.name} leg in dashboard lint_command, '
            f'got {checker_legs} from {cmd!r}'
        )
        assert checker_legs[0].endswith('dashboard/tests'), (
            f'Checker leg must target dashboard/tests: {checker_legs[0]!r}'
        )

    def test_script_runs_under_isolated_python3_proves_stdlib_only(self, tmp_path: Path):
        """Running under `python3 -I -S` proves the script imports only stdlib.

        `-I` alone does not block venv site-packages; `-I -S` additionally skips
        site.py, so any accidental third-party import raises ModuleNotFoundError
        at interpreter startup. Uses PATH python3 (not sys.executable) on purpose:
        that mirrors the hook's runtime assumption.

        Two cases, so a third-party import added inside a scan-only or
        print-only code path cannot slip past: an empty directory (startup), and
        a violating file (parse, scan AND the violation-printing branch).
        """
        if shutil.which('python3') is None:
            pytest.skip('python3 not found on PATH — cannot verify hook runtime assumption')

        empty = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert empty.returncode == 0, (
            f'Script failed under python3 -I -S (unexpected import?):\n{empty.stderr}'
        )
        assert empty.stdout == ''

        (tmp_path / 'test_bad.py').write_text(_MODULE_LOCAL_CLIENT_FIXTURE)
        bad = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True, text=True,
        )
        assert bad.returncode == 1, (
            f'Expected exit 1 under python3 -I -S:\n{bad.stdout}\n{bad.stderr}'
        )
        assert 'test_bad.py' in bad.stdout
