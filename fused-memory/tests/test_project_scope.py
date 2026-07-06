"""Tests for ProjectId/ProjectRoot/ProjectScope — type-level project identity.

Mirrors the class-based style of tests/test_scope.py and the
pytest.raises(InputValidationError) convention of tests/test_validation.py.
"""

import dataclasses
import json
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.utils.validation import InputValidationError


class TestProjectIdAndProjectRoot:
    """ProjectId and ProjectRoot are importable NewTypes wrapping str."""

    def test_project_id_is_a_str_newtype(self):
        """ProjectId.__supertype__ is str, and calling it is the str identity."""
        # __supertype__ exists at runtime for any NewType, but pyright's basic-mode
        # model of NewType doesn't expose it as a known attribute (it types the
        # NewType callable as a plain function) — a documented pyright limitation,
        # not a bug in this code.
        assert ProjectId.__supertype__ is str  # pyright: ignore[reportFunctionMemberAccess]
        assert ProjectId('dark_factory') == 'dark_factory'

    def test_project_root_is_a_str_newtype(self):
        """ProjectRoot.__supertype__ is str, and calling it is the str identity."""
        # See test_project_id_is_a_str_newtype above: pyright doesn't model
        # NewType.__supertype__ even though it exists at runtime.
        assert ProjectRoot.__supertype__ is str  # pyright: ignore[reportFunctionMemberAccess]
        assert ProjectRoot('/home/leo/src/dark-factory') == '/home/leo/src/dark-factory'


class TestProjectScopeConstruction:
    """ProjectScope happy path — construction and attribute read-back."""

    def test_constructs_and_reads_back_values(self):
        scope = ProjectScope(
            ProjectId('dark_factory'), ProjectRoot('/home/leo/src/dark-factory')
        )
        assert scope.project_id == 'dark_factory'
        assert scope.project_root == '/home/leo/src/dark-factory'


class TestProjectScopeValidation:
    """__post_init__ rejects empty project_id and non-absolute/empty project_root."""

    def test_empty_project_id_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId(''), ProjectRoot('/home/leo/src/dark-factory'))

    def test_relative_project_root_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId('dark_factory'), ProjectRoot('relative/path'))

    def test_empty_project_root_raises(self):
        with pytest.raises(InputValidationError):
            ProjectScope(ProjectId('dark_factory'), ProjectRoot(''))


class TestProjectScopeFrozen:
    """ProjectScope instances are immutable at runtime."""

    def test_mutation_raises_frozen_instance_error(self):
        scope = ProjectScope(
            ProjectId('dark_factory'), ProjectRoot('/home/leo/src/dark-factory')
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment on a frozen dataclass is reportAttributeAccessIssue).
            setattr(scope, 'project_id', ProjectId('y'))  # noqa: B010


# ---------------------------------------------------------------------------
# Type-gate rejection fixture.
#
# Generated into tmp_path at test run time (never committed) because pyright's
# include=["src", "tests"] would otherwise type-check these deliberate misuse
# statements as part of the repo's own unscoped pyright run, turning it red.
# tmp_path lives outside include, so the repo stays green while this test still
# exercises real pyright over the fixture.
# ---------------------------------------------------------------------------

# Valid usage only — must produce ZERO pyright errors.
_FIXTURE_VALID_SETUP = """\
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope

pid = ProjectId('dark_factory')
root = ProjectRoot('/abs/root')
scope = ProjectScope(pid, root)


def takes_root(r: ProjectRoot) -> None: ...


takes_root(root)
"""

# Deliberate misuse — each line below must produce at least one pyright error
# of the rule noted in its comment (see _EXPECTED_RULE_BY_MISUSE_LINE below).
_FIXTURE_MISUSE = """\
ProjectScope(root, pid)  # transposed positional args -> 2 reportArgumentType
ProjectScope('dark_factory', '/x')  # bare str, not the NewType -> 2 reportArgumentType
takes_root(scope.project_id)  # ProjectId where ProjectRoot expected -> 1 reportArgumentType
takes_root('/plain/str')  # bare str where ProjectRoot expected -> 1 reportArgumentType
scope.project_root = root  # frozen dataclass mutation -> 1 reportAttributeAccessIssue
"""

_FIXTURE_SOURCE = _FIXTURE_VALID_SETUP + '\n' + _FIXTURE_MISUSE

# 0-indexed (LSP-style) pyright line numbers at or after this value fall in the
# misuse section; any diagnostic before it would mean pyright rejected the valid setup.
_VALID_SETUP_LINE_COUNT = len(_FIXTURE_VALID_SETUP.splitlines())

# _FIXTURE_SOURCE joins the two blocks with an extra blank line (the literal
# '\n' between them), so the misuse block's first source line is one past the
# valid-setup line count.
_MISUSE_START_LINE = _VALID_SETUP_LINE_COUNT + 1

# Expected pyright rule for each misuse statement, keyed by its 0-indexed
# (LSP-style) line number in the full fixture. Checked as "at least one error
# of this rule on this line" rather than an exact total-error count, so a
# pyright version bump that splits/merges diagnostics or adds a new inferred
# error doesn't turn this suite red -- see amendment note on the test below.
_EXPECTED_RULE_BY_MISUSE_LINE = {
    _MISUSE_START_LINE + 0: 'reportArgumentType',  # ProjectScope(root, pid) -- transposed
    _MISUSE_START_LINE + 1: 'reportArgumentType',  # ProjectScope('dark_factory', '/x') -- bare str
    _MISUSE_START_LINE + 2: 'reportArgumentType',  # takes_root(scope.project_id) -- wrong NewType
    _MISUSE_START_LINE + 3: 'reportArgumentType',  # takes_root('/plain/str') -- bare str
    _MISUSE_START_LINE + 4: 'reportAttributeAccessIssue',  # scope.project_root = root -- frozen mutation
}


def _resolve_pyright_launcher() -> list[str] | None:
    """Return the argv prefix to invoke pyright, or None if unavailable.

    Tries the ``pyright`` console script first (repo dev-dependency), falling
    back to the ``pyright`` package via ``python -m pyright``. Returns None
    (rather than raising) so the caller can pytest.skip() in environments
    without pyright installed.
    """
    if shutil.which('pyright'):
        return ['pyright']
    if find_spec('pyright') is not None:
        return [sys.executable, '-m', 'pyright']
    return None


class TestProjectScopeTypeGateRejection:
    """pyright rejects transposed/bare-str/frozen-mutation ProjectScope misuse.

    This is the task's headline negative signal: NewType + frozen dataclass
    turns the recurring project_id/project_root transposition bug into a
    pyright compile-time error rather than a runtime bug.
    """

    @pytest.mark.timeout(120)
    def test_pyright_rejects_transposition_and_frozen_mutation(self, tmp_path: Path):
        launcher = _resolve_pyright_launcher()
        if launcher is None:
            pytest.skip('pyright unavailable (neither `pyright` nor `python -m pyright` launched)')

        fused_memory_src = Path(__file__).resolve().parent.parent / 'src'
        pyrightconfig = {
            'typeCheckingMode': 'basic',
            'pythonVersion': '3.11',
            'extraPaths': [str(fused_memory_src)],
        }
        config_path = tmp_path / 'pyrightconfig.json'
        config_path.write_text(json.dumps(pyrightconfig))

        fixture_path = tmp_path / 'fixture.py'
        fixture_path.write_text(_FIXTURE_SOURCE)

        try:
            result = subprocess.run(
                [*launcher, '--outputjson', '--project', str(config_path), str(fixture_path)],
                capture_output=True,
                text=True,
                timeout=100,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            pytest.skip(f'pyright launch failed: {exc}')

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(
                f'pyright did not produce valid JSON output.\n'
                f'returncode: {result.returncode}\n'
                f'stdout: {result.stdout!r}\n'
                f'stderr: {result.stderr!r}'
            )

        errors = [d for d in payload.get('generalDiagnostics', []) if d.get('severity') == 'error']

        # (a) Zero errors on the valid-setup lines -- pyright must not reject
        # correct usage. Asserted directly (not just "every error is past the
        # boundary") so a missing/empty errors list can't vacuously pass.
        setup_errors = [
            e for e in errors if e['range']['start']['line'] < _MISUSE_START_LINE
        ]
        assert not setup_errors, (
            f'pyright reported errors on the valid-setup lines (expected zero): '
            f'{[(e["range"]["start"]["line"], e["message"]) for e in setup_errors]}'
        )

        # (b) Each misuse line produces at least one error of its expected
        # rule. This checks the essential invariant -- pyright catches every
        # documented misuse -- without pinning the exact diagnostic count,
        # which is what made this test brittle across pyright versions
        # (amendment: relaxed from an exact `len(errors) == 7` +
        # per-rule-total assertion pinned to a single pyright 1.1.408 run).
        errors_by_line: dict[int, list[dict]] = {}
        for e in errors:
            errors_by_line.setdefault(e['range']['start']['line'], []).append(e)

        for line, expected_rule in _EXPECTED_RULE_BY_MISUSE_LINE.items():
            line_errors = errors_by_line.get(line, [])
            matching = [e for e in line_errors if e.get('rule') == expected_rule]
            assert matching, (
                f'Expected at least one {expected_rule} error on fixture line {line} '
                f'({_FIXTURE_MISUSE.splitlines()[line - _MISUSE_START_LINE]!r}), got: '
                f'{[(e.get("rule"), e["message"]) for e in line_errors]}'
            )
