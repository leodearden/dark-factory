"""Tests for check_bare_magicmock_config.py lint checker.

Tests for the AST-based lint check that flags bare MagicMock() assignments to
config-named variables (config, cfg, *_config, *_cfg) in test files unless
preceded by a structured exemption comment.
See task 1372 (lint guard) and task 1339/1313/1064 (migration).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Load the checker script via importlib to avoid sys.path pollution.
# fused-memory/scripts/ is not on PYTHONPATH per pyproject.toml (pythonpath=['src']).
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_bare_magicmock_config.py'


def _load_checker() -> types.ModuleType:
    """Load the checker module from its script path."""
    spec = importlib.util.spec_from_file_location('check_bare_magicmock_config', SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_checker = _load_checker()
find_violations = _checker.find_violations


class TestFindViolationsConfigNameDetection:
    """Core detection: flag bare MagicMock() assigned to config-named variables."""

    def test_flags_config_equals_bare_magicmock(self):
        """config = MagicMock() → exactly 1 violation with correct attributes."""
        source = 'config = MagicMock()\n'
        violations = find_violations(source, 'test_example.py')
        assert len(violations) == 1
        v = violations[0]
        assert v.filename == 'test_example.py'
        assert v.lineno == 1
        assert v.col_offset == 0
        assert 'mock_orch_config' in v.message
        assert 'MagicMock(spec_set=pydantic_spec(...))' in v.message
        assert '1339' in v.message

    def test_flags_cfg_equals_bare_magicmock(self):
        """cfg = MagicMock() → 1 violation."""
        source = 'cfg = MagicMock()\n'
        violations = find_violations(source, 'test_cfg.py')
        assert len(violations) == 1
        v = violations[0]
        assert v.filename == 'test_cfg.py'
        assert v.lineno == 1

    def test_flags_orch_config_suffix_name(self):
        """orch_config = MagicMock() → violation (matches *_config suffix)."""
        source = 'orch_config = MagicMock()\n'
        violations = find_violations(source, 'test_suffix.py')
        assert len(violations) == 1

    def test_flags_mock_cfg_suffix_name(self):
        """mock_cfg = MagicMock() → violation (matches *_cfg suffix)."""
        source = 'mock_cfg = MagicMock()\n'
        violations = find_violations(source, 'test_suffix.py')
        assert len(violations) == 1

    def test_ignores_mcp_generic_name(self):
        """mcp = MagicMock() → no violation (generic name, not a config name)."""
        source = 'mcp = MagicMock()\n'
        violations = find_violations(source, 'test_generic.py')
        assert violations == []

    def test_ignores_mock_generic_name(self):
        """mock = MagicMock() → no violation (generic name, not a config name)."""
        source = 'mock = MagicMock()\n'
        violations = find_violations(source, 'test_generic.py')
        assert violations == []


class TestFindViolationsSpecHandling:
    """Spec-handling: specced calls are never violations; unspecced non-spec-kwarg calls are."""

    def test_no_violation_for_spec_keyword(self):
        """config = MagicMock(spec=OrchestratorConfig) → no violation."""
        source = 'config = MagicMock(spec=OrchestratorConfig)\n'
        violations = find_violations(source, 'test_spec.py')
        assert violations == []

    def test_no_violation_for_spec_set_keyword(self):
        """config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig)) → no violation."""
        source = 'config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))\n'
        violations = find_violations(source, 'test_spec_set.py')
        assert violations == []

    def test_no_violation_for_positional_spec(self):
        """config = MagicMock(SomeClass) → no violation (first positional IS spec)."""
        source = 'config = MagicMock(SomeClass)\n'
        violations = find_violations(source, 'test_positional.py')
        assert violations == []

    def test_violation_for_name_kwarg_only(self):
        """config = MagicMock(name='cfg') → violation (name= is cosmetic, not a spec)."""
        source = "config = MagicMock(name='cfg')\n"
        violations = find_violations(source, 'test_name_only.py')
        assert len(violations) == 1

    def test_violation_for_attribute_form_mock_dot(self):
        """config = mock.MagicMock() → violation (attribute form still targeted)."""
        source = 'config = mock.MagicMock()\n'
        violations = find_violations(source, 'test_attr.py')
        assert len(violations) == 1

    def test_violation_for_attribute_form_unittest_mock(self):
        """config = unittest.mock.MagicMock() → violation (deep attribute form)."""
        source = 'config = unittest.mock.MagicMock()\n'
        violations = find_violations(source, 'test_attr_deep.py')
        assert len(violations) == 1

    def test_no_violation_for_plain_mock(self):
        """config = Mock() → no violation (only MagicMock is targeted)."""
        source = 'config = Mock()\n'
        violations = find_violations(source, 'test_mock.py')
        assert violations == []

    def test_no_violation_for_create_autospec(self):
        """config = create_autospec(X) → no violation (only MagicMock is targeted)."""
        source = 'config = create_autospec(SomeClass)\n'
        violations = find_violations(source, 'test_autospec.py')
        assert violations == []

    def test_starred_positional_arg_is_treated_as_unspecced(self):
        """config = MagicMock(*args) → violation (Starred spread cannot be inspected at AST time)."""
        source = 'config = MagicMock(*args)\n'
        violations = find_violations(source, 'test_starred.py')
        assert len(violations) == 1, (
            'MagicMock(*args) cannot be statically verified as specced; should flag as violation'
        )

    def test_spec_equals_none_is_treated_as_unspecced(self):
        """config = MagicMock(spec=None) → violation (None is not a real spec)."""
        source = 'config = MagicMock(spec=None)\n'
        violations = find_violations(source, 'test_spec_none.py')
        assert len(violations) == 1, (
            'MagicMock(spec=None) is equivalent to bare MagicMock(); should flag'
        )

    def test_spec_set_equals_none_is_treated_as_unspecced(self):
        """config = MagicMock(spec_set=None) → violation (None is not a real spec)."""
        source = 'config = MagicMock(spec_set=None)\n'
        violations = find_violations(source, 'test_spec_set_none.py')
        assert len(violations) == 1, (
            'MagicMock(spec_set=None) is equivalent to bare MagicMock(); should flag'
        )

    def test_double_starred_kwargs_is_treated_as_unspecced(self):
        """config = MagicMock(**spec_kwargs) → violation (**kwargs spread is opaque at AST time)."""
        source = 'config = MagicMock(**spec_kwargs)\n'
        violations = find_violations(source, 'test_double_starred.py')
        assert len(violations) == 1, (
            'MagicMock(**spec_kwargs) cannot be statically verified as specced; '
            'a **kwargs spread is opaque at AST-inspection time so flagging is safer '
            'than a false negative (mirrors the *args conservative stance). '
            f'Got {len(violations)} violations (expected 1).'
        )


class TestFindViolationsExemption:
    """Exemption comment: # noqa: bare-magicmock — <reason> suppresses the violation."""

    def test_exemption_em_dash_suppresses_violation(self):
        """# noqa: bare-magicmock — reason directly above → no violation."""
        source = (
            '# noqa: bare-magicmock — needed for legacy fixture migration\nconfig = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt.py')
        assert violations == []

    def test_exemption_ascii_hyphen_suppresses_violation(self):
        """# noqa: bare-magicmock - reason with ASCII hyphen → no violation."""
        source = (
            '# noqa: bare-magicmock - legacy interface, cannot add spec yet\nconfig = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt_hyphen.py')
        assert violations == []

    def test_exemption_with_blank_line_between_comment_and_assignment(self):
        """Blank lines between exemption comment and assignment are tolerated."""
        source = (
            '# noqa: bare-magicmock — bridging task 1339 migration\n\n    \nconfig = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt_blank.py')
        assert violations == []

    def test_no_exemption_for_bare_noqa_no_reason(self):
        """# noqa: bare-magicmock (no separator, no reason) → still a violation."""
        source = '# noqa: bare-magicmock\nconfig = MagicMock()\n'
        violations = find_violations(source, 'test_no_reason.py')
        assert len(violations) == 1

    def test_no_exemption_for_separator_only_empty_reason(self):
        """# noqa: bare-magicmock — (em-dash but no reason text) → still a violation."""
        source = '# noqa: bare-magicmock —\nconfig = MagicMock()\n'
        violations = find_violations(source, 'test_empty_reason.py')
        assert len(violations) == 1

    def test_no_exemption_when_intervening_code_line_between_comment_and_assignment(self):
        """Intervening non-blank code line breaks the exemption."""
        source = '# noqa: bare-magicmock — some reason\nsome_code = 42\nconfig = MagicMock()\n'
        violations = find_violations(source, 'test_broken_exemption.py')
        assert len(violations) == 1

    def test_no_exemption_for_unrelated_comment_above(self):
        """An unrelated comment immediately above → still a violation."""
        source = '# just a regular comment\nconfig = MagicMock()\n'
        violations = find_violations(source, 'test_unrelated_comment.py')
        assert len(violations) == 1

    def test_inline_trailing_noqa_is_intentionally_not_honored(self):
        """Inline trailing # noqa: bare-magicmock is NOT honored — only the preceding line is checked.

        The exemption contract is defined as: the nearest preceding non-blank source line
        must match the # noqa: bare-magicmock — <reason> pattern.  An inline comment
        on the same assignment line is NOT consulted — this is an intentional design
        choice to keep the contract surface minimal.
        """
        source = 'config = MagicMock()  # noqa: bare-magicmock — inline reason\n'
        violations = find_violations(source, 'test_inline_noqa.py')
        assert len(violations) == 1, (
            'Inline trailing # noqa: bare-magicmock should NOT be honored; '
            'only the nearest preceding non-blank line is checked. '
            f'Got {len(violations)} violations (expected 1).'
        )


class TestFindViolationsAnnAssign:
    """AnnAssign branch: config: Foo = MagicMock() is detected; annotation-only and exempted forms are not."""

    def test_annotated_assignment_with_magicmock_is_a_violation(self):
        """config: Foo = MagicMock() → exactly 1 violation with correct lineno/col/message."""
        source = 'config: Foo = MagicMock()\n'
        violations = find_violations(source, 'test_annassign.py')
        assert len(violations) == 1, (
            f'config: Foo = MagicMock() should produce 1 violation; got {len(violations)}'
        )
        v = violations[0]
        assert v.lineno == 1
        assert v.col_offset == 0
        assert 'mock_orch_config' in v.message

    def test_annotation_only_no_value_is_not_a_violation(self):
        """config: Foo (no value, node.value is None) → 0 violations."""
        source = 'config: Foo\n'
        violations = find_violations(source, 'test_ann_only.py')
        assert violations == [], (
            f'Annotation-only assignment (no value) should produce no violation; got {violations}'
        )

    def test_exemption_above_annotated_assignment_suppresses_violation(self):
        """# noqa: bare-magicmock — reason directly above config: Foo = MagicMock() → 0 violations."""
        source = (
            '# noqa: bare-magicmock — exempted annotated assignment\nconfig: Foo = MagicMock()\n'
        )
        violations = find_violations(source, 'test_ann_exempt.py')
        assert violations == [], (
            f'Exemption comment above annotated assignment should suppress violation; got {violations}'
        )


class TestFindViolationsOutputOrder:
    """find_violations() returns violations sorted ascending by (lineno, col_offset)."""

    def test_violations_sorted_ascending_by_lineno(self):
        """Violations at different nesting depths come out in source order, not BFS order.

        ast.walk() is breadth-first: module-level nodes before function-body nodes,
        outer function body before inner function body.  If find_violations() returns
        raw BFS order, top_config (line 5) comes first, then mid_config (line 4) and
        inner_config (line 3) — descending, which exposes the sort bug.

        To make BFS vs. source-order observable we place the module-level assignment
        LAST in the source so that BFS order (module scope → outer fn → inner fn)
        yields [5, 4, 3] while correct source order is [3, 4, 5].
        """
        # Source order:   inner_config (line 3) → mid_config (line 4) → top_config (line 5)
        # BFS order:      top_config (line 5) → mid_config (line 4) → inner_config (line 3)
        # Expected order: [3, 4, 5]  (ascending source order)
        source = (
            'def outer():\n'  # line 1
            '    def inner():\n'  # line 2
            '        inner_config = MagicMock()\n'  # line 3
            '    mid_config = MagicMock()\n'  # line 4
            'top_config = MagicMock()\n'  # line 5
        )
        violations = find_violations(source, 'test_order.py')
        assert len(violations) == 3, f'Expected 3 violations, got {len(violations)}: {violations}'
        linenos = [v.lineno for v in violations]
        assert linenos == [3, 4, 5], (
            f'find_violations() must return violations in source order [3, 4, 5]; got {linenos}'
        )

    def test_violations_sorted_ascending_by_col_offset_within_same_line(self):
        """Within the same line, violations are ordered by col_offset ascending."""
        # NOTE: This is a guard, not a true adversarial regression test.
        # Chained-assign targets (config = cfg = MagicMock()) are appended to
        # ast.Assign.targets left-to-right, so col_offsets are inherently
        # col_offset-ascending; col_offsets == sorted(col_offsets) would hold
        # even without the sorted() call in find_violations().  No Python
        # construct yields same-line violations in descending col_offset order,
        # so the col_offset sort path cannot be adversarially isolated.
        # The test documents and guards the ordering contract.
        # Python AST: 'config = cfg = MagicMock()' — both on line 1.
        # config is at col_offset 0; cfg is at col_offset 9.
        source = 'config = cfg = MagicMock()\n'
        violations = find_violations(source, 'test_col_order.py')
        assert len(violations) == 2
        col_offsets = [v.col_offset for v in violations]
        assert col_offsets == [0, 9], (
            f'find_violations() must sort by col_offset within same line; expected [0, 9], got {col_offsets}'
        )


class TestFindViolationsMultiTarget:
    """Multi-target and chained assignment: each ast.Name config-target is evaluated independently."""

    def test_chained_assign_mock_then_config_yields_one_violation(self):
        """mock = config = MagicMock() → exactly 1 violation for the 'config' binding."""
        source = 'mock = config = MagicMock()\n'
        violations = find_violations(source, 'test_chained.py')
        assert len(violations) == 1, (
            'mock = config = MagicMock() should produce exactly 1 violation (for config); '
            f'got {len(violations)}'
        )
        v = violations[0]
        assert v.lineno == 1
        assert 'mock_orch_config' in v.message

    def test_chained_assign_two_config_names_yields_two_violations(self):
        """config = cfg = MagicMock() → exactly 2 violations (one per config-named target)."""
        source = 'config = cfg = MagicMock()\n'
        violations = find_violations(source, 'test_chained_two.py')
        assert len(violations) == 2, (
            'config = cfg = MagicMock() should produce 2 violations (one per config target); '
            f'got {len(violations)}'
        )

    def test_chained_assign_no_config_names_yields_zero_violations(self):
        """mock = other = MagicMock() → 0 violations (neither name matches config-name set)."""
        source = 'mock = other = MagicMock()\n'
        violations = find_violations(source, 'test_chained_none.py')
        assert violations == [], (
            'mock = other = MagicMock() should produce no violations (no config names); '
            f'got {violations}'
        )

    def test_exemption_above_chained_assign_suppresses_violation(self):
        """# noqa exemption above mock = config = MagicMock() → 0 violations (shared value/exemption)."""
        source = (
            '# noqa: bare-magicmock — needed for legacy fixture migration\n'
            'mock = config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_chained_exempt.py')
        assert violations == [], (
            'Exemption comment above a chained assign should suppress all config-target violations; '
            f'got {violations}'
        )


# ---------------------------------------------------------------------------
# Reusable source snippets for CLI tests
# ---------------------------------------------------------------------------

_VIOLATION_SOURCE = 'config = MagicMock()\n'

_CLEAN_SOURCE = '# noqa: bare-magicmock — exempted for CLI clean test\nconfig = MagicMock()\n'


def _assert_violation_output(stdout: str, bad_file: Path) -> None:
    """Assert that checker stdout matches the violation-report contract.

    Checks: full bad_file path present, mock_orch_config present,
    MagicMock(spec_set=pydantic_spec(...)) present, task 1339 reference present.

    Shared by TestCliExitCodes (sys.executable) and TestStdlibOnlyProof
    (python3 -I -S) so both tests track the same output contract.
    """
    assert str(bad_file) in stdout, f'Expected bad_file path in violation output, got: {stdout!r}'
    assert 'mock_orch_config' in stdout, (
        f'Expected mock_orch_config in violation output, got: {stdout!r}'
    )
    assert 'MagicMock(spec_set=pydantic_spec(...))' in stdout, (
        f'Expected MagicMock(spec_set=pydantic_spec(...)) in violation output, got: {stdout!r}'
    )
    assert '1339' in stdout, f'Expected task 1339 reference in violation output, got: {stdout!r}'


class TestCliExitCodes:
    """CLI exit-code contract: 1 on violations, 0 on clean input."""

    def test_cli_exits_nonzero_and_prints_violations_on_bad_file(self, tmp_path: Path):
        """Violations file → returncode 1, stdout contains path/alternatives/1339."""
        bad_file = tmp_path / 'test_bad.py'
        bad_file.write_text(_VIOLATION_SOURCE)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(bad_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        _assert_violation_output(result.stdout, bad_file)

    def test_cli_exits_zero_on_clean_file(self, tmp_path: Path):
        """Clean file (exempted) → returncode 0, stdout empty."""
        clean_file = tmp_path / 'test_clean.py'
        clean_file.write_text(_CLEAN_SOURCE)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(clean_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == ''


class TestCliDirectoryScan:
    """Directory-mode scans test_*.py and conftest.py recursively, ignores other .py files."""

    def test_cli_recursively_scans_directory_for_test_files_and_conftest(self, tmp_path: Path):
        """Dir scan: test_example.py + conftest.py flagged; other_file.py ignored."""
        subdir = tmp_path / 'sub'
        subdir.mkdir()

        # Should be scanned and violate
        (subdir / 'test_example.py').write_text(_VIOLATION_SOURCE)
        (subdir / 'conftest.py').write_text(_VIOLATION_SOURCE)
        # Should NOT be scanned (not a test_*.py or conftest.py)
        (subdir / 'other_file.py').write_text(_VIOLATION_SOURCE)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = result.stdout
        assert 'test_example.py' in output
        assert 'conftest.py' in output
        assert 'other_file.py' not in output


class TestCliErrorHandling:
    """main() path/read-error handling: fail fast on missing explicit paths,
    accumulate mid-scan OSErrors without dropping already-collected violations.
    """

    def test_missing_explicit_file_path_fails_fast_with_exit_2(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A missing explicit path → exit 2, no scan work done (no read_text calls).

        bad_file listed FIRST to prove Phase 1 validates ALL paths before Phase 2 reads.
        """
        bad_file = tmp_path / 'test_bad.py'
        bad_file.write_text(_VIOLATION_SOURCE)
        missing = tmp_path / 'nonexistent.py'  # deliberately NOT created

        real_read_text = Path.read_text
        read_text_calls: list[str] = []

        def spy_read_text(self, *args, **kwargs):
            read_text_calls.append(self.name)
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', spy_read_text)

        exit_code = _checker.main([str(bad_file), str(missing)])

        captured = capsys.readouterr()
        assert exit_code == 2
        assert 'nonexistent.py' in captured.err
        assert captured.out == ''
        assert read_text_calls == [], (
            f'Phase 1 should fail fast before any read_text call, '
            f'but read_text was invoked on: {read_text_calls}'
        )

    def test_transient_os_error_does_not_hide_violations_from_other_files(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A mid-scan OSError on one file must not discard violations from other files."""
        good_file = tmp_path / 'test_good.py'
        good_file.write_text(_VIOLATION_SOURCE)
        broken_file = tmp_path / 'test_broken.py'
        broken_file.write_text('# placeholder — read_text will be monkeypatched to raise')

        real_read_text = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self.name == 'test_broken.py':
                raise OSError('simulated transient read error')
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', fake_read_text)

        exit_code = _checker.main([str(tmp_path)])

        captured = capsys.readouterr()
        # Read error → fatal exit precedence over plain violations exit (1).
        assert exit_code == 2
        # Violation from the readable file IS still reported.
        assert 'test_good.py' in captured.out
        assert 'mock_orch_config' in captured.out
        # Read failure from the broken file is reported on stderr.
        assert 'test_broken.py' in captured.err
        assert 'simulated transient read error' in captured.err


class TestStdlibOnlyProof:
    """Running the script under python3 -I -S proves it imports only stdlib modules.

    ``python3 -I`` alone does NOT block venv site-packages on this machine (python 3.14,
    uv-managed venv): ``-I`` only disables *user* site-packages (implies ``-s``), not system
    or venv site-packages.  ``-I -S`` additionally skips site.py, so venv site-packages are
    never added to sys.path — any accidental third-party import in the script would raise
    ModuleNotFoundError at interpreter startup, loudly failing this test.

    Three cases are exercised:
      1. Empty directory: no test_*.py → exit 0, empty stdout (startup isolation).
      2. Clean file: parse/scan runs → exit 0, empty stdout (scan path isolation).
      3. Violation file: exit 1, stdout has violation + alternatives + 1339 (print path).
    """

    def test_script_runs_under_isolated_python3_proves_stdlib_only(self, tmp_path: Path):
        """python3 -I -S three-case stdlib-only proof."""
        if shutil.which('python3') is None:
            pytest.skip('python3 not found on PATH — cannot verify hook runtime assumption')

        # --- Case 1: empty directory (startup + import isolation) ---
        result = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f'Script exited non-zero under python3 -I -S:\n'
            f'  stdout: {result.stdout!r}\n'
            f'  stderr: {result.stderr!r}'
        )
        assert result.stdout == '', (
            f'Expected empty stdout (no scan targets in empty dir), got: {result.stdout!r}'
        )

        # --- Case 2: clean file (parse/scan path isolation) ---
        clean_file = tmp_path / 'test_clean.py'
        clean_file.write_text(_CLEAN_SOURCE)
        result2 = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, (
            f'Script exited non-zero (clean scan) under python3 -I -S:\n'
            f'  stdout: {result2.stdout!r}\n'
            f'  stderr: {result2.stderr!r}'
        )
        assert result2.stdout == '', f'Expected empty stdout (clean file), got: {result2.stdout!r}'

        # --- Case 3: violation file (print-violations branch isolation) ---
        bad_file = tmp_path / 'test_bad.py'
        bad_file.write_text(_VIOLATION_SOURCE)
        result3 = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(bad_file)],
            capture_output=True,
            text=True,
        )
        assert result3.returncode == 1, (
            f'Script should exit 1 under python3 -I -S, got {result3.returncode}:\n'
            f'  stdout: {result3.stdout!r}\n'
            f'  stderr: {result3.stderr!r}'
        )
        _assert_violation_output(result3.stdout, bad_file)


class TestHooksIntegration:
    """hooks/project-checks must invoke the bare-magicmock-config checker."""

    def test_hook_invokes_check_with_python3_not_uv_run(self):
        """The bare-magicmock check invocation must use a python3 token, not uv run,
        and the gate must target the five test directories.

        Word-boundary regex r'\\bpython3(?:\\.\\d+)?\\b' accepts plain `python3`,
        versioned `python3.11`, and absolute paths, while rejecting `mypython3`.

        The invocation filter checks
        `'check_bare_magicmock_config.py' in line.split('#')[0]` so that the script
        name must appear in the non-comment portion of the line. This excludes both
        full-line bash comments and inline comments.

        Scan-dir coverage: commit 2a527c12c9 ("scope pre-commit to staged diff")
        moved the five scan directories off the script-invocation line and onto the
        `git diff --cached ... -- <dirs>` selection line (the `staged_mm=`
        assignment); matching staged files are then piped to the script via xargs.
        The directories therefore live on the selection line, not the invocation
        line, so the coverage assertion scans the whole gate (selection +
        invocation lines), each via its non-comment portion (`line.split('#')[0]`)
        so comment-only occurrences don't satisfy it.
        """
        import re as _re  # noqa: PLC0415 — avoid polluting module namespace

        hooks_path = Path(__file__).parent.parent.parent / 'hooks' / 'project-checks'
        content = hooks_path.read_text(encoding='utf-8')
        invocation_lines = [
            line
            for line in content.splitlines()
            if 'check_bare_magicmock_config.py' in line.split('#')[0]
        ]
        assert invocation_lines, (
            'No invocation of check_bare_magicmock_config.py found in hooks/project-checks'
        )
        for line in invocation_lines:
            assert _re.search(r'\bpython3(?:\.\d+)?\b', line), (
                f'Expected a python3 token in invocation, got: {line!r}'
            )
            assert 'uv run' not in line, (
                f'Found uv run in bare-magicmock check invocation (should use plain python3): {line!r}'
            )
        # Assert ALL five configured scan directories appear in the bare-magicmock
        # gate. The gate is the contiguous block that begins at the staged-file
        # selection (`staged_mm=` — a multi-line `git diff --cached ... -- <dirs>`
        # whose `-- <dirs>` pathspec carries the five directories on a backslash
        # continuation line) and ends at the script-invocation line. Scanning the
        # whole block (each line's non-comment portion) keeps a drop of any single
        # directory immediately catchable regardless of which continuation line
        # carries the pathspec.
        all_lines = content.splitlines()
        start_idx = next(
            (i for i, line in enumerate(all_lines) if 'staged_mm=' in line.split('#')[0]),
            None,
        )
        end_idx = next(
            (
                i
                for i, line in enumerate(all_lines)
                if 'check_bare_magicmock_config.py' in line.split('#')[0]
            ),
            None,
        )
        assert start_idx is not None and end_idx is not None and start_idx <= end_idx, (
            'Could not locate the bare-magicmock gate block (staged_mm selection '
            'through check_bare_magicmock_config.py invocation) in hooks/project-checks'
        )
        gate_code = '\n'.join(
            line.split('#')[0] for line in all_lines[start_idx : end_idx + 1]
        )
        _EXPECTED_SCAN_DIRS = [
            'shared/tests',
            'escalation/tests',
            'fused-memory/tests',
            'orchestrator/tests',
            'dashboard/tests',
        ]
        for expected_dir in _EXPECTED_SCAN_DIRS:
            assert expected_dir in gate_code, (
                f'Expected scan target {expected_dir!r} in the bare-magicmock gate '
                f'(staged_mm selection + check_bare_magicmock_config.py invocation) '
                f'in hooks/project-checks, got: {gate_code!r}'
            )


# ===========================================================================
# Rule B — `bare-dataclass-double` (task 4016)
#
# A SECOND, independently-named rule carried by the same script: a
# position-blind walk flagging unspecced MagicMocks shaped like a registered
# stdlib dataclass (VerifyResult today).  Rule A (`bare-magicmock`, config-name,
# Assign/AnnAssign-only) is unchanged; every test above this banner pins it.
# ===========================================================================

# The 16 real fields of orchestrator.verify.VerifyResult (verify.py:3216).
# Duplicated here deliberately: this test is the thing that would notice if the
# script's registry copy silently drifted from the real dataclass.
_VERIFY_RESULT_FIELDS = frozenset({
    'passed',
    'test_output',
    'lint_output',
    'type_output',
    'summary',
    'timed_out',
    'cause_hint',
    'category',
    'worktree_log_paths',
    'archive_log_paths',
    'contention',
    'plan',
    'failing_test_ids',
    'failing_leg_categories',
    'trivial',
    'duration_secs',
})


class TestDataclassShapeRegistry:
    """The `_DATACLASS_SHAPES` registry that drives Rule B's anchor+overlap match."""

    def test_registry_exists_and_is_a_nonempty_tuple(self):
        """_DATACLASS_SHAPES is a non-empty tuple (immutable — the rule's whole input)."""
        shapes = _checker._DATACLASS_SHAPES
        assert isinstance(shapes, tuple), (
            f'_DATACLASS_SHAPES must be a tuple (immutable registry); got {type(shapes)}'
        )
        assert shapes, '_DATACLASS_SHAPES must not be empty — Rule B would match nothing'

    def test_registry_holds_exactly_the_verify_result_entry_today(self):
        """v1 registers exactly one shape: VerifyResult."""
        names = [s.name for s in _checker._DATACLASS_SHAPES]
        assert names == ['VerifyResult'], (
            f'v1 registers exactly one shape (VerifyResult); got {names}. '
            'Adding a shape is a deliberate widening — update this test with it.'
        )

    def test_verify_result_fields_match_the_real_dataclass(self):
        """The registry's field literal equals VerifyResult's 16 real field names."""
        shape = _checker._DATACLASS_SHAPES[0]
        assert shape.fields == _VERIFY_RESULT_FIELDS, (
            'Registry field set drifted from orchestrator/src/orchestrator/verify.py:3216.\n'
            f'  missing from registry: {sorted(_VERIFY_RESULT_FIELDS - shape.fields)}\n'
            f'  extra in registry:     {sorted(shape.fields - _VERIFY_RESULT_FIELDS)}'
        )
        assert isinstance(shape.fields, frozenset), (
            f'fields must be a frozenset for cheap set algebra; got {type(shape.fields)}'
        )

    def test_verify_result_anchor_is_passed(self):
        """anchors == {'passed'} — VerifyResult's first field and the census tell."""
        shape = _checker._DATACLASS_SHAPES[0]
        assert shape.anchors == frozenset({'passed'}), (
            f"VerifyResult's anchor must be exactly {{'passed'}}; got {set(shape.anchors)}"
        )

    def test_verify_result_min_field_matches_is_two(self):
        """min_field_matches == 2 — the overlap floor that rejects a lone MagicMock(passed=True)."""
        shape = _checker._DATACLASS_SHAPES[0]
        assert shape.min_field_matches == 2, (
            f'min_field_matches must be 2 (measured overlap floor); got {shape.min_field_matches}'
        )

    def test_anchors_are_a_subset_of_fields_for_every_shape(self):
        """An anchor outside its own field set could never be reached by the overlap floor.

        anchors ⊄ fields would be a silently dead registry entry: the anchor gate
        could pass while contributing nothing toward min_field_matches, making the
        effective floor stricter than declared.  Asserted for EVERY shape so a
        future registration cannot introduce the defect.
        """
        for shape in _checker._DATACLASS_SHAPES:
            assert shape.anchors <= shape.fields, (
                f'{shape.name}: anchors must be a subset of fields; '
                f'stray anchors={sorted(shape.anchors - shape.fields)}'
            )

    def test_every_shape_names_its_module_and_factory_remedy(self):
        """Each shape carries the provenance the violation message is built from."""
        for shape in _checker._DATACLASS_SHAPES:
            assert shape.module, f'{shape.name}: module must be set (message provenance)'
            assert shape.factory, f'{shape.name}: factory must name the canonical remedy'
        verify_result = _checker._DATACLASS_SHAPES[0]
        assert verify_result.module == 'orchestrator.verify'
        assert verify_result.factory == '_fake_verify_result'


class TestDataclassDoublePositionBlindness:
    """Rule B flags a VerifyResult-shaped double in ANY syntactic position.

    Position-blindness is the whole point of the widening: Rule A inspects only
    ast.Assign/ast.AnnAssign, so all ten ``return MagicMock(...)`` sites behind
    task 3980 were invisible to it.
    """

    def test_flags_double_in_return_position(self):
        """return MagicMock(passed=..., summary=...) → 1 violation (the exact 3980 shape)."""
        source = "def f():\n    return MagicMock(passed=False, summary='x')\n"
        violations = find_violations(source, 'test_return.py')
        assert len(violations) == 1, (
            'A VerifyResult-shaped double in RETURN position must be flagged — this is '
            f'the exact shape task 3980 removed from ten sites; got {violations}'
        )
        v = violations[0]
        assert v.lineno == 2, f'violation anchors at the MagicMock( line; got {v.lineno}'

    def test_flags_double_assigned_to_a_non_config_name(self):
        """bare = MagicMock(passed=..., summary=...) → 1 violation (Rule A's name gate bypassed).

        ``bare`` is not config/cfg/*_config/*_cfg, so Rule A ignores it entirely.
        Rule B does not consult the binding name at all.
        """
        source = "bare = MagicMock(passed=False, summary='x')\n"
        violations = find_violations(source, 'test_nonconfig_name.py')
        assert len(violations) == 1, (
            "Rule A's config-name gate must not apply to Rule B; a double bound to a "
            f'non-config name must still be flagged; got {violations}'
        )

    def test_flags_double_in_call_argument_position(self):
        """handler(MagicMock(passed=..., summary=...)) → 1 violation (argument position)."""
        source = "handler(MagicMock(passed=True, summary='x'))\n"
        violations = find_violations(source, 'test_argpos.py')
        assert len(violations) == 1, (
            f'A double passed directly as a call argument must be flagged; got {violations}'
        )

    def test_flags_double_inside_comprehension_and_lambda(self):
        """A double buried in a comprehension body and in a lambda body is still flagged."""
        comprehension = "doubles = [MagicMock(passed=True, summary='x') for _ in range(3)]\n"
        violations = find_violations(comprehension, 'test_comprehension.py')
        assert len(violations) == 1, (
            f'A double constructed inside a comprehension must be flagged; got {violations}'
        )

        lam = "make = lambda: MagicMock(passed=False, summary='x')\n"
        violations = find_violations(lam, 'test_lambda.py')
        assert len(violations) == 1, (
            f'A double constructed inside a lambda body must be flagged; got {violations}'
        )

    def test_return_position_violation_carries_rule_b_remedy_not_rule_a(self):
        """A Rule B violation speaks Rule B's vocabulary, never _VIOLATION_MSG."""
        source = "def f():\n    return MagicMock(passed=False, summary='x')\n"
        violations = find_violations(source, 'test_msg_split.py')
        assert len(violations) == 1
        message = violations[0].message
        assert message != _checker._VIOLATION_MSG, (
            "Rule B must not reuse Rule A's message — the remedies are unrelated "
            '(mock_orch_config/pydantic_spec vs _fake_verify_result/spec=VerifyResult)'
        )
        assert 'bare-dataclass-double' in message, (
            f"Rule B's message must name its own noqa code; got {message!r}"
        )


class TestDataclassDoubleNegatives:
    """Precision: shapes Rule B must NOT flag, drawn from the measured repo-wide census.

    Each negative below corresponds to a real construction pattern that occurs in the
    seven scanned tests/ directories.  A rule that flagged any of them would be
    unshippable — hence the positive floor case at the end, which makes it impossible
    to satisfy this class with a rule that simply flags nothing.
    """

    def test_positive_floor_case_still_flags(self):
        """MagicMock(passed=True, summary='x') → 1 violation (anchor + exactly 2 fields).

        Asserted FIRST so the negatives below cannot be trivially satisfied by a
        rule that matches nothing at all.
        """
        source = "d = MagicMock(passed=True, summary='x')\n"
        violations = find_violations(source, 'test_floor.py')
        assert len(violations) == 1, (
            'anchor present + exactly min_field_matches fields must flag; '
            f'got {violations}'
        )

    def test_spec_keyword_exempts(self):
        """MagicMock(spec=VerifyResult, passed=..., summary=...) → 0 violations."""
        source = "d = MagicMock(spec=VerifyResult, passed=False, summary='x')\n"
        assert find_violations(source, 'test_spec_kw.py') == []

    def test_spec_set_keyword_exempts(self):
        """MagicMock(spec_set=VerifyResult, passed=..., summary=...) → 0 violations."""
        source = "d = MagicMock(spec_set=VerifyResult, passed=False, summary='x')\n"
        assert find_violations(source, 'test_spec_set_kw.py') == []

    def test_positional_spec_exempts(self):
        """MagicMock(VerifyResult, passed=..., summary=...) → 0 (first positional IS spec)."""
        source = "d = MagicMock(VerifyResult, passed=False, summary='x')\n"
        assert find_violations(source, 'test_positional_spec.py') == [], (
            "MagicMock's first positional argument IS spec — a positionally-specced "
            'double is the remedy, not a violation'
        )

    def test_single_field_is_below_the_overlap_floor(self):
        """MagicMock(passed=True) alone → 0 violations (anchor present, only 1 field).

        This is precisely what the overlap floor buys: a stray ``passed=`` on an
        unrelated object (a fake process result, a predicate double) is not a
        VerifyResult impersonation.
        """
        source = 'd = MagicMock(passed=True)\n'
        violations = find_violations(source, 'test_one_field.py')
        assert violations == [], (
            'a lone passed= kwarg is below the 2-field overlap floor and must NOT '
            f'flag; got {violations}'
        )

    def test_return_value_kwarg_does_not_flag(self):
        """MagicMock(return_value=x) → 0 violations (660 unspecced occurrences repo-wide)."""
        source = 'd = MagicMock(return_value=sentinel)\n'
        assert find_violations(source, 'test_return_value.py') == []

    def test_side_effect_kwarg_does_not_flag(self):
        """MagicMock(side_effect=e) → 0 violations (161 unspecced occurrences repo-wide)."""
        source = 'd = MagicMock(side_effect=RuntimeError("boom"))\n'
        assert find_violations(source, 'test_side_effect.py') == []

    def test_plain_mock_is_not_targeted(self):
        """Mock(passed=..., summary=...) → 0 violations (only MagicMock is targeted)."""
        source = "d = Mock(passed=False, summary='x')\n"
        assert find_violations(source, 'test_plain_mock_shape.py') == []

    def test_async_mock_is_not_targeted(self):
        """AsyncMock(passed=..., summary=...) → 0 violations (only MagicMock is targeted)."""
        source = "d = AsyncMock(passed=False, summary='x')\n"
        assert find_violations(source, 'test_async_mock_shape.py') == []

    def test_kwargs_spread_alone_does_not_flag(self):
        """MagicMock(**kw) → 0 violations (no LITERAL anchor kwarg is visible).

        A ``**spread`` contributes a keyword whose ``.arg`` is None.  Rule B matches on
        literal kwarg NAMES only, so a spread can never satisfy the anchor gate.
        (Rule A's separate, conservative stance — flagging ``config = MagicMock(**kw)``
        as unspecced — is unaffected and pinned by its own test above.)
        """
        source = 'd = MagicMock(**kw)\n'
        violations = find_violations(source, 'test_spread_only.py')
        assert violations == [], (
            f'a **kwargs spread exposes no literal anchor name and must not flag; got {violations}'
        )

    def test_two_real_fields_without_the_anchor_does_not_flag(self):
        """MagicMock(summary='x', timed_out=False) → 0 violations (anchor absent).

        Two genuine VerifyResult field names are matched, but ``passed`` — the anchor
        — is missing, so the overlap floor alone must not be sufficient.
        """
        source = "d = MagicMock(summary='x', timed_out=False)\n"
        violations = find_violations(source, 'test_no_anchor.py')
        assert violations == [], (
            'the anchor gate is mandatory — two field matches without passed= must '
            f'NOT flag; got {violations}'
        )


# The verbatim shape task 3980 removed from ten sites (test_merge_speculation.py:4275).
# Note `verify_skipped=`, which is a MergeOutcome field (merge_types.py:945), NOT a
# VerifyResult field — a bare MagicMock accepts it without objection. This is the case
# the "kwargs are a subset of the dataclass's fields" rule would have MISSED.
_TASK_3980_SHAPE = (
    'def make():\n'
    '    return MagicMock(\n'
    "        passed=False, summary='tests failed', test_output='FAIL',\n"
    "        lint_output='', type_output='', category='', timed_out=False,\n"
    '        verify_skipped=False,\n'
    '    )\n'
)


class TestDataclassDoubleMessage:
    """Rule B's violation message: names the shape, the drift, the remedy, and its own code."""

    def test_task_3980_shape_is_flagged_despite_a_non_field_kwarg(self):
        """The verbatim 3980 shape flags — the regression-critical case a subset rule misses.

        ``verify_skipped`` is not a VerifyResult field, so ``kwargs <= fields`` is FALSE
        here. Under the anchor+overlap rule the non-field kwarg is drift evidence, not
        an exemption, and the site is correctly flagged.
        """
        violations = find_violations(_TASK_3980_SHAPE, 'test_3980.py')
        assert len(violations) == 1, (
            'the verbatim task-3980 shape MUST be flagged; a non-field kwarg is drift '
            f'evidence, never an exemption. got {violations}'
        )

    def test_message_names_shape_remedy_and_origin_tasks(self):
        """The message names VerifyResult, both remedies, dataclasses.fields, and 3477/3980."""
        message = find_violations(_TASK_3980_SHAPE, 'test_3980.py')[0].message
        for needle in (
            'VerifyResult',
            '_fake_verify_result',
            'spec=VerifyResult',
            'dataclasses.fields',
            '3477',
            '3980',
        ):
            assert needle in message, (
                f'Rule B message must name {needle!r} so the remedy is actionable '
                f'without leaving the error; got {message!r}'
            )

    def test_message_names_the_drift_kwarg(self):
        """The message calls out `verify_skipped` as not-a-VerifyResult-field.

        Naming the specific unknown kwarg is what turns the message from "this looks
        wrong" into evidence: it is the strongest signal that the double has drifted
        from the type it impersonates.
        """
        message = find_violations(_TASK_3980_SHAPE, 'test_3980.py')[0].message
        assert 'verify_skipped' in message, (
            f'the drift kwarg must be named as evidence; got {message!r}'
        )

    def test_message_does_not_leak_rule_a_pydantic_vocabulary(self):
        """Rule A's pydantic remedies must not appear in a stdlib-dataclass message.

        ``pydantic_spec`` reads ``model_fields`` and requires a BaseModel — it is
        unusable for VerifyResult. Offering it here would send the reader down a
        dead end.
        """
        message = find_violations(_TASK_3980_SHAPE, 'test_3980.py')[0].message
        for forbidden in ('mock_orch_config', 'pydantic_spec'):
            assert forbidden not in message, (
                f"Rule A's {forbidden!r} must not leak into Rule B's message — it is "
                f'unusable for a stdlib dataclass; got {message!r}'
            )

    def test_message_instructs_the_rule_b_noqa_code(self):
        """The message tells the reader the correct, rule-specific suppression code."""
        message = find_violations(_TASK_3980_SHAPE, 'test_3980.py')[0].message
        assert '# noqa: bare-dataclass-double' in message, (
            f'the message must instruct the Rule B noqa code verbatim; got {message!r}'
        )

    def test_rule_b_path_runs_under_isolated_python3_stdlib_only(self, tmp_path: Path):
        """The registry, walk and message builder add no third-party import.

        Reuses TestStdlibOnlyProof's harness: under ``python3 -I -S`` no venv
        site-packages are on sys.path, so any accidental third-party import in the
        Rule B path would raise ModuleNotFoundError at startup instead of reporting.
        """
        if shutil.which('python3') is None:
            pytest.skip('python3 not found on PATH — cannot verify hook runtime assumption')

        bad_file = tmp_path / 'test_bad_double.py'
        bad_file.write_text(_TASK_3980_SHAPE)
        result = subprocess.run(
            ['python3', '-I', '-S', str(SCRIPT_PATH), str(bad_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, (
            f'Rule B should exit 1 under python3 -I -S, got {result.returncode}:\n'
            f'  stdout: {result.stdout!r}\n'
            f'  stderr: {result.stderr!r}'
        )
        for needle in (
            str(bad_file),
            'VerifyResult',
            '_fake_verify_result',
            'verify_skipped',
            'bare-dataclass-double',
        ):
            assert needle in result.stdout, (
                f'Expected {needle!r} in Rule B stdout under python3 -I -S, '
                f'got: {result.stdout!r}'
            )


_RULE_B_SOURCE = "bare = MagicMock(passed=False, summary='x')\n"
_RULE_A_SOURCE = 'config = MagicMock()\n'


class TestDataclassDoubleExemption:
    """Rule B honours its OWN noqa code, on the preceding non-blank line, with a reason.

    The two rules deliberately do NOT share a suppression code: their remedies are
    unrelated, so a pragma written for one is never informed consent for the other.
    """

    def test_em_dash_exemption_suppresses_rule_b(self):
        """# noqa: bare-dataclass-double — reason above the call → 0 violations."""
        source = '# noqa: bare-dataclass-double — deliberate mutation leg\n' + _RULE_B_SOURCE
        assert find_violations(source, 'test_b_exempt.py') == []

    def test_ascii_hyphen_exemption_suppresses_rule_b(self):
        """ASCII-hyphen separator is accepted, same as Rule A."""
        source = '# noqa: bare-dataclass-double - deliberate mutation leg\n' + _RULE_B_SOURCE
        assert find_violations(source, 'test_b_exempt_hyphen.py') == []

    def test_exemption_tolerates_intervening_blank_lines(self):
        """Blank lines between the pragma and the call are tolerated."""
        source = (
            '# noqa: bare-dataclass-double — deliberate mutation leg\n\n    \n' + _RULE_B_SOURCE
        )
        assert find_violations(source, 'test_b_exempt_blank.py') == []

    def test_no_exemption_without_a_reason(self):
        """# noqa: bare-dataclass-double with no reason after the separator → still flagged."""
        for header in (
            '# noqa: bare-dataclass-double\n',
            '# noqa: bare-dataclass-double —\n',
        ):
            violations = find_violations(header + _RULE_B_SOURCE, 'test_b_no_reason.py')
            assert len(violations) == 1, (
                f'a reasonless pragma must NOT suppress Rule B; header={header!r} '
                f'gave {violations}'
            )

    def test_intervening_code_line_breaks_the_exemption(self):
        """A non-blank, non-matching line between pragma and call breaks the exemption."""
        source = (
            '# noqa: bare-dataclass-double — a reason\nsome_code = 42\n' + _RULE_B_SOURCE
        )
        assert len(find_violations(source, 'test_b_broken.py')) == 1

    def test_inline_trailing_exemption_is_not_honored(self):
        """Inline trailing placement is not honored for Rule B either."""
        source = (
            "bare = MagicMock(passed=False, summary='x')"
            '  # noqa: bare-dataclass-double — inline\n'
        )
        assert len(find_violations(source, 'test_b_inline.py')) == 1, (
            'only the nearest PRECEDING non-blank line is consulted, for both rules'
        )

    def test_rule_a_code_does_not_suppress_rule_b(self):
        """CROSS-RULE ISOLATION: # noqa: bare-magicmock must NOT suppress a Rule B violation.

        Otherwise a pragma written years ago for a config assignment would silently
        exempt a VerifyResult-shaped double that later lands on the following line.
        """
        source = '# noqa: bare-magicmock — legacy config exemption\n' + _RULE_B_SOURCE
        violations = find_violations(source, 'test_cross_a_to_b.py')
        assert len(violations) == 1, (
            "Rule A's noqa code must not suppress Rule B — the remedies are unrelated, "
            f'so it is not informed consent; got {violations}'
        )

    def test_rule_b_code_does_not_suppress_rule_a(self):
        """CROSS-RULE ISOLATION, the other direction: bare-dataclass-double ⊅ bare-magicmock."""
        source = '# noqa: bare-dataclass-double — a reason\n' + _RULE_A_SOURCE
        violations = find_violations(source, 'test_cross_b_to_a.py')
        assert len(violations) == 1, (
            f"Rule B's noqa code must not suppress a Rule A violation; got {violations}"
        )

    def test_rule_a_exemption_still_works_unchanged(self):
        """Regression pin: parameterising _is_exempted did not change Rule A's behaviour."""
        source = '# noqa: bare-magicmock — needed for legacy fixture migration\n' + _RULE_A_SOURCE
        assert find_violations(source, 'test_a_still_exempt.py') == [], (
            "Rule A's own exemption must remain bit-identical after parameterisation"
        )


# The 11 files carrying pre-existing Rule B debt, from the AST census over all seven
# scanned tests/ directories (95 sites total). test_merge_speculation.py is
# deliberately ABSENT: its single deliberate double gets a per-site pragma instead,
# so task 3980's freshly-cleaned module stays fully covered.
_EXPECTED_DEBT_PATHS = frozenset({
    'orchestrator/tests/test_merge_queue.py',
    'orchestrator/tests/test_concurrent_verify_boundary.py',
    'orchestrator/tests/test_merge_queue_permit_conservation.py',
    'orchestrator/tests/test_merge_queue_resolve_release.py',
    'orchestrator/tests/test_merge_queue_request_liveness.py',
    'orchestrator/tests/test_coalesce_integration_gate.py',
    'orchestrator/tests/test_merge_item_union.py',
    'orchestrator/tests/test_merge_queue_equivalence.py',
    'orchestrator/tests/test_merge_queue_lifecycle_registry.py',
    'orchestrator/tests/test_merge_queue_metrics.py',
    'orchestrator/tests/test_merge_queue_single_writer_asserts.py',
})

_REPO_ROOT = Path(__file__).parent.parent.parent


class TestDataclassDoubleDebtBaseline:
    """The shrink-only per-file debt baseline that lets Rule B ship default-ON.

    96 pre-existing sites across 12 files mean a hot default-on rule would turn
    orchestrator/tests' lint_command red immediately and stall the merge lane
    repo-wide. The baseline is per-FILE (line numbers churn on every edit above
    them) and opt-OUT (a brand-new offending file must be covered by default —
    an opt-in list would exempt exactly the third file this task exists to catch).
    """

    def test_debt_baseline_holds_exactly_the_eleven_measured_paths(self):
        """_DATACLASS_DOUBLE_DEBT == the 11 census paths — no more, no less."""
        debt = _checker._DATACLASS_DOUBLE_DEBT
        assert set(debt) == _EXPECTED_DEBT_PATHS, (
            'Debt baseline drifted from the measured census.\n'
            f'  unexpected additions: {sorted(set(debt) - _EXPECTED_DEBT_PATHS)}\n'
            f'  missing entries:      {sorted(_EXPECTED_DEBT_PATHS - set(debt))}\n'
            'The list is SHRINK-ONLY: entries may be removed as files are migrated, '
            'never added.'
        )

    def test_test_merge_speculation_is_not_grandfathered(self):
        """test_merge_speculation.py must NOT be on the baseline.

        Blanket-suppressing Rule B there would silently un-cover the eleven other
        doubles task 3980 just removed, the moment anyone reintroduced one.
        """
        assert 'orchestrator/tests/test_merge_speculation.py' not in _checker._DATACLASS_DOUBLE_DEBT, (
            'test_merge_speculation.py must stay OFF the debt baseline — its one '
            'deliberate double carries a per-site pragma so the rest of the module '
            'stays covered (task 3980 regression)'
        )

    def test_same_source_opposite_verdicts_by_filename(self):
        """The identical offending source is suppressed in a debt file and flagged elsewhere."""
        suppressed = find_violations(_RULE_B_SOURCE, 'orchestrator/tests/test_merge_queue.py')
        assert suppressed == [], (
            f'Rule B must be suppressed in a debt-listed file; got {suppressed}'
        )
        flagged = find_violations(_RULE_B_SOURCE, 'orchestrator/tests/test_brand_new.py')
        assert len(flagged) == 1, (
            'the SAME source in a non-debt file must still flag — otherwise the '
            f'baseline is not a baseline but a global off switch; got {flagged}'
        )

    def test_suppression_works_for_absolute_paths(self):
        """An absolute path ending in the debt components is suppressed too.

        The CLI passes repo-relative paths; pytest passes tmp_path absolutes. Both
        must reach the same verdict or the baseline would be invisible to one caller.
        """
        absolute = str(_REPO_ROOT / 'orchestrator' / 'tests' / 'test_merge_queue.py')
        assert find_violations(_RULE_B_SOURCE, absolute) == [], (
            f'an absolute path to a debt file must be suppressed; filename={absolute!r}'
        )

    def test_matching_is_path_component_aware_not_substring(self):
        """Trailing-COMPONENT matching: a substring match must not grandfather an unrelated file."""
        # Real trailing components → suppressed (this is what makes absolute paths work).
        assert find_violations(_RULE_B_SOURCE, 'evil/orchestrator/tests/test_merge_queue.py') == [], (
            'a path whose real trailing components are a debt entry is suppressed'
        )
        # Substring of a filename, but not a component boundary → NOT suppressed.
        not_suppressed = find_violations(
            _RULE_B_SOURCE, 'orchestrator/tests/not_test_merge_queue.py'
        )
        assert len(not_suppressed) == 1, (
            'not_test_merge_queue.py merely CONTAINS a debt filename as a substring; '
            f'a substring match must not grandfather it. got {not_suppressed}'
        )
        # Same basename at a different root → NOT suppressed (fewer components match).
        bare = find_violations(_RULE_B_SOURCE, 'test_merge_queue.py')
        assert len(bare) == 1, (
            'a bare basename at another root shares only ONE trailing component and '
            f'must not be suppressed. got {bare}'
        )

    def test_debt_baseline_suppresses_rule_b_only(self):
        """A Rule A violation in a debt-listed file is still reported.

        The baseline grandfathers dataclass-double debt, not all mock-spec discipline.
        """
        violations = find_violations(_RULE_A_SOURCE, 'orchestrator/tests/test_merge_queue.py')
        assert len(violations) == 1, (
            'the debt baseline must suppress Rule B ONLY — a bare config MagicMock in '
            f'a debt-listed file is still a Rule A violation; got {violations}'
        )
        assert 'mock_orch_config' in violations[0].message

    def test_every_debt_entry_resolves_to_an_existing_file(self):
        """A deleted or renamed file must not leave a stale blanket suppression behind."""
        missing = [
            entry for entry in _checker._DATACLASS_DOUBLE_DEBT if not (_REPO_ROOT / entry).is_file()
        ]
        assert missing == [], (
            f'Debt baseline entries no longer exist in the repo: {missing}. '
            'A stale entry silently suppresses Rule B for a path nothing occupies — '
            'and would grandfather a NEW file created at that path. Remove them.'
        )


class TestFusedMemoryTestsDirectoryClean:
    """Regression guard: fused-memory/tests must have zero bare MagicMock() config sites.

    Task 1531 migrated all 26 violation sites in fused-memory/tests off bare
    MagicMock() and onto MagicMock(spec_set=pydantic_spec(FusedMemoryConfig)).
    This test re-runs the hook script against the real fused-memory/tests
    directory so that any future reintroduction is caught at pytest time (not
    only at commit time, since the hook is repo-wide and operators may skip it
    locally).
    """

    def test_fused_memory_tests_directory_has_zero_violations(self):
        """Running the checker against fused-memory/tests must exit 0 with empty stdout."""
        tests_dir = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(tests_dir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f'check_bare_magicmock_config found violations in fused-memory/tests '
            f'(task 1531 regression). Fix by wrapping each bare MagicMock() config '
            f'with MagicMock(spec_set=pydantic_spec(FusedMemoryConfig)).\n'
            f'Violations:\n{result.stdout}'
        )
