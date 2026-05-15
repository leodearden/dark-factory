"""Tests for check_bare_magicmock_config.py lint checker.

Tests for the AST-based lint check that flags bare MagicMock() assignments to
config-named variables (config, cfg, *_config, *_cfg) in test files unless
preceded by a structured exemption comment.
See task 1372 (lint guard) and task 1339/1313/1064 (migration).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

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


class TestFindViolationsExemption:
    """Exemption comment: # noqa: bare-magicmock — <reason> suppresses the violation."""

    def test_exemption_em_dash_suppresses_violation(self):
        """# noqa: bare-magicmock — reason directly above → no violation."""
        source = (
            '# noqa: bare-magicmock — needed for legacy fixture migration\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt.py')
        assert violations == []

    def test_exemption_ascii_hyphen_suppresses_violation(self):
        """# noqa: bare-magicmock - reason with ASCII hyphen → no violation."""
        source = (
            '# noqa: bare-magicmock - legacy interface, cannot add spec yet\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt_hyphen.py')
        assert violations == []

    def test_exemption_with_blank_line_between_comment_and_assignment(self):
        """Blank lines between exemption comment and assignment are tolerated."""
        source = (
            '# noqa: bare-magicmock — bridging task 1339 migration\n'
            '\n'
            '    \n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_exempt_blank.py')
        assert violations == []

    def test_no_exemption_for_bare_noqa_no_reason(self):
        """# noqa: bare-magicmock (no separator, no reason) → still a violation."""
        source = (
            '# noqa: bare-magicmock\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_no_reason.py')
        assert len(violations) == 1

    def test_no_exemption_for_separator_only_empty_reason(self):
        """# noqa: bare-magicmock — (em-dash but no reason text) → still a violation."""
        source = (
            '# noqa: bare-magicmock —\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_empty_reason.py')
        assert len(violations) == 1

    def test_no_exemption_when_intervening_code_line_between_comment_and_assignment(self):
        """Intervening non-blank code line breaks the exemption."""
        source = (
            '# noqa: bare-magicmock — some reason\n'
            'some_code = 42\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_broken_exemption.py')
        assert len(violations) == 1

    def test_no_exemption_for_unrelated_comment_above(self):
        """An unrelated comment immediately above → still a violation."""
        source = (
            '# just a regular comment\n'
            'config = MagicMock()\n'
        )
        violations = find_violations(source, 'test_unrelated_comment.py')
        assert len(violations) == 1


# ---------------------------------------------------------------------------
# Reusable source snippets for CLI tests
# ---------------------------------------------------------------------------

_VIOLATION_SOURCE = 'config = MagicMock()\n'

_CLEAN_SOURCE = (
    '# noqa: bare-magicmock — exempted for CLI clean test\n'
    'config = MagicMock()\n'
)


def _assert_violation_output(stdout: str, bad_file: 'Path') -> None:
    """Assert that checker stdout matches the violation-report contract.

    Checks: full bad_file path present, mock_orch_config present,
    MagicMock(spec_set=pydantic_spec(...)) present, task 1339 reference present.

    Shared by TestCliExitCodes (sys.executable) and TestStdlibOnlyProof
    (python3 -I -S) so both tests track the same output contract.
    """
    assert str(bad_file) in stdout, (
        f'Expected bad_file path in violation output, got: {stdout!r}'
    )
    assert 'mock_orch_config' in stdout, (
        f'Expected mock_orch_config in violation output, got: {stdout!r}'
    )
    assert 'MagicMock(spec_set=pydantic_spec(...))' in stdout, (
        f'Expected MagicMock(spec_set=pydantic_spec(...)) in violation output, got: {stdout!r}'
    )
    assert '1339' in stdout, (
        f'Expected task 1339 reference in violation output, got: {stdout!r}'
    )


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

    def test_cli_recursively_scans_directory_for_test_files_and_conftest(
        self, tmp_path: Path
    ):
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
