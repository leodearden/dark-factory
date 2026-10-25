"""Tests for _lint_cli.py, the CLI and pragma contract shared by the fused-memory/scripts/check_*.py lints.

Drives only the public surface: ``Violation``, ``exemption_pattern``,
``is_exempted``, ``discover_files`` and ``run_cli``.  ``run_cli`` is exercised
with a stub ``find_violations`` so these tests pin the shared driver, not any one
checker's rule.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

# fused-memory/scripts/ is not on PYTHONPATH, so load the module by path.
_lint_cli = load_script_module(Path(__file__).parent.parent / 'scripts' / '_lint_cli.py')
Violation = _lint_cli.Violation
exemption_pattern = _lint_cli.exemption_pattern
is_exempted = _lint_cli.is_exempted
discover_files = _lint_cli.discover_files
run_cli = _lint_cli.run_cli


class TestExemptionPattern:
    """``# noqa: <code> <separator> <reason>``: a separator and a non-empty reason are mandatory."""

    @pytest.mark.parametrize(
        'line',
        [
            '# noqa: rule-a — a reason',
            '# noqa: rule-a - a reason',
            '# noqa: rule-a -- a reason',
            '#noqa:rule-a—reason',
        ],
    )
    def test_accepts_em_dash_ascii_hyphen_and_repeated_hyphen_separators(self, line: str):
        assert exemption_pattern('rule-a').match(line)

    @pytest.mark.parametrize(
        'line',
        [
            '# noqa: rule-a',
            '# noqa: rule-a —',
            '# noqa: rule-a -   ',
        ],
    )
    def test_rejects_a_pragma_without_a_reason(self, line: str):
        assert not exemption_pattern('rule-a').match(line)

    def test_is_keyed_on_its_code(self):
        """A pragma for one rule is not informed consent for another, in either direction."""
        assert not exemption_pattern('rule-a').match('# noqa: rule-b — r')
        assert not exemption_pattern('rule-b').match('# noqa: rule-a — r')
        assert exemption_pattern('rule-b').match('# noqa: rule-b — r')


class TestIsExempted:
    """Only the nearest preceding non-blank line can exempt the node below it."""

    @staticmethod
    def _exempted(source: str, lineno: int, code: str = 'rule-a') -> bool:
        return is_exempted(source.splitlines(), lineno, code)

    def test_pragma_on_preceding_line_exempts(self):
        source = '# noqa: rule-a — reason\nnode = 1\n'
        assert self._exempted(source, 2)

    def test_indented_pragma_exempts(self):
        source = 'def f():\n    # noqa: rule-a — reason\n    node = 1\n'
        assert self._exempted(source, 3)

    def test_blank_and_whitespace_only_lines_between_pragma_and_node_are_tolerated(self):
        source = '# noqa: rule-a — reason\n\n   \n\t\nnode = 1\n'
        assert self._exempted(source, 5)

    def test_intervening_non_blank_line_breaks_the_exemption(self):
        source = '# noqa: rule-a — reason\nother = 0\nnode = 1\n'
        assert not self._exempted(source, 3)

    def test_inline_trailing_pragma_does_not_exempt(self):
        source = 'before = 0\nnode = 1  # noqa: rule-a — reason\n'
        assert not self._exempted(source, 2)

    def test_first_line_has_nothing_above_it(self):
        source = 'node = 1  # noqa: rule-a — reason\n'
        assert not self._exempted(source, 1)

    def test_another_codes_pragma_does_not_exempt(self):
        source = '# noqa: rule-b — reason\nnode = 1\n'
        assert not self._exempted(source, 2, code='rule-a')
        assert self._exempted(source, 2, code='rule-b')


class TestDiscoverFiles:
    """The sorted, de-duplicated union of each glob's recursive matches."""

    def test_returns_the_sorted_union_across_nested_directories(self, tmp_path: Path):
        nested = tmp_path / 'pkg' / 'deeper'
        nested.mkdir(parents=True)
        wanted = [
            tmp_path / 'test_top.py',
            tmp_path / 'conftest.py',
            tmp_path / 'pkg' / '_helper.py',
            nested / 'test_deep.py',
            nested / 'conftest.py',
        ]
        for f in wanted:
            f.write_text('')

        found = discover_files(tmp_path, ('test_*.py', 'conftest.py', '_*.py'))

        assert found == sorted(wanted)

    def test_overlapping_globs_yield_each_file_once(self, tmp_path: Path):
        (tmp_path / 'test_alpha.py').write_text('')
        (tmp_path / 'test_beta.py').write_text('')

        found = discover_files(tmp_path, ('test_*.py', 'test_a*.py'))

        assert found == [tmp_path / 'test_alpha.py', tmp_path / 'test_beta.py']

    def test_excludes_files_matching_no_glob(self, tmp_path: Path):
        (tmp_path / 'test_kept.py').write_text('')
        (tmp_path / 'helpers.py').write_text('')
        (tmp_path / 'notes.txt').write_text('')

        assert discover_files(tmp_path, ('test_*.py',)) == [tmp_path / 'test_kept.py']


class _RecordingFinder:
    """A ``find_violations`` stub: one Violation per ``BAD`` token, returned in REVERSE order.

    Reversing makes an unsorted driver visible.  Every call is recorded so a test
    can prove a file was, or was not, read.
    """

    message = 'BAD token found'

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def scanned(self) -> list[str]:
        return [filename for _source, filename in self.calls]

    def __call__(self, source: str, filename: str) -> list:
        self.calls.append((source, filename))
        found = [
            Violation(filename, lineno, col, self.message)
            for lineno, line in enumerate(source.splitlines(), start=1)
            for col in range(len(line))
            if line.startswith('BAD', col)
        ]
        return list(reversed(found))


def _ruff_line(path: Path, lineno: int, col: int) -> str:
    return f'{path}:{lineno}:{col}: {_RecordingFinder.message}'


@pytest.fixture
def finder() -> _RecordingFinder:
    return _RecordingFinder()


def _run(argv: list[str], finder: _RecordingFinder, **overrides) -> int:
    kwargs = {
        'description': 'stub lint',
        'discover': functools.partial(discover_files, globs=('test_*.py',)),
        'find_violations': finder,
    }
    kwargs.update(overrides)
    return run_cli(argv, **kwargs)


class TestRunCliReporting:
    """Exit 0 clean, 1 on violations; ruff-style lines sorted across files."""

    def test_clean_files_exit_zero_with_empty_stdout(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        clean = tmp_path / 'test_clean.py'
        clean.write_text('fine = 1\n')

        assert _run([str(clean)], finder) == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        assert finder.scanned == [str(clean)]

    def test_violations_exit_one_with_ruff_style_lines(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        bad = tmp_path / 'test_bad.py'
        bad.write_text('fine = 1\n    BAD\n')

        assert _run([str(bad)], finder) == 1
        assert capsys.readouterr().out.splitlines() == [_ruff_line(bad, 2, 4)]

    def test_output_is_sorted_by_file_line_and_column_across_files(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        """Explicit args in reverse name order, the stub returning each file's hits reversed."""
        z = tmp_path / 'test_z.py'
        z.write_text('BAD\nfine\nBAD BAD\n')
        a = tmp_path / 'test_a.py'
        a.write_text('fine\nBAD\n')

        assert _run([str(z), str(a)], finder) == 1
        assert capsys.readouterr().out.splitlines() == [
            _ruff_line(a, 2, 0),
            _ruff_line(z, 1, 0),
            _ruff_line(z, 3, 0),
            _ruff_line(z, 3, 4),
        ]

    def test_description_is_the_help_text(self, finder: _RecordingFinder, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(['--help'], finder, description='A uniquely worded stub description.')
        assert exc.value.code == 0
        assert 'A uniquely worded stub description.' in capsys.readouterr().out

    def test_no_paths_is_a_usage_error(self, finder: _RecordingFinder):
        with pytest.raises(SystemExit) as exc:
            _run([], finder)
        assert exc.value.code == 2
        assert finder.calls == []


class TestRunCliErrors:
    """Exit 2: a missing explicit path fails fast; a read error keeps the other files' results."""

    def test_missing_explicit_path_fails_fast_before_any_read(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        valid = tmp_path / 'test_valid.py'
        valid.write_text('BAD\n')
        missing = tmp_path / 'test_missing.py'

        assert _run([str(valid), str(missing)], finder) == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert f'error: {missing}: No such file or directory' in captured.err
        assert finder.calls == []

    def test_undecodable_file_is_a_read_error_that_keeps_other_violations(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        """The bad file sorts FIRST, so an early return on it would print nothing."""
        unreadable = tmp_path / 'test_a_unreadable.py'
        unreadable.write_bytes(b'\xff\xfe not utf-8 at all\n')
        violating = tmp_path / 'test_b_violating.py'
        violating.write_text('BAD\n')

        assert _run([str(tmp_path)], finder) == 2
        captured = capsys.readouterr()
        assert f'error reading {unreadable}:' in captured.err
        assert captured.out.splitlines() == [_ruff_line(violating, 1, 0)]

    def test_os_error_on_one_discovered_file_keeps_other_violations(
        self, tmp_path: Path, finder: _RecordingFinder, monkeypatch, capsys
    ):
        broken = tmp_path / 'test_a_broken.py'
        broken.write_text('BAD\n')
        good = tmp_path / 'test_b_good.py'
        good.write_text('BAD\n')

        real_read_text = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if self.name == broken.name:
                raise OSError('simulated transient read error')
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', fake_read_text)

        assert _run([str(tmp_path)], finder) == 2
        captured = capsys.readouterr()
        assert captured.out.splitlines() == [_ruff_line(good, 1, 0)]
        assert f'error reading {broken}: simulated transient read error' in captured.err


class TestRunCliFileSelection:
    """Directories expand through ``discover``; ``is_scannable`` gates every file before it is read."""

    def test_directory_arg_expands_to_exactly_what_discover_returns(
        self, tmp_path: Path, finder: _RecordingFinder
    ):
        """The checker's own ``discover`` is the one directory expansion the CLI uses."""
        chosen = tmp_path / 'helpers.py'
        for f in (chosen, tmp_path / 'test_not_chosen.py'):
            f.write_text('BAD\n')
        asked: list[Path] = []

        def discover(directory: Path) -> list[Path]:
            asked.append(directory)
            return [chosen]

        assert _run([str(tmp_path)], finder, discover=discover) == 1
        assert asked == [tmp_path]
        assert finder.scanned == [str(chosen)]

    def test_default_is_scannable_admits_every_explicit_file(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        """Explicit paths bypass ``discover``: hooks hand over staged files as-is."""
        helper = tmp_path / 'helpers.py'
        helper.write_text('BAD\n')

        assert _run([str(helper)], finder) == 1
        assert capsys.readouterr().out.splitlines() == [_ruff_line(helper, 1, 0)]

    def test_is_scannable_filters_an_explicit_file_before_it_is_read(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        """An undecodable explicit conftest.py would exit 2 if it were ever read."""
        conftest = tmp_path / 'conftest.py'
        conftest.write_bytes(b'\xff\xfe not utf-8\n')
        ok = tmp_path / 'test_ok.py'
        ok.write_text('fine\n')

        exit_code = _run(
            [str(conftest), str(ok)],
            finder,
            is_scannable=lambda f: Path(f).name != 'conftest.py',
        )

        assert exit_code == 0
        assert capsys.readouterr().err == ''
        assert finder.scanned == [str(ok)]

    def test_is_scannable_filters_a_discovered_file_before_it_is_read(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        (tmp_path / 'test_skipped.py').write_bytes(b'\xff\xfe not utf-8\n')
        ok = tmp_path / 'test_ok.py'
        ok.write_text('fine\n')

        exit_code = _run(
            [str(tmp_path)],
            finder,
            is_scannable=lambda f: Path(f).name != 'test_skipped.py',
        )

        assert exit_code == 0
        assert capsys.readouterr().err == ''
        assert finder.scanned == [str(ok)]

    def test_missing_explicit_path_fails_even_when_is_scannable_would_reject_it(
        self, tmp_path: Path, finder: _RecordingFinder, capsys
    ):
        """The existence check precedes the scannability filter."""
        missing = tmp_path / 'conftest.py'

        exit_code = _run(
            [str(missing)],
            finder,
            is_scannable=lambda f: Path(f).name != 'conftest.py',
        )

        assert exit_code == 2
        assert f'error: {missing}: No such file or directory' in capsys.readouterr().err
