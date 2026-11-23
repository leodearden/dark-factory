"""Tests for _lint_cli.py, the CLI, pragma and per-file debt-budget contract shared by the fused-memory/scripts/check_*.py lints.

Drives only the public surface: ``Violation``, ``exemption_pattern``,
``is_exempted``, ``discover_files``, ``run_cli``, ``DebtEntry``, ``DebtBaseline``
and ``apply_debt_budget``.  ``run_cli`` is exercised with a stub
``find_violations``, and the budget with stub message builders, so these tests
pin the shared machinery, not any one checker's rule.
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


_MERGE_QUEUE_KEY = 'orchestrator/tests/test_merge_queue.py'


def _baseline(debt: dict[str, int]):
    return _lint_cli.DebtBaseline.from_mapping(debt)


class TestDebtBaselineEntryFor:
    """``entry_for`` matches a scanned filename to its row on TRAILING PATH COMPONENTS.

    The nine lint call sites pass repo-relative paths while pytest passes
    absolutes, so both spellings must reach the same row; a substring must not.
    """

    def test_returns_the_row_under_the_baselines_own_key(self):
        entry = _baseline({_MERGE_QUEUE_KEY: 6}).entry_for(_MERGE_QUEUE_KEY)
        assert entry == _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 6)

    def test_an_absolute_spelling_matches_the_same_row(self):
        """The returned key is the baseline's spelling, which a remedy can quote verbatim."""
        entry = _baseline({_MERGE_QUEUE_KEY: 6}).entry_for(f'/home/someone/repo/{_MERGE_QUEUE_KEY}')
        assert entry == _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 6)

    def test_a_backslash_spelling_matches_the_same_row(self):
        entry = _baseline({_MERGE_QUEUE_KEY: 6}).entry_for(
            'C:\\repo\\orchestrator\\tests\\test_merge_queue.py'
        )
        assert entry == _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 6)

    @pytest.mark.parametrize(
        'filename',
        [
            'orchestrator/tests/not_test_merge_queue.py',
            'test_merge_queue.py',
            'dashboard/tests/test_merge_queue.py',
        ],
    )
    def test_a_filename_sharing_fewer_trailing_components_is_not_matched(self, filename: str):
        assert _baseline({_MERGE_QUEUE_KEY: 6}).entry_for(filename) is None

    def test_rows_sharing_a_basename_are_told_apart_by_their_package_directory(self):
        baseline = _baseline({
            'orchestrator/tests/test_harness.py': 2,
            'fused-memory/tests/test_harness.py': 7,
        })
        assert baseline.entry_for('orchestrator/tests/test_harness.py') == _lint_cli.DebtEntry(
            'orchestrator/tests/test_harness.py', 2
        )
        assert baseline.entry_for('/abs/fused-memory/tests/test_harness.py') == _lint_cli.DebtEntry(
            'fused-memory/tests/test_harness.py', 7
        )
        assert baseline.entry_for('dashboard/tests/test_harness.py') is None

    def test_a_non_debt_file_is_none_and_a_zero_budget_row_is_not(self):
        """None means "no budget recorded"; a zero budget is a row of its own."""
        baseline = _baseline({_MERGE_QUEUE_KEY: 0})
        assert baseline.entry_for('orchestrator/tests/test_brand_new.py') is None
        assert baseline.entry_for(_MERGE_QUEUE_KEY) == _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 0)

    def test_the_baseline_is_a_snapshot_of_its_source_mapping(self):
        debt = {_MERGE_QUEUE_KEY: 6}
        baseline = _baseline(debt)
        debt[_MERGE_QUEUE_KEY] = 99
        debt['orchestrator/tests/test_brand_new.py'] = 1
        assert baseline.entry_for(_MERGE_QUEUE_KEY) == _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 6)
        assert baseline.entry_for('orchestrator/tests/test_brand_new.py') is None


_BUDGETED_FILE = '/abs/orchestrator/tests/test_merge_queue.py'


def _overrun_msg(entry, found: int) -> str:
    return f'overrun {entry.key} budget={entry.budget} found={found}'


def _slack_msg(entry, found: int) -> str:
    return f'slack {entry.key} budget={entry.budget} found={found}'


def _apply(found: list, budget: int | None) -> list:
    entry = None if budget is None else _lint_cli.DebtEntry(_MERGE_QUEUE_KEY, budget)
    return _lint_cli.apply_debt_budget(
        found,
        entry,
        filename=_BUDGETED_FILE,
        build_overrun_msg=_overrun_msg,
        build_slack_msg=_slack_msg,
    )


def _sites(*positions: tuple[int, int]) -> list:
    return [Violation(_BUDGETED_FILE, lineno, col, 'raw site') for lineno, col in positions]


class TestApplyDebtBudget:
    """One rule's per-file violation COUNT against its budget: four branches."""

    def test_a_non_debt_file_reports_every_violation_unchanged(self):
        found = _sites((9, 0), (2, 4), (2, 0))
        assert _apply(found, None) == found

    def test_exactly_at_budget_is_silent(self):
        assert _apply(_sites((9, 0), (2, 4), (2, 0)), 3) == []

    def test_over_budget_reports_the_last_sites_in_source_order_with_the_overrun_message(self):
        found = _sites((9, 0), (2, 4), (5, 1), (2, 0))
        message = _overrun_msg(_lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 2), 4)
        assert _apply(found, 2) == [
            Violation(_BUDGETED_FILE, 5, 1, message),
            Violation(_BUDGETED_FILE, 9, 0, message),
        ]

    @pytest.mark.parametrize('found_count', [0, 2])
    def test_under_budget_reports_one_file_level_slack_violation(self, found_count: int):
        found = _sites(*((lineno, 0) for lineno in range(10, 10 + found_count)))
        message = _slack_msg(_lint_cli.DebtEntry(_MERGE_QUEUE_KEY, 3), found_count)
        assert _apply(found, 3) == [Violation(_BUDGETED_FILE, 1, 0, message)]

    def test_both_budget_builders_are_required_and_keyword_only(self):
        """A rule must not be able to inherit another rule's remedy by omission.

        The overrun parameter once carried ``= None`` falling back to Rule B's own
        builder. Every call site passed it, so the fallback was dead, but a new
        rule calling ``apply_debt_budget(found, entry)`` would have silently
        prescribed another rule's remedy, with no type error and no failing test.
        The slack builder carries the same guard. Requiredness is the whole guard,
        so it is pinned here rather than trusted.

        Every call passes entry None, so no debt entry is ever built here.
        """
        apply_debt_budget = _lint_cli.apply_debt_budget

        def _builder(_entry, _found) -> str:
            return ''

        builders = {'build_overrun_msg': _builder, 'build_slack_msg': _builder}
        assert apply_debt_budget([], None, filename=_BUDGETED_FILE, **builders) == [], (
            'the complete call must succeed, so each TypeError below has one cause'
        )
        for name in builders:
            others = {key: value for key, value in builders.items() if key != name}
            with pytest.raises(TypeError):
                apply_debt_budget([], None, filename=_BUDGETED_FILE, **others)
            with pytest.raises(TypeError):
                apply_debt_budget([], None, _builder, filename=_BUDGETED_FILE, **others)
        with pytest.raises(TypeError):
            apply_debt_budget([], None, _BUDGETED_FILE, _builder, _builder)
