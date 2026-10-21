"""The ratchet-baseline merge driver: its 3-way rules, and git actually using it.

``scripts/merge_lane_ratchet_merge.py`` composes two branches' moves of the
committed baseline instead of letting git's line merge conflict on a shared row.
The rules are pinned on synthetic images through ``merge_baselines`` and the
``merge`` verb git invokes; the wiring is pinned end to end in throwaway repos
that carry the REAL driver, instrument and ``.gitattributes`` -- through
``git merge`` (in the main checkout and a linked worktree), through the lane
probe's ``git merge-tree``, and with the driver uninstalled, where the merge
must conflict exactly as it did before the driver existed.

The driver's module is imported after ``sys.path`` gains ``scripts/``, the
shape ``test_orchestrator_governed_lists.py`` uses, because the shared
fixtures module imports only the instrument.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _merge_lane_ratchet_fixtures import metrics, synthetic_report
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

import merge_lane_ratchet_merge as driver

_REPO_ROOT = Path(__file__).parents[2]


def _with(mutate) -> dict:
    report = copy.deepcopy(synthetic_report())
    mutate(report)
    return report


def _set(path: tuple, value: object):
    """A mutation setting the value at *path* inside a report."""

    def mutate(report: dict) -> None:
        node = report
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return mutate


def _a_py(measure: str, value: int):
    return _set(('files', 'a.py', measure), value)


class TestCounts:
    """``A == B -> A``, ``A == O -> B``, ``B == O -> A``, else ``A + B - O``."""

    @pytest.mark.parametrize('ours_moved', [True, False], ids=['ours-moved', 'theirs-moved'])
    def test_one_side_moving_a_count_takes_that_side(self, ours_moved: bool) -> None:
        moved = _with(_a_py('lines', 1005))
        base = synthetic_report()
        ours, theirs = (moved, base) if ours_moved else (base, moved)

        assert driver.merge_baselines(base, ours, theirs)['files']['a.py']['lines'] == 1005

    def test_both_sides_making_one_move_take_it_once(self) -> None:
        moved = _with(_a_py('lines', 1005))

        merged = driver.merge_baselines(synthetic_report(), moved, copy.deepcopy(moved))

        assert merged['files']['a.py']['lines'] == 1005

    @pytest.mark.parametrize(
        ('ours', 'theirs', 'composed'),
        [(1005, 1010, 1015), (990, 1010, 1000), (995, 990, 985)],
        ids=['both-up', 'opposite', 'both-down'],
    )
    def test_two_different_moves_compose(self, ours: int, theirs: int, composed: int) -> None:
        merged = driver.merge_baselines(
            synthetic_report(), _with(_a_py('lines', ours)), _with(_a_py('lines', theirs))
        )

        assert merged['files']['a.py']['lines'] == composed

    def test_measures_of_one_row_compose_independently(self) -> None:
        ours = _with(_a_py('lines', 1005))
        theirs = _with(lambda r: r['files']['a.py'].update(lines=1010, cognitive=125))

        row = driver.merge_baselines(synthetic_report(), ours, theirs)['files']['a.py']

        assert (row['lines'], row['cognitive'], row['prose_lines']) == (1015, 125, 400)

    def test_function_scores_and_private_reads_compose_too(self) -> None:
        ours = _with(lambda r: (r['functions'].update({'a.py::f': 43}),
                                r['tests']['t1.py'].update(private_reads=22)))
        theirs = _with(lambda r: (r['functions'].update({'a.py::f': 41}),
                                  r['tests']['t1.py'].update(private_reads=25)))

        merged = driver.merge_baselines(synthetic_report(), ours, theirs)

        assert merged['functions']['a.py::f'] == 44
        assert merged['tests']['t1.py']['private_reads'] == 27

    def test_a_composition_below_zero_is_refused(self) -> None:
        # Both sides deleted the same lines: no tree holds a negative count.
        with pytest.raises(driver.MergeRefused):
            driver.merge_baselines(
                synthetic_report(),
                _with(_set(('files', 'b.py', 'lines'), 50)),
                _with(_set(('files', 'b.py', 'lines'), 40)),
            )


class TestRows:
    """A row's presence: one side's add or delete stands; two incompatible ones refuse."""

    _NEW = {'lines': 10, 'prose_lines': 1, 'cognitive': 1,
            'function_local_imports': 0, 'reexport_names': 0}

    def test_a_row_one_side_added_stands(self) -> None:
        theirs = _with(_set(('files', 'c.py'), self._NEW))

        merged = driver.merge_baselines(synthetic_report(), synthetic_report(), theirs)

        assert merged['files']['c.py'] == self._NEW

    def test_rows_each_side_added_both_stand(self) -> None:
        ours = _with(_set(('functions', 'a.py::new_ours'), 3))
        theirs = _with(_set(('functions', 'b.py::new_theirs'), 4))

        merged = driver.merge_baselines(synthetic_report(), ours, theirs)

        assert merged['functions']['a.py::new_ours'] == 3
        assert merged['functions']['b.py::new_theirs'] == 4

    def test_one_row_added_differently_on_both_sides_is_refused(self) -> None:
        with pytest.raises(driver.MergeRefused):
            driver.merge_baselines(
                synthetic_report(),
                _with(_set(('functions', 'a.py::new'), 3)),
                _with(_set(('functions', 'a.py::new'), 4)),
            )

    def test_a_row_one_side_deleted_and_the_other_left_is_deleted(self) -> None:
        ours = _with(lambda r: r['functions'].pop('b.py::g'))

        merged = driver.merge_baselines(synthetic_report(), ours, synthetic_report())

        assert 'b.py::g' not in merged['functions']

    def test_a_row_deleted_on_one_side_and_changed_on_the_other_is_refused(self) -> None:
        with pytest.raises(driver.MergeRefused):
            driver.merge_baselines(
                synthetic_report(),
                _with(lambda r: r['functions'].pop('b.py::g')),
                _with(_set(('functions', 'b.py::g'), 9)),
            )


class TestLists:
    """Name sets merge as sets; ordered path lists merge in place, or refuse."""

    def test_patch_targets_union_what_each_side_added(self) -> None:
        ours = _with(_set(('tests', 't1.py', 'patch_targets'), ['bar', 'foo', 'zed']))
        theirs = _with(_set(('tests', 't1.py', 'patch_targets'), ['alpha', 'bar']))

        merged = driver.merge_baselines(synthetic_report(), ours, theirs)

        # theirs removed 'foo', ours kept it: a removal one side made stands.
        assert merged['tests']['t1.py']['patch_targets'] == ['alpha', 'bar', 'zed']

    def test_cluster_paths_each_side_inserted_elsewhere_both_land_in_place(self) -> None:
        ours = _with(_set(('params', 'cluster_paths'), ['a.py', 'x.py', 'b.py']))
        theirs = _with(_set(('params', 'cluster_paths'), ['a.py', 'b.py', 'y.py']))

        merged = driver.merge_baselines(synthetic_report(), ours, theirs)

        assert merged['params']['cluster_paths'] == ['a.py', 'x.py', 'b.py', 'y.py']

    def test_a_removal_and_an_insertion_elsewhere_both_apply(self) -> None:
        ours = _with(_set(('enumeration', 'resolved'), ['b.py']))
        theirs = _with(_set(('enumeration', 'resolved'), ['a.py', 'b.py', 'c.py']))

        merged = driver.merge_baselines(synthetic_report(), ours, theirs)

        assert merged['enumeration']['resolved'] == ['b.py', 'c.py']

    def test_two_insertions_at_one_place_are_refused(self) -> None:
        # The images cannot say which comes first; git's line merge conflicts too.
        with pytest.raises(driver.MergeRefused):
            driver.merge_baselines(
                synthetic_report(),
                _with(_set(('params', 'cluster_paths'), ['a.py', 'x.py', 'b.py'])),
                _with(_set(('params', 'cluster_paths'), ['a.py', 'y.py', 'b.py'])),
            )


class TestScalars:
    """A scalar one side changed is taken; two different changes refuse."""

    def test_a_scalar_one_side_changed_is_taken(self) -> None:
        theirs = _with(_set(('params', 'complexipy_version'), '6.3.0'))

        merged = driver.merge_baselines(synthetic_report(), synthetic_report(), theirs)

        assert merged['params']['complexipy_version'] == '6.3.0'

    @pytest.mark.parametrize(
        'path',
        [('params', 'complexipy_version'), ('params', 'file_line_ceiling'), ('schema_version',)],
        ids=['version', 'ceiling', 'schema'],
    )
    def test_two_different_changes_to_a_scalar_are_refused(self, path: tuple) -> None:
        # Ints outside the per-path sections are settings, never summed.
        ours_value, theirs_value = ('6.3.0', '6.4.0') if 'version' in path[-1] else (2, 3)
        with pytest.raises(driver.MergeRefused) as excinfo:
            driver.merge_baselines(
                synthetic_report(), _with(_set(path, ours_value)), _with(_set(path, theirs_value))
            )
        assert path[-1] in str(excinfo.value)


def _image(directory: Path, name: str, report: dict, *, readme: str | None = None) -> Path:
    target = directory / name
    text = (
        metrics.render_baseline(report)
        if readme is None
        else metrics.render_baseline(report, readme=readme)
    )
    target.write_text(text, encoding='utf-8')
    return target


class TestTheMergeVerb:
    """``merge %O %A %B``, the driver's contract with git: the result in %A, exit 0 or 1."""

    def test_the_merge_is_written_in_the_committed_format(self, tmp_path: Path) -> None:
        base = _image(tmp_path, 'O', synthetic_report())
        ours = _image(tmp_path, 'A', _with(_a_py('lines', 1005)))
        theirs = _image(tmp_path, 'B', _with(_a_py('lines', 1010)))

        assert driver.main(['merge', str(base), str(ours), str(theirs)]) == 0

        assert ours.read_text(encoding='utf-8') == metrics.render_baseline(
            _with(_a_py('lines', 1015))
        )

    def test_the_merged_images_readme_is_carried_through(self, tmp_path: Path) -> None:
        # The driver may run another checkout's instrument than the tree it
        # merges; the merged tree's README is the one the gate compares with.
        base = _image(tmp_path, 'O', synthetic_report())
        ours = _image(tmp_path, 'A', synthetic_report(), readme='a reworded README')
        theirs = _image(tmp_path, 'B', _with(_a_py('lines', 1010)))

        assert driver.main(['merge', str(base), str(ours), str(theirs)]) == 0

        assert json.loads(ours.read_text(encoding='utf-8'))['_README'] == 'a reworded README'

    def test_a_refusal_leaves_gits_conflict_markers_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = _image(tmp_path, 'O', synthetic_report())
        ours = _image(tmp_path, 'A', _with(_set(('params', 'complexipy_version'), '6.3.0')))
        theirs = _image(tmp_path, 'B', _with(_set(('params', 'complexipy_version'), '6.4.0')))

        assert driver.main(['merge', str(base), str(ours), str(theirs)]) == 1

        text = ours.read_text(encoding='utf-8')
        assert '<<<<<<< ours' in text and '>>>>>>> theirs' in text
        err = capsys.readouterr().err
        assert 'complexipy_version' in err and metrics.BASELINE_RELPATH in err

    def test_any_failure_is_a_conflict_with_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # git reads every non-zero exit as a conflict; the resolver must still
        # meet git's markers rather than a silently kept ours.
        def explode(*images: dict) -> dict:
            raise KeyError('a bug the rules never anticipated')

        monkeypatch.setattr(driver, 'merge_baselines', explode)
        base = _image(tmp_path, 'O', synthetic_report())
        ours = _image(tmp_path, 'A', _with(_a_py('lines', 1005)))
        theirs = _image(tmp_path, 'B', _with(_a_py('lines', 1010)))

        assert driver.main(['merge', str(base), str(ours), str(theirs)]) == 1

        assert '<<<<<<< ours' in ours.read_text(encoding='utf-8')

    def test_an_unreadable_image_is_a_conflict(self, tmp_path: Path) -> None:
        # git hands an EMPTY base for an add/add.
        base = tmp_path / 'O'
        base.write_text('', encoding='utf-8')
        ours = _image(tmp_path, 'A', synthetic_report())
        theirs = _image(tmp_path, 'B', _with(_a_py('lines', 1010)))

        assert driver.main(['merge', str(base), str(ours), str(theirs)]) == 1


#: What a throwaway repo needs for git to run THIS checkout's driver. Copied
#: verbatim, so a rename or a dropped line in any of them reddens the cases.
_DRIVER_FILES = (
    '.gitattributes',
    'pyproject.toml',
    'scripts/merge_lane_ratchet_merge.py',
    'scripts/merge_lane_metrics.py',
    'scripts/source_measures.py',
    'scripts/_checkout.py',
    'shared/src/shared/__init__.py',
    'shared/src/shared/safe_io.py',
)


class _Repo:
    """A throwaway repo carrying the real driver files and a seed baseline on main."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / 'repo'
        self.root.mkdir()
        self.env = git_env_with_ceiling(self.root)
        bootstrap = subprocess.run(
            ['git', 'init', '-b', 'main'], cwd=self.root, env=self.env,
            capture_output=True, text=True, check=False,
        )
        assert bootstrap.returncode == 0, bootstrap.stderr
        assert_isolated_git_repo(self.root)
        self.git('config', 'user.name', 'ratchet driver test')
        self.git('config', 'user.email', 'ratchet@example.invalid')
        self.git('config', 'core.hooksPath', str(self.root / '.git' / 'hooks'))
        self.git('config', 'rerere.enabled', 'false')
        for relpath in _DRIVER_FILES:
            target = self.root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO_ROOT / relpath, target)
        self.write_baseline(synthetic_report())
        self.commit_all('seed')

    def git(
        self, *args: str, check: bool = True, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            ['git', *args], cwd=cwd or self.root, env=self.env,
            capture_output=True, text=True, check=False,
        )
        if check:
            assert proc.returncode == 0, f'git {" ".join(args)}: {proc.stderr}'
        return proc

    def install(self) -> None:
        """The real ``install`` verb, run from the copied driver (``-B``: no bytecode)."""
        proc = subprocess.run(
            [sys.executable, '-B', str(self.root / driver.DRIVER_RELPATH),
             'install', '--repo', str(self.root)],
            env=self.env, capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, proc.stderr

    def write_baseline(self, report: dict) -> None:
        target = self.root / metrics.BASELINE_RELPATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(metrics.render_baseline(report), encoding='utf-8')

    def commit_all(self, message: str) -> None:
        self.git('add', '-A')
        self.git('commit', '--quiet', '-m', message)

    def branch(self, name: str, report: dict, entry: dict) -> None:
        """Branch *name* off main: the baseline at *report*, one entry recorded."""
        self.git('switch', '--quiet', '-c', name, 'main')
        self.write_baseline(report)
        metrics.append_authorization(self.root / metrics.LEDGER_DIR_RELPATH, entry)
        self.commit_all(f'{name}: raise and record it')

    def baseline(self, cwd: Path | None = None) -> str:
        return ((cwd or self.root) / metrics.BASELINE_RELPATH).read_text(encoding='utf-8')


def _entry(task_id: str, was: int, now: int) -> dict:
    return metrics.authorization_record(
        metrics.RaiseAuthorization(task_id=task_id, reason=f'net-additive {task_id}'),
        [metrics.Violation.rose('lines', 'a.py', was, now)],
    )


_OURS = _with(lambda r: r['files']['a.py'].update(lines=1005, cognitive=123))
_THEIRS = _with(_a_py('lines', 1007))


def _expected() -> str:
    return metrics.render_baseline(
        driver.merge_baselines(synthetic_report(), _OURS, _THEIRS)
    )


class TestGitMergesThroughTheDriver:
    """Two branches each raising one row and each recording it, merged by real git."""

    @staticmethod
    def _two_branches(tmp_path: Path, *, installed: bool = True) -> _Repo:
        repo = _Repo(tmp_path)
        if installed:
            repo.install()
        repo.branch('task/1', _OURS, _entry('1', 1000, 1005))
        repo.branch('task/2', _THEIRS, _entry('2', 1000, 1007))
        return repo

    def test_git_merge_is_clean_and_holds_the_composed_baseline(self, tmp_path: Path) -> None:
        repo = self._two_branches(tmp_path)
        repo.git('switch', '--quiet', 'task/1')

        merged = repo.git('merge', '--no-edit', 'task/2', check=False)

        assert merged.returncode == 0, merged.stdout + merged.stderr
        assert repo.baseline() == _expected()
        assert json.loads(repo.baseline())['files']['a.py']['lines'] == 1012
        assert {entry['task_id'] for entry in metrics.load_ledger(
            repo.root / metrics.LEDGER_DIR_RELPATH
        )} == {'1', '2'}
        assert repo.git('status', '--porcelain').stdout == ''

    def test_the_lane_probe_composes_and_leaves_the_checkout_clean(
        self, tmp_path: Path
    ) -> None:
        # GitOps.merge_tree_conflicts's argv, at the repository root: the
        # driver's transient .merge_file_* images must be gone when it returns.
        repo = self._two_branches(tmp_path)

        probe = repo.git('merge-tree', '--write-tree', '--name-only', 'task/1', 'task/2', check=False)

        assert probe.returncode == 0, probe.stdout + probe.stderr
        tree = probe.stdout.split()[0]
        assert repo.git('cat-file', '-p', f'{tree}:{metrics.BASELINE_RELPATH}').stdout == _expected()
        assert repo.git('status', '--porcelain', '--ignored').stdout == ''

    def test_a_linked_worktree_merges_through_the_shared_config(self, tmp_path: Path) -> None:
        repo = self._two_branches(tmp_path)
        worktree = tmp_path / 'linked'
        repo.git('worktree', 'add', '--quiet', str(worktree), 'task/1')

        merged = repo.git('merge', '--no-edit', 'task/2', check=False, cwd=worktree / 'scripts')

        assert merged.returncode == 0, merged.stdout + merged.stderr
        assert repo.baseline(cwd=worktree) == _expected()

    def test_without_the_driver_installed_the_merge_conflicts_as_before(
        self, tmp_path: Path
    ) -> None:
        # The wanted polarity: an uninstalled driver is a loud conflict, never
        # a silent pass.
        repo = self._two_branches(tmp_path, installed=False)
        repo.git('switch', '--quiet', 'task/1')

        merged = repo.git('merge', '--no-edit', 'task/2', check=False)
        probe = repo.git('merge-tree', '--write-tree', '--name-only', 'task/1', 'task/2', check=False)

        assert merged.returncode == 1, merged.stdout + merged.stderr
        assert '<<<<<<<' in repo.baseline()
        assert probe.returncode == 1 and metrics.BASELINE_RELPATH in probe.stdout

    def test_a_pair_the_driver_refuses_stops_the_merge(self, tmp_path: Path) -> None:
        repo = _Repo(tmp_path)
        repo.install()
        repo.branch(
            'task/1', _with(_set(('params', 'complexipy_version'), '6.3.0')), _entry('1', 1, 2)
        )
        repo.branch(
            'task/2', _with(_set(('params', 'complexipy_version'), '6.4.0')), _entry('2', 1, 3)
        )
        repo.git('switch', '--quiet', 'task/1')

        merged = repo.git('merge', '--no-edit', 'task/2', check=False)

        assert merged.returncode == 1, merged.stdout + merged.stderr
        assert '<<<<<<< ours' in repo.baseline()
        assert 'complexipy_version' in merged.stdout + merged.stderr


class TestTheWiringIsStructurallyPinned:
    def test_gitattributes_routes_the_baseline_to_the_installed_driver(self) -> None:
        lines = (_REPO_ROOT / '.gitattributes').read_text(encoding='utf-8').splitlines()

        assert f'{metrics.BASELINE_RELPATH} merge={driver.DRIVER_NAME}' in lines

    def test_the_driver_command_runs_this_file_from_the_checkout_root(self) -> None:
        assert (_REPO_ROOT / driver.DRIVER_RELPATH).resolve() == Path(driver.__file__).resolve()
        assert driver.DRIVER_COMMAND.endswith(f'{driver.DRIVER_RELPATH} merge %O %A %B')
