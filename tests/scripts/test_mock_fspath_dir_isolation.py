"""Guard tests: a mock can never leave a mock-derived directory tree in the CWD (task 5145).

Task 4592 committed 72 files under ``AsyncMock/mock.data_dir/<id>/``: the old
guard and ``.gitignore`` knew only the ``MagicMock`` spelling. This covers the
generalised guard in ``df_pytest_isolation``, shaped like the other
``test_*_isolation.py`` family members.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from nested_pytest_session import binding_conftest, run_nested_pytest  # noqa: E402

import df_pytest_isolation  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    MOCK_FSPATH_ROOT_NAMES,
    fixture_marker,
    mock_derived_stray_dir_reason,
    mock_derived_stray_dirs,
)

# Reached through the module, never imported: importing a fixture into a test
# module registers it there and would shadow the wiring under test.
_GUARD_NAME = '_df_no_mock_derived_stray_dirs'


class TestMockFspathRootNames:
    """The name set covers every real mock-derived tree, and nothing else."""

    @pytest.mark.parametrize(
        'build',
        [
            lambda: AsyncMock().data_dir,
            lambda: MagicMock().project_root,
            lambda: NonCallableMagicMock(),
        ],
        ids=['AsyncMock-child-task-4592', 'MagicMock-child-task-3223', 'NonCallableMagicMock'],
    )
    def test_a_real_mock_path_leads_with_a_covered_name(
        self, build: Callable[[], os.PathLike[str]],
    ) -> None:
        leading = Path(os.fspath(build())).parts[0]

        assert leading in MOCK_FSPATH_ROOT_NAMES, (
            f'os.fspath() of this mock leads with {leading!r}, which the guard '
            f'does not look for: {MOCK_FSPATH_ROOT_NAMES!r}'
        )

    @pytest.mark.parametrize('name', ['Mock', 'NonCallableMock', 'PropertyMock'])
    def test_a_non_pathlike_mock_class_is_not_covered(self, name: str) -> None:
        """These classes cannot produce a path, so a dir by that name is never ours to delete."""
        assert name not in MOCK_FSPATH_ROOT_NAMES

    def test_it_is_a_sorted_tuple(self) -> None:
        assert isinstance(MOCK_FSPATH_ROOT_NAMES, tuple)
        assert list(MOCK_FSPATH_ROOT_NAMES) == sorted(MOCK_FSPATH_ROOT_NAMES)


class TestMockDerivedStrayDirs:
    """The detector, against ``tmp_path`` standing in for the CWD."""

    def test_an_empty_dir_has_none(self, tmp_path: Path) -> None:
        assert mock_derived_stray_dirs(tmp_path) == []

    @pytest.mark.parametrize('name', MOCK_FSPATH_ROOT_NAMES)
    def test_each_mock_derived_tree_is_found(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name / 'mock.x' / '1').mkdir(parents=True)

        assert mock_derived_stray_dirs(tmp_path) == [tmp_path / name]

    def test_unrelated_dirs_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / 'Mock').mkdir()
        (tmp_path / 'data').mkdir()

        assert mock_derived_stray_dirs(tmp_path) == []

    def test_a_regular_file_is_never_reported(self, tmp_path: Path) -> None:
        """The fspath shape is always a directory, and rmtree on a file would be wrong."""
        (tmp_path / 'AsyncMock').write_text('')

        assert mock_derived_stray_dirs(tmp_path) == []

    def test_several_are_all_reported_in_name_order(self, tmp_path: Path) -> None:
        for name in reversed(MOCK_FSPATH_ROOT_NAMES):
            (tmp_path / name).mkdir()

        assert mock_derived_stray_dirs(tmp_path) == [
            tmp_path / name for name in MOCK_FSPATH_ROOT_NAMES
        ]


class TestMockDerivedStrayDirReason:
    def test_no_strays_is_none(self) -> None:
        assert mock_derived_stray_dir_reason('tests/test_x.py::test_y', []) is None

    def test_it_names_the_test_the_tree_and_the_remedy(self, tmp_path: Path) -> None:
        stray = tmp_path / 'AsyncMock'
        reason = mock_derived_stray_dir_reason('tests/test_x.py::test_y', [stray])

        assert reason is not None
        assert 'tests/test_x.py::test_y' in reason
        assert str(stray) in reason
        assert 'os.PathLike' in reason
        assert 'tmp_path' in reason


class TestTheGuardFixtureIsArmed:
    """Shape pinned structurally; ``tests/scripts`` itself does not bind the guard."""

    def test_the_fixture_exists(self) -> None:
        assert hasattr(df_pytest_isolation, _GUARD_NAME), (
            f'df_pytest_isolation defines no {_GUARD_NAME}; the pure helpers above '
            'protect nothing on their own.'
        )

    def test_the_fixture_is_function_scoped_and_autouse(self) -> None:
        """Function scope is what lets the failure name the test that leaked."""
        marker = fixture_marker(getattr(df_pytest_isolation, _GUARD_NAME))

        assert marker.scope == 'function', f'{_GUARD_NAME} scope is {marker.scope!r}'
        assert marker.autouse is True, f'{_GUARD_NAME} must be autouse — nothing requests it'


def _is_ignored(relpath: str) -> subprocess.CompletedProcess[str]:
    """Ask git whether *relpath* is ignored; it answers for paths that need not exist.

    Exit 0 is ignored and 1 is not, so the code is read rather than checked.
    """
    return subprocess.run(
        ['git', 'check-ignore', '-q', '--', relpath],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


class TestEveryMockDerivedShapeIsGitIgnored:
    """``.gitignore`` follows ``MOCK_FSPATH_ROOT_NAMES``; a name it lags on goes red here.

    The subproject prefixes matter because verify runs ``cd <subproject> &&
    pytest``, so a leaked tree lands under the subproject, not the repo root.
    """

    @pytest.mark.parametrize('prefix', ['', 'fused-memory/', 'orchestrator/'])
    @pytest.mark.parametrize('name', MOCK_FSPATH_ROOT_NAMES)
    def test_both_shapes_are_ignored(self, name: str, prefix: str) -> None:
        shapes = {
            'fspath DIRECTORY': f'{prefix}{name}/mock.data_dir/1/probe',
            'str()/repr FILE': f"{prefix}<{name} name='mock.x' id='1'>",
        }
        uncovered = {
            shape: (relpath, probed.returncode, probed.stderr)
            for shape, relpath in shapes.items()
            if (probed := _is_ignored(relpath)).returncode != 0
        }

        assert not uncovered, (
            f'mock-derived paths a `git add -- .` would stage: {uncovered!r} '
            '(shape: (path, git check-ignore exit code, stderr)). Add '
            f'`{name}/` and `<{name}*` to the .gitignore block "Test SQLite '
            'artifacts — MagicMock path side-effects".'
        )

    def test_the_probe_can_say_no(self) -> None:
        """Non-vacuity control: a tracked, unignored file is reported as not ignored."""
        probed = _is_ignored('df_pytest_isolation.py')

        assert probed.returncode == 1, (
            f'git check-ignore exited {probed.returncode} for a tracked file, so '
            f'the probe proves nothing. stderr={probed.stderr!r}'
        )


def _nested_run(root: Path, body: str, *, imports: str = ''):
    source = f'import os\nfrom pathlib import Path\n{imports}\n\ndef test_leaks():\n{body}\n'
    return run_nested_pytest(root, {
        'conftest.py': binding_conftest(_GUARD_NAME),
        'test_x.py': source,
    })


class TestTheGuardFailsTheRun:
    """What the guard does to a real nested session whose test leaks a tree."""

    @pytest.mark.parametrize('mock_class', ['AsyncMock', 'MagicMock'])
    def test_a_leaked_tree_fails_the_run_and_is_removed(
        self, tmp_path: Path, mock_class: str,
    ) -> None:
        root = tmp_path / mock_class.lower()
        result = _nested_run(
            root,
            f'    Path(os.fspath({mock_class}().data_dir)).mkdir(parents=True)',
            imports=f'from unittest.mock import {mock_class}',
        )
        combined = result.stdout + result.stderr

        assert result.returncode != 0, (
            f'a test that leaked a {mock_class}/ tree exited 0 — the guard is inert. '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert '1 passed' in combined, (
            'the nested TEST must still pass on its own, or this proves nothing '
            f'about the guard. stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'test_leaks' in combined, (
            f'the failure does not name the leaking test. '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert mock_class in combined, (
            f'the failure does not name the stray tree. '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert not (root / mock_class).exists(), (
            f'the guard failed the run but left {root / mock_class} behind. '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )

    def test_a_clean_test_exits_zero(self, tmp_path: Path) -> None:
        """Non-vacuity control: the identical harness, minus the leak."""
        result = _nested_run(tmp_path / 'clean', '    pass')

        assert result.returncode == 0, (
            f'the control run failed for an unrelated reason: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )

    def test_unrelated_dirs_survive_and_pass(self, tmp_path: Path) -> None:
        """The guard is narrow: it destroys nothing it does not own."""
        root = tmp_path / 'unrelated'
        result = _nested_run(
            root,
            "    Path('Mock/x').mkdir(parents=True)\n    Path('data').mkdir()",
        )

        assert result.returncode == 0, (
            f'unrelated directories failed the run: '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert (root / 'Mock' / 'x').is_dir(), (
            f'the guard deleted Mock/, which no mock can produce. '
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert (root / 'data').is_dir(), (
            f'the guard deleted data/. stdout={result.stdout!r} stderr={result.stderr!r}'
        )
