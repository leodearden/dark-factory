"""What the merge-lane ratchet suites share: the instrument, the one seed report,
the throwaway repo the git-driving suites build history in, and what a gate
copied into that repo needs beside it.

The ``scripts/`` modules are imported HERE, once, so no suite carries its own
``sys.path`` bootstrap; orchestrator/pyproject.toml's [tool.pyright] extraPaths
already carries "../scripts", so this needs no type-checker config.

``synthetic_report`` is the ONE seed every suite measures against. A second
declaration would drift, and the suites would then disagree about what a raise
looks like. ``RatchetRepo`` and the record builders are here for the same
reason: the commit-time and the advance-time gate suites both build real
history, and a second throwaway-repo class would re-declare the esc-3072-3
bootstrap. A helper lives here only when more than one ratchet suite uses it;
``_orch_helpers`` is the home for what the whole test tree shares.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Self

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

_CHECKOUT = Path(__file__).parents[2]
_SCRIPTS = _CHECKOUT / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402
import source_measures  # type: ignore[import-not-found]  # noqa: E402

__all__ = [
    'GATE_IMPORT_CLOSURE',
    'RatchetRepo',
    'a_py_cognitive_raise',
    'blank_file_entry',
    'metrics',
    'record',
    'source_measures',
    'synthetic_report',
]


#: What either gate script needs beside it to run in a copied checkout: the
#: gates' git plumbing, the instrument, and the checkout binding with the
#: pyproject.toml it reads. A suite installing a gate adds only that gate and the
#: hooks that run it, so a new import is named here once.
GATE_IMPORT_CLOSURE = (
    'scripts/ratchet_revisions.py',
    'scripts/merge_lane_metrics.py',
    'scripts/source_measures.py',
    'scripts/_checkout.py',
    'pyproject.toml',
)


def synthetic_report() -> dict:
    """Two cluster files, three functions, two test files -- a COMPLETE report.

    Every number is small and hand-chosen so a perturbation reads at a
    glance, and the cluster totals the ratchet compares are DERIVED from
    these entries by ``derive_totals`` rather than stated a second time.
    """
    return {
        'schema_version': 1,
        'params': {
            'complexipy_version': '6.2.0',
            'cluster_paths': ['a.py', 'b.py'],
            'file_line_ceiling': 1500,
            'new_function_cognitive_ceiling': 15,
        },
        'enumeration': {
            'requested': ['a.py', 'b.py'],
            'resolved': ['a.py', 'b.py'],
            # The LIVE shape: the test-tree half is two counts, and it is the one
            # key render_baseline drops on the way to the committed file.
            'test_tree': {'requested': 9, 'resolved': 2},
            'unreadable': [],
            'complete': True,
        },
        'files': {
            'a.py': {
                'lines': 1000,
                'prose_lines': 400,
                'cognitive': 120,
                'function_local_imports': 3,
                'reexport_names': 5,
            },
            'b.py': {
                'lines': 200,
                'prose_lines': 50,
                'cognitive': 30,
                'function_local_imports': 1,
                'reexport_names': 0,
            },
        },
        'functions': {'a.py::f': 40, 'a.py::C::m': 12, 'b.py::g': 7},
        'tests': {
            't1.py': {'patch_targets': ['foo', 'bar'], 'private_reads': 20},
            't2.py': {'patch_targets': ['bar', 'baz'], 'private_reads': 5},
        },
    }


def blank_file_entry(**overrides: int) -> dict:
    """A cluster-file entry with every measure zero, then *overrides* applied."""
    entry = {
        'lines': 0,
        'prose_lines': 0,
        'cognitive': 0,
        'function_local_imports': 0,
        'reexport_names': 0,
    }
    entry.update(overrides)
    return entry


class RatchetRepo:
    """A throwaway repo carrying the ratchet artifacts at their real paths.

    Every git invocation goes through :meth:`git`, so ``cwd`` and the ceiling
    environment are established in exactly one place and no case builder can
    shell out unguarded (esc-3072-3).
    """

    def __init__(self, root: Path) -> None:
        # FIRST, before any subprocess: a rejected root must write nothing.
        assert_isolated_git_repo(root)
        self.root = root
        self.env = git_env_with_ceiling(root)

    @staticmethod
    def _init_main(tmp_path: Path) -> Path:
        """``git init -b main`` in a fresh ``tmp_path/repo``; the new root."""
        root = tmp_path / 'repo'
        root.mkdir()
        # The pre-flight cannot precede `git init` -- it refuses any directory
        # that is not ALREADY a repo root, which is what this call creates. The
        # ceiling is applied to the bootstrap on its own, and `git init` only
        # ever writes into its own cwd.
        bootstrap = subprocess.run(
            ['git', 'init', '-b', 'main'],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=git_env_with_ceiling(root),
            check=False,
        )
        assert bootstrap.returncode == 0, bootstrap.stderr
        return root

    @classmethod
    def seeded(cls, tmp_path: Path, report: dict | None = None) -> Self:
        """``git init`` plus one commit on main carrying a baseline and no entry.

        HEAD is left on main, with hooks pinned off.
        """
        repo = cls(cls._init_main(tmp_path))
        repo.git('config', 'user.name', 'ratchet gate test')
        repo.git('config', 'user.email', 'ratchet@example.invalid')
        # Pinned explicitly so an operator's GLOBAL core.hooksPath cannot reach
        # into a throwaway repo and run real hooks over it.
        repo.git('config', 'core.hooksPath', str(repo.root / '.git' / 'hooks'))
        repo.write_baseline(report if report is not None else synthetic_report())
        repo.commit_all('seed the ratchet artifacts')
        return repo

    @classmethod
    def unborn(cls, tmp_path: Path) -> Self:
        """An initialized repo with NO commit yet -- HEAD names nothing."""
        return cls(cls._init_main(tmp_path))

    def attempt(self, *args: str) -> subprocess.CompletedProcess[str]:
        """A real git command, hooks and all, whose outcome is the caller's assertion."""
        return subprocess.run(
            ['git', *args],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )

    def git(self, *args: str, check: bool = True) -> str:
        proc = self.attempt(*args)
        if check:
            assert proc.returncode == 0, (
                f'git {" ".join(args)} failed (rc={proc.returncode}): '
                f'{proc.stderr.strip()}'
            )
        return proc.stdout.strip()

    def commit_all(self, message: str) -> str:
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def write_file(self, relpath: str, text: str) -> Path:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        return target

    def write_baseline(self, report: dict) -> Path:
        return self.write_file(metrics.BASELINE_RELPATH, metrics.render_baseline(report))

    def write_baseline_text(self, text: str) -> Path:
        """Raw bytes, for a blob ``render_baseline`` would never produce."""
        return self.write_file(metrics.BASELINE_RELPATH, text)

    def write_entry(self, record: dict) -> str:
        """Record *record* as the writer does; its path, repo-relative."""
        written = metrics.append_authorization(self.root / metrics.LEDGER_DIR_RELPATH, record)
        return written.relative_to(self.root).as_posix()

    def stage(self, *relpaths: str) -> None:
        self.git('add', '--', *relpaths)

    def install_from_checkout(self, *relpaths: str) -> None:
        """Copy THIS checkout's *relpaths* in verbatim and executable, at the same paths.

        Copied rather than restated, so a renamed gate or a dropped hook section
        stops the copy working and the case that relies on it goes red.
        """
        for relpath in relpaths:
            target = self.root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_CHECKOUT / relpath, target)
            target.chmod(0o755)


def record(raises: list[metrics.Violation], task_id: str = '5722') -> dict:
    """The ledger record `--authorize-raise <task_id>` appends for *raises*."""
    return metrics.authorization_record(
        metrics.RaiseAuthorization(task_id=task_id, reason=f'net-additive {task_id}'),
        raises,
    )


def a_py_cognitive_raise(was: int, now: int, task_id: str) -> dict:
    """The record `--authorize-raise` appends when a.py's `cognitive` rises *was* -> *now*."""
    b_py_cognitive = synthetic_report()['files']['b.py']['cognitive']
    return record(
        [
            metrics.Violation.rose('cognitive', 'a.py', was, now),
            metrics.Violation.rose(
                'total:cognitive',
                metrics.CLUSTER_TOTAL_KEY,
                was + b_py_cognitive,
                now + b_py_cognitive,
            ),
        ],
        task_id,
    )
