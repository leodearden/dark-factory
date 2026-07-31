"""Guard tests: pytest's basetemp can never leak git writes into a live worktree.

Incident esc-3072-3.  Git repository discovery walks UP the directory tree, so
a pytest basetemp nested inside a live task worktree
(``.worktrees/<task>/.pytest-tmp/``) makes every ``cwd=tmp_path`` git call
silently retarget *the enclosing worktree* — three blobs were written into a
real task's object store and ``foo.py`` was staged at stages 1/2/3.

Task 3182 shipped two per-call, opt-in layers in ``_orch_helpers``
(``assert_isolated_git_repo``, ``git_env_with_ceiling``).  This module covers
the two suite-wide, opt-out-impossible layers in the root ``df_pytest_isolation``
module:

* a session-scoped autouse fixture that sets ``GIT_CEILING_DIRECTORIES`` to the
  running basetemp, so the upward walk physically cannot leave it; and
* a collection-time ``--basetemp`` rejection, so a basetemp aimed inside a
  worktree fails loudly before a single test runs.

``_init_repo`` and ``_object_store`` are copied (not imported — ``_orch_helpers``
is not importable from this root-level suite) from
``orchestrator/tests/test_git_repo_isolation_guard.py`` so the containment proof
here is shaped identically to 3182's and the two read as one defence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import git_ceiling_value  # noqa: E402

_CEILING_KEY = 'GIT_CEILING_DIRECTORIES'


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit.  Creates its own
    target, so it cannot escape into an enclosing repo.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test'], cwd=path, check=True, capture_output=True,
    )
    (path / 'README.md').write_text('# sentinel\n')
    subprocess.run(['git', 'add', '-A'], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'initial'], cwd=path, check=True, capture_output=True,
    )
    return path


def _object_store(repo: Path) -> set[str]:
    """Snapshot of every loose/packed object file under ``<repo>/.git/objects``.

    Used to prove that a refused call left the victim's object store byte-for-
    byte unchanged — not merely that the command exited non-zero.
    """
    objects = repo / '.git' / 'objects'
    if not objects.is_dir():
        return set()
    return {str(p.relative_to(objects)) for p in objects.rglob('*') if p.is_file()}


def _incident_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Rebuild the esc-3072-3 shape: basetemp nested inside a live worktree.

    Returns ``(victim_repo, simulated_basetemp, nested_non_repo_dir)``.
    """
    victim = _init_repo(tmp_path / 'live-worktree')
    simulated_basetemp = victim / '.pytest-tmp'
    nested = simulated_basetemp / 'test_x0' / 'sub'
    nested.mkdir(parents=True)
    return victim, simulated_basetemp, nested


class TestGitCeilingValueConstruction:
    """git_ceiling_value builds an entry git will actually honour."""

    def test_returns_an_absolute_path_for_a_relative_basetemp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Git silently IGNORES non-absolute ceiling entries.

        A relative entry is not a weaker ceiling, it is *no* ceiling — the walk
        proceeds exactly as if the variable were unset.  Same rationale as
        ``_orch_helpers.git_env_with_ceiling``.
        """
        (tmp_path / 'bt').mkdir()
        monkeypatch.chdir(tmp_path)

        value = git_ceiling_value('bt')

        assert Path(value).is_absolute()
        assert value == str((tmp_path / 'bt').resolve())

    def test_resolves_symlinks(self, tmp_path: Path) -> None:
        """Git compares the ceiling against the RESOLVED cwd.

        An unresolved entry never matches, so the containment is inert.
        """
        real = tmp_path / 'real-basetemp'
        real.mkdir()
        link = tmp_path / 'link-to-basetemp'
        link.symlink_to(real)

        assert git_ceiling_value(link) == str(real.resolve())

    def test_preserves_a_pre_existing_value_and_appends(self, tmp_path: Path) -> None:
        """An operator- or CI-set ceiling must never be clobbered.

        Git treats the variable as a colon-separated list where ANY entry can
        stop the walk, so appending is strictly additive containment.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        value = git_ceiling_value(basetemp, existing='/opt/ci-ceiling')

        assert value == f'/opt/ci-ceiling:{basetemp.resolve()}'

    def test_preserves_every_entry_of_a_multi_entry_value(self, tmp_path: Path) -> None:
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        value = git_ceiling_value(basetemp, existing='/opt/a:/opt/b')

        assert value.split(':') == ['/opt/a', '/opt/b', str(basetemp.resolve())]

    def test_is_idempotent_when_the_entry_is_already_present(self, tmp_path: Path) -> None:
        """Conftests nest: a root-rootdir run loads BOTH the root conftest and a
        subproject conftest, so the value must not accumulate duplicates.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()
        once = git_ceiling_value(basetemp)

        twice = git_ceiling_value(basetemp, existing=once)

        assert twice == once

    def test_is_idempotent_amid_other_entries(self, tmp_path: Path) -> None:
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()
        existing = f'/opt/a:{basetemp.resolve()}:/opt/b'

        assert git_ceiling_value(basetemp, existing=existing) == existing

    def test_an_empty_existing_value_does_not_produce_a_leading_separator(
        self, tmp_path: Path,
    ) -> None:
        """An empty entry is git's "subsequent entries are not symlinks" marker,
        not a no-op — never emit one by accident.
        """
        basetemp = tmp_path / 'bt'
        basetemp.mkdir()

        for empty in (None, ''):
            value = git_ceiling_value(basetemp, existing=empty)
            assert value == str(basetemp.resolve())
            assert '' not in value.split(':')


class TestGitCeilingValueContainsRealGit:
    """The premise proof: the value actually stops real git's upward walk."""

    def test_a_write_from_inside_the_basetemp_cannot_reach_the_enclosing_repo(
        self, tmp_path: Path,
    ) -> None:
        """The esc-3072-3 shape, refused.

        The object store is the assertion that matters — a non-zero exit code
        alone would not prove nothing was written.
        """
        victim, simulated_basetemp, nested = _incident_layout(tmp_path)
        before = _object_store(victim)
        env = os.environ.copy()
        env[_CEILING_KEY] = git_ceiling_value(simulated_basetemp)

        result = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, env=env, input=b'leaked payload', capture_output=True,
        )

        assert result.returncode != 0
        assert _object_store(victim) == before

    def test_control_without_the_ceiling_the_write_does_reach_the_repo(
        self, tmp_path: Path,
    ) -> None:
        """Non-vacuity control: the SAME command, ceiling popped, escapes.

        Without this the containment test above could be passing for any
        unrelated reason (a bad git invocation, a missing binary).
        """
        victim, _simulated_basetemp, nested = _incident_layout(tmp_path)
        before = _object_store(victim)
        env = os.environ.copy()
        env.pop(_CEILING_KEY, None)

        result = subprocess.run(
            ['git', 'hash-object', '-w', '--stdin'],
            cwd=nested, env=env, input=b'leaked payload', capture_output=True,
        )

        assert result.returncode == 0, result.stderr.decode()
        assert _object_store(victim) != before

    def test_a_legitimate_repo_under_the_basetemp_still_resolves(
        self, tmp_path: Path,
    ) -> None:
        """No false containment.

        Every legitimate ``tmp_path`` repo and linked worktree lives UNDER the
        basetemp; the ceiling must leave all of them working, or a suite-wide
        ceiling is unshippable.  Git stops only when the walk would ascend INTO
        or above a ceiling entry, so a repo root strictly below it is found
        normally.
        """
        _victim, simulated_basetemp, _nested = _incident_layout(tmp_path)
        inner = _init_repo(simulated_basetemp / 'test_x0' / 'repo')
        deep = inner / 'deep'
        deep.mkdir()
        env = os.environ.copy()
        env[_CEILING_KEY] = git_ceiling_value(simulated_basetemp)

        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=deep, env=env, capture_output=True, text=True,
        )

        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()).resolve() == inner.resolve()
