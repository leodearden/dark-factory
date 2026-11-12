"""Real-git fixtures for tests whose verdict is git semantics: throwaway repos
(signing off, hooks disabled) plus a runner that scripts chosen git
subcommands over them.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import subprocess
from collections.abc import Mapping, Sequence
from typing import NamedTuple, TypedDict

from shared.git_async import GitResult, run_git

# ---------------------------------------------------------------------------
# Real-git repo builders
# ---------------------------------------------------------------------------


def git_stdout(path, *args) -> str:
    """Run ``git -C <path> <args>``; return its stripped stdout.

    Raises CalledProcessError on a non-zero exit.
    """
    return subprocess.run(
        ['git', '-C', str(path), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _init_single_commit_repo(path, files: dict[str, str], message: str) -> str:
    """Create a repo at *path* whose ``main`` holds one commit writing *files*
    (name -> content); return that commit's full SHA.

    Signing is off and ``core.hooksPath`` points nowhere, so neither a
    developer's global gpgsign setting nor any hook can hang or fail a later
    git command in the repo.
    """
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
    git_stdout(path, 'config', 'user.email', 't@e.example')
    git_stdout(path, 'config', 'user.name', 'T')
    git_stdout(path, 'config', 'commit.gpgsign', 'false')
    git_stdout(path, 'config', 'core.hooksPath', '/dev/null')
    for name, content in files.items():
        (path / name).write_text(content)
    git_stdout(path, 'add', '-A')
    git_stdout(path, 'commit', '-q', '--no-verify', '-m', message)
    return git_stdout(path, 'rev-parse', 'HEAD')


def _init_git_repo(path) -> str:
    """Create a minimal git repo at path with one seed.txt commit; return full SHA.

    A ``done_provenance={'kind': 'merged', 'commit': ...}`` write needs a real
    ``main`` branch for the interceptor's ``_verify_commit_on_main`` ancestor
    backstop (``git merge-base --is-ancestor <sha> main``) to resolve against.
    """
    return _init_single_commit_repo(path, {'seed.txt': 'seed\n'}, 'seed')


def init_ab_commit_repo(path) -> str:
    """Create a repo at *path* whose ``main`` holds one commit adding a.txt and
    b.txt; return that commit's full SHA.

    The single-parent prefix of :func:`init_merge_history_repo`.
    """
    return _init_single_commit_repo(
        path, {'a.txt': 'a\n', 'b.txt': 'b\n'}, 'feat: ship a + b',
    )


class MergeHistoryShas(TypedDict):
    single: str
    merge: str
    conflict_merge: str


def init_merge_history_repo(path) -> MergeHistoryShas:
    """Build a repo with a single-parent commit, a clean --no-ff merge, and a
    --no-ff merge that resolves a real conflict; return their full SHAs.

    Commit graph on ``main``:

      1. ``single`` — :func:`init_ab_commit_repo`'s commit adding a.txt +
         b.txt (a genuine single-parent commit, used to pin
         ``--first-parent -m`` as a no-op on non-merge commits).
      2. a ``feature`` branch off commit 1 adds feature.py plus
         extra_1.py..extra_4.py (five files total, so max_files truncation is
         reachable for merge provenance — see
         tests/test_stages.py::TestDoneProvenanceSection::test_merge_commit_file_list_respects_max_files_truncation);
         main then merges it via ``git merge --no-ff`` — recorded as
         ``merge``. Plain ``git show --name-only`` reports NOTHING for this
         commit (git's combined diff is empty for a clean merge); this is the
         fixture that reproduces the Stage-2 provenance bug.
      3. a ``conflict`` branch off commit 1 edits a.txt and adds
         conflict_only.py; main separately edits a.txt after ``merge``.
         Merging ``conflict`` into main therefore CONFLICTS on a.txt —
         resolved by hand and committed, recorded as ``conflict_merge``. This
         pins the OTHER, worse symptom: plain ``git show --name-only`` on a
         merge that resolved a conflict reports ONLY the conflict-resolution
         file(s), silently omitting every non-conflicting file the branch
         also brought in (measured on this repo's own history: commit
         a75568bd11 showed 1 file under the bare invocation vs. 6 real
         first-parent files, none of which was that 1 file — see
         tests/test_stages.py::TestDoneProvenanceSection::test_merge_commit_with_resolved_conflict_reports_full_first_parent_file_list).
         ``conflict_only.py`` deliberately does NOT reuse the feature/extra_*
         names from scenario 2: this branch forks off ``single`` (before the
         feature merge), so those names are free to reuse on disk, but doing
         so would make feature.py conflict too (both sides would have
         independently "added" it), muddying the one conflict this scenario
         needs.

    The branch/merge sequence follows
    tests/test_audit_found_on_main_provenance.py::_build_test_repo.
    """
    single = init_ab_commit_repo(path)

    git_stdout(path, 'checkout', '-q', '-b', 'feature')
    (path / 'feature.py').write_text('feature = 1\n')
    for i in range(1, 5):
        (path / f'extra_{i}.py').write_text(f'extra_{i} = 1\n')
    git_stdout(path, 'add', '-A')
    git_stdout(path, 'commit', '-q', '--no-verify', '-m', 'feat: add feature')
    git_stdout(path, 'checkout', '-q', 'main')
    git_stdout(path, 'merge', '--no-ff', '--no-verify', '-q', '-m', 'Merge feature into main', 'feature')
    merge = git_stdout(path, 'rev-parse', 'HEAD')

    # A second branch that edits a.txt independently of main's own later
    # edit of the same file — merging it conflicts, exercising the
    # (worse) symptom where a bare `git show` on a resolved-conflict
    # merge reports only the conflicted path and hides everything else.
    git_stdout(path, 'checkout', '-q', '-b', 'conflict', single)
    (path / 'a.txt').write_text('a-conflict-branch\n')
    (path / 'conflict_only.py').write_text('conflict_only = 1\n')
    git_stdout(path, 'add', '-A')
    git_stdout(path, 'commit', '-q', '--no-verify', '-m', 'feat: edit a.txt on conflict branch')
    git_stdout(path, 'checkout', '-q', 'main')
    (path / 'a.txt').write_text('a-main-edit\n')
    git_stdout(path, 'add', '-A')
    git_stdout(path, 'commit', '-q', '--no-verify', '-m', 'chore: edit a.txt on main after feature merge')

    # Deliberately not using `git_stdout` (check=True): a real conflict makes
    # `git merge` exit 1 and leave conflict markers instead of a commit.
    conflict_result = subprocess.run(
        ['git', '-C', str(path), 'merge', '--no-ff', '--no-verify', '-q',
         '-m', 'Merge conflict into main', 'conflict'],
        capture_output=True, text=True,
    )
    assert conflict_result.returncode != 0, (
        'expected a.txt to conflict (both branches edited it since the '
        'common ancestor); merge unexpectedly succeeded cleanly'
    )
    (path / 'a.txt').write_text('a-resolved\n')
    git_stdout(path, 'add', 'a.txt')
    git_stdout(path, 'commit', '-q', '--no-verify', '-m', 'Merge conflict into main (resolved)')
    conflict_merge = git_stdout(path, 'rev-parse', 'HEAD')

    return {'single': single, 'merge': merge, 'conflict_merge': conflict_merge}


# ---------------------------------------------------------------------------
# Scripted git runner
# ---------------------------------------------------------------------------


_GIT_OPTIONS_TAKING_A_VALUE = frozenset({'-C', '-c'})


class GitCall(NamedTuple):
    """One call a :class:`ScriptedGitRunner` received."""

    argv: tuple[str, ...]
    timeout: float | None

    @property
    def subcommand(self) -> str | None:
        """The git subcommand: the first token after ``git`` and its global
        options (``-C <path>``, ``-c <key=value>``, bare flags)."""
        tokens = iter(self.argv[1:])
        for token in tokens:
            if token in _GIT_OPTIONS_TAKING_A_VALUE:
                next(tokens, None)
            elif not token.startswith('-'):
                return token
        return None


class ScriptedGitRunner:
    """A ``shared.git_async.run_git`` stand-in that scripts chosen subcommands.

    *script* maps a git subcommand (``'merge-base'``, ``'rev-parse'``,
    ``'show'``) to the ``GitResult`` it returns or the exception it raises. A
    key matches only the call's :attr:`GitCall.subcommand`, never a path, ref
    or other argument. A call whose subcommand is not scripted goes to the
    real ``run_git``, so a test faking ``merge-base`` still resolves its SHA
    for real against :func:`_init_git_repo`'s repo. Every call is recorded in
    ``calls``.
    """

    def __init__(self, script: Mapping[str, GitResult | Exception]):
        self._script = dict(script)
        self.calls: list[GitCall] = []

    async def __call__(self, cmd: Sequence[str], *, timeout: float | None = None) -> GitResult:
        call = GitCall(tuple(cmd), timeout)
        self.calls.append(call)
        outcome = self._script.get(call.subcommand) if call.subcommand else None
        if outcome is None:
            return await run_git(cmd, timeout=timeout)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
