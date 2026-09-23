"""One definition of "a directory that looks like a git checkout" (task 4722).

`CodebaseVerifier.verify()` refuses any `codebase_root` that is not a
directory containing a `.git` entry (PRD D4's fail-closed pre-flight), so
every test that drives a real verifier has to build one.  That scaffolding
was written three times across the task's diff; since
`verify._resolve_codebase_root`'s notion of "usable" is a shared contract,
its test-side mirror lives HERE, in one place — the next change to it
(accepting a bare directory, say, or an ancestor `.git`) is a one-file edit
rather than a hunt through four sites that must not drift apart.

Lives outside conftest.py, under the `_fm_helpers.py` convention documented
in tests/conftest.py, for two reasons: a `tests/reconciliation/conftest.py`
fixture would be invisible to `tests/test_targeted.py` one directory up, and
a uniquely-named sibling module avoids the `sys.modules['conftest']`
collision that arises when a workspace-root pytest loads several
subprojects' conftests in one process.  tests/conftest.py puts this
directory on `sys.path`, so `from _git_root_helper import make_git_root`
resolves from `tests/` and `tests/reconciliation/` alike.
"""

from pathlib import Path


def make_git_root(base: Path, name: str = 'repo', *, dot_git: str = 'dir') -> Path:
    """Create and return ``base / name``, shaped like a checkout verify() accepts.

    ``dot_git='dir'`` (the default) is an ordinary clone, where ``.git`` is a
    directory.

    ``dot_git='file'`` is a ``git worktree`` checkout, where ``.git`` is a
    FILE holding a ``gitdir:`` pointer.  It is an explicit variant rather
    than a second helper because it is the shape this factory runs every
    task in: an implementation testing ``.is_dir()`` on that entry would
    refuse exactly the population the pre-flight exists to serve, so the
    distinction has to stay visible at the call site.
    """
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    dot_git_entry = root / '.git'
    if dot_git == 'dir':
        dot_git_entry.mkdir(exist_ok=True)
    elif dot_git == 'file':
        dot_git_entry.write_text('gitdir: /home/leo/src/project/.git/worktrees/wt\n')
    else:
        raise ValueError(f"dot_git must be 'dir' or 'file', got {dot_git!r}")
    return root
