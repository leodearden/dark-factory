"""Suite-wide pytest isolation: git can never escape the basetemp into a repo.

Incident esc-3072-3.  Git repository discovery walks UP the directory tree, so
a git command run with ``cwd=<some tmp dir>`` does not operate on the directory
the caller named — it operates on *whatever repo encloses that directory*.  When
pytest's basetemp lives inside a live task worktree
(``.worktrees/<task>/.pytest-tmp/``), that enclosing repo is production state:
three blobs were written into a real task's object store and ``foo.py`` was
staged at stages 1/2/3, leaving ``UU foo.py`` in its index.

THE ONE NON-OBVIOUS FACT — the ceiling must equal the basetemp ITSELF:

    Git stops the upward walk only when the walk would ascend INTO or above a
    ``GIT_CEILING_DIRECTORIES`` entry.  Everything strictly BELOW an entry is
    still inspected.  So a ceiling at an *ancestor* of the basetemp — the
    tempting ``/tmp``, or ``tempfile.gettempdir()`` — is entirely inert against
    this incident: the walk from ``/tmp/…/.pytest-tmp/test_x0/sub`` still finds
    the repo sitting below ``/tmp``.  Verified against real git before this
    module was written.  Anyone "simplifying" this to a value computable in
    ``pytest_configure`` (where basetemp is not yet derivable without the
    private ``config._tmp_path_factory``) silently disarms the whole defence.

That precision is also what makes a suite-wide ceiling SAFE: a repo created
under the basetemp — every legitimate ``tmp_path`` repo and linked worktree —
sits below the ceiling entry and keeps resolving normally.

Complementary per-call layers from task 3182, which remain in force and are
strictly tighter than this one (``cwd.parent`` rather than the basetemp):

* ``_orch_helpers.assert_isolated_git_repo`` — a pure-filesystem pre-flight
  that runs BEFORE any subprocess, so a rejected call writes nothing anywhere.
  A ceiling cannot give that guarantee: it makes git fail mid-sequence, after
  an earlier ``git hash-object -w`` has already written its blobs.
* ``_orch_helpers.git_env_with_ceiling`` — the same ceiling mechanism applied
  per call, on a private env copy.

Import constraint: STDLIB + PYTEST ONLY.  Every subproject conftest imports this
module, so it must import cleanly inside every member venv — escalation's lacks
aiosqlite and stubs ``shared`` in ``sys.modules``, so nothing under ``shared/src``
may be depended on here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_CEILING_ENV = 'GIT_CEILING_DIRECTORIES'

# This repo's own definition of "this directory is a live worktree root",
# taken verbatim from .gitignore:15-17 rather than invented in parallel here.
_WORKTREE_ROOT_COMPONENTS = frozenset({
    '.worktrees',
    '.worktrees-orphaned',
    '.eval-worktrees',
})


def git_ceiling_value(basetemp: str | os.PathLike[str], existing: str | None = None) -> str:
    """Build the ``GIT_CEILING_DIRECTORIES`` value that contains *basetemp*.

    *basetemp* is made absolute and symlink-resolved: git IGNORES non-absolute
    entries outright, and compares entries against the RESOLVED cwd, so an
    unresolved or relative entry is not a weaker ceiling but no ceiling at all.

    Any *existing* value is PRESERVED and the basetemp entry appended after it.
    Git treats the variable as a colon-separated list where any single entry can
    stop the walk, so appending is strictly additive containment — overwriting
    would silently discard an operator- or CI-set ceiling.

    The append is idempotent: conftests nest (a run whose rootdir is the repo
    root loads both the root conftest and a subproject conftest), so re-entry
    must not accumulate duplicate entries.
    """
    entry = str(Path(basetemp).resolve())
    if not existing:
        return entry
    if entry in existing.split(':'):
        return existing
    return f'{existing}:{entry}'


def basetemp_rejection_reason(path: str | os.PathLike[str]) -> str | None:
    """Explain why *path* is an unsafe pytest basetemp, or ``None`` if it is fine.

    Deliberately NARROW: it refuses only a basetemp sitting inside one of this
    repo's own gitignored worktree roots.  The tempting general rule — "reject
    any basetemp with an enclosing git repo" — would hard-fail every developer
    and CI setup that legitimately points ``--basetemp`` inside a checkout,
    turning a safety net into an outage.

    Matching is on resolved path COMPONENTS, never substrings: a directory
    merely *containing* the name (``my.worktrees-backup``) is a different,
    perfectly safe directory.  Resolution is what catches the shape that
    actually caused the incident — ``--basetemp=.pytest-tmp`` run from inside
    the worktree, where the flag value never mentions ``.worktrees`` at all.
    """
    resolved = Path(path).resolve()
    offenders = _WORKTREE_ROOT_COMPONENTS.intersection(resolved.parts)
    if not offenders:
        return None
    return (
        f'unsafe --basetemp: {resolved} is inside a live task worktree '
        f'({"/".join(sorted(offenders))}).\n'
        'Git repository discovery walks UP the directory tree, so every '
        'git command a test runs under that basetemp resolves against the '
        "enclosing worktree's repo and mutates production state "
        '(incident esc-3072-3: blobs written and a file staged at stages '
        '1/2/3 in a live task). The untracked tree it creates can also be '
        'swept into a commit by `git add -A` or trip a clean-worktree gate.\n'
        'Fix: pass --basetemp somewhere OUTSIDE the worktree, or omit it '
        'entirely and let pytest default to /tmp/pytest-of-<user>/.'
    )


def reject_unsafe_basetemp(config: object) -> None:
    """Fail collection loudly when ``--basetemp`` points inside a worktree.

    Reads only the PUBLIC ``config.option.basetemp`` CLI option, which is
    populated by the time ``pytest_configure`` runs.  It deliberately does not
    consult ``config._tmp_path_factory``: that attribute is private and its
    availability during a *conftest* ``pytest_configure`` depends on plugin
    hook ordering.

    A no-op when no ``--basetemp`` was passed — the verify lane's case, which
    runs bare ``uv run pytest tests/`` and lands in ``/tmp/pytest-of-<user>/``.

    ``pytest.UsageError`` is the right register here: pytest renders it as a
    clean ``ERROR: ...`` and exits without a traceback, which is what an
    operator/agent misconfiguration deserves.
    """
    basetemp = getattr(getattr(config, 'option', None), 'basetemp', None)
    if not basetemp:
        return
    reason = basetemp_rejection_reason(basetemp)
    if reason is not None:
        raise pytest.UsageError(reason)
