"""shared.delivered_check_scope — what a delivered_check's ``paths`` name at
a commit (task 6480).

A ``kind: grep``/``path`` check is only as live as its ``paths``. An entry
naming a file the mainline has since deleted or moved away, or a ``.py``
reduced to a ``sys.modules[__name__]`` alias shim, can never go green, so the
check's dependents wedge behind what reads as an undelivered capability (task
5036 left ``orchestrator/src/orchestrator/merge_queue.py`` a shim under live
checks; 5596). This module classifies each literal entry at a resolved commit
and owns the staleness policy the authoring lint
(``shared.delivered_check_polarity``) and the corpus audit
(``scripts/audit_delivered_checks.py``) share. The policy is stated in
``docs/task-authoring.md`` §3.3. stdlib only, and never raises.
"""

from __future__ import annotations

import functools
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

__all__ = [
    'GIT_PROBE_FAILURES',
    'GIT_TIMEOUT_SECS',
    'STALE_PATH_CODES',
    'STALE_PATH_REASONS',
    'SYS_MODULES_SHIM_PATTERN',
    'PathState',
    'ScopePath',
    'classify_scope_paths',
    'is_literal_scope_path',
    'resolve_commit',
    'stale_scope_paths',
]

#: Wall-clock ceiling for ONE authoring-time git probe (``grep``,
#: ``ls-tree``, ``ls-files``). Generous relative to a real probe
#: (milliseconds on this repo) because exceeding it is not a verdict — it
#: degrades to ``ERRORED``, which is REPORTED rather than blocking, so a
#: slow disk delays a commit_planning call instead of rejecting a healthy
#: batch.
GIT_TIMEOUT_SECS: float = 30.0

#: Everything ``subprocess.run`` can raise for one git probe, all of which mean
#: "unevaluable", never a verdict: ``OSError`` (no ``git``, exec failure),
#: ``SubprocessError`` (chiefly a timeout) and ``ValueError`` — an argv
#: element carrying a NUL byte, or a lone surrogate that cannot be encoded
#: (``UnicodeEncodeError`` is a ``ValueError``), both refused before git runs.
GIT_PROBE_FAILURES = (OSError, subprocess.SubprocessError, ValueError)

#: A module-level rebind of the module object: the file still exists, but its
#: code lives wherever the alias points. POSIX ERE, for ``git grep -E``.
SYS_MODULES_SHIM_PATTERN = r'^sys\.modules\[__name__\][[:space:]]*='

_PATHSPEC_GLOB_CHARS = frozenset('*?[')


class PathState(Enum):
    LIVE = 'live'
    SYS_MODULES_SHIM = 'sys_modules_shim'
    REMOVED = 'removed'
    NEVER_EXISTED = 'never_existed'


@dataclass(frozen=True)
class ScopePath:
    """One scope entry at a commit; *removed_in* is ``'<short sha> <subject>'``
    of the mainline commit that removed a REMOVED entry, when git names one."""

    path: str
    state: PathState
    removed_in: str | None = None


#: The finding code each stale state is reported under, by the lint and the
#: audit alike.
STALE_PATH_CODES = MappingProxyType({
    PathState.SYS_MODULES_SHIM: 'shim_path',
    PathState.REMOVED: 'removed_path',
})

_NEVER_GREEN = (
    'so the expect=present check can never go green, and its dependents wedge '
    'behind what reads as an undelivered capability.'
)

#: Why each stale code blocks, and its repair: the one explanation the lint's
#: messages and the audit's ``reason`` both carry.
STALE_PATH_REASONS = MappingProxyType({
    'shim_path': (
        'the path is a sys.modules alias shim: the file only rebinds '
        'sys.modules[__name__] to the module it aliases, so the code the check '
        f'asserts lives elsewhere, {_NEVER_GREEN} Repath `paths` to the module '
        'the shim aliases.'
    ),
    'removed_path': (
        'the path was deleted or moved away on the mainline, '
        f'{_NEVER_GREEN} Repath `paths` to where the code lives now.'
    ),
})

_STALE_STATES_BY_KIND: dict[str, frozenset[PathState]] = {
    'grep': frozenset({PathState.SYS_MODULES_SHIM, PathState.REMOVED}),
    'path': frozenset({PathState.REMOVED}),
}


def stale_scope_paths(
    kind: object, expect: object, scope: Iterable[ScopePath]
) -> tuple[ScopePath, ...]:
    """The entries of *scope* that leave a ``kind``/``expect`` check unable to
    go green, in input order.

    Only ``expect='present'`` is subject: a dead scope makes an absent check
    PASS, which the polarity 2x2 and the audit's ``vacuous_live_gate`` already
    report (``docs/task-authoring.md`` §3.3). A path check is not stale on a
    shim, because the shim file exists.
    """
    stale_states = _STALE_STATES_BY_KIND.get(kind) if isinstance(kind, str) else None
    if expect != 'present' or stale_states is None:
        return ()
    return tuple(entry for entry in scope if entry.state in stale_states)


def _git(
    repo_root: str | Path, args: list[str], timeout_secs: float
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ['git', '-C', str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
    except GIT_PROBE_FAILURES:
        return None


def resolve_commit(
    repo_root: str | Path, ref: str, timeout_secs: float = GIT_TIMEOUT_SECS
) -> str | None:
    """The commit *ref* names in *repo_root*, or ``None`` if it names none."""
    completed = _git(
        repo_root, ['rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}'], timeout_secs
    )
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def is_literal_scope_path(path: object) -> bool:
    """A plain repo path: not a glob, not pathspec magic, not empty. Only
    these are classified; the rest name no single path."""
    return (
        isinstance(path, str)
        and bool(path.strip('/'))
        and not path.startswith(':')
        and _PATHSPEC_GLOB_CHARS.isdisjoint(path)
    )


def _with_ancestors(names: Iterable[str]) -> frozenset[str]:
    """*names* plus every directory above each, so a literal entry naming a
    directory is answered by set membership."""
    found: set[str] = set()
    for name in names:
        parts = name.split('/')
        found.update('/'.join(parts[:depth]) for depth in range(1, len(parts) + 1))
    return frozenset(found)


def _nul_records(stdout: str) -> list[str]:
    return [record.strip('\n') for record in stdout.split('\0') if record.strip('\n')]


def _tree_files(
    repo_root: str | Path, sha: str, literals: list[str], timeout_secs: float
) -> frozenset[str] | None:
    completed = _git(
        repo_root,
        ['ls-tree', '-r', '-z', '--full-tree', '--name-only', sha, '--', *literals],
        timeout_secs,
    )
    if completed is None or completed.returncode != 0:
        return None
    return frozenset(_nul_records(completed.stdout))


def _shim_files(
    repo_root: str | Path, sha: str, py_files: list[str], timeout_secs: float
) -> frozenset[str] | None:
    if not py_files:
        return frozenset()
    completed = _git(
        repo_root,
        ['grep', '-l', '-z', '-E', '-e', SYS_MODULES_SHIM_PATTERN, sha, '--', *py_files],
        timeout_secs,
    )
    if completed is None or completed.returncode >= 2:
        return None
    prefix = f'{sha}:'
    return frozenset(
        record.removeprefix(prefix) for record in _nul_records(completed.stdout)
    )


@functools.lru_cache(maxsize=16)
def _mainline_deletions(
    repo_root: str, sha: str, timeout_secs: float
) -> frozenset[str] | None:
    """Every path a first-parent commit reachable from *sha* deleted, with its
    ancestor directories.

    Memoised on the RESOLVED sha, which content-addresses its history, never
    on a ref name. A failure is memoised too: retrying a timed-out walk would
    cost every later caller the full timeout, and main moves on the next merge.
    """
    completed = _git(
        repo_root,
        [
            'log', '--first-parent', '--no-renames', '--diff-filter=D',
            '--format=', '--name-only', '-z', sha, '--',
        ],
        timeout_secs,
    )
    if completed is None or completed.returncode != 0:
        return None
    return _with_ancestors(_nul_records(completed.stdout))


def _removed_in(
    repo_root: str | Path, sha: str, path: str, timeout_secs: float
) -> str | None:
    completed = _git(
        repo_root,
        [
            'log', '-1', '--first-parent', '--no-renames', '--diff-filter=D',
            '--format=%h %s', sha, '--', path,
        ],
        timeout_secs,
    )
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def classify_scope_paths(
    paths: Iterable[str],
    *,
    repo_root: str | Path,
    ref: str,
    timeout_secs: float = GIT_TIMEOUT_SECS,
) -> dict[str, ScopePath] | None:
    """Classify each LITERAL entry of *paths* at the commit *ref* resolves to.

    Keyed by the exact input string; entries failing
    :func:`is_literal_scope_path` are left out. ``None`` means git could not answer
    (non-repo root, unresolvable *ref*, a value no argv can carry, a timeout),
    never "nothing is stale". A missing entry is REMOVED only when a
    first-parent commit deleted it or everything under it; otherwise it is
    the forward-looking NEVER_EXISTED. Costs a constant number of git calls,
    plus one per REMOVED entry to name the commit.
    """
    literals = list(dict.fromkeys(p for p in paths if is_literal_scope_path(p)))
    if not literals:
        return {}
    sha = resolve_commit(repo_root, ref, timeout_secs)
    if sha is None:
        return None
    tree_files = _tree_files(repo_root, sha, literals, timeout_secs)
    if tree_files is None:
        return None
    tree = _with_ancestors(tree_files)
    keys = {p: p.rstrip('/') for p in literals}
    shims = _shim_files(
        repo_root,
        sha,
        sorted({key for key in keys.values() if key in tree_files and key.endswith('.py')}),
        timeout_secs,
    )
    if shims is None:
        return None
    deletions: frozenset[str] | None = frozenset()
    if any(key not in tree for key in keys.values()):
        deletions = _mainline_deletions(str(repo_root), sha, timeout_secs)
    if deletions is None:
        return None

    census: dict[str, ScopePath] = {}
    for path, key in keys.items():
        if key in shims:
            census[path] = ScopePath(path, PathState.SYS_MODULES_SHIM)
        elif key in tree:
            census[path] = ScopePath(path, PathState.LIVE)
        elif key in deletions:
            census[path] = ScopePath(
                path, PathState.REMOVED, _removed_in(repo_root, sha, key, timeout_secs)
            )
        else:
            census[path] = ScopePath(path, PathState.NEVER_EXISTED)
    return census
