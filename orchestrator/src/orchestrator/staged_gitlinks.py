"""Keep gitlinks that the index newly introduces out of a WIP-save commit.

An embedded git repository inside a worktree (pytest ``--basetemp`` scratch, a
fixture's own ``git init``, a stray linked worktree) is staged by ``git add`` as
a gitlink, mode 160000.  A WIP-save commit must never introduce one, while a
registered submodule's pointer bump (160000 to 160000) is legitimate.
:func:`unstage_new_gitlinks` is the step a committer runs after staging;
:func:`newly_staged_gitlinks` is its parser.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

GITLINK_MODE = '160000'

#: Runs a git argv in ``cwd`` and returns ``(returncode, stdout, stderr)``.
#: Injected because the caller, ``orchestrator/src/orchestrator/git_ops.py``,
#: owns the spawn seam.
GitRunner = Callable[..., Awaitable[tuple[int, str, str]]]

_STAGED_RAW_DIFF_ARGV: tuple[str, ...] = (
    'git', 'diff', '--cached', '--raw', '-z', '--no-renames', '--ignore-submodules=none',
)


async def unstage_new_gitlinks(worktree: Path, run: GitRunner) -> tuple[str, ...]:
    """Unstage every gitlink the index newly introduces and return their paths.

    The embedded repositories stay on disk.  Raises :class:`RuntimeError` when
    either git call fails, so an unchecked index never reaches a commit.
    """
    rc, raw_diff, err = await run(list(_STAGED_RAW_DIFF_ARGV), cwd=worktree)
    if rc != 0:
        raise RuntimeError(
            f'could not list staged changes in {worktree} to screen for '
            f'embedded-repo gitlinks (rc={rc}): {err.strip()}'
        )
    paths = newly_staged_gitlinks(raw_diff)
    if not paths:
        return ()
    rc, _, err = await run(
        ['git', '--literal-pathspecs', 'reset', '-q', 'HEAD', '--', *paths], cwd=worktree,
    )
    if rc != 0:
        raise RuntimeError(
            f'could not unstage embedded-repo gitlink(s) {list(paths)} '
            f'in {worktree} (rc={rc}): {err.strip()}'
        )
    logger.warning(
        'left %d embedded git repositor%s out of the commit in %s: %s',
        len(paths), 'y' if len(paths) == 1 else 'ies', worktree, ', '.join(paths),
    )
    return paths


def newly_staged_gitlinks(raw_diff: str) -> tuple[str, ...]:
    """Return, in input order, every path whose staged mode becomes a gitlink.

    *raw_diff* is the stdout of ``git diff --cached --raw -z --no-renames``:
    NUL-separated pairs of a ``:old new sha sha status`` record and its path.
    """
    tokens = raw_diff.split('\0')
    if tokens[-1] == '':
        tokens.pop()
    if len(tokens) % 2:
        raise ValueError(f'staged raw diff has a record without a path: {tokens[-1]!r}')
    return tuple(
        path
        for meta, path in zip(tokens[::2], tokens[1::2], strict=True)
        if _introduces_gitlink(meta)
    )


def _introduces_gitlink(meta: str) -> bool:
    fields = meta.split()
    if len(fields) != 5 or not fields[0].startswith(':'):
        raise ValueError(f'malformed staged raw-diff record: {meta!r}')
    old_mode, new_mode = fields[0][1:], fields[1]
    return new_mode == GITLINK_MODE and old_mode != GITLINK_MODE
