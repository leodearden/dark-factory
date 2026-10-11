"""Find gitlinks that a staged change set newly introduces.

An embedded git repository inside a worktree (pytest ``--basetemp`` scratch, a
fixture's own ``git init``, a stray linked worktree) is staged by ``git add`` as
a gitlink, mode 160000.  A WIP-save commit must never introduce one, while a
registered submodule's pointer bump (160000 to 160000) is legitimate.  Run
:data:`STAGED_RAW_DIFF_ARGV` in the worktree and hand its stdout to
:func:`newly_staged_gitlinks`; the parser relies on exactly those flags.
"""

from __future__ import annotations

GITLINK_MODE = '160000'

STAGED_RAW_DIFF_ARGV: tuple[str, ...] = (
    'git', 'diff', '--cached', '--raw', '-z', '--no-renames', '--ignore-submodules=none',
)


def newly_staged_gitlinks(raw_diff: str) -> tuple[str, ...]:
    """Return, in input order, every path whose staged mode becomes a gitlink."""
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
