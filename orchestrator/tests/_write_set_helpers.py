"""The one ``WriteSet`` builder the write-set and sandbox-event tests share."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from orchestrator.agents.write_set import WriteSet


def make_write_set(root: Path, **overrides: Path | tuple[Path, ...] | None) -> WriteSet:
    """A ``WriteSet`` laid out the way ``compute_write_set`` derives one, built
    without touching the filesystem.

    Every path except the ``/tmp`` and ``/dev`` system grants lies under *root*
    (pass ``tmp_path``). ``ensure_sandbox_dirs`` creates some of these fields,
    whether it is called directly, through ``TaskWorkflow._ensure_sandbox_dirs``
    or through ``_invoke``, and it can therefore only ever create directories
    under *root* without any test overriding a field. A field added to
    ``WriteSet`` later belongs under *root* too.
    """
    base = root / 'base'
    main_git = root / 'main' / '.git'
    home = root / 'home'
    write_set = WriteSet(
        worktree=base / 'wt',
        task_meta=base / '.task-meta' / 'wt',
        task_root=base / '.task-meta' / 'task-1',
        git_objects=main_git / 'objects',
        git_task_refs=main_git / 'refs' / 'heads' / 'task',
        git_task_reflogs=main_git / 'logs' / 'refs' / 'heads' / 'task',
        git_worktree_admin=main_git / 'worktrees' / 'wt',
        uv_cache=home / '.cache' / 'uv',
        tree_sitter_cache=home / '.cache' / 'tree-sitter',
        claude_fleet=home / '.claude' / 'fleet',
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
    return dataclasses.replace(write_set, **overrides)
