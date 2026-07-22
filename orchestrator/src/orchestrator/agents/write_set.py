"""Single-source writable-path derivation for sandboxed agent invocations.

``compute_write_set()`` is the SINGLE source of truth (PRD D11 / INV-5
no-lockstep-duplication) for the writable-path list that BOTH sandbox
backends consume via their existing ``writable_modules``/``writable_extras``
params (bwrap ``build_bwrap_command``, landlock ``build_landlock_command``,
dispatcher ``wrap_command``). This module is derivation-only — no backend or
call-site is wired up here (that's alpha3/alpha4).

Contract (PRD Sec Contract) — derived per-invocation from the worktree path:
  1. whole worktree root (D1)
  2. ``<base>/.task-meta/<name>/`` (via ``TaskArtifacts.meta_root_for``)
  3-6. main ``.git`` carve-outs (``objects/``, ``refs/heads/task/``,
       ``logs/refs/heads/task/``, ``worktrees/<name>/``) derived by parsing
       ``<worktree>/.git``'s ``gitdir:`` line
  7. ``~/.cache/uv/``
  8. ``~/.claude/fleet/`` (alpha1 FINAL-WRITABLE-LIST — see
     ``plans/os-sandbox-claude-home-writer-audit.md``)
  9-10. ``/tmp/``, ``/dev/``

Everything else is read-only. ``compute_write_set`` is a PURE derivation —
no filesystem side effects (no makedirs); existence/creation remains the
backend's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts


@dataclass(frozen=True)
class WriteSet:
    """The full writable-path set for one sandboxed agent invocation.

    One named field per contract row (INV-1: each row individually
    pinnable) plus ``writable_paths()``, the single aggregate list both
    sandbox backends consume (INV-5).
    """

    worktree: Path
    task_meta: Path
    git_objects: Path
    git_task_refs: Path
    git_task_reflogs: Path
    git_worktree_admin: Path
    uv_cache: Path
    claude_fleet: Path
    tmp: Path
    dev: Path

    def writable_paths(self) -> list[Path]:
        """The single aggregate writable-path list (INV-5) — contract
        order, order-preserving dedupe."""
        paths = [
            self.worktree,
            self.task_meta,
            self.git_objects,
            self.git_task_refs,
            self.git_task_reflogs,
            self.git_worktree_admin,
            self.uv_cache,
            self.claude_fleet,
            self.tmp,
            self.dev,
        ]
        seen: set[Path] = set()
        deduped: list[Path] = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
        return deduped


def compute_write_set(worktree: Path, *, home: Path | None = None) -> WriteSet:
    """Derive the full writable-path set for a sandboxed agent invocation
    rooted at ``worktree``.

    Pure: reads ``<worktree>/.git`` and resolves symlinks, but performs no
    filesystem writes (no makedirs) — existence/creation remains the
    backend's job (landlock_exec's ``_add_path`` already skips non-existent
    dirs).

    ``home`` defaults to ``Path.home()``; tests should inject a hermetic
    tmp dir.
    """
    home = home or Path.home()

    content = (worktree / '.git').read_text()
    _, _, value = content.partition('gitdir:')
    admindir = Path(value.strip()).resolve()
    main_git = admindir.parent.parent

    return WriteSet(
        worktree=worktree.resolve(),
        # .resolve() walks EVERY intermediate path component (not just the
        # trailing one), so a symlinked lane base (e.g. reify's symlinked
        # `.worktrees` mount, or a symlinked `<base>/.task-meta`) collapses
        # to the real directory landlock enforces on — see
        # TestTaskMetaSymlinkResolution in test_write_set.py.
        task_meta=TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve(),
        git_objects=(main_git / 'objects').resolve(),
        git_task_refs=(main_git / 'refs' / 'heads' / 'task').resolve(),
        git_task_reflogs=(main_git / 'logs' / 'refs' / 'heads' / 'task').resolve(),
        git_worktree_admin=admindir,
        uv_cache=(home / '.cache' / 'uv').resolve(),
        claude_fleet=(home / '.claude' / 'fleet').resolve(),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
