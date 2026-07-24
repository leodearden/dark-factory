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

import hashlib
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

    def digest(self) -> str:
        """Stable sha256 hex digest of this writable-set — the SINGLE owner
        of the digest definition (INV-5 no-lockstep-duplication).

        Hashes the newline-joined, contract-ordered ``writable_paths()``
        strings. Because ``writable_paths()`` is deterministic (contract order
        + order-preserving dedupe) and resolves to absolute paths, the digest
        is reproducible across processes (pure ``hashlib``) yet per-worktree
        DISTINCT — exactly the "diff exactly what a given invocation could
        touch" property the OS-sandbox ``sandbox_applied`` event needs for
        operator diffing (β2, PRD §Goal 3 / INV-2). The emit site reads this;
        it never recomputes the hash.
        """
        joined = '\n'.join(str(p) for p in self.writable_paths())
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()


_GITDIR_PREFIX = 'gitdir:'


def _parse_gitdir(worktree: Path) -> Path:
    """Read ``<worktree>/.git`` (a linked-worktree gitdir file) and return
    the resolved admindir it points at.

    Raises ``ValueError`` (naming the ``.git`` path) if it is missing, is a
    directory (a non-linked/main repo), or lacks a well-formed ``gitdir:``
    line with a non-empty value. A relative gitdir value is resolved
    against ``worktree`` — not the process cwd — matching git's own
    convention for relocatable worktrees.
    """
    git_path = worktree / '.git'
    if not git_path.is_file():
        raise ValueError(
            f'compute_write_set: expected a linked-worktree gitdir file at '
            f'{git_path}, but it is missing or not a regular file'
        )
    content = git_path.read_text()
    if not content.startswith(_GITDIR_PREFIX):
        raise ValueError(
            f"compute_write_set: {git_path} does not start with 'gitdir:': {content!r}"
        )
    value = content[len(_GITDIR_PREFIX):].strip()
    if not value:
        raise ValueError(f'compute_write_set: {git_path} has an empty gitdir value')
    admindir = Path(value)
    if not admindir.is_absolute():
        admindir = worktree / admindir
    return admindir.resolve()


def compute_write_set(worktree: Path, *, home: Path | None = None) -> WriteSet:
    """Derive the full writable-path set for a sandboxed agent invocation
    rooted at ``worktree``.

    Pure: reads ``<worktree>/.git`` and resolves symlinks, but performs no
    filesystem writes (no makedirs) — existence/creation remains the
    backend's job (landlock_exec's ``_add_path`` already skips non-existent
    dirs). Fails loudly (``ValueError``) on a missing/malformed/directory
    ``.git`` — fail-closed-friendly, honoring the loud-over-silent norm.

    ``home`` defaults to ``Path.home()``; tests should inject a hermetic
    tmp dir.
    """
    home = home or Path.home()

    admindir = _parse_gitdir(worktree)
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


def ensure_claude_fleet_dir(write_set: WriteSet) -> bool:
    """Best-effort, side-effecting pre-creation of ``write_set.claude_fleet``.

    ``compute_write_set`` is intentionally PURE (no makedirs — pinned by
    ``test_purity_no_makedirs_side_effect``), yet the ``~/.claude/fleet``
    carve-out is load-bearing and must EXIST on disk before the sandbox is
    built: neither backend can grant a nonexistent writable path from INSIDE
    the sandbox — bwrap ``build_bwrap_command`` binds a writable extra only
    ``if os.path.isdir(extra)`` (sandbox.py) and landlock ``_add_path`` skips a
    path that ``not os.path.isdir(path)`` (landlock_exec.py) — and the agent
    cannot create it itself because the parent ``~/.claude`` is read-only under
    both backends. So the orchestrator must materialize it OUTSIDE the sandbox,
    once, before dispatch (task 2996). This helper is co-located with the
    single-source ``WriteSet`` (PRD D11) but kept SEPARATE from the pure
    derivation.

    Returns ``True`` when the directory exists after the call (created or
    already present — ``exist_ok=True``), ``False`` when ``mkdir`` raises
    ``OSError``. Does no logging: the caller
    (``TaskWorkflow._ensure_sandbox_dirs``) owns the operator-facing warning
    and the ``task_id`` triage context.
    """
    try:
        write_set.claude_fleet.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
