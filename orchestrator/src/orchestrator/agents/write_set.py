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
  2b. the dispatch's task root, by default ``<base>/.task-meta/task-<id>/``
      (via ``TaskArtifacts.task_root_for``)
  3-6. main ``.git`` carve-outs (``objects/``, ``refs/heads/task/``,
       ``logs/refs/heads/task/``, ``worktrees/<name>/``) derived by parsing
       ``<worktree>/.git``'s ``gitdir:`` line
  7. ``~/.cache/uv/``
  8. ``~/.claude/fleet/`` (alpha1 FINAL-WRITABLE-LIST — see
     ``plans/os-sandbox-claude-home-writer-audit.md``)
  9-10. ``/tmp/``, ``/dev/``
  12-13. conditional: ``<project_root>/.git/`` plus the narrowest grant per
         declared docs path, present only when
         ``docs_scope.docs_only_paths(declared_files)`` is non-empty
         (``plans/os-sandbox-worktree-containment-prd.md`` §Docs-only main grant)

Everything else is read-only. The common dir's ``rr-cache`` is deliberately
not granted; sandboxed agents run with rerere off instead
(``orchestrator/src/orchestrator/agents/sandbox_git_config.py``).
``compute_write_set`` is a PURE derivation —
no filesystem side effects (no makedirs); existence/creation remains the
backend's job. ``WriteSet.is_foreign_task_meta`` is the query plan-tools uses
to flag a plan step naming another task's ``.task-meta`` (task 4825).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from orchestrator.agents.docs_scope import docs_only_paths
from orchestrator.artifacts import TaskArtifacts

logger = logging.getLogger(__name__)

_UNGRANTABLE_TOP_LEVEL = frozenset({'.git', '.taskmaster', '.worktrees'})


@dataclass(frozen=True)
class WriteSet:
    """The full writable-path set for one sandboxed agent invocation.

    One named field per contract row (INV-1: each row individually
    pinnable) plus ``writable_paths()``, the single aggregate list both
    sandbox backends consume (INV-5).
    """

    worktree: Path
    task_meta: Path
    task_root: Path
    git_objects: Path
    git_task_refs: Path
    git_task_reflogs: Path
    git_worktree_admin: Path
    uv_cache: Path
    claude_fleet: Path
    tmp: Path
    dev: Path
    docs_main_git: Path | None = None
    docs_main_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.docs_main_paths) != (self.docs_main_git is not None):
            raise ValueError(
                'WriteSet: docs_main_paths must be non-empty exactly when '
                f'docs_main_git is set; got docs_main_git={self.docs_main_git!r}, '
                f'docs_main_paths={self.docs_main_paths!r}'
            )

    def writable_paths(self) -> list[Path]:
        """The single aggregate writable-path list (INV-5) — contract
        order, order-preserving dedupe."""
        paths = [
            self.worktree,
            self.task_meta,
            self.task_root,
            self.git_objects,
            self.git_task_refs,
            self.git_task_reflogs,
            self.git_worktree_admin,
            self.uv_cache,
            self.claude_fleet,
            self.tmp,
            self.dev,
        ]
        if self.docs_main_git is not None:
            paths.append(self.docs_main_git)
        paths.extend(self.docs_main_paths)
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

    def is_foreign_task_meta(self, path: Path) -> bool:
        """Rows 2 and 2b are a dispatch's only ``.task-meta`` grants, so any
        other path inside the ``.task-meta`` base is another lane's or task's
        and unwritable here."""
        resolved = path.resolve()
        base = self.task_meta.parent
        return (
            resolved.is_relative_to(base)
            and resolved != base
            and not resolved.is_relative_to(self.task_meta)
            and not resolved.is_relative_to(self.task_root)
        )


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


def compute_write_set(
    worktree: Path, *, task_id: str, task_root: Path | None = None, home: Path | None = None,
    declared_files: object = None,
) -> WriteSet:
    """Derive the full writable-path set for a sandboxed agent invocation
    of task *task_id* rooted at ``worktree``.

    ``task_root`` is the task root the dispatch actually uses (the workflow
    passes its own, so a single-root eval dispatch is granted no
    ``task-<id>`` dir its parallel trials would share); ``None`` derives
    ``TaskArtifacts.task_root_for(worktree.parent, task_id)``.

    Pure: reads ``<worktree>/.git`` and resolves symlinks, but performs no
    filesystem writes (no makedirs) — existence/creation remains the
    backend's job (landlock_exec's ``_add_path`` already skips non-existent
    dirs). Fails loudly (``ValueError``) on a missing/malformed/directory
    ``.git``, and on a *task_id* ``TaskArtifacts.task_root_for`` refuses —
    fail-closed-friendly, honoring the loud-over-silent norm.

    ``home`` defaults to ``Path.home()``; tests should inject a hermetic
    tmp dir.

    ``declared_files`` is the task's declared file scope. When every entry is
    a docs path, the main checkout's ``.git`` and those paths are granted too
    (contract rows 12-13); any other value grants nothing extra.
    """
    return _derive_write_set(
        worktree,
        task_root if task_root is not None
        else TaskArtifacts.task_root_for(worktree.parent, task_id),
        home,
        declared_files,
    )


def write_set_of(artifacts: TaskArtifacts, *, home: Path | None = None) -> WriteSet:
    """The write set of the dispatch owning *artifacts*: its worktree, with
    its ``task_root`` as row 2b.

    Raises ``ValueError``, like ``compute_write_set``, on a worktree with no
    linked-worktree ``.git``.
    """
    return _derive_write_set(artifacts.worktree, artifacts.task_root, home)


def _derive_write_set(
    worktree: Path, task_root: Path, home: Path | None, declared_files: object = None,
) -> WriteSet:
    home = home or Path.home()

    admindir = _parse_gitdir(worktree)
    main_git = admindir.parent.parent
    docs_main_git, docs_main_paths = _docs_main_grant(main_git, declared_files)

    return WriteSet(
        worktree=worktree.resolve(),
        # .resolve() walks EVERY intermediate path component (not just the
        # trailing one), so a symlinked lane base (e.g. reify's symlinked
        # `.worktrees` mount, or a symlinked `<base>/.task-meta`) collapses
        # to the real directory landlock enforces on — see
        # TestTaskMetaSymlinkResolution in test_write_set.py.
        task_meta=TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve(),
        task_root=task_root.resolve(),
        git_objects=(main_git / 'objects').resolve(),
        git_task_refs=(main_git / 'refs' / 'heads' / 'task').resolve(),
        git_task_reflogs=(main_git / 'logs' / 'refs' / 'heads' / 'task').resolve(),
        git_worktree_admin=admindir,
        uv_cache=(home / '.cache' / 'uv').resolve(),
        claude_fleet=(home / '.claude' / 'fleet').resolve(),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
        docs_main_git=docs_main_git,
        docs_main_paths=docs_main_paths,
    )


def _docs_main_grant(
    main_git: Path, declared_files: object,
) -> tuple[Path | None, tuple[Path, ...]]:
    docs = docs_only_paths(declared_files)
    if not docs:
        return None, ()
    if main_git.name != '.git' or not main_git.is_dir():
        logger.warning(
            'compute_write_set: docs-only declared files %s, but the common git '
            'dir %s is not a <project_root>/.git; no main-checkout grant',
            [str(rel) for rel in docs], main_git,
        )
        return None, ()
    paths = _docs_main_grants(main_git.parent.resolve(), docs)
    return (main_git, paths) if paths else (None, ())


def _docs_main_grants(root: Path, docs: tuple[PurePosixPath, ...]) -> tuple[Path, ...]:
    grants: dict[Path, None] = {}
    for rel in docs:
        grant = _docs_grant(root, rel)
        if grant is not None:
            grants[grant] = None
    return tuple(grants)


def _docs_grant(root: Path, rel: PurePosixPath) -> Path | None:
    """The narrowest grant under *root* that lets *rel* be written: the path
    itself when it exists, else its nearest existing ancestor directory."""
    target = (root / rel).resolve()
    grant = target if target.exists() else next(a for a in target.parents if a.exists())
    reason = _ungrantable_reason(root, grant)
    if reason is None and not (grant.is_dir() or (grant == target and grant.is_file())):
        reason = f'{grant} is not a regular file or directory'
    if reason is not None:
        logger.warning(
            'compute_write_set: not granting declared docs path %s: %s', rel, reason,
        )
        return None
    return grant


def _ungrantable_reason(root: Path, grant: Path) -> str | None:
    if grant == root or not grant.is_relative_to(root):
        return f'it resolves to {grant}, not strictly inside {root}'
    top = grant.relative_to(root).parts[0]
    if top in _UNGRANTABLE_TOP_LEVEL:
        return f'it resolves into {root / top}, which is never granted'
    return None


def ensure_claude_fleet_dir(write_set: WriteSet) -> bool:
    """Best-effort, side-effecting pre-creation of ``write_set.claude_fleet``.

    ``compute_write_set`` is intentionally PURE (no makedirs — pinned by
    ``test_purity_no_makedirs_side_effect``), yet the ``~/.claude/fleet``
    carve-out is load-bearing and must EXIST on disk before the sandbox is
    built: neither backend can grant a nonexistent writable path from INSIDE
    the sandbox — bwrap ``build_bwrap_command`` binds a writable extra only if
    it is an existing file or dir (sandbox.py), and landlock ``_add_path``
    skips any other path (landlock_exec.py) — and the agent
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
