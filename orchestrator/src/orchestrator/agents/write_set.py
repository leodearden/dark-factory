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
  8. ``~/.cache/tree-sitter/`` (task 3382) — the tree-sitter grammar-cache
     SUBDIR specifically, never the ``~/.cache`` parent, which would hand
     every sandboxed role write access to every other tool cache there
  9. ``~/.claude/fleet/`` (alpha1 FINAL-WRITABLE-LIST — see
     ``plans/os-sandbox-claude-home-writer-audit.md``)
  10-11. ``/tmp/``, ``/dev/``
  12-13. conditional: ``<project_root>/.git/`` plus the narrowest grant per
         declared docs path (a directory grant only inside a top-level docs
         dir), present only when
         ``docs_scope.docs_only_paths(declared_files)`` is non-empty
         (``plans/os-sandbox-worktree-containment-prd.md`` §Docs-only main grant)

Everything else is read-only. The common dir's ``rr-cache`` is deliberately
not granted; sandboxed agents run with rerere off instead
(``orchestrator/src/orchestrator/agents/sandbox_git_config.py``).
``compute_write_set`` is a PURE derivation —
no filesystem side effects (no makedirs). The backends skip a nonexistent
writable path, so the carve-outs that must pre-exist are created outside the
sandbox by ``ensure_sandbox_dirs``. ``WriteSet.is_foreign_task_meta`` is the
query plan-tools uses to flag a plan step naming another task's
``.task-meta`` (task 4825).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from orchestrator.agents.docs_scope import docs_only_paths, is_in_docs_tree
from orchestrator.artifacts import TaskArtifacts

logger = logging.getLogger(__name__)


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
    tree_sitter_cache: Path
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
        if self.docs_main_git is None:
            return
        refused_roots = _docs_refused_roots(self.docs_main_git, self.worktree, self.task_meta)
        for path in self.docs_main_paths:
            reason = _docs_containment_refusal(self.docs_main_git.parent, path, refused_roots)
            if reason is not None:
                raise ValueError(f'WriteSet: docs_main_paths entry {path} is ungrantable: {reason}')

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
            self.tree_sitter_cache,
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
    resolved_worktree = worktree.resolve()
    # .resolve() walks EVERY intermediate path component (not just the
    # trailing one), so a symlinked lane base (e.g. reify's symlinked
    # `.worktrees` mount, or a symlinked `<base>/.task-meta`) collapses
    # to the real directory landlock enforces on — see
    # TestTaskMetaSymlinkResolution in test_write_set.py.
    task_meta = TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve()
    docs_main_git, docs_main_paths = _docs_main_rows(
        main_git, declared_files, _docs_refused_roots(main_git, resolved_worktree, task_meta),
    )

    return WriteSet(
        worktree=resolved_worktree,
        task_meta=task_meta,
        task_root=task_root.resolve(),
        git_objects=(main_git / 'objects').resolve(),
        git_task_refs=(main_git / 'refs' / 'heads' / 'task').resolve(),
        git_task_reflogs=(main_git / 'logs' / 'refs' / 'heads' / 'task').resolve(),
        git_worktree_admin=admindir,
        uv_cache=(home / '.cache' / 'uv').resolve(),
        # The tree-sitter grammar cache SUBDIR only — never the ~/.cache
        # parent, which would grant every other tool cache living there.
        tree_sitter_cache=(home / '.cache' / 'tree-sitter').resolve(),
        claude_fleet=(home / '.claude' / 'fleet').resolve(),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
        docs_main_git=docs_main_git,
        docs_main_paths=docs_main_paths,
    )


def _docs_refused_roots(main_git: Path, worktree: Path, task_meta: Path) -> tuple[Path, ...]:
    """What a docs grant may neither lie in nor contain: the git dir, the task
    store, and the lane and ``.task-meta`` bases, which keeps rows 2 and 2b
    the only ``.task-meta`` grants (``WriteSet.is_foreign_task_meta``).

    A base that holds the project root beside its lanes is dropped: every
    lane in it already lies outside the root."""
    root = main_git.parent
    candidates = (main_git, (root / '.taskmaster').resolve(), worktree.parent, task_meta.parent)
    return tuple(r for r in candidates if r == root or not root.is_relative_to(r))


def _docs_containment_refusal(
    root: Path, grant: Path, refused_roots: tuple[Path, ...],
) -> str | None:
    if grant == root or not grant.is_relative_to(root):
        return f'it resolves to {grant}, not strictly inside {root}'
    for refused in refused_roots:
        if grant.is_relative_to(refused) or refused.is_relative_to(grant):
            return f'{grant} overlaps {refused}, which is never granted'
    return None


def _docs_main_rows(
    main_git: Path, declared_files: object, refused_roots: tuple[Path, ...],
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
    paths = _resolve_docs_paths(main_git.parent.resolve(), docs, refused_roots)
    return (main_git, paths) if paths else (None, ())


def _resolve_docs_paths(
    root: Path, docs: tuple[PurePosixPath, ...], refused_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    grants: dict[Path, None] = {}
    for rel in docs:
        grant = _narrowest_grant(root, rel, refused_roots)
        if grant is not None:
            grants[grant] = None
    return tuple(grants)


def _narrowest_grant(
    root: Path, rel: PurePosixPath, refused_roots: tuple[Path, ...],
) -> Path | None:
    """The narrowest grant under *root* that lets *rel* be written: the path
    itself when it exists, else its nearest existing ancestor directory. A
    directory grant must lie in a top-level docs dir, so it never exposes a
    non-docs file."""
    target = (root / rel).resolve()
    grant = target if target.exists() else next(a for a in target.parents if a.exists())
    reason = (
        _docs_containment_refusal(root, grant, refused_roots)
        or _docs_shape_refusal(root, target, grant)
    )
    if reason is not None:
        logger.warning(
            'compute_write_set: not granting declared docs path %s: %s', rel, reason,
        )
        return None
    return grant


def _docs_shape_refusal(root: Path, target: Path, grant: Path) -> str | None:
    if grant.is_dir():
        if is_in_docs_tree(PurePosixPath(*grant.relative_to(root).parts)):
            return None
        return (
            f'{grant} is a directory outside a top-level docs dir, so granting it '
            'would expose non-docs files'
        )
    if grant != target:
        return f'its nearest existing ancestor {grant} is not a directory'
    if not grant.is_file():
        return f'{grant} is not a regular file or directory'
    return None


def ensure_sandbox_dirs(write_set: WriteSet) -> list[Path]:
    """Best-effort, side-effecting pre-creation of every carve-out that must
    exist before the sandbox is built: ``claude_fleet`` (task 2996) and
    ``tree_sitter_cache`` (task 3382).

    ``compute_write_set`` is intentionally PURE (no makedirs — pinned by
    ``test_purity_no_makedirs_side_effect``), yet neither backend can grant a
    nonexistent writable path: bwrap ``build_bwrap_command`` binds a writable
    extra only if it is an existing file or dir (sandbox.py), and landlock
    ``_add_path`` skips any other path (landlock_exec.py). Both merely warn and
    drop the grant. The agent cannot create these dirs from inside either,
    because their parents ``~/.claude`` and ``~/.cache`` are read-only under
    both backends. So the orchestrator materializes them OUTSIDE the sandbox,
    once, before dispatch. Kept SEPARATE from the pure derivation but
    co-located with the single-source ``WriteSet`` (PRD D11).

    ``uv_cache`` is deliberately absent: ``uv`` itself runs outside the sandbox
    (verify, cold pre-provisioning) and materializes ``~/.cache/uv`` there.
    Tree-sitter grammar builds run INSIDE the sandbox, so nothing outside it
    would ever create ``~/.cache/tree-sitter``.

    Returns the paths whose ``mkdir`` raised ``OSError``; an empty list means
    every carve-out exists after the call (created or already present). Each
    path is attempted independently, so one failure never suppresses the
    rest. Does no logging: the caller (``TaskWorkflow._ensure_sandbox_dirs``)
    owns the operator-facing warning and the ``task_id`` triage context.
    """
    failed: list[Path] = []
    for path in (write_set.claude_fleet, write_set.tree_sitter_cache):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            failed.append(path)
    return failed
