"""Git worktree and merge operations.

IMPORTANT — .task/ contamination prevention
============================================
The .task/ directory is an ephemeral scratch space for orchestrator agents.
It must NEVER reach the main branch.  If it does, every future worktree
inherits it, agents treat it as state, and cross-task contamination follows.

This module contains multiple redundant safeguards ("belts and braces"):

1. scrub_task_dir_from_tree() — removes .task/ from the git index in any
   worktree, amending the current commit.  Called after merges and during
   worktree creation.
2. _assert_no_task_dir() — hard assertion that a given commit SHA contains
   no .task/ entries.  Called before advance_main().
3. create_worktree() — scrubs inherited .task/ when main is contaminated.
4. commit() — post-staging safety net: unstages .task/ even if the pathspec
   exclusion (:!.task) was somehow bypassed.
5. merge_to_main() — scrubs .task/ after the merge commit is created.

If you are an AI agent reading this: DO NOT remove or weaken these guards.
They exist because .task/ contamination has happened repeatedly and caused
cascading bugs across all concurrent tasks.  The pre-commit hook, .gitignore,
and .task/.gitignore are NOT sufficient — agents bypass them routinely.
"""

import asyncio
import logging
import re
import shutil
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

from orchestrator.config import GitConfig
from orchestrator.worktree_identity import identities_match, read_worktree_title

logger = logging.getLogger(__name__)

# Return type for advance_main — lets callers distinguish transient
# (CAS) failures from permanent ones (not-a-descendant, contamination).
AdvanceResult = Literal[
    'advanced', 'cas_failed', 'not_descendant', 'contaminated',
    'stash_failed', 'wip_overlap', 'pop_conflict',
    'unmerged_state', 'pop_conflict_no_advance',
    'rebased_pending_reverify',
]


PushResult = Literal['pushed', 'noop', 'rejected', 'error']


# Return type for recover_red_main — distinguishes a successful CAS move
# from an atomic abort (another writer beat us to the ref).
RecoverResult = Literal['rewound', 'cas_failed', 'error']


class TrainMembership(TypedDict, total=False):
    """Train metadata passed from task.metadata.train.

    All keys are optional at the type level; _train_predecessor validates
    presence of required keys at runtime with diagnostic error messages.
    """
    id: str
    order: int
    members: list[str] | None


@dataclass(frozen=True)
class TrainPredecessor:
    """Resolved predecessor for a train member with order > 0."""
    task_id: str
    branch: str


@dataclass(frozen=True)
class TrainStackResult:
    """Result of stack_train_branches: which members survived and which were ejected.

    survivors: member ids that were successfully rebased into the linear stack
               (or are the anchor, which is always the base).
    ejected:   member ids that conflicted during stacking and were dropped;
               their branches are left clean (rebase aborted) so they can
               merge solo.
    """
    survivors: list[str]
    ejected: list[str]


# Default commit-citation pattern for ``find_task_citation_commit``.
#
# Matches dark-factory / reify conventions on main:
#   - Conventional-commit subjects that cite the task id in parens or
#     after a colon: ``impl(50): xyz`` / ``fix(50): xyz`` / ``test(50: ...)``.
#   - Subjects that mention the task branch directly: ``... task/50 ...``
#     anywhere in the subject line.
#   - The canonical no-ff merge subject ``Merge task/50 into <main>`` produced
#     by ``merge_to_main``.
#
# The ``{tid}`` placeholder is interpolated via ``str.format`` with the
# escaped task id; a ``\b`` (word boundary) at each side blocks substring
# overlap so ``task/3399`` doesn't match a row that cites ``task/339``.
DEFAULT_COMMIT_CITATION_PATTERN: str = (
    r'^(merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)'
    r'(\(\b{tid}\b[):]|.*\btask/{tid}\b)'
    r'|^Merge task/{tid} into '
)

# Fixed name for the persistent warm merge-verify worktree (task 1692).
# Lives at <worktree_base>/_merge-verify.  Excluded from prune and
# find_inflight enumeration (see _iter_merge_worktrees).
PERSISTENT_MERGE_WORKTREE_NAME: str = '_merge-verify'


class ScrubOutcome(Enum):
    """Outcome discriminant for :class:`ScrubResult`.

    Kept as a separate Enum so callers get IDE autocomplete and type-checking
    while the :class:`ScrubResult` dataclass carries the optional error payload.
    """
    CLEAN = auto()
    SCRUBBED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class ScrubResult:
    """Result of a ``scrub_task_dir_from_tree`` call.

    Distinguishes three outcomes so callers can react precisely:

    - ``ScrubOutcome.CLEAN``   — ``.task/`` was not present in the tree.
    - ``ScrubOutcome.SCRUBBED``— ``.task/`` was found and successfully removed.
    - ``ScrubOutcome.FAILED``  — ``.task/`` was found but removal failed.
                                  The ``error`` field contains the raw git stderr
                                  for operator diagnostics.
    """
    outcome: ScrubOutcome
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None and self.outcome is not ScrubOutcome.FAILED:
            raise ValueError(
                f'ScrubResult.error must only be set when outcome is FAILED, '
                f'got outcome={self.outcome!r} with error={self.error!r}'
            )
        if isinstance(self.error, str) and not self.error.strip():
            raise ValueError(
                'ScrubResult.error must not be an empty or whitespace-only string; '
                'use None instead'
            )

    def format_error(self, prefix: str = '') -> str:
        """Return prefix+error when error is set, otherwise empty string.

        Designed for safe interpolation into log messages and f-strings:
        - When error is set: returns ``f'{prefix}{self.error}'``
        - When error is None: returns ``''`` (nothing to show)

        Args:
            prefix: Optional string prepended to the error (e.g. ' Error: ', ': ').
        """
        if self.error is not None:
            return f'{prefix}{self.error}'
        return ''


# ---------------------------------------------------------------------------
# .task/ contamination helpers
# ---------------------------------------------------------------------------

async def scrub_task_dir_from_tree(
    cwd: Path, context: str, *, amend: bool = True,
) -> ScrubResult:
    """Remove .task/ from the git index if present.

    This is the primary defense against .task/ reaching main.  It checks
    whether any .task/ entries exist in the current HEAD's tree and, if so,
    removes them from the index.

    Args:
        cwd: Working directory (a worktree or the project root).
        context: Human-readable label for log messages (e.g. "post-merge",
                 "worktree-creation").
        amend: If True (default), amend the current commit to exclude .task/.
               If False, create a NEW commit for the removal.  Use amend=False
               when the current HEAD is shared with another branch (e.g. right
               after create_worktree where HEAD == main's tip).

    Returns:
        ``ScrubResult.CLEAN``    if ``.task/`` was not present in the tree.
        ``ScrubResult.SCRUBBED`` if ``.task/`` was found and successfully removed.
        ``ScrubResult.FAILED``   if ``.task/`` was found but removal failed
                                  (git rm or git commit returned non-zero).

    DO NOT REMOVE THIS FUNCTION.  It is the last reliable defense before
    .task/ reaches main via update-ref.  Note: git's ``reference-transaction``
    hook (git>=2.28) is the exception — it DOES fire on update-ref — and
    advance_main's main_gate mark (added in task 1678) sanctions that hook.
    All other git hooks (pre-commit, etc.) are still bypassed by update-ref.
    See also task 7 for the same stale "bypasses ALL hooks" assumption.
    """
    rc, tracked, _ = await _run(
        ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
        cwd=cwd,
    )
    if rc != 0 or not tracked.strip():
        return ScrubResult(outcome=ScrubOutcome.CLEAN)

    files = [f for f in tracked.strip().splitlines() if f.strip()]
    if not files:
        return ScrubResult(outcome=ScrubOutcome.CLEAN)

    logger.warning(
        '.task/ CONTAMINATION detected during %s — removing %d tracked file(s): %s',
        context, len(files), ', '.join(files[:10]),
    )

    # Remove from index (not filesystem — .task/ may still be needed as scratch)
    rc, _, err = await _run(['git', 'rm', '-r', '--cached', '--', '.task/'], cwd=cwd)
    if rc != 0:
        logger.error('.task/ scrub failed during %s: git rm --cached failed: %s', context, err)
        return ScrubResult(outcome=ScrubOutcome.FAILED, error=err.strip() or None)

    # Also remove from filesystem if present (cleanup inherited contamination)
    task_dir = cwd / '.task'
    if task_dir.exists():
        shutil.rmtree(task_dir)

    if amend:
        # Amend the current commit to exclude .task/ (used post-merge where
        # we own the merge commit and want a clean tree).
        rc, _, err = await _run(
            ['git', 'commit', '--amend', '--no-edit', '--allow-empty'],
            cwd=cwd,
        )
    else:
        # Create a new commit (used in create_worktree where HEAD is shared
        # with main — amending would rewrite main's history).
        rc, _, err = await _run(
            ['git', 'commit', '-m',
             'chore: remove .task/ contamination inherited from main\n\n'
             '.task/ is the orchestrator scratch directory and must never\n'
             'be on main.  This commit removes it from the branch tree.'],
            cwd=cwd,
        )

    if rc != 0:
        logger.error('.task/ scrub failed during %s: could not commit removal: %s', context, err)
        return ScrubResult(outcome=ScrubOutcome.FAILED, error=err.strip() or None)

    logger.info('.task/ scrub completed during %s — %d file(s) removed from tree', context, len(files))
    return ScrubResult(outcome=ScrubOutcome.SCRUBBED)


def _ensure_task_gitignore(worktree: Path) -> None:
    """Create .task/.gitignore with '*' if it doesn't exist.

    This is a defense-in-depth measure.  When an agent does ``git add .``
    or ``git add -A``, the nested .gitignore prevents .task/ contents from
    being staged — UNLESS files were previously explicitly added (tracked
    files override .gitignore).  The pre-commit hook is the primary guard;
    this is supplementary.
    """
    task_dir = worktree / '.task'
    task_dir.mkdir(exist_ok=True)
    gi = task_dir / '.gitignore'
    if not gi.exists():
        gi.write_text('# Auto-generated — prevents .task/ from being staged.\n*\n')


async def _assert_no_task_dir(sha: str, cwd: Path, context: str) -> None:
    """Raise RuntimeError if the given commit SHA contains any .task/ entries.

    This is a hard gate — if this fires, something upstream failed to scrub
    .task/ and we must NOT advance main.

    DO NOT CATCH THIS EXCEPTION to "work around" it.  Fix the root cause:
    find where .task/ was committed and add a scrub there.
    """
    rc, tracked, _ = await _run(
        ['git', 'ls-tree', '-r', '--name-only', sha, '--', '.task/'],
        cwd=cwd,
    )
    if rc == 0 and tracked.strip():
        files = tracked.strip().splitlines()
        raise RuntimeError(
            f'.task/ CONTAMINATION GATE FAILED ({context}): commit {sha[:8]} '
            f'contains {len(files)} .task/ file(s): {", ".join(files[:5])}. '
            f'Refusing to advance main.  This is a bug — .task/ should have '
            f'been scrubbed before reaching this point.'
        )


@dataclass
class MergeResult:
    success: bool
    conflicts: bool = False
    details: str = ''
    merge_commit: str | None = None
    pre_merge_sha: str | None = None
    merge_worktree: Path | None = None


class WarmLaneUnavailable(Enum):
    """Discriminated failure result from :meth:`acquire_warm_lane`.

    Returned instead of bare ``None`` so callers can distinguish:

    * ``EXHAUSTED`` — all pool lanes are ASSIGNED; signal backpressure / requeue.
    * ``FAULT`` — seed/worktree-add failure or absent seed script; signal blocked + L1.
    * ``DISK_PRESSURE`` — seed exited 75 (EX_TEMPFAIL); transient infra; requeue.
    * ``DISABLED`` — pool knob is off (``warm_lane_pool is None``); programming-error
      sentinel returned when :meth:`acquire_warm_lane` is called without first
      checking ``self.warm_lane_pool is not None``.  A disabled pool is NOT
      equivalent to backpressure — callers that receive ``DISABLED`` should NOT
      requeue; they should fall back to the cold path or raise.  The canonical
      caller (:meth:`create_worktree`) is already guarded and will never observe
      this value; it is provided solely to surface caller bugs clearly rather
      than silently requeuing forever.

    A lane is always released back to FREE before this value is returned, so no
    ASSIGNED lanes are ever leaked on failure.
    """
    EXHAUSTED = 'exhausted'
    FAULT = 'fault'
    DISK_PRESSURE = 'disk_pressure'
    DISABLED = 'disabled'


@dataclass
class WorktreeInfo:
    """Return value from create_worktree - captures worktree path and base commit.

    The base_commit is the SHA of main at worktree creation time, pinned to
    ensure stable diffs even if main advances during task execution.

    stale_commits: how far local main was behind the remote at worktree creation
    time.  None means either (a) the fetch was unavailable (no remote configured),
    or (b) the worktree is train-stacked — branched from a sibling's tip rather
    than from main, so the "behind remote" concept does not apply.  0 means
    already current.  A positive stale_commits value means the remote was ahead by
    N commits.  When local main has diverged (has unpushed commits), the worktree
    is based on local main despite the positive count — check this field together
    with base_commit to determine actual freshness.

    reify_debug_port: per-worktree reify-debug port allocated during provisioning
    by running scripts/setup-worktree-debug-port.sh in the worktree.  None when
    no such script is present (non-reify projects) or provisioning failed (fail-open).
    """
    path: Path
    base_commit: str
    stale_commits: int | None = None
    reify_debug_port: int | None = None


class ConflictProbe(NamedTuple):
    """Result of merge_tree_conflicts — a lightweight, tuple-destructurable probe.

    Fields
    ------
    clean:
        True  → branch_head would merge onto base_tip without conflicts.
        False → at least one file conflict was detected.
    conflicted_paths:
        Paths of conflicting files (relative to repo root).  Empty when clean.

    As a NamedTuple, supports both named-field access and tuple destructuring::

        probe = await git_ops.merge_tree_conflicts(base_tip, branch_head)
        # named-field access:
        if probe.clean: ...
        # tuple-destructuring (PRD §5.2 contract):
        clean, paths = probe
    """

    clean: bool
    conflicted_paths: list[str]


class WorktreeMissing(FileNotFoundError):
    """Raised when a subprocess cannot start because its ``cwd`` does not exist.

    The orchestrator races against humans who may delete a task's worktree
    out-of-band.  When that happens, ``asyncio.create_subprocess_exec`` raises
    a generic ``FileNotFoundError`` whose ``.filename`` is the missing
    directory.  We re-raise as this typed exception so callers can distinguish
    a missing worktree (recoverable: task may already be done) from a missing
    binary (real bug).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        super().__init__(f'Worktree missing: {self.path}')


class WarmLaneRequeue(Exception):
    """Base class for warm-lane failures that should requeue (not block + L1).

    Raised by :meth:`GitOps.create_worktree` when warm_lane_pool is enabled
    and the pool cannot allocate a seeded lane for a transient reason.
    Callers (workflow.run()) should return :attr:`WorkflowOutcome.REQUEUED`
    rather than letting this propagate to the broad ``except Exception`` handler.

    Subclasses:
        WarmLanePoolExhausted — all lanes ASSIGNED (backpressure).
        WarmLaneDiskPressure  — seed exited 75 (EX_TEMPFAIL, transient infra).
    """


class WarmLanePoolExhausted(WarmLaneRequeue):
    """All pool lanes are ASSIGNED; task must be requeued (backpressure).

    Scheduler should release the task and re-dispatch it when a lane frees up.
    """


class WarmLaneDiskPressure(WarmLaneRequeue):
    """Seed exited 75 (EX_TEMPFAIL) — transient disk pressure / infra issue.

    Task should be requeued with a ``warm_lane_disk_pressure (transient infra)``
    block-reason annotation so the requeue is distinguishable from backpressure.
    """


async def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run an arbitrary subprocess command and return (returncode, stdout, stderr).

    Used throughout for git invocations and for any other subprocess call
    (e.g. project setup scripts).  Raises :class:`WorktreeMissing` if ``cwd``
    is provided but does not exist, so the caller can distinguish a deleted
    worktree (recoverable race) from other ``FileNotFoundError``\\ s (e.g.
    missing binary on ``PATH``).
    """
    # Pre-flight: a missing cwd surfaces as a generic FileNotFoundError from
    # posix_spawn whose .filename is not reliably set.  Check explicitly so we
    # can raise a typed exception consumers can pattern-match on.
    if cwd is not None and not Path(cwd).is_dir():
        raise WorktreeMissing(cwd)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        # Race: cwd existed at the pre-flight check but vanished before spawn.
        # Re-classify as WorktreeMissing if cwd is now gone; otherwise the
        # error is about the binary itself.
        if cwd is not None and not Path(cwd).is_dir():
            raise WorktreeMissing(cwd) from e
        raise
    stdout, stderr = await proc.communicate()
    return proc.returncode if proc.returncode is not None else 1, stdout.decode().strip(), stderr.decode().strip()


def _merge_subject(branch: str, main_branch: str) -> str:
    """Return the canonical subject line for a no-ff merge of *branch* into *main_branch*.

    Single source of truth for the merge commit subject format consumed by
    ``find_merge_marker``, ``merge_to_main``, and the retry path in
    ``advance_main``.  Changing this function is the one place where the
    format needs to be updated — all three consumers will automatically
    use the new format.
    """
    return f'Merge {branch} into {main_branch}'


# Sentinel range used to represent files that are fully deleted or renamed.
# The range (0, 2**30) spans every plausible line number, so an intersection
# check against any real hunk range always returns True (not stackable).
_WHOLE_FILE_SENTINEL: tuple[int, int] = (0, 2**30)


def parse_diff_line_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff and return old-side (BASE) line ranges per file.

    Given the output of ``git diff <main>...<ref> --unified=0 --no-color``,
    returns a mapping of file path → list of (start, end) tuples representing
    old-side (BASE/main-relative) changed line ranges.  Using old-side ranges
    from both branches diffed against the same main makes ranges directly
    comparable for stackability checks.

    Pure insertion hunks (old_count == 0, e.g. ``@@ -7,0 +8,3 @@``) are
    mapped to a point range ``(old_start, old_start)`` so they are still
    comparable; ``@@ -N,0 ... @@`` anchors at line N (the line *before* the
    insertion in the old file).

    Deleted files (``+++ /dev/null``), pure renames (``rename from``), and
    renames with content changes (``--- a/old`` → ``+++ b/new``) are
    represented via ``_WHOLE_FILE_SENTINEL`` on the old-side path.  This
    ensures that a modify/delete or rename/modify pair between two tasks is
    always flagged non-stackable by the stackability gate.

    Returns an empty dict for an empty or header-only diff.
    """
    import re

    result: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    old_path: str | None = None  # from '--- a/<path>'; reset per diff block

    hunk_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            # Start of a new file block — reset per-file state.
            current_file = None
            old_path = None
        elif line.startswith('--- a/'):
            # Record old-side path for deletion / rename detection below.
            old_path = line[6:]
        elif line.startswith('+++ b/'):
            new_path = line[6:]
            # Rename with content changes: old_path ≠ new_path.  The old file is
            # gone; represent it with a sentinel so tasks touching the old name
            # are flagged non-stackable with this rename.
            if old_path and old_path != new_path and old_path not in result:
                result[old_path] = [_WHOLE_FILE_SENTINEL]
            current_file = new_path
            if current_file not in result:
                result[current_file] = []
        elif line.startswith('+++ /dev/null'):
            # File deletion: old file is completely gone.  Represent old_path
            # with the whole-file sentinel so any task modifying this file is
            # flagged non-stackable with the deletion.
            if old_path and old_path not in result:
                result[old_path] = [_WHOLE_FILE_SENTINEL]
            current_file = None  # no new file; skip hunk parsing
        elif line.startswith('rename from '):
            # Pure rename (R100) header — no --- / +++ lines follow for the old
            # path.  Add it with the sentinel so tasks touching the old name are
            # flagged.  For renames with content changes the --- a/ handler above
            # also runs, but the 'not in result' guard prevents a double-insert.
            renamed_from = line[len('rename from '):]
            if renamed_from not in result:
                result[renamed_from] = [_WHOLE_FILE_SENTINEL]
        elif current_file is not None:
            m = hunk_re.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                # Pure insertion: old_count == 0 → point range at old_start.
                end = old_start + max(old_count, 1) - 1
                result[current_file].append((old_start, end))

    return result


class GitOps:
    """Git worktree and merge operations."""

    def __init__(
        self,
        config: GitConfig,
        project_root: Path,
        *,
        warm_lane_pool_size: int = 0,
        merge_spec_warm_lane_pool_size: int = 0,
    ):
        self.config = config
        self.project_root = project_root
        self.worktree_base = (project_root / config.worktree_dir).resolve()
        # Warm-lane pool — None when knob off or size=0 (default-off, trivially
        # revertible, mirrors persistent_merge_worktree).  Size is passed from
        # OrchestratorConfig.max_concurrent_tasks by Harness at startup (D9).
        self._warm_lane_pool_size = warm_lane_pool_size
        if warm_lane_pool_size > 0 and config.warm_lane_pool:
            from orchestrator.warm_lane_pool import WarmLanePool
            self.warm_lane_pool: WarmLanePool | None = WarmLanePool(
                worktree_base=self.worktree_base,
                size=warm_lane_pool_size,
            )
        else:
            self.warm_lane_pool = None
        # Merge-speculation warm-lane pool — None when knob off or size=0.
        # Second WarmLanePool instance with name_prefix='_spec-' for K>1 LOCAL
        # speculative verify slots.  Size K = speculation_depth, sized from the
        # SAME shared K source as the worker (steps 5-6, harness.py).
        # Default-off, byte-identical at default, mirrors warm_lane_pool above.
        self._merge_spec_warm_lane_pool_size = merge_spec_warm_lane_pool_size
        if merge_spec_warm_lane_pool_size > 0 and config.merge_spec_warm_lane_pool:
            from orchestrator.warm_lane_pool import WarmLanePool as _WLP
            self.spec_warm_lane_pool: WarmLanePool | None = _WLP(
                worktree_base=self.worktree_base,
                size=merge_spec_warm_lane_pool_size,
                name_prefix='_spec-',
            )
        else:
            self.spec_warm_lane_pool = None
        # Serialize first-time `git worktree add` for _spec- lanes.
        # Git serializes worktree administration via a repo-level lock; concurrent
        # adds from the same project_root can transiently fail during the initial
        # K>1 warm-up burst.  Reset-in-place (already-registered) acquires are
        # per-lane and don't contend, so this only guards the one-time create path.
        self._spec_wt_create_lock: asyncio.Lock = asyncio.Lock()
        # Merge serialization is handled by MergeWorker in merge_queue.py.
        # See task 292 for design rationale (ghost loops, lock starvation,
        # branch drift at 64 max concurrency with external actors).

    async def _is_registered_worktree(self, path: Path) -> bool:
        """Check if *path* is a registered git worktree.

        Uses ``git worktree list --porcelain`` and matches by **canonical
        (resolved) path on both sides** — each listed ``worktree <path>``
        is ``Path.resolve()``-d and compared against ``path.resolve()``.
        This recognizes a registration recorded under a *symlink* path
        (e.g. reify's ``.worktrees`` symlinked to a mount after migration,
        esc-4146-268) that an exact-string compare would miss, while still
        rejecting stale directories (containing only .task/ state files)
        that were never registered.
        """
        resolved = path.resolve()
        rc, output, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.project_root,
        )
        if rc != 0:
            return False  # fail-safe is provided by the destroy gate in create_worktree
        for line in output.splitlines():
            if line.startswith('worktree ') and Path(line[9:]).resolve() == resolved:
                return True
        return False

    async def _freshen_main(self) -> tuple[str, int | None]:
        """Fetch from remote and return the freshest ref to use as worktree base.

        Returns:
            (ref, stale_commits) where:
            - ref: the git ref to pass to ``git worktree add`` / ``git rev-parse``
            - stale_commits: None  → fetch failed (no remote configured)
                             0     → local main is already current with remote
                             N > 0 → local main was N commits behind remote

        Design decisions:
        - Best-effort fetch: if fetch fails (no remote in tests), return
          (main_branch, None) silently — matches the pattern in
          _create_merge_worktree (line 578).
        - No mutation of local main ref: advance_main() uses CAS on the local
          main ref; updating it here could cause spurious CAS failures.  We
          return the remote-tracking ref (origin/main) as the start-point
          instead.
        - Divergence guard: if local main has commits not in origin/main (e.g.
          from advance_main calls not yet pushed), using origin/main would lose
          those commits.  In the diverged case we fall back to local main and
          log a warning.
        """
        remote_ref = f'{self.config.remote}/{self.config.main_branch}'

        # Best-effort fetch — silently ignore failure (no remote in tests)
        rc, _, _ = await _run(
            ['git', 'fetch', self.config.remote, self.config.main_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.debug(
                '_freshen_main: fetch from %s failed — using local %s',
                self.config.remote, self.config.main_branch,
            )
            return self.config.main_branch, None

        # Count commits local main is BEHIND origin/main
        rc, behind_out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{self.config.main_branch}..{remote_ref}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                '_freshen_main: rev-list (behind) failed (rc=%d) — using local %s',
                rc, self.config.main_branch,
            )
            return self.config.main_branch, None
        try:
            behind = int(behind_out.strip())
        except ValueError:
            logger.warning(
                '_freshen_main: unexpected behind-count output: %r', behind_out,
            )
            return self.config.main_branch, None

        if behind == 0:
            return self.config.main_branch, 0

        # Check for divergence: count commits local main is AHEAD of origin/main
        rc, ahead_out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{remote_ref}..{self.config.main_branch}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                '_freshen_main: rev-list (ahead) failed (rc=%d) — using local %s',
                rc, self.config.main_branch,
            )
            return self.config.main_branch, behind
        try:
            ahead = int(ahead_out.strip())
        except ValueError:
            logger.warning(
                '_freshen_main: unexpected ahead-count output: %r', ahead_out,
            )
            # Fall back to local main; report behind count as-is (ref is local, not remote)
            return self.config.main_branch, behind

        if ahead > 0:
            logger.warning(
                '_freshen_main: local %s diverged from %s (%d ahead, %d behind) '
                '— using local ref to avoid losing advance_main commits',
                self.config.main_branch, remote_ref, ahead, behind,
            )
            return self.config.main_branch, behind

        # Strictly behind: use remote-tracking ref as worktree start-point
        logger.info(
            '_freshen_main: local %s is %d commits behind %s — using %s',
            self.config.main_branch, behind, remote_ref, remote_ref,
        )
        return remote_ref, behind

    async def _train_predecessor(self, train: TrainMembership) -> TrainPredecessor:
        """Resolve the predecessor for a train member with order > 0.

        Reads train['members'][order - 1] and derives its branch name using
        the configured branch_prefix.  Raises ValueError when invariants are
        violated (order <= 0, members absent/None, members too short).
        """
        order = train.get('order', 0)
        if order <= 0:
            raise ValueError(
                f'_train_predecessor called with order={order!r}; '
                'must only be called when order > 0'
            )
        members = train.get('members')
        if not members or not isinstance(members, list):
            raise ValueError(
                f'_train_predecessor: members is {members!r}; '
                'expected a non-empty list of task ids'
            )
        if len(members) < order:
            raise ValueError(
                f'_train_predecessor: members has {len(members)} entries but '
                f'order={order} requires at least {order} entries; members={members!r}'
            )
        predecessor_id = str(members[order - 1])
        return TrainPredecessor(
            task_id=predecessor_id,
            branch=f'{self.config.branch_prefix}{predecessor_id}',
        )

    async def _resolve_predecessor_tip(
        self, train: TrainMembership, branch_name: str
    ) -> tuple[str, str]:
        """Resolve predecessor branch tip SHA and branch name for a train member.

        Returns ``(predecessor_sha, predecessor_branch)`` where
        *predecessor_sha* is the 40-char tip SHA of the predecessor's branch
        and *predecessor_branch* is its full branch name.

        Raises ``RuntimeError`` when the predecessor branch does not exist.
        Both the initial-create path and the reuse path share this invariant:
        the predecessor must be present before any successor can be created or
        requeued.  Centralising the guard here prevents the two callers from
        drifting apart — the pre-refactor duplication would have allowed the
        two error messages (and their None-guard logic) to diverge, silently
        reintroducing the main-rebase corruption this task fixes.
        """
        predecessor = await self._train_predecessor(train)
        predecessor_sha = await self.resolve_branch_sha(predecessor.branch)
        if predecessor_sha is None:
            raise RuntimeError(
                f'create_worktree: predecessor branch {predecessor.branch!r} '
                f'does not exist (train_id={train.get("id")!r}, '
                f'order={train.get("order")}, branch_name={branch_name!r}). '
                'The predecessor branch must exist before any successor can be '
                'created or requeued.'
            )
        return predecessor_sha, predecessor.branch

    async def create_worktree(
        self,
        branch_name: str,
        *,
        expected_title: str | None = None,
        train: TrainMembership | None = None,
    ) -> WorktreeInfo:
        """Create a git worktree for a task branch, based off main.

        Returns a WorktreeInfo with the worktree path and the base commit SHA
        (main's SHA at creation time) so diffs remain stable even if main
        advances during task execution.

        ``train`` — when supplied and ``train['order'] > 0``, the worktree is
        branched from the prior train member's branch tip instead of
        ``origin/main``.  See PRD § 9.4 for the full train-branching spec.
        ``train=None`` (default) and ``train['order'] == 0`` both fall through
        to the existing ``_freshen_main()`` path unchanged.

        If the worktree/branch already exist (e.g., from a requeued task),
        reuses them instead of failing — UNLESS ``expected_title`` is supplied
        and the existing worktree's stored title fails to match it (a recycled
        task id whose orphaned worktree holds unrelated WIP).  On mismatch the
        stale worktree is quarantined and a fresh one is created instead.
        ``expected_title=None`` (the default) skips this guard entirely, so all
        existing callers/tests are unaffected.
        """
        worktree_path = self.worktree_base / branch_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        full_branch = f'{self.config.branch_prefix}{branch_name}'

        # ── Ensure core.hooksPath is set ──────────────────────────────
        # The pre-commit hook in hooks/pre-commit strips .task/ from the
        # staging area on ALL branches.  core.hooksPath must point to
        # hooks/ (relative) so worktrees find the hook via their own
        # working tree.  This is idempotent — safe to run every time.
        await _run(
            ['git', 'config', 'core.hooksPath', 'hooks'],
            cwd=self.project_root,
        )

        # ── Resolve start-ref: train-predecessor tip or freshened main ──
        # PRD § 9.4: when a train member has order > 0, branch from the prior
        # member's branch tip so the chain is contiguous.  order=0 (degenerate
        # train) and train=None both fall through to _freshen_main().
        if train is not None and train.get('order', 0) > 0:
            # ── Train path: branch from predecessor's tip ─────────────────
            # PRD § 9.4: resolve the predecessor's branch and use its tip SHA
            # as start_ref so the new worktree stacks directly on top.
            # Guard (raise RuntimeError on None) is in _resolve_predecessor_tip.
            start_ref, _ = await self._resolve_predecessor_tip(train, branch_name)
            stale_commits = None  # "behind remote" does not apply to sibling branches
        else:
            # ── Freshen main from remote (best-effort) ────────────────────
            # If origin/main has advanced since session start, use the remote-
            # tracking ref as the worktree base so agents start from the freshest
            # code.  Falls back to local main silently when no remote is configured
            # (e.g. in test repos).  Never mutates the local main ref — that would
            # interfere with advance_main's CAS logic.
            start_ref, stale_commits = await self._freshen_main()
        logger.info(
            'create_worktree: freshening result: ref=%s, stale_commits=%s',
            start_ref, stale_commits,
        )

        # Capture the freshened ref's SHA (used as stable base for diffs)
        rc, base_sha, _ = await _run(
            ['git', 'rev-parse', start_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            if train is not None and train.get('order', 0) > 0:
                # start_ref was a SHA just verified by resolve_branch_sha; if
                # rev-parse fails here it indicates git state corruption, not a
                # missing remote ref.  Falling back to main would silently violate
                # the train-stacking invariant, so raise instead.
                raise RuntimeError(
                    f'create_worktree: rev-parse of confirmed predecessor SHA '
                    f'{start_ref!r} failed (rc={rc}); this is unexpected — '
                    f'the SHA was just resolved by resolve_branch_sha and should '
                    f'be stable'
                )
            logger.warning(
                'create_worktree: rev-parse %s failed (rc=%d) — falling back to local %s',
                start_ref, rc, self.config.main_branch,
            )
            start_ref = self.config.main_branch
            rc, base_sha, _ = await _run(
                ['git', 'rev-parse', start_ref],
                cwd=self.project_root,
            )
            if rc != 0:
                raise RuntimeError(
                    f'create_worktree: rev-parse of local {start_ref} also failed (rc={rc})'
                )

        # ── Warm-lane pool (ζ): try to allocate a pre-seeded lane ──────────
        # Only for non-train, non-reuse-by-name fresh dispatches.  Train-
        # stacked members (predecessor-tip branching) and identity-guard
        # reuse-by-name have bespoke logic below; pooling them is out of ζ's
        # scope and the cold path stays correct.
        # Pool exhaustion / absent seed script / seed failure all fall through
        # to the unchanged cold path below (inv.6: never block/deadlock).
        if (
            self.warm_lane_pool is not None
            and (train is None or train.get('order', 0) == 0)
            and not await self._is_registered_worktree(worktree_path)
        ):
            pool_info = await self.acquire_warm_lane(
                branch_name, start_ref, expected_title=expected_title,
            )
            if isinstance(pool_info, WorktreeInfo):
                # Carry stale_commits from the freshen result onto the pool info
                return WorktreeInfo(
                    path=pool_info.path,
                    base_commit=pool_info.base_commit,
                    stale_commits=stale_commits,
                    reify_debug_port=pool_info.reify_debug_port,
                )
            # β: pool is enabled — no cold-path fall-through.  Route discriminant
            # to the appropriate exception so callers get typed failure signals.
            if pool_info is WarmLaneUnavailable.EXHAUSTED:
                raise WarmLanePoolExhausted(
                    f'warm-lane pool exhausted for branch {branch_name!r}; requeue'
                )
            if pool_info is WarmLaneUnavailable.DISK_PRESSURE:
                raise WarmLaneDiskPressure(
                    f'warm-lane seed disk pressure for branch {branch_name!r}; requeue'
                )
            # FAULT or DISABLED → RuntimeError reuses existing blocked+L1 plumbing.
            # DISABLED is a programming error (caller bypassed the pool-enabled
            # guard); it is treated as a fault here so blocked+L1 surfaces the
            # bug rather than silently requeueing forever.
            raise RuntimeError(
                f'warm-lane acquire fault for branch {branch_name!r} '
                f'(seed/worktree-add failure, absent seed script, or pool disabled)'
            )

        # If worktree already exists, reuse it (common after requeue) —
        # but ONLY if it is a real registered git worktree.  A stale
        # directory (e.g. containing only .task/ state files from a previous
        # run) must be removed so a fresh worktree can be created.
        if worktree_path.exists():
            reuse_ok = await self._is_registered_worktree(worktree_path)
            # ── Identity guard (Fix C, defense-in-depth) ──────────────
            # A registered worktree whose stored title does not match the
            # live task's title is a recycled-id collision: the dir name
            # equals the new task's numeric id but the contents belong to a
            # deleted task.  Quarantine it (preserving its WIP) and fall
            # through to a fresh create.  identities_match fails open, so a
            # title-less legacy worktree is reused as before.
            if reuse_ok and expected_title is not None:
                stored_title = read_worktree_title(worktree_path)
                if not identities_match(stored_title, expected_title):
                    logger.warning(
                        'create_worktree: reuse identity MISMATCH for %s — '
                        'stored title %r != expected %r; quarantining and '
                        'creating fresh',
                        worktree_path, stored_title, expected_title,
                    )
                    await self.quarantine_worktree(
                        worktree_path, branch_name, 'reuse-identity-mismatch',
                    )
                    reuse_ok = False
            if reuse_ok:
                logger.info(f'Reusing existing worktree at {worktree_path} on branch {full_branch}')

                # Save any uncommitted tracked work before rebasing
                # (.task/ is gitignored so plan.json is unaffected)
                await self.commit(
                    worktree_path,
                    'chore: save WIP before requeue rebase',
                )

                # ── Resolve rebase target: train-predecessor tip or main ──────
                # Mirror the create-path start-ref logic (lines 777-791): a
                # stacked train member (order > 0) must rebase onto its
                # predecessor's tip so the stacking invariant is preserved
                # across requeues.  Rebasing onto main (the old unconditional
                # behaviour) corrupts the train by re-parenting the delta off
                # main instead of the predecessor's commits.
                if train is not None and train.get('order', 0) > 0:
                    # Guard (raise RuntimeError on None) is in _resolve_predecessor_tip.
                    rebase_target, base_ref = await self._resolve_predecessor_tip(
                        train, branch_name
                    )
                else:
                    rebase_target: str | None = None  # rebase_onto_main defaults to main
                    base_ref: str = self.config.main_branch

                # Rebase onto the resolved target (predecessor tip for stacked
                # trains, main for non-train / order==0).  Critical for plan
                # revalidation: the architect needs to see current file
                # contents from the right base.
                if await self.rebase_onto_main(worktree_path, onto=rebase_target):
                    # Re-capture base from worktree's own merge-base after the
                    # rebase completes.  merge-base from inside the worktree
                    # is race-immune to concurrent base-ref advances during
                    # rev-parse / rebase.
                    _, mb_out, _ = await _run(
                        ['git', 'merge-base', base_ref, 'HEAD'],
                        cwd=worktree_path,
                    )
                    actual_base = mb_out.strip() or base_sha.strip()
                else:
                    # Rebase failed (conflict) — continue on old base.
                    # Compute the actual merge-base so WorktreeInfo is truthful.
                    _, mb_out, _ = await _run(
                        ['git', 'merge-base', base_ref, 'HEAD'],
                        cwd=worktree_path,
                    )
                    actual_base = mb_out.strip() or base_sha.strip()
                    if train is not None and train.get('order', 0) > 0:
                        # For stacked train members a rebase conflict means the
                        # stacking invariant is broken — log at ERROR so the
                        # degradation is visible rather than silently shipped on
                        # a stale base.
                        logger.error(
                            'Rebase conflict for stacked train member %s '
                            '(train_id=%r, order=%s) — continuing on stale '
                            'base %s; stack integrity is degraded.',
                            worktree_path, train.get('id'),
                            train.get('order'), actual_base[:8],
                        )
                    else:
                        logger.warning(
                            'Rebase failed for reused worktree %s — continuing '
                            'on old base %s',
                            worktree_path, actual_base[:8],
                        )

                _ensure_task_gitignore(worktree_path)
                # Re-run on reuse so the requeued agent re-acquires a free
                # port and re-patches its .mcp.json.  The script must be
                # idempotent (return the same port for the same worktree dir)
                # to avoid leaking ports across requeues — see the docstring
                # of _provision_reify_debug_port for the full contract.
                port = await self._provision_reify_debug_port(worktree_path)
                return WorktreeInfo(
                    path=worktree_path,
                    base_commit=actual_base,
                    stale_commits=stale_commits,
                    reify_debug_port=port,
                )
            elif worktree_path.exists():
                # The directory exists but git does not recognize it as a
                # registered worktree.  Two very different cases share this
                # branch, and conflating them is what destroyed live work in
                # esc-4146-268 (the silent rmtree below):
                #   (a) a genuinely stale leftover — only .task/ residue (and/
                #       or empty), no .git link, no branch — safe to remove;
                #   (b) a de-registered LIVE worktree whose admin entry was
                #       lost (reify's symlink migration, or a stray prune) —
                #       its .git link or source files are still on disk, or
                #       its task branch still carries commits.  Deleting it
                #       destroys work, including the gitignored .task/plan.json
                #       that git cannot restore.
                # Discriminate git-independently (so the gate still holds under
                # ENOSPC / total git failure): a .git link present, or content
                # beyond .task/, or a branch with commits beyond main => live.
                # Mirror _cleanup_leftover_branch: raise rather than delete
                # anything live (RuntimeError from create_worktree routes to
                # blocked + L1, non-stranding via Harness Fix #1a — see below).
                entries = {p.name for p in worktree_path.iterdir()}
                has_git_link = '.git' in entries
                has_substantive_content = bool(entries - {'.task', '.git'})
                rc_v, _, _ = await _run(
                    ['git', 'rev-parse', '--verify', full_branch],
                    cwd=self.project_root,
                )
                branch_has_work = (
                    rc_v == 0
                    and await self._branch_has_commits_beyond_main(full_branch)
                )
                if has_git_link or has_substantive_content or branch_has_work:
                    raise RuntimeError(
                        f'create_worktree: refusing to delete directory '
                        f'{worktree_path} — it looks like a live worktree whose '
                        f'git registration was lost (a .git link is present, '
                        f'source files exist, or branch {full_branch!r} carries '
                        f'commits beyond {self.config.main_branch}). Deleting '
                        f'would destroy work, including the gitignored .task/ '
                        f'plan state that git cannot restore. Recover by '
                        f're-registering it (`git worktree repair '
                        f'{worktree_path}`) and re-dispatching, or quarantine it '
                        f'manually once any wanted work is preserved. (fail-safe; '
                        f'was the silent rmtree at git_ops.py:702)'
                    )
                logger.warning(
                    f'Directory {worktree_path} exists but is NOT a registered '
                    f'git worktree, and holds no live work (no .git link, only '
                    f'.task/ residue, branch has no commits beyond '
                    f'{self.config.main_branch}) — removing stale directory and '
                    f'creating fresh worktree'
                )
                shutil.rmtree(worktree_path)

        # If the branch ref already exists (stale from a previous run, or — the
        # 3576 trigger — still checked out in a leftover worktree), clean it up
        # ONLY when deterministically non-destructive.  The old code ran a blind
        # `git branch -D` and ignored its rc: when the branch was checked out in
        # a leftover worktree the delete silently failed, then `git worktree add`
        # raised the opaque "a branch named ... already exists" (2026-05-29).
        # Worse, a blind delete of a branch carrying commits beyond main would
        # have destroyed orphan work.  Hard rule: never delete uncommitted WIP
        # or orphan commits — prove the cleanup is non-destructive, else raise
        # (→ blocked + L1, now non-stranding via Harness Fix #1a).
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', full_branch],
            cwd=self.project_root,
        )
        if rc == 0:
            await self._cleanup_leftover_branch(full_branch, branch_name)

        # Create worktree with new branch from the freshened ref
        rc, out, err = await _run(
            ['git', 'worktree', 'add', '-b', full_branch, str(worktree_path), start_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create worktree: {err}')

        logger.info(
            'Created worktree at %s on branch %s (base=%s, stale_commits=%s)',
            worktree_path, full_branch, base_sha[:8], stale_commits,
        )

        # ── .task/.gitignore defense layer ────────────────────────────
        # Create .task/.gitignore with "*" so that broad "git add ."
        # commands in the worktree don't pick up .task/ contents.  This
        # is defense-in-depth — the pre-commit hook is the primary guard.
        _ensure_task_gitignore(worktree_path)

        # ── .task/ contamination guard ────────────────────────────────
        # If main is contaminated (has .task/ tracked), this worktree
        # inherits it.  Scrub it NOW before any agent code runs, so the
        # task starts from a clean tree.  The scrub amends the initial
        # commit on the new branch — harmless since nothing else has
        # been committed yet.
        # amend=False: HEAD is shared with main — must NOT amend the shared commit.
        # Instead, create a new commit on the branch to remove .task/.
        scrub_result = await scrub_task_dir_from_tree(
            worktree_path, 'worktree-creation', amend=False,
        )
        if scrub_result.outcome == ScrubOutcome.SCRUBBED:
            logger.warning(
                'MAIN IS CONTAMINATED — .task/ was inherited by new worktree %s. '
                'The contamination has been removed from this worktree, but main '
                'still carries .task/.  Run: git rm -r --cached .task/ on main.',
                worktree_path,
            )
        elif scrub_result.outcome == ScrubOutcome.FAILED:
            logger.error(
                '.task/ scrub FAILED during worktree-creation for %s — the index '
                'may still be contaminated.  The hard gate at advance_main will '
                'catch this if contamination reaches main.%s',
                worktree_path,
                scrub_result.format_error(prefix=' Error: '),
            )

        # Re-capture base from the worktree's own merge-base after positioning
        # AND the scrub_task_dir_from_tree call above.  merge-base from inside
        # the freshly-created worktree is race-immune to concurrent main
        # advances between rev-parse and `git worktree add`: it is the fork
        # point of HEAD with the freshened start_ref regardless of when main
        # advanced.  We use start_ref (the ref the worktree was actually based
        # on — may be origin/main when local main lags) rather than
        # self.config.main_branch, so the freshen-from-remote semantic is
        # preserved (see test_create_worktree_freshens_from_remote).
        _, mb_out, _ = await _run(
            ['git', 'merge-base', start_ref, 'HEAD'],
            cwd=worktree_path,
        )
        post_create_base = mb_out.strip() or base_sha.strip()
        port = await self._provision_reify_debug_port(worktree_path)
        return WorktreeInfo(
            path=worktree_path,
            base_commit=post_create_base,
            stale_commits=stale_commits,
            reify_debug_port=port,
        )

    async def _seed_warm_lane(self, lane_dir: Path, mode: str) -> int:
        """Run seed-warm-lane.sh to CoW-seed the lane's target/ from the warm base.

        Invokes ``<lane_dir>/scripts/seed-warm-lane.sh <base_target> <lane_dir> <mode>``
        where *base_target* is the resolved base path and *mode* is either
        ``'--fresh-checkout'`` or ``'--reset-in-place'``.

        **D8 gen-dir symlink resolution (reify activation-seam R1)**:
        When :attr:`warm_lane_base_target_path` is a symlink (e.g. ``target`` →
        ``.gen.N`` as created by ``ln -sfn .gen.N target`` in reify), the CONCRETE
        gen dir is resolved via ``base.parent / base.readlink()`` (relative-sibling
        join, NOT ``Path.resolve()``, to avoid tmp-prefix canonicalization drift).
        The resolved path is what gets passed to the script so ``cp -a --reflink``
        copies the gen dir contents rather than the symlink itself.

        For a plain directory or nonexistent path the raw path is passed unchanged
        (back-compat: default config where base == _merge-verify/target plain dir).

        The script lives in the LANE's own scripts dir (the lane's checked-out
        tree provides it, consistent with the debug-port script pattern).

        Returns:
            0   — script ran and exited 0 (seed succeeded, lane is warm).
            75  — script exited 75 (EX_TEMPFAIL, disk-pressure discriminant).
            1..N — script exited with any other non-zero code (generic fault).
            127 — script absent (command-not-found sentinel).
            127 — any unexpected exception (non-zero sentinel, never raises).

        Callers must use ``rc == 0`` for success and may inspect the exact
        code to discriminate disk-pressure (75) from generic fault (other non-zero).
        """
        try:
            script = lane_dir / 'scripts' / 'seed-warm-lane.sh'
            if not script.exists():
                logger.debug('_seed_warm_lane: seed script absent at %s', script)
                return 127  # command-not-found sentinel
            base_path = self.warm_lane_base_target_path
            if base_path.is_symlink():
                # D8: resolve relative-sibling symlink (target -> .gen.N) to the
                # concrete gen dir so the cp receives the directory, not the symlink.
                gen_dir = base_path.parent / base_path.readlink()
                base_target = str(gen_dir)
                # D8 reader-refcount GC: hold a shared lock (flock -s) on the per-gen
                # lock file for exactly the cp lifetime.  A concurrent reify GC rewrite
                # (flock -n -x) is deferred while the shared lock is held, preventing a
                # torn read (ENOENT mid-walk).  Lock auto-released on script exit.
                lock_path = gen_dir.with_name(gen_dir.name + '.lock')
                if not lock_path.exists():
                    # The lock file should be pre-created by reify alongside the gen dir.
                    # flock(1) creates it if absent, but the resulting inode is unshared
                    # with reify's exclusive locker — the GC protocol is desynced.
                    # Log at debug so a missing lock file surfaces rather than degrading
                    # silently to an ineffective lock.
                    logger.debug(
                        '_seed_warm_lane: lock file %s does not pre-exist — '
                        'flock will create a new inode unshared with reify GC; '
                        'this indicates a reify/DF gen-dir protocol mismatch',
                        lock_path,
                    )
                cmd = ['flock', '-s', str(lock_path), str(script), base_target, str(lane_dir), mode]
            else:
                base_target = str(base_path)
                cmd = [str(script), base_target, str(lane_dir), mode]
            rc, _, err = await _run(cmd, cwd=lane_dir)
            if rc != 0:
                logger.warning(
                    '_seed_warm_lane: script exited %d for %s (stderr=%r)',
                    rc, lane_dir, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_seed_warm_lane: unexpected error for %s', lane_dir, exc_info=True,
            )
            return 127  # exception sentinel (non-zero, non-75 → generic fault)

    async def refresh_warm_base(self, landed_commit: str | None = None) -> bool:
        """Advance the rolling warm base by running the refresh script.

        Invokes ``<_merge-verify>/scripts/refresh-warm-base.sh <advancing> <base>
        [--landed-commit <sha>]`` where:
        - *advancing* = ``persistent_merge_worktree_path / reap_build_artifact_dirs[0]``
          (the _merge-verify lane's target — the warm build of the just-landed commit)
        - *base* = ``warm_lane_base_target_path`` (the configured rolling base)
        - ``--landed-commit <sha>`` (D10) — appended when *landed_commit* is truthy.

        **D10 --landed-commit derivation**: when *landed_commit* is ``None`` (the
        default), it is derived from ``git rev-parse HEAD`` in the _merge-verify
        worktree (fail-soft: if rev-parse fails the flag is omitted).  The refresh
        only fires for the _merge-verify lane (merge_queue.py gate), whose HEAD IS
        the just-landed commit — the derived sha is exactly what reify's inv.9
        HEAD-match guard expects.  The optional *landed_commit* parameter allows
        deterministic test injection.

        **inv.9 promote-provenance** is enforced STRUCTURALLY: the advancing dir
        is hardcoded to the _merge-verify target and no caller-supplied advancing
        parameter is exposed, so DF can never source a task lane (un-landed WIP).
        The reify B12 guard inside refresh-warm-base.sh is defense-in-depth.

        Self-guards (all return False, never raise):
        - ``warm_lane_pool is None`` — pool knob off; refresh has no meaning.
        - ``advancing == base`` — no separate rolling base configured; degenerate
          self-copy skipped (default config has ``warm_lane_base_target_dir=None``
          so base == ``_merge-verify/target`` == advancing).
        - Script absent or non-zero exit — fail-soft.

        Returns:
            True  — script ran and exited 0 (base refreshed).
            False — any guard triggered or the script absent/failed (fail-soft;
                    never raises — a base-refresh hiccup cannot block merges).
        """
        if self.warm_lane_pool is None:
            return False

        try:
            advancing = self._merge_verify_artifact_path
            base = self.warm_lane_base_target_path

            if advancing.resolve() == base.resolve():
                logger.debug(
                    'refresh_warm_base: advancing == base (%s) — no separate rolling '
                    'base configured, skipping degenerate self-copy', advancing,
                )
                return False

            script = self.persistent_merge_worktree_path / 'scripts' / 'refresh-warm-base.sh'
            if not script.exists():
                logger.debug(
                    'refresh_warm_base: script absent at %s — no-op', script,
                )
                return False

            # D10: derive --landed-commit from HEAD of the _merge-verify worktree
            # (fail-soft: rev-parse failure omits the flag rather than blocking).
            if landed_commit is None:
                rc_h, head_out, _ = await _run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=self.persistent_merge_worktree_path,
                )
                if rc_h == 0 and head_out.strip():
                    landed_commit = head_out.strip()
                else:
                    logger.debug(
                        'refresh_warm_base: could not derive HEAD from %s (rc=%d) — '
                        'omitting --landed-commit flag',
                        self.persistent_merge_worktree_path, rc_h,
                    )

            argv = [str(script), str(advancing), str(base)]
            if landed_commit:
                argv += ['--landed-commit', landed_commit]

            rc, _, err = await _run(argv, cwd=self.persistent_merge_worktree_path,
            )
            if rc != 0:
                logger.warning(
                    'refresh_warm_base: script exited %d (stderr=%r)', rc, err,
                )
                return False
            return True
        except Exception:
            logger.warning('refresh_warm_base: unexpected error', exc_info=True)
            return False

    # ε: warm-lane disk-guard admission helpers --------------------------------

    async def _run_warm_lane_disk_guard(self) -> int:
        """Invoke ``<project_root>/scripts/warm-lane-disk-guard.sh check``.

        Mirrors the ``_seed_warm_lane``/``refresh_warm_base`` fail-soft helper
        pattern: absent script → 127 sentinel; any unexpected exception → 127;
        never raises.

        Returns:
            0   — healthy (disk pressure below threshold).
            75  — disk pressure (EX_TEMPFAIL, admission should block).
            127 — script absent or exception (fail-open sentinel).
            other non-zero — script error (treated as fail-open by caller).
        """
        try:
            script = self.project_root / 'scripts' / 'warm-lane-disk-guard.sh'
            if not script.exists():
                logger.debug(
                    '_run_warm_lane_disk_guard: script absent at %s — no-op', script,
                )
                return 127
            # NOTE: worktree_base may not exist yet on a fresh host where no lane
            # has been created.  In that case the real γ script will likely stat a
            # non-existent path and return a non-(0,75) exit code, which the caller
            # treats as fail-open (guard is an inert no-op until worktree_base first
            # appears).  This is deliberate: the guard cannot measure a volume that
            # does not yet exist, and the script-level fail-closed convention (df
            # failure → 75) only applies to volume access errors, not missing mount
            # points.  Seed the first lane to create worktree_base, then the guard
            # becomes active.
            cmd = [
                str(script), 'check',
                '--mount', str(self.worktree_base),
                '--min-free-gib', str(self.config.warm_lane_min_free_gib),
                '--min-free-inodes', str(self.config.warm_lane_min_free_inodes),
            ]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc not in (0, 75):
                logger.warning(
                    '_run_warm_lane_disk_guard: script exited %d (stderr=%r)', rc, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_warm_lane_disk_guard: unexpected error', exc_info=True,
            )
            return 127

    async def _run_warm_lane_gc_reclaim(self) -> int:
        """Invoke ``<project_root>/scripts/warm-lane-gc.sh reclaim``.

        Fail-soft: absent script → 127 sentinel; any unexpected exception → 127;
        never raises.  A non-zero exit is logged at WARNING and treated as
        'nothing reclaimed' by the caller (``_warm_lane_disk_admission_blocked``).

        Returns:
            0   — reclaim succeeded.
            127 — script absent or exception (fail-soft sentinel).
            other non-zero — reclaim script error (caller still re-checks).
        """
        try:
            script = self.project_root / 'scripts' / 'warm-lane-gc.sh'
            if not script.exists():
                logger.debug(
                    '_run_warm_lane_gc_reclaim: script absent at %s — no-op', script,
                )
                return 127
            cmd = [
                str(script), 'reclaim',
                '--mount', str(self.worktree_base),
            ]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc != 0:
                logger.warning(
                    '_run_warm_lane_gc_reclaim: script exited %d (stderr=%r)', rc, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_warm_lane_gc_reclaim: unexpected error', exc_info=True,
            )
            return 127

    async def _warm_lane_disk_admission_blocked(self) -> bool:
        """Run the ε disk-pressure admission check: check → reclaim → recheck.

        Mirrors ``merge_queue._ensure_verify_disk_space`` (check → prune → recheck →
        block) for the warm-lane acquire side.

        Logic:
            1. Run γ ``warm-lane-disk-guard.sh check``.
            2. Exit 0 or 127 or any non-(0,75) → admit (return False, fail-open).
            3. Exit 75 (disk pressure) → log WARNING, run δ ``warm-lane-gc.sh reclaim``
               (fail-soft; reclaim failure is logged but never blocks admission),
               re-run γ check.
            4. Re-check exit 75 → still pressured → block (return True).
               Re-check exit 0/other → recovered → admit (return False).

        Never raises.
        """
        rc = await self._run_warm_lane_disk_guard()
        if rc != 75:
            # 0 = healthy; 127 = absent (fail-open); other = script error (fail-open)
            return False
        # Disk pressure detected — attempt GC reclaim
        logger.warning(
            '_warm_lane_disk_admission_blocked: disk pressure detected (rc=75); '
            'invoking warm-lane-gc.sh reclaim before re-check',
        )
        await self._run_warm_lane_gc_reclaim()
        rc2 = await self._run_warm_lane_disk_guard()
        if rc2 == 75:
            logger.warning(
                '_warm_lane_disk_admission_blocked: disk pressure persists after '
                'reclaim (rc=75); blocking warm-lane admission',
            )
            return True
        return False

    async def acquire_spec_lane(
        self,
        merge_commit: str,
    ) -> tuple[Path, bool]:
        """Acquire a warm ``_spec-`` lane for a speculative merge verify (inv.8).

        **Create-once path** (lane not yet a registered worktree):
            ``git worktree add --detach <lane> <merge_commit>``, then seed via
            :meth:`_seed_warm_lane(lane, '--reset-in-place')`.

        **Reset-in-place path** (lane already registered):
            ``git reset --hard <merge_commit>`` + ONE ``git clean -xfd -e <dir>``
            invocation (one -e per :attr:`~GitConfig.reap_build_artifact_dirs`
            entry) mirroring :meth:`reset_persistent_merge_worktree`'s pattern,
            then always re-seeded via :meth:`_seed_warm_lane(lane, '--reset-in-place')`.

        **inv.8 always-re-seed-at-acquire**: target/ is re-CoW-seeded from the
        current warm base on every acquire, so each speculative verify starts
        from the latest base regardless of which lane it lands on.

        **Cold-fallback path** (pool exhausted or seed failure — inv.6):
            On any failure (pool exhausted, worktree creation failure, seed
            failure) the partially-acquired lane (if any) is released back to
            FREE and an ephemeral cold worktree is returned with warm=False.
            Logs at DEBUG; never raises (a fallback cannot block the scheduler).

        Args:
            merge_commit: The merge commit SHA to check out in the spec lane.

        Returns:
            ``(lane_path, True)`` when the warm spec lane is ready;
            ``(cold_path, False)`` on pool exhaustion or any failure (inv.6).
        """
        if self.spec_warm_lane_pool is None:
            # Knob off or size=0 — always cold (byte-identical to default).
            logger.debug(
                'acquire_spec_lane: spec pool absent (knob off) — cold fallback '
                'for %s', merge_commit[:8],
            )
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        lane = await self.spec_warm_lane_pool.try_acquire()
        if lane is None:
            # Pool exhausted — inv.6: cold ephemeral fallback, never block.
            logger.debug(
                'acquire_spec_lane: pool exhausted — cold fallback for %s',
                merge_commit[:8],
            )
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        # ── Create-once or reset-in-place ────────────────────────────────────
        if not await self._is_registered_worktree(lane):
            # Create-once: serialized via _spec_wt_create_lock so that the K>1
            # initial warm-up burst does not issue concurrent `git worktree add`
            # calls against the same repo-level git lock (which would cause
            # transient failures and a burst of cold fallbacks at warm-up time).
            # Reset-in-place acquires (already-registered path below) are
            # per-lane and don't contend, so no lock is needed there.
            async with self._spec_wt_create_lock:
                # Pool ownership is per-lane exclusive (try_acquire assigns a
                # unique lane to exactly one caller), so no double-check is needed
                # here — we're the only caller that can be creating this lane.
                # The lock purely serializes the repo-level git worktree add
                # against other lanes' concurrent first-time creates.
                # Self-heal a stale unregistered directory first
                # (mirrors reset_persistent_merge_worktree's self-heal pattern).
                if lane.exists():
                    import shutil as _shutil
                    _shutil.rmtree(lane)
                lane.parent.mkdir(parents=True, exist_ok=True)
                rc, _, err = await _run(
                    ['git', 'worktree', 'add', '--detach', str(lane), merge_commit],
                    cwd=self.project_root,
                )
                if rc != 0:
                    logger.warning(
                        'acquire_spec_lane: worktree add failed for %s (rc=%d,'
                        ' err=%r) — releasing lane, cold fallback', lane, rc, err,
                    )
                    await self.spec_warm_lane_pool.release(lane)
                    wt = await self.create_throwaway_verify_worktree(merge_commit)
                    return wt, False
                logger.info(
                    'acquire_spec_lane: created %s (HEAD=%s)', lane, merge_commit[:8],
                )
        else:
            # Reset-in-place: mirror reset_persistent_merge_worktree's exact sequence
            # (git reset --hard + ONE git clean with all -e flags in a single pass).
            rc, _, err = await _run(
                ['git', 'reset', '--hard', merge_commit],
                cwd=lane,
            )
            if rc != 0:
                logger.debug(
                    'acquire_spec_lane: reset --hard failed for %s (rc=%d, err=%r)'
                    ' — releasing lane, cold fallback', lane, rc, err,
                )
                await self.spec_warm_lane_pool.release(lane)
                wt = await self.create_throwaway_verify_worktree(merge_commit)
                return wt, False
            clean_cmd = ['git', 'clean', '-xfd']
            for artifact_dir in self.config.reap_build_artifact_dirs:
                clean_cmd += ['-e', artifact_dir]
            rc, _, err = await _run(clean_cmd, cwd=lane)
            if rc != 0:
                logger.debug(
                    'acquire_spec_lane: git clean failed for %s (rc=%d, err=%r)'
                    ' — releasing lane, cold fallback', lane, rc, err,
                )
                await self.spec_warm_lane_pool.release(lane)
                wt = await self.create_throwaway_verify_worktree(merge_commit)
                return wt, False
            logger.info(
                'acquire_spec_lane: reset %s to HEAD=%s', lane, merge_commit[:8],
            )

        # ── inv.8: ALWAYS re-seed target/ from the current warm base ─────────
        rc = await self._seed_warm_lane(lane, '--reset-in-place')
        if rc != 0:
            # Seed failed — release the partially-acquired lane back to FREE
            # (inv.6: cold fallback on seed failure, never block scheduler).
            logger.debug(
                'acquire_spec_lane: seed failed for %s (rc=%d) — releasing lane, cold fallback',
                lane, rc,
            )
            await self.spec_warm_lane_pool.release(lane)
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        return lane, True

    async def release_spec_lane(self, lane: Path, *, warm: bool) -> None:
        """Release a ``_spec-`` lane after a speculative merge verify.

        **Warm path** (``warm=True`` — pool lane):
            ``await self.spec_warm_lane_pool.release(lane)`` — flips
            ASSIGNED→FREE retaining the worktree and ``target/`` on disk so
            the next :meth:`acquire_spec_lane` can re-seed from a warm base
            (CoW-cheap, harmless to retain).  The worktree is never removed.

        **Cold path** (``warm=False`` — ephemeral fallback worktree):
            ``await self.cleanup_merge_worktree(lane)`` — removes the
            throwaway ``_merge-<uuid>`` worktree created by
            :meth:`create_throwaway_verify_worktree`.

        **Idempotent**: releasing a FREE lane, an already-cleaned path, or an
        unknown path is a no-op (never raises).  Mirrors the
        ``release_warm_lane`` best-effort / never-raise contract so a hiccup
        cannot strand the scheduler.

        Args:
            lane: Path returned by :meth:`acquire_spec_lane`.
            warm: True when *lane* is a warm ``_spec-`` pool lane; False when
                it is a cold ephemeral fallback worktree.
        """
        try:
            if warm and self.spec_warm_lane_pool is not None:
                await self.spec_warm_lane_pool.release(lane)
                logger.debug('release_spec_lane: released warm lane %s', lane)
            else:
                await self.cleanup_merge_worktree(lane)
                logger.debug('release_spec_lane: cleaned up cold lane %s', lane)
        except Exception:
            logger.warning(
                'release_spec_lane: error releasing %s (warm=%s)',
                lane, warm, exc_info=True,
            )

    async def acquire_warm_lane(
        self,
        branch_name: str,
        start_ref: str,
        *,
        expected_title: str | None = None,
    ) -> 'WorktreeInfo | WarmLaneUnavailable':
        """Allocate a FREE warm lane, seed/reset it, and return a WorktreeInfo.

        **Create-once path** (lane not yet a registered worktree):
            ``git worktree add -b task/<branch_name> <lane> <start_ref>``,
            then seed via :meth:`_seed_warm_lane(lane, '--fresh-checkout')`.

        **Reset-in-place path** (lane already registered — added in step-10):
            Handled by :meth:`_reset_warm_lane` then always re-seeded from the
            current base via ``_seed_warm_lane(lane, '--fresh-checkout')`` (D10
            always-re-seed-at-acquire).  The lane is the ``§9.5`` within-assignment
            reset_lane primitive; the re-seed layers on top to deliver at-head
            warmth for the NEW task.

        **Identity guard** (``expected_title``, step-26):
            When *expected_title* is supplied, any reuse candidate (in-memory
            map hit *or* on-disk plan.json match) has its stored title checked
            via :func:`identities_match`.  On mismatch the stale assignment is
            dropped from the pool and the lane is reset in-place (fresh path) —
            so a recycled-id task never inherits the prior task's ``.task/``
            state.  ``expected_title=None`` (default) disables the guard and
            all existing callers/tests are unaffected.

        **ε pre-acquire disk-guard** (``config.warm_lane_disk_guard``, task-1860):
            When the knob is True, runs γ ``warm-lane-disk-guard.sh check`` →
            on exit 75 (EX_TEMPFAIL/disk pressure) invokes δ ``warm-lane-gc.sh
            reclaim`` to free stale capacity → re-checks.  If STILL pressured,
            returns ``WarmLaneUnavailable.DISK_PRESSURE`` so
            :meth:`create_worktree` raises :class:`WarmLaneDiskPressure` →
            workflow requeues (exit-75) instead of proceeding into an ENOSPC
            build.  Runs BEFORE :func:`acquire_for` so all idle lanes remain
            FREE for δ's reclaim.  Fail-open on absent scripts (rc 127): no
            guard / nothing reclaimed — byte-identical to today until reify
            γ/δ are deployed.

        Returns:
            WorktreeInfo  — success; lane is ASSIGNED and seeded.
            WarmLaneUnavailable.EXHAUSTED — all pool lanes are ASSIGNED
                (backpressure; caller should requeue).
            WarmLaneUnavailable.FAULT — seed/worktree-add failure or absent
                seed script (infra fault; caller should escalate blocked+L1).
            WarmLaneUnavailable.DISK_PRESSURE — pre-acquire ε disk-guard
                detected persistent pressure (rc=75 after reclaim), OR seed
                exited 75 (EX_TEMPFAIL); transient disk pressure (caller should
                requeue with annotation).
            WarmLaneUnavailable.DISABLED — pool knob is off; this is a
                programming error (callers MUST guard with
                ``if self.warm_lane_pool is not None`` before calling).
                Returned rather than raising so the function contract
                "never raises internally" holds; callers that receive
                DISABLED must NOT requeue — they should fall back to the
                cold path or raise immediately.

            Never raises internally; the lane is always released before
            returning any WarmLaneUnavailable value.
        """
        if self.warm_lane_pool is None:
            # Programming error: callers must guard with
            # `if self.warm_lane_pool is not None` before invoking.
            # A disabled pool is NOT 'backpressure' — returning EXHAUSTED
            # here would silently requeue a task forever if a future caller
            # skips the guard.  DISABLED is a distinct sentinel so callers
            # that don't handle it hit an explicit error path rather than
            # infinite requeue.
            return WarmLaneUnavailable.DISABLED

        # ε: pre-acquire disk-guard (check → reclaim → recheck → DISK_PRESSURE/exit-75).
        # Running BEFORE acquire_for keeps all idle lanes FREE so δ's reclaim can reset
        # them.  On still-pressured (rc=75 after reclaim), return DISK_PRESSURE so
        # create_worktree raises WarmLaneDiskPressure → workflow requeues as transient
        # infra (exit-75) instead of proceeding into an ENOSPC build that SIGBUSes the
        # linker.  Fail-open (absent script rc=127) → byte-identical to today.
        #
        # Concurrency note: the guard + δ reclaim run WITHOUT holding warm_lane_pool
        # _lock.  This is intentional — serializing on the pool lock would prevent
        # concurrent acquires from even attempting during a reclaim.  The safety
        # invariant is delegated to the γ/δ scripts: warm-lane-gc.sh MUST only reset
        # lanes that have been idle beyond a configurable safety threshold (e.g. no
        # active acquire_for in flight for those lanes).  Because acquire_for itself
        # holds the pool lock for the registration step, a lane that is mid-acquire
        # will be ASSIGNED in pool state and δ must skip ASSIGNED lanes.  Until reify
        # δ (task-4717) ships with that guarantee documented, the guard is fail-open
        # (rc 127) and the window is a no-op; document it here so the δ author knows
        # the expected contract.
        if self.config.warm_lane_disk_guard and await self._warm_lane_disk_admission_blocked():
            return WarmLaneUnavailable.DISK_PRESSURE

        acq = await self.warm_lane_pool.acquire_for(branch_name)
        if acq is None:
            return WarmLaneUnavailable.EXHAUSTED  # Pool exhausted → backpressure
        lane, reused = acq

        full_branch = f'{self.config.branch_prefix}{branch_name}'

        try:
            if reused:
                # ── Reuse path: live requeue of same task on same lane ────
                # Identity guard: if expected_title is set, verify the stored
                # title matches before reusing .task/plan.json + WIP.
                # Fail-open: identities_match returns True when either side is
                # empty, so a title-less lane always reuses as before.
                _ident_ok = (
                    expected_title is None
                    or identities_match(read_worktree_title(lane), expected_title)
                )
                if _ident_ok:
                    # .task/plan.json is preserved; WIP is committed + rebased.
                    return await self._reuse_warm_lane(lane, full_branch)
                # Mismatch: stale assignment from a recycled id — drop it and
                # reset in-place so the new task starts clean.
                logger.warning(
                    'acquire_warm_lane: in-memory reuse identity MISMATCH for '
                    '%s — expected %r; running fresh reset',
                    lane, expected_title,
                )
                self.warm_lane_pool.drop_assignment(branch_name)
                await self._reset_warm_lane(lane, full_branch, start_ref)
                # β: THIN re-seed — rm target/ before seeding so the re-seed is a
                # fresh CoW copy rather than an in-place overlay on stale blobs.
                # Defensive rmtree is robust even if reify α's replace-capable seed
                # is not yet landed (an emptied target/ seeds cleanly either way).
                shutil.rmtree(lane / 'target', ignore_errors=True)
                rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if rc != 0:
                    if rc == 127:
                        logger.warning(
                            'acquire_warm_lane: recycle re-seed — seed script absent '
                            'for lane %s (rc=127) — check seed-warm-lane.sh '
                            'deployment; EVERY task on this host will fault while '
                            'pool is enabled and the script is missing',
                            lane,
                        )
                    else:
                        logger.warning(
                            'acquire_warm_lane: recycle re-seed failed (rc=%d) for '
                            '%s; releasing lane',
                            rc, lane,
                        )
                    await self.warm_lane_pool.release(lane)
                    return (
                        WarmLaneUnavailable.DISK_PRESSURE if rc == 75
                        else WarmLaneUnavailable.FAULT
                    )
                # Falls through to shared tail

            elif not await self._is_registered_worktree(lane):
                # ── Create-once branch ────────────────────────────────────
                # Self-heal: remove a stale unregistered directory so
                # git worktree add doesn't refuse a non-empty dir.
                if lane.exists():
                    logger.warning(
                        'acquire_warm_lane: lane %s exists but is not registered; '
                        'removing stale directory (self-heal)', lane,
                    )
                    shutil.rmtree(lane)
                lane.parent.mkdir(parents=True, exist_ok=True)

                git_add_rc, _, err = await _run(
                    ['git', 'worktree', 'add', '-b', full_branch, str(lane), start_ref],
                    cwd=self.project_root,
                )
                if git_add_rc != 0:
                    logger.warning(
                        'acquire_warm_lane: git worktree add failed for %s: %s', lane, err,
                    )
                    await self.warm_lane_pool.release(lane)
                    return WarmLaneUnavailable.FAULT

                seed_rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if seed_rc != 0:
                    # Seed failed — remove the just-created worktree and release.
                    # rc=127 specifically means seed-warm-lane.sh is absent from
                    # the lane's scripts/ dir — a deploy/config error that will
                    # affect EVERY task on this host while the pool is enabled,
                    # producing one BLOCKED+L1 escalation per dispatched task.
                    # Operators should check that seed-warm-lane.sh is present
                    # and executable in the lane's checked-out scripts/ directory.
                    if seed_rc == 127:
                        logger.warning(
                            'acquire_warm_lane: seed script absent for lane %s '
                            '(rc=127) — check seed-warm-lane.sh deployment; '
                            'EVERY task on this host will fault while pool is '
                            'enabled and the script is missing',
                            lane,
                        )
                    else:
                        logger.warning(
                            'acquire_warm_lane: seed failed (rc=%d) for %s; '
                            'removing worktree',
                            seed_rc, lane,
                        )
                    await _run(
                        ['git', 'worktree', 'remove', str(lane), '--force'],
                        cwd=self.project_root,
                    )
                    await self.warm_lane_pool.release(lane)
                    return (
                        WarmLaneUnavailable.DISK_PRESSURE if seed_rc == 75
                        else WarmLaneUnavailable.FAULT
                    )
            else:
                # ── Already-registered lane — check on-disk backstop first ─
                # If the lane still carries THIS task's plan.json (e.g. after a
                # process restart that cleared the in-memory _assignments map),
                # treat it as a REUSE: restore the assignment and route to
                # _reuse_warm_lane so .task/plan.json + WIP are preserved.
                # Identity guard: if expected_title is set and the stored title
                # does not match, treat as fresh (recycled-id guard).
                # Any read/parse error falls safe toward the fresh reset path.
                disk_reuse = False
                try:
                    plan_path = lane / '.task' / 'plan.json'
                    if plan_path.exists():
                        import json as _json
                        data = _json.loads(plan_path.read_text())
                        if data.get('task_id') == branch_name:
                            # Check identity guard before declaring reuse
                            _disk_ident_ok = (
                                expected_title is None
                                or identities_match(
                                    read_worktree_title(lane), expected_title,
                                )
                            )
                            if _disk_ident_ok:
                                # Record the mapping and route to reuse
                                self.warm_lane_pool.note_assignment(branch_name, lane)
                                disk_reuse = True
                            else:
                                logger.warning(
                                    'acquire_warm_lane: disk backstop identity '
                                    'MISMATCH for %s — expected %r; fresh reset',
                                    lane, expected_title,
                                )
                except Exception:
                    # Fail safe: unreadable or non-JSON plan → treat fresh
                    pass

                if disk_reuse:
                    return await self._reuse_warm_lane(lane, full_branch)

                # ── Fresh reset-in-place (new task on a recycled FREE lane) ─
                await self._reset_warm_lane(lane, full_branch, start_ref)
                # β: THIN re-seed — rm target/ before seeding (no retained bloat).
                # Defensive rmtree is robust even if reify α's replace-capable seed
                # is not yet landed (an emptied target/ seeds cleanly either way).
                shutil.rmtree(lane / 'target', ignore_errors=True)
                rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if rc != 0:
                    if rc == 127:
                        logger.warning(
                            'acquire_warm_lane: reset-in-place re-seed — seed script '
                            'absent for lane %s (rc=127) — check seed-warm-lane.sh '
                            'deployment; EVERY task on this host will fault while '
                            'pool is enabled and the script is missing',
                            lane,
                        )
                    else:
                        logger.warning(
                            'acquire_warm_lane: recycle re-seed failed (rc=%d) for '
                            '%s; releasing lane',
                            rc, lane,
                        )
                    await self.warm_lane_pool.release(lane)
                    return (
                        WarmLaneUnavailable.DISK_PRESSURE if rc == 75
                        else WarmLaneUnavailable.FAULT
                    )

            # ── Shared tail: gitignore, scrub, base, debug-port ──────────
            _ensure_task_gitignore(lane)
            await scrub_task_dir_from_tree(lane, 'warm-lane-acquire', amend=False)

            _, mb_out, _ = await _run(
                ['git', 'merge-base', start_ref, 'HEAD'],
                cwd=lane,
            )
            _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
            base_commit = mb_out.strip() or head_sha.strip()

            port = await self._provision_reify_debug_port(lane)
            logger.info(
                'acquire_warm_lane: acquired %s on branch %s (base=%s, port=%s)',
                lane, full_branch, base_commit[:8] if base_commit else '?', port,
            )
            return WorktreeInfo(path=lane, base_commit=base_commit, reify_debug_port=port)

        except Exception:
            logger.warning(
                'acquire_warm_lane: unexpected error for %s; releasing', lane, exc_info=True,
            )
            await self.warm_lane_pool.release(lane)
            return WarmLaneUnavailable.FAULT

    async def _reuse_warm_lane(
        self, lane_dir: Path, full_branch: str,
    ) -> 'WorktreeInfo':
        """Handle a live-requeue: same task on the same lane (in-memory map hit).

        Mirrors the cold-requeue reuse block (git_ops.py ~806-858):
        1. Commit any uncommitted WIP so it is preserved across the rebase.
        2. Rebase onto main (best-effort; log failure and continue on old base).
        3. Re-ensure the task .gitignore.
        4. Recompute ``base_commit`` as ``merge-base main HEAD``.
        5. Re-provision the debug port (inv.7).

        ``.task/plan.json`` is NOT committed by ``commit()`` (the
        ``:!.task`` pathspec excludes it) and survives the rebase intact
        because git rebase only touches tracked files.

        Returns:
            WorktreeInfo for the reused lane.  Never raises — exceptions
            propagate to the caller's try/except-release wrapper.
        """
        # 1. Commit WIP (returns None if nothing to commit — that's fine)
        await self.commit(lane_dir, 'chore: save WIP before requeue rebase')

        # 2. Rebase onto main (best-effort; failure leaves the branch on old base)
        rebased = await self.rebase_onto_main(lane_dir)
        if not rebased:
            logger.info(
                '_reuse_warm_lane: rebase failed for %s; continuing on old base', lane_dir,
            )

        # 3. Re-ensure task .gitignore (idempotent)
        _ensure_task_gitignore(lane_dir)

        # 4. Recompute base: merge-base between main_branch and HEAD
        _, mb_out, _ = await _run(
            ['git', 'merge-base', self.config.main_branch, 'HEAD'],
            cwd=lane_dir,
        )
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane_dir)
        base_commit = mb_out.strip() or head_sha.strip()

        # 5. Re-provision debug port (inv.7)
        port = await self._provision_reify_debug_port(lane_dir)

        logger.info(
            '_reuse_warm_lane: reused %s on branch %s (base=%s, port=%s)',
            lane_dir, full_branch, base_commit[:8] if base_commit else '?', port,
        )
        return WorktreeInfo(path=lane_dir, base_commit=base_commit, reify_debug_port=port)

    async def _reset_warm_lane(
        self, lane_dir: Path, full_branch: str, target_commit: str,
    ) -> None:
        """Reset an already-registered lane to *target_commit* on *full_branch*.

        Implements reset-determinism (inv.1) + warmth retention:
        ``git checkout -B <full_branch> <target_commit>`` establishes the new
        task branch and updates the tracked tree; then a SINGLE
        ``git clean -xfd -e <dir>`` (one -e per reap_build_artifact_dirs)
        removes stray untracked files while retaining all artifact dirs —
        mirroring reset_persistent_merge_worktree's single-pass clean so
        >1 artifact dir all survive (per-dir-loop bug from κ, step-19).

        Added as a stub here (step-8); fully exercised by step-10 tests.
        """
        # -f (force) discards local modifications to tracked files before the
        # branch switch — required when the lane is being reused from a prior
        # task that left uncommitted work.  -B creates or resets the branch.
        rc, _, err = await _run(
            ['git', 'checkout', '-f', '-B', full_branch, target_commit],
            cwd=lane_dir,
        )
        if rc != 0:
            raise RuntimeError(
                f'_reset_warm_lane: checkout -f -B {full_branch} {target_commit} '
                f'failed for {lane_dir}: {err}'
            )
        # Single invocation excluding ALL artifact dirs at once — every
        # configured build-output dir survives in one pass.  A per-dir loop
        # would leave NONE surviving with >1 dir (κ step-19 regression).
        clean_cmd = ['git', 'clean', '-xfd']
        for artifact_dir in self.config.reap_build_artifact_dirs:
            clean_cmd += ['-e', artifact_dir]
        rc, _, err = await _run(clean_cmd, cwd=lane_dir)
        if rc != 0:
            raise RuntimeError(
                f'_reset_warm_lane: git clean failed for {lane_dir}: {err}'
            )

    async def release_warm_lane(self, lane_dir: Path, branch_name: str) -> None:
        """Release a warm lane back to the FREE pool.

        Steps:
        1. ``git -C <lane> checkout --detach`` — detach HEAD so the just-used
           branch is deletable while the lane directory stays in place.
        2. ``git branch -D task/<branch_name>`` — best-effort; logs on failure.
        3. ``await self.warm_lane_pool.release(lane_dir)`` — flip ASSIGNED→FREE.

        Never removes the worktree; ``target/`` is left in place incidentally
        (CoW-cheap, harmless).  The *next* :meth:`acquire_warm_lane` always
        re-seeds from the current base (D10), so a released lane's target/
        drift is irrelevant — retention is no longer a contract promise.

        Fully best-effort / never-raise (mirrors ``cleanup_merge_worktree``
        contract) so a hiccup cannot strand the scheduler.
        """
        if self.warm_lane_pool is None:
            return

        full_branch = f'{self.config.branch_prefix}{branch_name}'
        try:
            # Detach HEAD so the task branch can be deleted while the lane
            # remains checked out warm (target/ survives).
            rc, _, err = await _run(
                ['git', 'checkout', '--detach'],
                cwd=lane_dir,
            )
            if rc != 0:
                logger.warning(
                    'release_warm_lane: checkout --detach failed for %s: %s', lane_dir, err,
                )
        except Exception:
            logger.warning(
                'release_warm_lane: checkout --detach error for %s', lane_dir, exc_info=True,
            )

        try:
            rc, _, err = await _run(
                ['git', 'branch', '-D', full_branch],
                cwd=self.project_root,
            )
            if rc != 0:
                logger.warning(
                    'release_warm_lane: branch -D %s failed: %s', full_branch, err,
                )
        except Exception:
            logger.warning(
                'release_warm_lane: branch -D error for %s', full_branch, exc_info=True,
            )

        await self.warm_lane_pool.release(lane_dir)
        logger.info('release_warm_lane: released %s (branch %s)', lane_dir, full_branch)

    async def release_lane_for_terminal_task(
        self,
        task_id_or_branch: str,
        *,
        allow_disk_backstop: bool = False,
    ) -> bool:
        """Idempotent, never-raise primitive — release the warm lane assigned to a terminal task.

        Shared by all terminal-exit paths (B1/B2/B3 event wiring + A periodic
        reconciler) so that B+A double-fire is a harmless no-op.

        Resolution order:
        1. Strip ``config.branch_prefix`` to get the bare task id.
        2. In-memory: ``pool.assignment_for(bare_id)`` — O(1), covers the
           common live-process path.  This is the ONLY resolution used when
           ``allow_disk_backstop=False`` (the default).  When the in-memory
           lookup returns None the primitive returns ``False`` immediately —
           a true no-op (no disk scan, no redundant ``cleanup_worktree`` /
           ``git branch -D`` retry).
        3. On-disk backstop: ``_find_lane_by_plan_task_id(bare_id)`` — scans
           ``worktree_base/_lane-*/.task/plan.json``; used ONLY when
           ``allow_disk_backstop=True`` (reserved for the lost-map /
           post-restart path: :meth:`_mark_in_progress_done`).  A theft guard
           checks whether the resolved lane has since been re-acquired by a
           different live task; if so, the release is refused (returns False)
           to prevent stealing a live task's lane via a stale plan.json.

        If neither resolves (or the theft guard refuses), returns ``False``
        (no lane to free — already free or not assigned to this task).
        Otherwise routes through :meth:`cleanup_worktree` (pool-aware: calls
        :meth:`release_warm_lane` for warm lanes) and returns ``True``.

        All exceptions are caught and logged; never raises.

        Args:
            task_id_or_branch: Task id or full branch name (prefix stripped).
            allow_disk_backstop: When True, fall back to scanning ``plan.json``
                files on disk if the in-memory assignment map has no entry.
                Defaults to False — callers on B1/B2/B3/A paths must NOT
                pass this; it is reserved for ``_mark_in_progress_done``
                (T9 lost-map / post-restart path).

        Returns:
            ``True`` if a lane was found and freed, ``False`` otherwise.
        """
        if self.warm_lane_pool is None:
            return False

        # Strip the branch prefix so we work with the bare task id throughout
        prefix = self.config.branch_prefix  # e.g. 'task/'
        bare_id = (
            task_id_or_branch[len(prefix):]
            if task_id_or_branch.startswith(prefix)
            else task_id_or_branch
        )

        # Resolve lane via in-memory assignment first
        lane = self.warm_lane_pool.assignment_for(bare_id)

        # Optionally fall back to the on-disk plan.json backstop (opt-in only)
        if lane is None and allow_disk_backstop:
            disk_lane = self._find_lane_by_plan_task_id(bare_id)
            if disk_lane is not None:
                # Theft guard: check if the disk-resolved lane has since been
                # re-acquired by a DIFFERENT live task (stale plan.json window).
                # Use assignments_snapshot() — no lock needed (single event loop).
                snap = self.warm_lane_pool.assignments_snapshot()
                holder = next(
                    (br for br, ln in snap.items() if ln == disk_lane), None
                )
                if holder is not None and holder != bare_id:
                    logger.warning(
                        'release_lane_for_terminal_task: theft guard refused — '
                        'disk-resolved lane %s for task %r is now held by %r',
                        disk_lane, bare_id, holder,
                    )
                    disk_lane = None
            lane = disk_lane

        if lane is None:
            logger.debug(
                'release_lane_for_terminal_task: no lane found for %r — already free or unknown',
                bare_id,
            )
            return False

        try:
            await self.cleanup_worktree(lane, bare_id)
            logger.info(
                'release_lane_for_terminal_task: released lane %s for task %r',
                lane, bare_id,
            )
            return True
        except Exception:
            logger.warning(
                'release_lane_for_terminal_task: cleanup error for %s task %r',
                lane, bare_id, exc_info=True,
            )
            return False

    def _find_lane_by_plan_task_id(self, task_id: str) -> Path | None:
        """Scan ``worktree_base/_lane-*/.task/plan.json`` for *task_id*.

        On-disk backstop for :meth:`release_lane_for_terminal_task`: used when
        ``pool.assignment_for`` returns None (e.g. post-restart where the
        in-memory assignment map was not rebuilt for a terminal task).

        Uses a local ``import json as _json`` (mirrors git_ops.py:1816 idiom —
        no module-level ``import json``).  Hoisted once above the loop rather
        than repeated per entry (import is cached after first call, but
        re-executing the statement inside the loop is needlessly noisy).
        Catches ``(ValueError, OSError)`` per corrupt/missing plan; never raises.

        Returns the lane ``Path`` whose ``plan.json`` carries
        ``str(plan.get('task_id')) == task_id``, or ``None`` if not found.
        """
        pool = self.warm_lane_pool
        if pool is None:
            return None
        base = self.worktree_base
        try:
            if not base.exists():
                return None
            entries = list(base.iterdir())
        except OSError:
            return None
        import json as _json  # local-import idiom; hoisted above loop
        for entry in entries:
            if not entry.is_dir():
                continue
            if not pool.is_lane(entry):
                continue
            try:
                plan_path = entry / '.task' / 'plan.json'
                if not plan_path.exists():
                    continue
                data = _json.loads(plan_path.read_text())
                if str(data.get('task_id')) == task_id:
                    return entry
            except (ValueError, OSError):
                continue
        return None

    async def _provision_reify_debug_port(self, worktree_path: Path) -> int | None:
        """Run setup-worktree-debug-port.sh in the worktree and return the allocated port.

        Best-effort and fail-open: returns None on any miss or failure so
        worktree creation is never blocked by a debug-port hiccup.

        **Idempotency contract**: This helper is invoked on *both* the
        fresh-create and reuse/requeue return paths of ``create_worktree``.
        On reuse the script is re-run to re-acquire a free port and re-patch
        ``<worktree>/.mcp.json``.  The script (``scripts/setup-worktree-
        debug-port.sh`` in the provisioned worktree) is therefore expected to
        be idempotent with respect to the worktree directory — successive calls
        for the same worktree must return the same port rather than allocating
        a new one each time.  If the script is not idempotent it may churn
        (leak) ports across requeues; the existence guard and ``try/except``
        wrapper below ensure this function itself is always safe to call, but
        port stability is the script's responsibility.
        """
        try:
            script = worktree_path / 'scripts' / 'setup-worktree-debug-port.sh'
            if not script.exists():
                return None
            rc, out, err = await _run([str(script), str(worktree_path)], cwd=worktree_path)
            if rc != 0:
                logger.warning(
                    '_provision_reify_debug_port: script exited %d for %s (stderr=%r)',
                    rc, worktree_path, err,
                )
                return None
            lines = [line for line in out.splitlines() if line.strip()]
            return int(lines[-1])
        except (ValueError, IndexError):
            logger.warning(
                '_provision_reify_debug_port: could not parse port from stdout for %s',
                worktree_path,
            )
            return None
        except Exception:
            logger.warning(
                '_provision_reify_debug_port: unexpected error for %s',
                worktree_path, exc_info=True,
            )
            return None

    async def _worktree_holding_branch(self, full_branch: str) -> Path | None:
        """Path of the registered worktree that has *full_branch* checked out.

        Returns ``None`` when no worktree holds it (a dangling ref) or when
        ``git worktree list`` errors.  Callers treat ``None`` conservatively —
        a dangling ref has no working tree to be dirty, so only commits-beyond-
        main can carry work, and that is checked separately and fail-safe.
        """
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=self.project_root,
        )
        if rc != 0:
            return None
        target = f'refs/heads/{full_branch}'
        current: Path | None = None
        for line in out.splitlines():
            if line.startswith('worktree '):
                current = Path(line[len('worktree '):].strip())
            elif line.startswith('branch ') and line[len('branch '):].strip() == target:
                return current
        return None

    async def _branch_has_commits_beyond_main(self, full_branch: str) -> bool:
        """Whether *full_branch* carries commits beyond main.

        **Fail-safe ``True``** on any git error or unparseable output — never
        report a branch as empty (safe to delete) when we cannot prove it.
        """
        rc, out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{self.config.main_branch}..{full_branch}'],
            cwd=self.project_root,
        )
        if rc != 0:
            return True
        try:
            return int(out.strip()) > 0
        except ValueError:
            return True

    async def _cleanup_leftover_branch(
        self, full_branch: str, branch_name: str,
    ) -> None:
        """Remove a leftover branch ref ONLY when provably non-destructive.

        Called by ``create_worktree`` when ``full_branch`` already exists.
        Raises :class:`RuntimeError` (→ the task blocks with an L1, now
        non-stranding via Harness Fix #1a) rather than deleting anything when
        the leftover carries commits beyond main, has a dirty working tree, or
        its state cannot be verified.  Hard rule: never destroy WIP / orphan
        commits — escalate when not deterministically certain.
        """
        holding = await self._worktree_holding_branch(full_branch)
        if holding is not None and holding.exists():
            # Branch is checked out in a live tree — full check (commits beyond
            # main OR dirty working tree, fail-safe True on any error).
            unsafe = await self.worktree_has_unsaved_work(holding, branch_name)
        else:
            # Dangling ref, or a worktree admin entry whose directory is gone
            # (e.g. rmtree'd above but still tracked by git — the 3576 shape):
            # no live working tree to be dirty, so only commits-beyond-main
            # can carry work.
            unsafe = await self._branch_has_commits_beyond_main(full_branch)

        if unsafe:
            raise RuntimeError(
                f'create_worktree: refusing to delete leftover branch '
                f'{full_branch!r} — it carries commits beyond '
                f'{self.config.main_branch}, has uncommitted changes, or its '
                f'state could not be verified (fail-safe). This would destroy '
                f'work. Inspect it and, once any wanted work is preserved, '
                f'remove it manually: '
                f'`git worktree remove --force <path>` (if checked out) then '
                f'`git branch -D {full_branch}`.'
            )

        # Provably empty AND clean → safe to remove.  Clear any worktree (and
        # its admin entry) holding the branch first, else `git branch -D` fails
        # with "branch is checked out" (the silent-failure that caused 3576).
        if holding is not None:
            rc_rm, _, err_rm = await _run(
                ['git', 'worktree', 'remove', '--force', str(holding)],
                cwd=self.project_root,
            )
            if rc_rm != 0:
                logger.warning(
                    'create_worktree: `git worktree remove` for leftover %s '
                    'failed (rc=%d): %s — pruning admin entries and retrying '
                    'branch delete', holding, rc_rm, err_rm.strip(),
                )
            await _run(['git', 'worktree', 'prune'], cwd=self.project_root)

        rc_del, _, err_del = await _run(
            ['git', 'branch', '-D', full_branch], cwd=self.project_root,
        )
        if rc_del != 0:
            raise RuntimeError(
                f'create_worktree: failed to delete provably-empty leftover '
                f'branch {full_branch!r} (rc={rc_del}): {err_del.strip()}. It '
                f'may still be checked out in a worktree; remove that worktree '
                f'first (`git worktree list` to find it).'
            )

        # Re-verify the ref is actually gone before `git worktree add` collides.
        rc_chk, _, _ = await _run(
            ['git', 'rev-parse', '--verify', full_branch], cwd=self.project_root,
        )
        if rc_chk == 0:
            raise RuntimeError(
                f'create_worktree: leftover branch {full_branch!r} still '
                f'present after `git branch -D`; aborting rather than colliding '
                f'on `git worktree add`.'
            )
        logger.info(
            'create_worktree: removed provably-empty leftover branch %s '
            '(no commits beyond main, clean/no working tree)', full_branch,
        )

    async def commit(self, worktree: Path, message: str) -> str | None:
        """Stage all changes and commit. Returns sha or None if nothing to commit.

        The :!.task pathspec SHOULD prevent .task/ from being staged, but
        agents can (and have) staged .task/ files via direct git commands
        before this method runs.  The post-staging check catches that case.
        """
        # Stage all — :!.task excludes .task/ from staging
        await _run(['git', 'add', '-A', '--', '.', ':!.task', ':!.claude'], cwd=worktree)

        # ── Post-staging .task/ safety net ────────────────────────────
        # If .task/ files are staged (e.g. an agent ran "git add .task/"
        # before we got here), unstage them.  This is a belt-and-braces
        # check — the pathspec above should handle it, but agents bypass it.
        rc, staged_task, _ = await _run(
            ['git', 'diff', '--cached', '--name-only', '--', '.task/'],
            cwd=worktree,
        )
        if rc == 0 and staged_task.strip():
            logger.warning(
                '.task/ CONTAMINATION caught in commit() — %d file(s) were staged '
                'despite :!.task pathspec (an agent likely ran "git add .task/" directly). '
                'Unstaging now: %s',
                len(staged_task.strip().splitlines()),
                staged_task.strip()[:200],
            )
            await _run(['git', 'reset', 'HEAD', '--', '.task/'], cwd=worktree)

        # Check for changes
        rc, _, _ = await _run(['git', 'diff', '--cached', '--quiet'], cwd=worktree)
        if rc == 0:
            return None  # nothing staged

        rc, out, err = await _run(['git', 'commit', '-m', message], cwd=worktree)
        if rc != 0:
            raise RuntimeError(f'Commit failed: {err}')

        # Get sha
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return sha

    async def get_diff_from_main(self, worktree: Path) -> str:
        """Get diff of worktree branch vs main (dynamic — may be empty if main moved)."""
        _, diff, _ = await _run(
            ['git', 'diff', f'{self.config.main_branch}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_diff_from_base(self, worktree: Path, base_commit: str) -> str:
        """Get diff of worktree HEAD vs a fixed base commit.

        Use this instead of get_diff_from_main when main may have advanced
        since the worktree was created (e.g. during review stage).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{base_commit}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_changed_line_ranges(
        self, ref: str,
    ) -> dict[str, list[tuple[int, int]]]:
        """Return old-side (BASE/main) changed line ranges for *ref* vs main.

        Runs ``git diff {main}...{ref} --unified=0 --no-color`` in
        ``self.project_root`` and delegates parsing to
        :func:`parse_diff_line_ranges`.  Using ``--unified=0`` gives exact
        hunk boundaries with no context padding, so the old-side ranges are
        the minimal set of lines actually modified.  The ``main...{ref}``
        three-dot syntax diffs *ref* against the merge-base of main and ref,
        so both tasks diffed against the same main share BASE coordinates that
        are directly comparable for stackability.

        Returns an empty dict when the diff is empty (no changes vs main).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{self.config.main_branch}...{ref}',
             '--unified=0', '--no-color'],
            cwd=self.project_root,
        )
        return parse_diff_line_ranges(diff)

    async def get_current_branch(self, worktree: Path) -> str:
        """Get the current branch name in a worktree."""
        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=worktree,
        )
        return branch

    async def get_main_sha(self) -> str:
        """Return current main branch SHA."""
        _, sha, _ = await _run(
            ['git', 'rev-parse', self.config.main_branch],
            cwd=self.project_root,
        )
        return sha.strip()

    async def resolve_branch_sha(self, branch_name: str) -> str | None:
        """Resolve a branch name to its 40-char commit SHA via ``git rev-parse --verify``.

        Uses ``refs/heads/{branch_name}`` to constrain resolution to local
        branches, preventing ambiguous resolution against tags or remote refs
        that happen to share the same name.

        Returns the SHA on success, or None when the ref does not exist or
        cannot be resolved (e.g. branch deleted post-merge, malformed name).
        """
        rc, sha, _ = await _run(
            ['git', 'rev-parse', '--verify', f'refs/heads/{branch_name}'],
            cwd=self.project_root,
        )
        return sha if rc == 0 else None

    async def find_merge_marker(self, branch: str) -> str | None:
        """Search main's history for a merge commit whose subject matches
        ``Merge {branch} into {main_branch}``.

        This is the companion check to ``is_ancestor`` for the case where the
        branch ref has already been deleted (e.g., ``cleanup_worktree`` ran
        after ``advance_main`` but before ``set_task_status('done')``).

        **Branch-existence gate**: calls ``resolve_branch_sha(branch)`` first.
        If it returns non-None the branch still exists, so this method returns
        None immediately — the caller should rely on ``is_ancestor`` instead.
        This prevents finding a stale merge marker from a *previous* run of a
        re-opened task that shared the same branch name.

        **Subject pattern**: the exact output of ``_merge_subject(branch,
        self.config.main_branch)`` matched with ``--fixed-strings`` (literal
        match — no BRE metacharacter interpretation, so branch names like
        ``task/v1.0`` are safe).  Because ``_merge_subject`` is also called
        by ``merge_to_main`` and the retry path in ``advance_main``, writer
        and reader share the same derivation and can never silently drift
        apart.  Substring-safety is preserved: ``'Merge task/1 into main'``
        cannot appear inside ``'Merge task/10 into main'`` because the ``0``
        after ``task/1`` falls where the pattern has a space.

        Args:
            branch: Full prefixed branch name, e.g. ``'task/123'``.
                    Same convention as ``is_ancestor`` and ``resolve_branch_sha``.

        Returns:
            The 40-char merge commit SHA on success, or None when the branch
            still exists, the branch never existed, or no matching marker was
            found on main.
        """
        # Gate: if the branch ref still exists, caller should use is_ancestor.
        if await self.resolve_branch_sha(branch) is not None:
            return None

        # Branch is gone — search main for a merge commit with the expected subject.
        # Pattern derivation shared with merge_to_main — see docstring for substring-safety argument.
        grep_pattern = _merge_subject(branch, self.config.main_branch)
        rc, out, _ = await _run(
            [
                'git', 'log', self.config.main_branch,
                '--fixed-strings',
                f'--grep={grep_pattern}',
                '--max-count=1',
                '--format=%H',
            ],
            cwd=self.project_root,
        )
        if rc != 0 or not out:
            return None
        return out

    async def find_task_citation_commit(
        self, tid: str, *, pattern_template: str | None = None,
    ) -> str | None:
        """Search main's history for a commit whose subject cites *tid*.

        Used by the reconciler to gate the ``is_ancestor==True`` fast-path:
        ``is_ancestor`` returns True trivially for zero-commit branches
        whose tip equals the main HEAD at branch-create time, which
        false-positives blocked/escalated tasks.  Requiring a positive
        citation on main rejects that degenerate case.

        Args:
            tid: Bare task id (no ``task/`` prefix); the prefix is added
                by the default pattern where appropriate.
            pattern_template: Optional override for the citation pattern.
                Defaults to ``DEFAULT_COMMIT_CITATION_PATTERN``.  Empty
                string disables the check by returning None immediately
                (caller opt-out for projects without citation conventions).

        Returns:
            The 40-char commit SHA of the most recent matching commit on
            main, or None when no commit cites the task or the pattern is
            disabled.
        """
        template = (
            pattern_template
            if pattern_template is not None
            else DEFAULT_COMMIT_CITATION_PATTERN
        )
        if template == '':
            return None
        pattern = template.format(tid=re.escape(tid))
        rc, out, _ = await _run(
            [
                'git', 'log', self.config.main_branch,
                '--extended-regexp',
                f'--grep={pattern}',
                '--max-count=1',
                '--format=%H',
            ],
            cwd=self.project_root,
        )
        if rc != 0 or not out:
            return None
        return out

    async def rebase_onto_main(self, worktree: Path, onto: str | None = None) -> bool:
        """Rebase the task branch in *worktree* onto *onto* (default: main).

        When *onto* is None (the default), rebases onto the configured
        ``main_branch`` — identical to the original behaviour, keeping all
        existing callers byte-compatible.

        When *onto* is provided (e.g. a sibling branch like ``task/123``),
        rebases the branch in *worktree* onto that ref instead.  This is used
        by ``stack_train_branches`` to chain members into a linear stack.

        Returns True on success.  On failure, aborts the rebase so the
        worktree is left in a clean state, and returns False.

        Caller must NOT hold ``_merge_lock`` — this is designed to run
        outside the lock so multiple tasks can rebase concurrently in
        their own worktrees.
        """
        target = onto if onto is not None else self.config.main_branch
        rc, _, err = await _run(
            ['git', 'rebase', target],
            cwd=worktree,
        )
        if rc != 0:
            await _run(['git', 'rebase', '--abort'], cwd=worktree)
            logger.info(f'Pre-merge rebase failed in {worktree}: {err}')
            return False
        return True

    async def merge_tree_conflicts(
        self,
        base_tip: str,
        branch_head: str,
    ) -> ConflictProbe:
        """Probe whether *branch_head* would merge cleanly onto *base_tip*.

        Uses ``git merge-tree --write-tree --name-only`` to perform an
        object-store-only 3-way merge (git auto-computes the merge-base).

        **Object-store-only contract** — MUST NOT touch worktree_base or any
        warm-lane path.  This primitive runs at ``self.project_root`` (the
        in-repo object store) and writes only loose tree/blob objects.
        Consumers δ (conflict-graph), η (bounce), and ι (drift metric) all
        depend on this ref-stable, disk-free, idempotent contract.

        Returns
        -------
        ConflictProbe(clean=True, conflicted_paths=[])
            when branch_head merges cleanly onto base_tip.
        ConflictProbe(clean=False, conflicted_paths=[...])
            when at least one conflict is detected; paths are relative to
            repo root, byte-faithful (``-c core.quotePath=false`` disables
            octal quoting of non-ASCII / special characters).

        Raises
        ------
        RuntimeError
            when git exits with a code other than 0 or 1 (e.g. 128 for an
            unknown / bad object name).  Silently returning clean=True would
            admit a broken branch into the verify frontier; returning
            clean=False would falsely bounce a mergeable branch.  Loud failure
            for a caller bug is the correct contract for a foundation primitive.
        """
        rc, out, err = await _run(
            [
                'git', '-c', 'core.quotePath=false',
                'merge-tree', '--write-tree', '--name-only',
                base_tip, branch_head,
            ],
            cwd=self.project_root,
        )
        if rc == 0:
            return ConflictProbe(clean=True, conflicted_paths=[])
        if rc == 1:
            # stdout layout (git 2.43.0, --name-only):
            #   line 0: merged tree OID (discard)
            #   lines 1..k: conflicted file paths (one per line, deduplicated)
            #   blank line: section boundary
            #   remaining lines: informational messages ("Auto-merging…" etc.)
            lines = out.split('\n')
            paths: list[str] = []
            for line in lines[1:]:  # skip the tree OID on line 0
                if line == '':  # blank line = section boundary
                    break
                paths.append(line)
            return ConflictProbe(clean=False, conflicted_paths=paths)
        raise RuntimeError(
            f'git merge-tree failed (rc={rc}) for {base_tip}..{branch_head}: {err}'
        )

    async def stack_train_branches(self, member_ids: list[str]) -> TrainStackResult:
        """Materialize a linear branch stack for a merge-train formation.

        The anchor (``member_ids[0]``) is always the stack base and always
        survives — it is NOT rebased (the _do_train_merge tip-rebase at
        merge time handles the anchor→main rebase).

        Each successor member's worktree (``self.worktree_base / member_id``)
        is rebased onto the last-surviving member's branch
        (``self.config.branch_prefix + last_good_id``) via
        ``rebase_onto_main(wt, onto=...)``.

        On a clean rebase the member is appended to *survivors* and becomes
        the new last-good predecessor for the next member.

        On a rebase conflict the member is added to *ejected*; the last-good
        predecessor is NOT advanced, so the next member re-links onto the last
        survivor (re-link invariant).  The conflicting branch is left clean by
        rebase_onto_main's ``git rebase --abort``.

        A missing worktree directory is treated as an eject (defensive;
        logged at WARNING level).

        Args:
            member_ids: Ordered list of member task ids, anchor first.

        Returns:
            TrainStackResult(survivors, ejected).
        """
        if not member_ids:
            return TrainStackResult(survivors=[], ejected=[])

        anchor_id = member_ids[0]
        survivors: list[str] = [anchor_id]
        ejected: list[str] = []
        last_good_id = anchor_id

        for member_id in member_ids[1:]:
            wt_path = self.worktree_base / member_id
            if not wt_path.is_dir():
                logger.warning(
                    'stack_train_branches: worktree %s not found for member %s — ejecting',
                    wt_path, member_id,
                )
                ejected.append(member_id)
                # Do not advance last_good_id — next member re-links onto last survivor.
                continue

            onto_branch = f'{self.config.branch_prefix}{last_good_id}'
            success = await self.rebase_onto_main(wt_path, onto=onto_branch)
            if success:
                survivors.append(member_id)
                last_good_id = member_id
            else:
                ejected.append(member_id)
                # Do not advance last_good_id.

        return TrainStackResult(survivors=survivors, ejected=ejected)

    async def materialize_member_solo(
        self,
        member_id: str,
        predecessor_ref: str,
        *,
        solo_prefix: str = '_solo-',
    ) -> WorktreeInfo | None:
        """Un-stack a train member's own delta onto current main.

        Creates an isolated ``<solo_prefix><member_id>`` branch starting at the
        member's current branch tip (``branch_prefix + member_id``) and checks
        it out in a new worktree at ``worktree_base / <solo_prefix><member_id>``.
        Then runs::

            git rebase --onto <main_branch> <predecessor_ref>

        inside that worktree, replaying only the commits between
        ``predecessor_ref`` and the member branch tip onto the current main
        HEAD.  This is the opposite of the cumulative stacking performed by
        ``stack_train_branches``: it extracts the member's *own* delta.

        On success returns a :class:`WorktreeInfo` whose *path* is the solo
        worktree directory and *base_commit* is the rebased tip SHA.  The
        caller is responsible for cleaning up the worktree via
        :meth:`cleanup_merge_worktree` when done.

        On rebase conflict (non-zero rc): aborts the rebase, removes the
        worktree and its temporary branch, and returns ``None``.  No dangling
        ``_solo-*`` worktrees or branches are left behind.

        Does NOT hold ``_merge_lock`` — runs outside the lock like
        ``rebase_onto_main`` and ``stack_train_branches``.

        Args:
            member_id: The member's task id (used to locate its branch/worktree).
            predecessor_ref: The ref that forms the base of the member's own
                commits — ``task/<predecessor_id>`` for non-anchors, or
                ``self.config.main_branch`` for the anchor.
            solo_prefix: Prefix for the temporary branch/worktree name.
                Defaults to ``'_solo-'``.

        Returns:
            WorktreeInfo on success, None on conflict.
        """
        member_branch = f'{self.config.branch_prefix}{member_id}'
        solo_name = f'{solo_prefix}{member_id}'
        solo_wt = self.worktree_base / solo_name
        solo_wt.parent.mkdir(parents=True, exist_ok=True)

        # Defensively pre-clean any stale solo artifacts for this member from a
        # prior crashed or un-torn-down attribution run.  A leaked _solo-<id>
        # branch/worktree from the previous attempt would make `git worktree add
        # -b _solo-<id>` fail (rc!=0) → materialize returns None → member is
        # mis-classified as 'unstackable'.  All three commands are best-effort
        # (we ignore their rc) — if there is nothing stale they are no-ops.
        await _run(['git', 'worktree', 'remove', str(solo_wt), '--force'],
                   cwd=self.project_root)
        await _run(['git', 'worktree', 'prune'], cwd=self.project_root)
        await _run(['git', 'branch', '-D', solo_name], cwd=self.project_root)

        # Create a temporary branch _solo-<member_id> starting at the member's
        # current tip and check it out in an isolated worktree.  Using -b with
        # the member branch as start-point avoids modifying the original branch.
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', solo_name, str(solo_wt), member_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'materialize_member_solo: failed to create worktree %s for member %s: %s',
                solo_wt, member_id, err,
            )
            return None

        # Rebase the temporary branch's commits (predecessor_ref..HEAD) onto main.
        rc, _, err = await _run(
            ['git', 'rebase', '--onto', self.config.main_branch, predecessor_ref],
            cwd=solo_wt,
        )
        if rc != 0:
            # Abort the rebase and clean up both the worktree and temp branch.
            await _run(['git', 'rebase', '--abort'], cwd=solo_wt)
            logger.info(
                'materialize_member_solo: rebase conflict for member %s '
                '(predecessor=%s): %s — cleaning up',
                member_id, predecessor_ref, err,
            )
            # Remove the worktree first, then delete the branch.
            rm_rc, _, rm_err = await _run(
                ['git', 'worktree', 'remove', str(solo_wt), '--force'],
                cwd=self.project_root,
            )
            if rm_rc != 0:
                logger.warning(
                    'materialize_member_solo: failed to remove worktree %s: %s',
                    solo_wt, rm_err,
                )
            del_rc, _, del_err = await _run(
                ['git', 'branch', '-D', solo_name],
                cwd=self.project_root,
            )
            if del_rc != 0:
                logger.warning(
                    'materialize_member_solo: failed to delete branch %s: %s',
                    solo_name, del_err,
                )
            return None

        # Resolve the rebased tip SHA.
        _, tip_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=solo_wt)
        tip_sha = tip_sha_raw.strip()

        logger.info(
            'materialize_member_solo: member %s un-stacked onto main, '
            'solo branch %s tip=%s worktree=%s',
            member_id, solo_name, tip_sha, solo_wt,
        )
        return WorktreeInfo(path=solo_wt, base_commit=tip_sha)

    async def delete_solo_branch(self, solo_branch: str) -> None:
        """Delete a bare ``_solo-<id>`` branch and prune any stale worktree entry.

        Companion to :meth:`materialize_member_solo`.  Called by
        ``_attribute_train_failure`` (and its tests) to tear down the temporary
        solo branch after a passer has been landed (or when the all-pass
        interaction case land-nothing branch is taken).

        Unlike :meth:`cleanup_worktree`, this method operates on the BARE
        branch name (e.g. ``_solo-b2``) — **NOT** prefixed with
        ``config.branch_prefix`` — because the solo branch is created without
        the prefix by :meth:`materialize_member_solo`.

        Best-effort / never-raises: non-zero rc from either git command is
        logged as a warning and ignored, mirroring the
        :meth:`cleanup_merge_worktree` never-raise contract.

        Args:
            solo_branch: Bare solo branch name (e.g. ``'_solo-b2'``).
        """
        # Prune first so that a removed (but not yet pruned) worktree entry
        # does not keep git from deleting the branch.
        prune_rc, _, prune_err = await _run(
            ['git', 'worktree', 'prune'], cwd=self.project_root,
        )
        if prune_rc != 0:
            logger.warning('delete_solo_branch: worktree prune failed: %s', prune_err)

        del_rc, _, del_err = await _run(
            ['git', 'branch', '-D', solo_branch], cwd=self.project_root,
        )
        if del_rc != 0:
            logger.warning(
                'delete_solo_branch: failed to delete branch %s: %s',
                solo_branch, del_err,
            )
        else:
            logger.debug('delete_solo_branch: deleted %s', solo_branch)

    async def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return True if *ancestor* is an ancestor of *descendant*."""
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', ancestor, descendant],
            cwd=self.project_root,
        )
        return rc == 0

    async def has_uncommitted_work(self, worktree: Path) -> bool:
        """Return True if worktree has staged or unstaged changes outside .task/."""
        rc, output, _ = await _run(
            ['git', 'status', '--porcelain', '--', '.', ':!.task'],
            cwd=worktree,
        )
        return rc == 0 and bool(output.strip())

    async def get_changed_files(self, from_sha: str, to_sha: str) -> list[str]:
        """Return list of files changed between two commits."""
        _, output, _ = await _run(
            ['git', 'diff', '--name-only', from_sha, to_sha],
            cwd=self.project_root,
        )
        return [f for f in output.strip().splitlines() if f.strip()]

    async def get_rebase_distance(self, old_base: str, new_base: str) -> int:
        """Count commits in ``old_base..new_base`` (i.e. how far main advanced).

        Returns the exact git rev-list count, or ``-1`` on any git error or
        unparseable output.  -1 is a distinct sentinel so an unmeasurable
        distance is never mistaken for a 0-commit (no-op) rebase.

        Modelled on the ``rev-list --count`` + int-parse + fail-safe pattern
        used in ``_freshen_main`` (line ~561) and
        ``_branch_has_commits_beyond_main`` (line ~1508).
        """
        rc, out, _ = await _run(
            ['git', 'rev-list', '--count', f'{old_base}..{new_base}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'get_rebase_distance: rev-list failed (rc=%d) for %s..%s',
                rc, old_base, new_base,
            )
            return -1
        try:
            return int(out.strip())
        except ValueError:
            logger.warning(
                'get_rebase_distance: unexpected output %r for %s..%s',
                out, old_base, new_base,
            )
            return -1

    async def get_merge_diff_files(
        self, base_sha: str, head_sha: str,
    ) -> tuple[list[str], Exception | None]:
        """Files changed by the merge ``base_sha..head_sha``, excluding ``.task/``.

        Returns a ``(files, error)`` tuple — **total, never raises**:

        * ``(files, None)`` on success.  An empty ``files`` list is a
          legitimate outcome (revert merges, ``.task/``-only merges).
        * ``([], exception)`` on any error: ``rc != 0`` from ``git diff``
          (returns :class:`subprocess.CalledProcessError`) or an unexpected
          raise from the subprocess helper (e.g. ``WorktreeMissing``,
          ``FileNotFoundError``).

        Callers should branch on ``err is not None`` (not ``resolver_failed``),
        because an empty diff is a valid non-error outcome for this function.

        Used by ``TaskWorkflow._reconcile_metadata_files_for_done`` to write
        the actually-changed paths into ``metadata.files`` instead of the
        architect's ``plan.files`` (which the merge may have squashed or
        refactored away).  Uses ``--no-renames`` so a rename surfaces as
        both add+delete; downstream consumers can decide whether to keep
        or drop the deleted path.
        """
        cmd = [
            'git', 'diff', '--name-only', '--no-renames',
            base_sha, head_sha, '--', ':!.task/',
        ]
        try:
            rc, output, stderr = await _run(cmd, cwd=self.project_root)
        except Exception as exc:
            return [], exc
        if rc != 0:
            logger.warning(
                'get_merge_diff_files: git diff %s..%s failed (rc=%s): %s',
                base_sha, head_sha, rc, (stderr or '').strip()[:200],
            )
            return [], subprocess.CalledProcessError(rc, cmd, output=output, stderr=stderr)
        return [f for f in output.strip().splitlines() if f.strip()], None

    async def get_files_touched_in_branch(
        self, base_sha: str, branch_head: str,
    ) -> list[str]:
        """Files touched by any commit in ``base_sha..branch_head``.

        Union of file paths that appeared in any commit on the branch
        (history-based, not just the diff).  Used by the pre-merge
        Decision-1 check: an architect-declared plan target is "touched"
        if it appears in this set.

        Excludes ``.task/`` and uses ``--no-renames`` so a rename
        surfaces both old and new paths (the old path is "touched" too).

        Returns ``[]`` on git error so the helper fails open — its
        consumer logs and proceeds rather than blocking the merge on
        a transient diff error.
        """
        rc, output, stderr = await _run(
            [
                'git', 'log', '--name-only', '--no-renames',
                '--pretty=format:', f'{base_sha}..{branch_head}',
                '--', ':!.task/',
            ],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'get_files_touched_in_branch: git log %s..%s failed (rc=%s): %s',
                base_sha, branch_head, rc, (stderr or '').strip()[:200],
            )
            return []
        seen: set[str] = set()
        for ln in output.splitlines():
            ln = ln.strip()
            if ln:
                seen.add(ln)
        return sorted(seen)

    async def merge_to_main(
        self,
        worktree: Path,
        branch: str,
        base_sha: str | None = None,
    ) -> MergeResult:
        """Merge a task branch into main using a temporary merge worktree.

        Creates a disposable worktree, performs the merge there, and returns
        the result.  The caller is responsible for calling :meth:`advance_main`
        after verification and :meth:`cleanup_merge_worktree` when done.

        When *base_sha* is provided the merge worktree is created at that
        commit rather than current main HEAD.  This supports speculative
        merges where N+1 is merged against N's merge commit SHA.

        Never touches ``project_root``'s working tree or index.
        Called by the MergeWorker (serialized via the merge queue).
        """
        full_branch = f'{self.config.branch_prefix}{branch}'
        merge_wt: Path | None = None

        try:
            merge_wt, pre_merge_sha = await self._create_merge_worktree(base_sha)

            # Pre-merge cleanup: remove .task/ from filesystem if inherited
            # from a contaminated main.  This is NOT sufficient on its own
            # because `git merge` will re-introduce .task/ from the branch.
            # The real fix is the post-merge scrub below.
            task_dir = merge_wt / '.task'
            if task_dir.exists():
                shutil.rmtree(task_dir)

            # Merge with no-ff
            rc, out, err = await _run(
                ['git', 'merge', '--no-ff', full_branch,
                 '-m', _merge_subject(full_branch, self.config.main_branch)],
                cwd=merge_wt,
            )

            if rc != 0:
                if 'CONFLICT' in out or 'CONFLICT' in err:
                    conflict_details = await self.get_conflict_details(merge_wt)
                    return MergeResult(
                        success=False, conflicts=True,
                        details=conflict_details,
                        pre_merge_sha=pre_merge_sha,
                        merge_worktree=merge_wt,
                    )
                # Non-conflict failure — clean up immediately
                await self.cleanup_merge_worktree(merge_wt)
                return MergeResult(
                    success=False, details=f'{out}\n{err}',
                    pre_merge_sha=pre_merge_sha,
                )

            # ── Post-merge .task/ scrub (CRITICAL) ────────────────────
            # The merge commit now exists.  If the task branch had .task/
            # tracked (common — agents commit it despite safeguards), the
            # merge commit contains those files.  We MUST remove them
            # before this commit reaches main via advance_main().
            #
            # scrub_task_dir_from_tree() checks git ls-tree, runs
            # git rm --cached, and amends the merge commit in-place.
            # This is the single most important .task/ defense.
            scrub_result = await scrub_task_dir_from_tree(merge_wt, f'post-merge({full_branch})')
            if scrub_result.outcome == ScrubOutcome.FAILED:
                logger.error(
                    '.task/ scrub FAILED post-merge for %s — aborting merge; '
                    'no advance_main will run.',
                    full_branch,
                )
                await self.cleanup_merge_worktree(merge_wt)
                _detail = f'.task/ scrub failed post-merge for {full_branch}{scrub_result.format_error(prefix=": ")}'
                return MergeResult(
                    success=False,
                    details=_detail,
                    pre_merge_sha=pre_merge_sha,
                )

            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=merge_wt)
            return MergeResult(
                success=True, merge_commit=sha,
                pre_merge_sha=pre_merge_sha,
                merge_worktree=merge_wt,
            )

        except BaseException:
            if merge_wt:
                await self.cleanup_merge_worktree(merge_wt)
            raise

    async def _create_merge_worktree(
        self, base_sha: str | None = None,
    ) -> tuple[Path, str]:
        """Create a temporary detached worktree at *base_sha* (or main HEAD).

        When *base_sha* is None the worktree is created at current main HEAD
        (normal case).  When *base_sha* is provided the worktree is created
        at that exact commit, supporting speculative merges where N+1 is
        merged against N's merge commit.
        """
        import uuid
        merge_id = uuid.uuid4().hex[:8]
        merge_wt = self.worktree_base / f'_merge-{merge_id}'
        merge_wt.parent.mkdir(parents=True, exist_ok=True)

        if base_sha is None:
            # Fetch latest (best-effort — no remote in tests)
            await _run(
                ['git', 'fetch', self.config.remote, self.config.main_branch],
                cwd=self.project_root,
            )
            # Capture current main SHA
            _, pre_merge_sha, _ = await _run(
                ['git', 'rev-parse', self.config.main_branch],
                cwd=self.project_root,
            )
            checkout_ref = self.config.main_branch
        else:
            pre_merge_sha = base_sha
            checkout_ref = base_sha.strip()

        # Detached worktree avoids "branch already checked out" error
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(merge_wt), checkout_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create merge worktree: {err}')

        logger.info(f'Created merge worktree at {merge_wt} (HEAD={pre_merge_sha[:8]})')
        return merge_wt, pre_merge_sha.strip()

    async def cleanup_merge_worktree(self, merge_wt: Path) -> None:
        """Remove a temporary merge worktree.

        **Persistent-worktree exemption**: if *merge_wt* resolves to
        :attr:`persistent_merge_worktree_path`, this method is a **no-op**
        (the warm worktree survives across attempts and across verify failures,
        so ``target/`` warmth is preserved).  The ephemeral removal path is
        unchanged for all other ``_merge-*`` worktrees.
        """
        if merge_wt.resolve() == self.persistent_merge_worktree_path.resolve():
            logger.debug('persistent merge worktree retained: %s', merge_wt)
            return

        rc, _, err = await _run(
            ['git', 'worktree', 'remove', str(merge_wt), '--force'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to remove merge worktree {merge_wt}: {err}')
        else:
            logger.info(f'Cleaned up merge worktree {merge_wt}')

    async def create_throwaway_verify_worktree(self, merge_commit: str) -> Path:
        """Create an ephemeral ``_merge-<uuid>`` worktree at *merge_commit*.

        Thin public wrapper over :meth:`_create_merge_worktree` for use by
        the warm-vs-cold shadow compare (PRD §10 invariant 6(b)).  The
        returned worktree is:

        * Checked out at *merge_commit* (detached HEAD).
        * Named ``_merge-<uuid>`` — NEVER the fixed ``_merge-verify`` path.
        * Intended for a single cold verify run; callers must remove it via
          :meth:`cleanup_merge_worktree` (in a ``finally`` block) after use.

        Unlike the warm :meth:`reset_persistent_merge_worktree` path, this
        worktree has no retained ``target/`` warmth — it is a true from-scratch
        cold verify worktree (PRD §10 invariant 6(b): "cold throwaway").

        Args:
            merge_commit: The merge commit SHA to check out in the new worktree.

        Returns:
            Path to the freshly created ephemeral worktree directory.
        """
        wt, _ = await self._create_merge_worktree(base_sha=merge_commit)
        return wt

    @property
    def persistent_merge_worktree_path(self) -> Path:
        """Fixed path for the persistent warm merge-verify worktree.

        Always ``<worktree_base>/_merge-verify``.  The path is independent of
        the ``git.persistent_merge_worktree`` knob — the property always
        returns the canonical location so callers can compare against it even
        when the feature is off.
        """
        return self.worktree_base / PERSISTENT_MERGE_WORKTREE_NAME

    @property
    def _merge_verify_artifact_path(self) -> Path:
        """Path of the build-artifact directory inside the persistent _merge-verify worktree.

        ``persistent_merge_worktree_path / reap_build_artifact_dirs[0]``
        (``<worktree_base>/_merge-verify/target`` by default when
        ``reap_build_artifact_dirs`` is empty).

        This is the single canonical spelling used as the *advancing* side in
        :meth:`refresh_warm_base` and as the derived default in
        :attr:`warm_lane_base_target_path`, so both stay in sync automatically.
        """
        artifact_dir = (
            self.config.reap_build_artifact_dirs[0]
            if self.config.reap_build_artifact_dirs
            else 'target'
        )
        return self.persistent_merge_worktree_path / artifact_dir

    @property
    def warm_lane_base_target_path(self) -> Path:
        """Absolute path of the warm BASE target/ to CoW-seed lane target/ from.

        Resolution order:
        1. ``config.warm_lane_base_target_dir`` when explicitly set (override).
        2. :attr:`_merge_verify_artifact_path`
           (derived default: ``<worktree_base>/_merge-verify/target``).
        """
        if self.config.warm_lane_base_target_dir is not None:
            return Path(self.config.warm_lane_base_target_dir)
        return self._merge_verify_artifact_path

    #: Name of the counter file used to persist the verify attempt count across
    #: stateless CLI invocations.  Scope is **per-project-worktree** — the file
    #: lives under ``worktree_base`` (``project_root / config.worktree_dir``), so
    #: a single laptop host running ``verify-merge`` for multiple projects keeps
    #: independent counters per project.  The file is never inside a registered
    #: worktree, so it is never pruned or git-cleaned.
    _VERIFY_ATTEMPT_COUNTER_FILENAME: str = '.merge_verify_host_attempts'

    def _bump_host_verify_attempt_count(self) -> int:
        """Read, increment, and persist the per-project-worktree verify attempt counter.

        The counter is stored as a plain integer in
        ``<worktree_base>/.merge_verify_host_attempts`` so that it survives
        across the stateless ``orchestrator verify-merge`` CLI invocations
        (each invocation is a fresh process; an in-memory counter cannot
        persist on the laptop host).  The counter is **per-project-worktree**:
        a single host running verify-merge for multiple projects has one
        independent counter file per project under that project's
        ``worktree_base``.

        A missing or corrupt counter file is treated as count 0 so the next
        call returns 1 — fail-safe, no exception raised.

        The non-atomic read / modify / write is safe because the per-host
        serial invariant enforced by
        :func:`~orchestrator.merge_queue.enforce_persistent_worktree_serial_lane`
        guarantees that at most one ``verify-merge`` process runs at a time on
        this host for this project.

        Returns:
            The new 1-based attempt count after the increment.
        """
        counter_file = self.worktree_base / self._VERIFY_ATTEMPT_COUNTER_FILENAME
        # Read existing count; treat missing/corrupt file as 0 (fail-safe)
        try:
            current = int(counter_file.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            current = 0
        new_count = current + 1
        # Ensure worktree_base exists before writing
        self.worktree_base.mkdir(parents=True, exist_ok=True)
        counter_file.write_text(str(new_count))
        return new_count

    async def acquire_host_verify_worktree(self, merge_sha: str) -> Path:
        """Acquire a verify worktree for the laptop (host-side) verify-merge CLI.

        Mirrors :func:`~orchestrator.merge_queue._acquire_warm_verify_worktree`
        for the off-host CLI path (PRD §8 η / §A invariant 4).  Picks between
        the warm fixed-path worktree and a fresh ephemeral worktree based on
        the ``git.persistent_merge_worktree`` knob and the per-host safety
        valve (PRD §10 invariant 6).

        **Warm path** (knob ON, safety valve not due):
            Calls :meth:`reset_persistent_merge_worktree` which creates or
            resets-in-place the fixed ``_merge-verify`` worktree retaining
            build-artifact dirs (invariants 1+4).  Returns the fixed path.

        **Ephemeral path** (knob OFF or safety valve due):
            Calls :meth:`_create_merge_worktree` for a fresh ``_merge-<uuid>``
            worktree.  The valve fires on ``attempt % every_n == 0`` (1-based,
            every_n > 0), mirroring :func:`~orchestrator.merge_queue._safety_valve_due`
            but inlined to avoid a git_ops→merge_queue import cycle.  Returns
            the ephemeral path; ``cleanup_merge_worktree`` will remove it in
            the caller's finally block (invariant 6: cold verify, target NOT
            retained).

        Args:
            merge_sha: The merge commit SHA to check out (passed to
                :meth:`reset_persistent_merge_worktree` or
                :meth:`_create_merge_worktree` as appropriate).

        Returns:
            The worktree path to use for verification.
        """
        if not self.config.persistent_merge_worktree:
            # Knob off — ephemeral path (byte-identical to today's behavior)
            wt, _ = await self._create_merge_worktree(base_sha=merge_sha)
            return wt

        # Bump the disk-persistent counter; check the valve predicate inline
        attempt = self._bump_host_verify_attempt_count()
        every_n = self.config.persistent_merge_worktree_safety_valve_every_n
        # Inlined from merge_queue._safety_valve_due to avoid an import cycle
        # (merge_queue already imports git_ops).
        due = every_n > 0 and attempt > 0 and attempt % every_n == 0

        if due:
            # Safety-valve fired: use a fresh ephemeral worktree so that a
            # true cold verify runs without a retained target/ (invariant 6).
            wt, _ = await self._create_merge_worktree(base_sha=merge_sha)
            return wt

        # Warm path: reset the fixed worktree in place (invariants 1+4).
        return await self.reset_persistent_merge_worktree(merge_sha)

    async def reset_persistent_merge_worktree(self, merge_commit: str) -> Path:
        """Create or reset-in-place the persistent warm merge-verify worktree.

        **Create-once path** (worktree not yet registered):
            ``git worktree add --detach <fixed_path> <merge_commit>``

        **Reset-in-place path** (worktree already registered):
            ``git reset --hard <merge_commit>`` followed by
            ``git clean -xfd -e <dir>`` for each dir in
            ``config.reap_build_artifact_dirs`` — so the source tree is
            bit-identical to a fresh checkout of *merge_commit* while
            build-artifact dirs (e.g. ``target/``) are retained (PRD §10
            invariant 1: source bit-identical to fresh checkout; build-cache
            dirs retained for warmth).

        Returns the fixed path (:attr:`persistent_merge_worktree_path`).
        Raises :exc:`RuntimeError` on git failure (mirrors
        :meth:`_create_merge_worktree`).
        """
        warm_path = self.persistent_merge_worktree_path

        if not await self._is_registered_worktree(warm_path):
            # Create-once branch — self-heal a stale unregistered directory first.
            # A previous run may have left the directory on disk without a git
            # worktree registration (e.g. worktree metadata pruned after a crash).
            # `git worktree add` refuses a non-empty directory, permanently
            # wedging the warm path until manual cleanup.  Removing the orphaned
            # directory here mirrors the stale-directory removal in create_worktree
            # and makes the create-once path self-healing.
            if warm_path.exists():
                logger.warning(
                    'Persistent merge worktree path %s exists on disk but is not '
                    'a registered git worktree; removing stale directory to allow '
                    'fresh creation (self-heal)',
                    warm_path,
                )
                shutil.rmtree(warm_path)
            warm_path.parent.mkdir(parents=True, exist_ok=True)
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '--detach', str(warm_path), merge_commit],
                cwd=self.project_root,
            )
            if rc != 0:
                raise RuntimeError(
                    f'Failed to create persistent merge worktree at {warm_path}: {err}'
                )
            logger.info(
                'Created persistent merge worktree at %s (HEAD=%s)',
                warm_path, merge_commit[:8],
            )
        else:
            # Reset-in-place branch (added in step-6)
            rc, _, err = await _run(
                ['git', 'reset', '--hard', merge_commit],
                cwd=warm_path,
            )
            if rc != 0:
                raise RuntimeError(
                    f'Failed to reset persistent merge worktree {warm_path} '
                    f'to {merge_commit}: {err}'
                )
            # Single invocation excluding ALL artifact dirs at once — every
            # configured build-output dir (e.g. build AND dist) survives in
            # one pass.  A per-dir loop would call ``git clean -xfd -e build``
            # (deleting dist/) then ``git clean -xfd -e dist`` (deleting
            # build/), so with >1 dir NONE survive (step-19 regression).
            clean_cmd = ['git', 'clean', '-xfd']
            for artifact_dir in self.config.reap_build_artifact_dirs:
                clean_cmd += ['-e', artifact_dir]
            rc, _, err = await _run(clean_cmd, cwd=warm_path)
            if rc != 0:
                raise RuntimeError(
                    f'Failed to clean persistent merge worktree {warm_path}: {err}'
                )
            logger.info(
                'Reset persistent merge worktree %s to HEAD=%s',
                warm_path, merge_commit[:8],
            )

        return warm_path

    async def _iter_merge_worktrees(self):
        """Yield ``(wt_path, wt_resolved)`` pairs for registered ``_merge-*`` worktrees.

        Private async-generator helper shared by :meth:`prune_stale_merge_worktrees`
        and :meth:`find_inflight_merge_worktree`.  Enumerates via
        ``git worktree list --porcelain``, filtering to direct children of
        ``worktree_base`` whose name starts with ``_merge-``.  Yields nothing
        on git error (fail-closed).

        *wt_path* is the raw path from porcelain output (used for git commands).
        *wt_resolved* is the resolved path (used for identity comparisons such
        as the ``keep`` exclusion in :meth:`prune_stale_merge_worktrees`).

        **Persistent-worktree exemption**: the fixed
        :data:`PERSISTENT_MERGE_WORKTREE_NAME` (``_merge-verify``) is always
        skipped so that both :meth:`prune_stale_merge_worktrees` (PRD §10
        invariant 4) and :meth:`find_inflight_merge_worktree` never touch or
        return the warm worktree.
        """
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.project_root,
        )
        if rc != 0:
            return

        for line in out.splitlines():
            if not line.startswith('worktree '):
                continue
            wt_path = Path(line[len('worktree '):].strip())
            try:
                wt_resolved = wt_path.resolve()
            except OSError:
                wt_resolved = wt_path
            if wt_resolved.parent != self.worktree_base:
                continue
            if not wt_resolved.name.startswith('_merge-'):
                continue
            # Exempt the persistent warm merge-verify worktree — prune and
            # find_inflight must never touch it (invariant 4).
            if wt_resolved.name == PERSISTENT_MERGE_WORKTREE_NAME:
                continue
            yield wt_path, wt_resolved

    async def prune_stale_merge_worktrees(
        self, keep: Path | Collection[Path] | None = None,
    ) -> list[str]:
        """Force-remove leftover ``_merge-*`` worktrees; return paths removed.

        Disk-pressure recovery helper.  A crashed or abandoned merge can leave
        ``_merge-<id>`` worktrees behind under ``worktree_base``, each holding
        a full checkout — dead weight that contributes to ENOSPC.  This
        force-removes every such *registered* worktree EXCEPT *keep* (the merge
        worktree currently in use), then runs ``git worktree prune`` to clear
        stale admin entries.

        *keep* may be a single :class:`~pathlib.Path`, a collection (set, list,
        …) of paths, or ``None`` (remove all).  Every path in the keep-set is
        resolved before comparison so symlinks and relative paths are handled
        correctly.

        NEVER touches task worktrees (``worktree_base/<task_id>``) — those hold
        live builds.  Only paths that are direct children of ``worktree_base``
        AND whose name starts with ``_merge-`` are eligible, so a task whose id
        happens to start with ``_merge`` cannot be caught (task ids are not
        prefixed that way).  Enumerates via :meth:`_iter_merge_worktrees`
        (``git worktree list --porcelain``), so a half-created directory git
        doesn't track is never removed.

        The persistent warm merge-verify worktree
        (:data:`PERSISTENT_MERGE_WORKTREE_NAME`) is always exempted via
        :meth:`_iter_merge_worktrees` — it is never removed by prune
        (PRD §10 invariant 4).
        """
        removed: list[str] = []
        if isinstance(keep, (str, bytes)):
            raise TypeError(
                f'prune_stale_merge_worktrees: keep must be a Path, '
                f'Collection[Path], or None — got {type(keep).__name__!r}; '
                f'wrap in Path(...) if a string path was intended'
            )
        if keep is None:
            keep_resolved: set[Path] = set()
        elif isinstance(keep, Path):
            keep_resolved = {keep.resolve()}
        else:
            keep_resolved = {p.resolve() for p in keep}

        async for wt_path, wt_resolved in self._iter_merge_worktrees():
            if wt_resolved in keep_resolved:
                continue
            rc_rm, _, err = await _run(
                ['git', 'worktree', 'remove', '--force', str(wt_path)],
                cwd=self.project_root,
            )
            if rc_rm == 0:
                removed.append(str(wt_path))
            else:
                logger.warning(
                    'prune_stale_merge_worktrees: failed to remove %s: %s',
                    wt_path, err.strip(),
                )

        if removed:
            await _run(['git', 'worktree', 'prune'], cwd=self.project_root)
            logger.info(
                'prune_stale_merge_worktrees: removed %d stale merge '
                'worktree(s)', len(removed),
            )
        return removed

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:
        """Find an on-disk ``_merge-*`` worktree whose HEAD matches *branch*.

        Uses :meth:`_iter_merge_worktrees` to enumerate candidates (direct
        children of ``worktree_base`` whose name starts with ``_merge-``).
        For each candidate, reads its HEAD commit subject with
        ``git log -1 --format=%s`` and compares it by **literal equality** to
        ``_merge_subject(f'{branch_prefix}{branch}', main_branch)``.

        Returns the first matching :class:`~pathlib.Path`, or ``None`` if no
        match is found.

        Fail-closed on git errors: a candidate whose ``git log`` fails is
        skipped (logged at WARNING level) rather than raising — avoids
        crashing the coalesce dispatch on a partially-written worktree.

        Crash-safety / cross-restart source of truth: even if the in-memory
        ``InFlightMergeRegistry`` was cleared by a process restart, an
        in-progress merger's ``_merge-*`` worktree persists on disk and is
        correctly detected here.
        """
        target_subject = _merge_subject(
            f'{self.config.branch_prefix}{branch}',
            self.config.main_branch,
        )

        async for wt_path, _ in self._iter_merge_worktrees():
            # Read HEAD commit subject of this candidate
            rc_log, subject, err_log = await _run(
                ['git', 'log', '-1', '--format=%s'],
                cwd=wt_path,
            )
            if rc_log != 0:
                logger.warning(
                    'find_inflight_merge_worktree: git log failed for %s: %s',
                    wt_path, err_log.strip(),
                )
                continue
            if subject.strip() == target_subject:
                return wt_path

        return None

    # ── PHASE 4: Speculative merge-verify pipeline ────────────────────
    #
    # Once the merge queue (task 292) is stable and we have metrics on
    # queue depth and cycle time, consider a 2-step speculative pipeline:
    #
    #   Worker A (merger):   dequeue → merge_wt → git merge → scrub
    #   Worker B (verifier): verify → CAS update-ref → notify
    #
    # While B verifies merge N, A speculatively merges N+1 against N's
    # merge SHA (not current main).  If N succeeds, N+1 is already a
    # descendant — CAS works immediately.  If N fails, discard N+1 and
    # re-merge against actual main.  Cap speculation depth at 1.
    #
    # Expected throughput gain: ~2-3x when queue depth >3, because
    # verification (~15-25s) dominates cycle time and is fully overlapped.
    #
    # Key risk: verification validity.  N+1 is verified against a tree
    # that includes N's changes.  If N is later rejected, N+1 passed
    # verification against a state that never existed on main.  Mitigated
    # by scoped verification (task_files only) and depth-1 cap.
    #
    # Unblock condition: merge queue metrics showing sustained queue
    # depth >3 and merge cycle time dominating task throughput.
    # See blocked task that depends on task 292.
    # ─────────────────────────────────────────────────────────────────

    async def advance_main(
        self,
        merge_sha: str,
        merge_worktree: Path | None = None,
        branch: str | None = None,
        max_attempts: int = 3,
        expected_main: str | None = None,
        reverify_on_rebase: bool = False,
    ) -> AdvanceResult:
        """Advance main branch ref to *merge_sha* atomically.

        Uses ``update-ref`` to advance the ref, then syncs the working tree
        via ``read-tree`` when project_root has main checked out.  Uncommitted
        changes are stashed before the advance and popped after, so user work
        survives and merge conflicts become visible markers rather than silent
        reverts (see incident ``0ea23cb5c``).

        Returns an :data:`AdvanceResult` literal:

        * ``'advanced'`` — success.
        * ``'cas_failed'`` — CAS ``update-ref`` failed (transient; caller
          can re-enqueue).
        * ``'not_descendant'`` — merge commit couldn't become a descendant
          of main after *max_attempts* (permanent; stop retrying).
        * ``'contaminated'`` — ``.task/`` contamination gate failed
          (permanent; stop retrying).
        * ``'stash_failed'`` — ``git stash push`` failed before the advance
          (permanent; halt merge to prevent code loss).
        * ``'pop_conflict_no_advance'`` — CAS ``update-ref`` failed AND the
          subsequent stash pop conflicted.  The merge did NOT land.  WIP is
          preserved on a ``wip/recovery-*`` branch; routes to a human-level
          escalation.
        * ``'unmerged_state'`` — ``project_root`` already has unresolved merge
          conflicts in its index (UU/AA/DD paths detected via
          ``git status --porcelain``).  Halts immediately; manual cleanup of
          the conflict markers is required before retrying.  Routes to a
          human-level escalation, not the steward corrective loop.

        When *branch* is provided and a rebase fails, the method will abort
        the rebase, reset to current main, and re-merge *branch* before
        retrying.  Up to *max_attempts* rounds are attempted.

        When *expected_main* is provided, the final ``update-ref`` uses a
        compare-and-swap: ``git update-ref refs/heads/main <new> <old>``.
        If main has moved (external actor), update-ref fails atomically
        and this method returns ``'cas_failed'``.

        IMPORTANT: This method is the LAST checkpoint before code reaches
        main.  update-ref bypasses most git hooks (including pre-commit),
        so the .task/ contamination gate here is the final defense.
        Exception: git's ``reference-transaction`` hook (git>=2.28) DOES
        fire on update-ref — advance_main's main_gate mark (task 1678)
        sanctions that hook by writing a sentinel immediately before the
        CAS so reify-style projects record the move as SANCTIONED rather
        than UNSANCTIONED.  See also task 7 for the same stale assumption.

        On a successful 'advanced' return, ``self._last_advanced_sha`` holds
        the SHA actually placed on main.  When CAS retry rebases the merge
        commit, the post-rebase SHA is captured here — callers must read
        this side channel for done_provenance instead of the pre-rebase
        ``MergeResult.merge_commit`` (which is stale after a rebase).
        """
        full_branch = f'{self.config.branch_prefix}{branch}' if branch else None
        rebased = False  # Track whether any rebase/re-merge occurred this call

        # Derive the verified branch tip from M^2 — the exact branch commit
        # that merge_to_main incorporated (--no-ff guarantees M^2 is the
        # branch commit verify ran against).  Captured ONCE here, before the
        # CAS loop, so the re-merge fallback can pin to this SHA rather than
        # re-resolving the moving full_branch ref.
        verified_branch_tip: str | None = None
        _vbt_rc, _vbt_sha, _ = await _run(
            ['git', 'rev-parse', f'{merge_sha}^2'],
            cwd=self.project_root,
        )
        if _vbt_rc == 0 and _vbt_sha.strip():
            verified_branch_tip = _vbt_sha.strip()

        for attempt in range(max_attempts):
            # ── .task/ contamination gate (FINAL DEFENSE) ─────────────
            try:
                await _assert_no_task_dir(
                    merge_sha, self.project_root,
                    f'advance_main(attempt={attempt + 1})',
                )
            except RuntimeError as e:
                logger.error(str(e))
                return 'contaminated'

            rc, _, _ = await _run(
                ['git', 'merge-base', '--is-ancestor',
                 self.config.main_branch, merge_sha],
                cwd=self.project_root,
            )
            if rc == 0:
                break  # merge_sha is a descendant of main — safe to advance

            if merge_worktree is None:
                logger.warning(
                    f'Cannot fast-forward: {merge_sha[:8]} is not a descendant '
                    f'of {self.config.main_branch} (no merge worktree for retry)'
                )
                return 'not_descendant'

            logger.info(
                f'advance_main attempt {attempt + 1}/{max_attempts}: '
                f'main advanced past {merge_sha[:8]}'
            )

            # Try rebasing the merge commit onto current main
            rebase_rc, _, rebase_err = await _run(
                ['git', 'rebase', self.config.main_branch],
                cwd=merge_worktree,
            )
            if rebase_rc == 0:
                _, new_sha, _ = await _run(
                    ['git', 'rev-parse', 'HEAD'], cwd=merge_worktree,
                )
                merge_sha = new_sha.strip()
                rebased = True
                continue  # re-check is_ancestor at top of loop

            # Rebase failed — abort and try a fresh re-merge if we have
            # the branch name
            logger.warning(
                f'Rebase failed (attempt {attempt + 1}): {rebase_err}'
            )
            await _run(['git', 'rebase', '--abort'], cwd=merge_worktree)

            if full_branch is None:
                # No branch to re-merge from — cannot recover
                continue

            # Reset merge worktree to current main and re-merge.
            # Pin to the verified branch tip (M^2) so that post-verify
            # commits pushed to the task branch cannot silently land on main.
            # Fall back to full_branch only if M^2 was unresolvable (defensive).
            _remerge_target = verified_branch_tip if verified_branch_tip else full_branch

            # Divergence canary: if the live branch ref has advanced past
            # verified M^2, emit a structured WARNING so any future stale-tip
            # mismatch is self-evident in logs.  Fail-open: a rev-parse error
            # must not block the advance.
            if verified_branch_tip and full_branch:
                _live_rc, _live_sha, _ = await _run(
                    ['git', 'rev-parse', full_branch],
                    cwd=self.project_root,
                )
                if _live_rc == 0:
                    _live_tip = _live_sha.strip()
                    if _live_tip != verified_branch_tip:
                        logger.warning(
                            'advance_main: branch ref diverged from verified M^2 '
                            'during re-merge fallback — pinning to verified tip. '
                            'branch=%s verified_tip=%s live_ref_tip=%s',
                            branch or full_branch,
                            verified_branch_tip[:8],
                            _live_tip[:8],
                        )

            await _run(
                ['git', 'reset', '--hard', self.config.main_branch],
                cwd=merge_worktree,
            )
            merge_rc, merge_out, merge_err = await _run(
                ['git', 'merge', '--no-ff', _remerge_target,
                 '-m', _merge_subject(full_branch, self.config.main_branch)],
                cwd=merge_worktree,
            )
            if merge_rc != 0:
                # True conflict with current main — stop retrying
                logger.warning(
                    f'Re-merge failed (true conflict): {merge_out}\n{merge_err}'
                )
                return 'not_descendant'

            scrub_result = await scrub_task_dir_from_tree(
                merge_worktree, f'advance_main-retry({attempt + 1})',
            )
            if scrub_result.outcome == ScrubOutcome.FAILED:
                logger.error(
                    '.task/ scrub FAILED during advance_main-retry(%d) — index may '
                    'be contaminated; _assert_no_task_dir will catch it.%s',
                    attempt + 1, scrub_result.format_error(prefix=' Error: '),
                )
            _, new_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=merge_worktree,
            )
            merge_sha = new_sha.strip()
            rebased = True
            continue  # re-check is_ancestor at top of loop
        else:
            # Exhausted all attempts
            logger.warning(
                f'Cannot fast-forward after {max_attempts} attempts: '
                f'{merge_sha[:8]} is not a descendant of '
                f'{self.config.main_branch}'
            )
            return 'not_descendant'

        # ── Reverify-on-rebase gate ──────────────────────────────────
        # When reverify_on_rebase is set and a rebase (or re-merge) occurred,
        # park merge_worktree at the rebased SHA and hand control back to the
        # caller WITHOUT advancing main.  The caller must intersect the
        # intervening delta with the branch-touched file set; if overlapping it
        # must re-verify the rebased tree before calling advance_main again.
        if reverify_on_rebase and rebased:
            self._last_advanced_sha = merge_sha
            self._rebased_from = expected_main  # original base provided by caller
            _, onto_sha, _ = await _run(
                ['git', 'rev-parse', self.config.main_branch],
                cwd=self.project_root,
            )
            self._rebased_onto = onto_sha.strip()
            logger.info(
                'advance_main: reverify_on_rebase — rebased tree parked at '
                '%s; returning rebased_pending_reverify (no update-ref)',
                merge_sha[:8],
            )
            return 'rebased_pending_reverify'

        # ── Pre-advance unmerged state guard ────────────────────────
        # Belt-and-braces: reject immediately if project_root already has
        # unresolved merge conflicts in the index.  Any git stash push in
        # this state would fail with "fatal: needs merge", producing
        # 'stash_failed' and hiding the real problem.  Detecting here
        # produces a distinct 'unmerged_state' code that routes to a
        # human-escalation path instead of the steward corrective loop.
        _unmerged_entry_paths = await self._detect_unmerged_paths(self.project_root)
        if _unmerged_entry_paths:
            logger.critical(
                'CRITICAL: project_root has %d pre-existing unresolved merge '
                'conflict(s) (%s) — halting advance_main to prevent data loss. '
                'Manual cleanup required before retrying.',
                len(_unmerged_entry_paths),
                ', '.join(_unmerged_entry_paths[:10]),
            )
            return 'unmerged_state'

        # ── Working-tree protection ──────────────────────────────────
        # When project_root has main checked out, update-ref will desync
        # the working tree from HEAD.  Stash any uncommitted work first,
        # sync after, then pop.  This prevents silent reverts (see 0ea23cb5c).
        is_on_main = False
        did_stash = False

        rc, current_branch, _ = await _run(
            ['git', 'symbolic-ref', '--short', 'HEAD'],
            cwd=self.project_root,
        )
        if rc == 0 and current_branch.strip() == self.config.main_branch:
            is_on_main = True

            # Check for uncommitted changes (staged or unstaged)
            _, porcelain, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
            )
            if porcelain.strip():
                # ── WIP overlap check ────────────────────────────────
                # Before stashing, check if dirty tracked files overlap
                # with the merge diff.  If they do, abort the advance
                # to prevent stash-pop conflicts that destroy WIP.
                #
                # Use git diff to get tracked dirty filenames reliably.
                # Porcelain parsing is fragile because _run strips stdout,
                # which eats the leading space from " M filename" status.
                # Exclude .task/ (ephemeral) and the worktree dir (managed by git).
                wt_dir = self.config.worktree_dir
                _, unstaged_files, _ = await _run(
                    ['git', 'diff', '--name-only', '--',
                     '.', ':!.task', f':!{wt_dir}'],
                    cwd=self.project_root,
                )
                _, staged_files, _ = await _run(
                    ['git', 'diff', '--name-only', '--cached', '--',
                     '.', ':!.task', f':!{wt_dir}'],
                    cwd=self.project_root,
                )
                dirty_tracked = {
                    f.strip() for f in
                    (unstaged_files + '\n' + staged_files).splitlines()
                    if f.strip()
                }
                if dirty_tracked:
                    _, merge_diff_files, _ = await _run(
                        ['git', 'diff', '--name-only',
                         await self.get_main_sha(), merge_sha],
                        cwd=self.project_root,
                    )
                    merge_files = {
                        f.strip() for f in merge_diff_files.splitlines() if f.strip()
                    }
                    overlap = dirty_tracked & merge_files
                    if overlap:
                        self._last_overlap_files = sorted(overlap)
                        logger.warning(
                            'WIP overlap detected: %d file(s) overlap merge diff '
                            'for %s — aborting advance to prevent stash-pop '
                            'conflict. Overlapping: %s',
                            len(overlap), branch or merge_sha[:8],
                            ', '.join(sorted(overlap)[:10]),
                        )
                        return 'wip_overlap'

                # Only stash if there are tracked dirty files.  Untracked-only
                # (??) entries survive read-tree without conflict — stashing
                # them risks spurious pop failures (e.g. .worktrees/).
                if dirty_tracked:
                    stash_rc, _, stash_err = await _run(
                        ['git', 'stash', 'push', '-m',
                         f'merge-queue: pre-advance for {branch or merge_sha[:8]}'],
                        cwd=self.project_root,
                    )
                    if stash_rc != 0:
                        logger.error(
                            'CRITICAL: git stash push failed before advance_main '
                            '— halting merge to prevent code loss. error=%s',
                            stash_err,
                        )
                        return 'stash_failed'
                    did_stash = True
                    logger.info('Stashed uncommitted changes before advance_main')

        # ── Main-gate mark (best-effort) ─────────────────────────────────
        # Run the project-configurable sentinel command immediately before
        # the update-ref so that reify's reference-transaction hook
        # (git>=2.28, which DOES fire on update-ref) sees the one-shot
        # marker and records this advance as SANCTIONED.  Skipped when the
        # field is unset (feature off — other projects unaffected).
        # Non-zero return is logged as WARNING but never aborts the advance:
        # the task's whole purpose is to prevent queue bricking; under
        # reify ENFORCE a failed mark simply lets update-ref abort →
        # existing 'cas_failed' handling.  Re-runs on every invocation
        # that reaches this point so the one-shot sentinel is refreshed on
        # caller-level CAS retries.
        #
        # SUCCESS PATH: the project's reference-transaction hook is
        # responsible for consuming the sentinel after the successful
        # update-ref.  A missing or non-consuming hook (absent hook, or
        # git < 2.28) leaves the mark stale; the exposure is bounded to
        # at most ONE intervening non-orchestrator move before the next
        # advance_main invocation re-marks + consumes it.
        if self.config.main_gate_mark_command:
            mark_rc, _, mark_err = await _run(
                ['sh', '-c', self.config.main_gate_mark_command],
                cwd=self.project_root,
            )
            if mark_rc != 0:
                logger.warning(
                    'main_gate_mark_command returned non-zero rc=%d: %s',
                    mark_rc, mark_err,
                )

        # All checks passed — advance the ref (CAS when expected_main provided)
        update_cmd = [
            'git', 'update-ref',
            f'refs/heads/{self.config.main_branch}', merge_sha,
        ]
        if expected_main is not None:
            update_cmd.append(expected_main)
        rc, _, err = await _run(update_cmd, cwd=self.project_root)
        if rc != 0:
            # ── Main-gate unmark (best-effort cleanup) ────────────────────
            # A mark written immediately before this failed/aborted update-ref
            # may not have been consumed by the aborted reference-transaction;
            # clear it now so it cannot falsely sanction a later non-
            # orchestrator move.  Runs at the TOP of rc!=0 so it covers both
            # the 'cas_failed' and 'pop_conflict_no_advance' return paths.
            #
            # When main_gate_unmark_command is unset the residual exposure is
            # bounded: a lingering mark sanctions at most ONE intervening move
            # before the next advance_main invocation re-marks+consumes it.
            # This is documented and accepted ("prefer explicit cleanup").
            if self.config.main_gate_unmark_command:
                unmark_rc, _, unmark_err = await _run(
                    ['sh', '-c', self.config.main_gate_unmark_command],
                    cwd=self.project_root,
                )
                if unmark_rc != 0:
                    logger.warning(
                        'main_gate_unmark_command returned non-zero rc=%d: %s',
                        unmark_rc, unmark_err,
                    )

            # Restore stash before returning — ref didn't move.
            # Use _safe_stash_pop_with_recovery so that a pop conflict here
            # does NOT leave UU markers in project_root and is escalated to
            # humans rather than silently cascading to 'stash_failed' on the
            # next cycle.
            if did_stash:
                pop_ok, recovery = await self._safe_stash_pop_with_recovery(
                    branch or merge_sha[:8],
                )
                if not pop_ok:
                    self._last_recovery_branch = recovery
                    logger.critical(
                        'CRITICAL: stash pop conflicted during CAS-failure recovery '
                        '(task %s). WIP preserved on recovery branch: %s. '
                        'Halting — manual intervention required.',
                        branch or merge_sha[:8], recovery,
                    )
                    return 'pop_conflict_no_advance'
            if expected_main is not None:
                logger.warning(
                    f'CAS update-ref failed (expected {expected_main[:8]}): {err}'
                )
            else:
                logger.error(f'update-ref failed: {err}')
            return 'cas_failed'

        logger.info(f'Advanced {self.config.main_branch} to {merge_sha[:8]}')

        # ── Sync working tree to new HEAD ────────────────────────────
        # update-ref moved the ref but left the working tree stale.
        # read-tree syncs the index and working tree to the new HEAD.
        # Then pop the stash to restore any in-progress user work.
        if is_on_main:
            sync_rc, _, sync_err = await _run(
                ['git', 'read-tree', '-u', '--reset', 'HEAD'],
                cwd=self.project_root,
            )
            if sync_rc != 0:
                logger.error(
                    'read-tree failed after advancing main — working tree '
                    'is stale. error=%s', sync_err,
                )

            if did_stash:
                pop_ok, recovery = await self._safe_stash_pop_with_recovery(
                    branch or merge_sha[:8],
                )
                if not pop_ok:
                    self._last_recovery_branch = recovery
                    logger.warning(
                        'Stash pop conflicted after merge advance (task %s). '
                        'WIP preserved on recovery branch: %s',
                        branch or merge_sha[:8], recovery,
                    )
                    # Main was advanced before the stash pop ran — record the
                    # actually-on-main SHA so callers handling done_wip_recovery
                    # can record correct done_provenance.
                    self._last_advanced_sha = merge_sha
                    return 'pop_conflict'

        # Main was advanced — expose the post-rebase SHA so callers can record
        # done_provenance against the SHA actually on main, not the stale
        # pre-rebase SHA from MergeResult.merge_commit.
        self._last_advanced_sha = merge_sha
        return 'advanced'

    async def recover_red_main(
        self,
        target_sha: str,
        expected_main: str,
    ) -> RecoverResult:
        """Move refs/heads/main to *target_sha* atomically, sanctioning it for reify's gate.

        This is the enforce-safe break-glass recovery operation: a SINGLE CAS
        update-ref that drops a bad merge in one move (no rewind-then-readvance).
        Mirrors advance_main's main_gate mark → update-ref → unmark-on-failure
        sequence so that reify's reference-transaction hook records the move as
        SANCTIONED under ENFORCE.

        Args:
            target_sha:    The good SHA to restore main to (the pre-bad-merge state).
            expected_main: The current (bad) value of refs/heads/main; used as the
                           CAS old-value so a concurrent ref-move aborts cleanly.

        Returns:
            ``'rewound'``    — update-ref succeeded; main now points at target_sha.
            ``'cas_failed'`` — another writer moved the ref first; no change made.

        Note:
            The caller (skill) must ensure project_root is clean before invoking
            this method.  recover_red_main does NOT stash/pop uncommitted WIP —
            that dance is out of scope for a break-glass operation.
        """
        # ── Check working tree ────────────────────────────────────────────
        is_on_main = False
        rc, branch_out, _ = await _run(
            ['git', 'symbolic-ref', '--short', 'HEAD'],
            cwd=self.project_root,
        )
        if rc == 0 and branch_out.strip() == self.config.main_branch:
            is_on_main = True

        # ── Pre-validate SHAs (distinguish bad-input from CAS mismatch) ─────
        # A non-existent or non-commit SHA would make update-ref fail with a
        # repo-level error indistinguishable from a CAS mismatch.  Fail early
        # with 'error' so the runbook routes to "fix the SHA" rather than the
        # retry loop intended for genuine CAS races.
        for _sha_label, _sha_val in (
            ('target_sha', target_sha),
            ('expected_main', expected_main),
        ):
            _v_rc, _, _v_err = await _run(
                ['git', 'rev-parse', '--verify', f'{_sha_val}^{{commit}}'],
                cwd=self.project_root,
            )
            if _v_rc != 0:
                logger.error(
                    'recover_red_main: %s %r does not resolve to a commit; '
                    'fix the SHA before retrying. detail=%s',
                    _sha_label, (_sha_val or '')[:8], _v_err.strip(),
                )
                return 'error'

        # ── Main-gate mark (best-effort) ──────────────────────────────────
        # Run the project-configurable sentinel command immediately before
        # update-ref so reify's reference-transaction hook sees the marker
        # and records this recovery as SANCTIONED.  Mirrors advance_main's
        # 1678 sanction block exactly.  Non-zero return is logged as WARNING
        # but never aborts the recovery.
        if self.config.main_gate_mark_command:
            mark_rc, _, mark_err = await _run(
                ['sh', '-c', self.config.main_gate_mark_command],
                cwd=self.project_root,
            )
            if mark_rc != 0:
                logger.warning(
                    'main_gate_mark_command returned non-zero rc=%d: %s',
                    mark_rc, mark_err,
                )

        # ── CAS move of refs/heads/main ───────────────────────────────────
        rc, _, err = await _run(
            ['git', 'update-ref',
             f'refs/heads/{self.config.main_branch}',
             target_sha, expected_main],
            cwd=self.project_root,
        )
        if rc != 0:
            # ── Main-gate unmark (best-effort cleanup) ────────────────────
            if self.config.main_gate_unmark_command:
                unmark_rc, _, unmark_err = await _run(
                    ['sh', '-c', self.config.main_gate_unmark_command],
                    cwd=self.project_root,
                )
                if unmark_rc != 0:
                    logger.warning(
                        'main_gate_unmark_command returned non-zero rc=%d: %s',
                        unmark_rc, unmark_err,
                    )
            logger.warning(
                'recover_red_main: CAS update-ref failed (expected %s): %s',
                expected_main[:8], err,
            )
            return 'cas_failed'

        logger.info(
            'recover_red_main: rewound %s to %s',
            self.config.main_branch, target_sha[:8],
        )

        # ── Sync working tree to new HEAD ─────────────────────────────────
        if is_on_main:
            # Warn if the tree is dirty — read-tree will silently discard
            # uncommitted tracked changes.  The caller/runbook is expected to
            # clean up first; this is a last-resort advisory so an operator
            # under stress isn't silently burned.
            _st_rc, _st_out, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
            )
            if _st_rc == 0 and _st_out.strip():
                logger.warning(
                    'recover_red_main: working tree has uncommitted changes '
                    '(git status --porcelain non-empty); '
                    'git read-tree will silently discard tracked WIP. '
                    'Ensure project_root is clean before break-glass recovery.',
                )
            sync_rc, _, sync_err = await _run(
                ['git', 'read-tree', '-u', '--reset', 'HEAD'],
                cwd=self.project_root,
            )
            if sync_rc != 0:
                logger.error(
                    'read-tree failed after recover_red_main — working tree '
                    'is stale. error=%s', sync_err,
                )

        return 'rewound'

    async def push_main(self) -> PushResult:
        """Push local main to ``<remote>/<main_branch>`` as a fast-forward.

        Best-effort mirror step for ``advance_main``: keeps origin in sync
        without ever blocking the merge worker. Local main is the source of
        truth; this is a one-way replication. Never raises and never uses
        ``--force``.

        Returns:
            ``'pushed'``   — push succeeded.
            ``'noop'``     — disabled via ``config.push_after_advance``.
            ``'rejected'`` — non-fast-forward (origin diverged); logged at ERROR.
            ``'error'``    — network / auth / other transient failure;
                             logged at WARNING.
        """
        if not self.config.push_after_advance:
            return 'noop'

        refspec = f'{self.config.main_branch}:{self.config.main_branch}'
        rc, _, err = await _run(
            ['git', 'push', self.config.remote, refspec],
            cwd=self.project_root,
        )
        if rc == 0:
            logger.info(
                'Pushed %s to %s', self.config.main_branch, self.config.remote,
            )
            return 'pushed'

        # Classify the failure. git push surfaces non-ff in stderr with one of
        # several phrasings depending on version/locale.
        err_lower = err.lower()
        if any(s in err_lower for s in ('non-fast-forward', 'fetch first', '! [rejected]')):
            logger.error(
                'Push of %s to %s rejected (non-fast-forward) — origin has '
                'diverged. NOT force-pushing. stderr=%s',
                self.config.main_branch, self.config.remote, err,
            )
            return 'rejected'

        logger.warning(
            'Push of %s to %s failed (rc=%d) — leaving origin behind; '
            'next successful push will catch up. stderr=%s',
            self.config.main_branch, self.config.remote, rc, err,
        )
        return 'error'

    async def _create_recovery_branch_from_stash(self, label: str) -> str:
        """Create a branch from the current stash to preserve WIP, then clean up.

        1. Create a deterministic branch name.
        2. ``git branch <name> stash@{0}`` — makes the stash commit reachable.
        3. ``git stash drop`` — safe now (WIP reachable via branch).
        4. ``git read-tree -u --reset HEAD`` — clean working tree (removes
           conflict markers and UU state).

        Returns the recovery branch name.
        """
        from datetime import UTC, datetime

        iso = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
        name = f'wip/recovery-{label}-{iso}'

        # Create branch pointing at the stash commit
        await _run(
            ['git', 'branch', name, 'stash@{0}'],
            cwd=self.project_root,
        )
        # Drop the stash entry (WIP is now reachable via the branch)
        await _run(['git', 'stash', 'drop'], cwd=self.project_root)
        # Reset working tree to HEAD (removes conflict markers / UU state)
        await _run(
            ['git', 'read-tree', '-u', '--reset', 'HEAD'],
            cwd=self.project_root,
        )
        return name

    async def _safe_stash_pop_with_recovery(
        self, label: str,
    ) -> tuple[bool, str | None]:
        """Pop ``stash@{0}`` and preserve WIP on a recovery branch if it conflicts.

        1. Run ``git stash pop``.
        2. Check return code AND ``_detect_unmerged_paths`` — either signal
           is sufficient to declare failure (belt-and-braces).
        3. On failure: call ``_create_recovery_branch_from_stash(label)``
           which saves the stash to a branch, drops the stash entry, and
           resets the working tree to HEAD.
        4. Return ``(True, None)`` on clean pop, or
           ``(False, recovery_branch_name)`` on conflict.
        """
        pop_rc, _, pop_err = await _run(['git', 'stash', 'pop'], cwd=self.project_root)
        unmerged = await self._detect_unmerged_paths(self.project_root)

        if pop_rc != 0 or unmerged:
            logger.warning(
                'Stash pop failed (rc=%d, unmerged=%s, err=%s) for label %r — '
                'creating recovery branch to preserve WIP.',
                pop_rc, unmerged or [], pop_err, label,
            )
            recovery = await self._create_recovery_branch_from_stash(label)
            return (False, recovery)

        return (True, None)

    async def has_dirty_working_tree(self) -> str:
        """Return names of tracked dirty files, or empty string if clean.

        Excludes .task/ (ephemeral scratch) and untracked files.
        """
        _, unstaged, _ = await _run(
            ['git', 'diff', '--name-only', '--', '.', ':!.task'],
            cwd=self.project_root,
        )
        _, staged, _ = await _run(
            ['git', 'diff', '--name-only', '--cached', '--', '.', ':!.task'],
            cwd=self.project_root,
        )
        files = {f.strip() for f in (unstaged + '\n' + staged).splitlines() if f.strip()}
        return '\n'.join(sorted(files))

    async def _detect_unmerged_paths(self, cwd: Path) -> list[str]:
        """Return sorted list of file paths that are in an unmerged state.

        Uses ``git status --porcelain`` XY parsing — a path is unmerged if
        either the index (X) or working-tree (Y) column is ``U``, OR if both
        columns are the same add/delete marker (``AA`` or ``DD``).

        Returns an empty list when the tree is clean or fully merged.
        """
        _, porcelain, _ = await _run(
            ['git', 'status', '--porcelain'],
            cwd=cwd,
        )
        unmerged: list[str] = []
        for line in porcelain.splitlines():
            if len(line) < 4:
                continue
            xy = line[:2]
            path = line[3:]
            if 'U' in xy or xy in ('AA', 'DD'):
                unmerged.append(path.strip())
        return sorted(unmerged)

    async def get_conflict_details(self, cwd: Path) -> str:
        """Parse conflict markers and return structured description."""
        _, status, _ = await _run(['git', 'diff', '--name-only', '--diff-filter=U'], cwd=cwd)
        if not status:
            return 'No conflicting files detected'

        details = [f'Conflicting files:\n{status}\n']
        for filepath in status.splitlines():
            filepath = filepath.strip()
            if filepath:
                _, diff, _ = await _run(['git', 'diff', '--', filepath], cwd=cwd)
                details.append(f'--- {filepath} ---\n{diff[:2000]}')

        return '\n'.join(details)

    async def abort_merge(self, cwd: Path) -> None:
        """Abort an in-progress merge."""
        await _run(['git', 'merge', '--abort'], cwd=cwd)
        logger.info('Merge aborted')

    async def rename_worktree(
        self,
        old_path: Path,
        new_path: Path,
        old_branch: str,
        new_branch: str,
    ) -> None:
        """Rename a registered worktree and its branch atomically.

        Used by the auto-eval hook to preserve the original
        attempt's branch + worktree (suffixed ``-skip-attempt``) so the
        full-architect redo can use the original branch name without
        clobbering the artefacts of the optimistic-path attempt.

        Args:
            old_path: Current worktree path (registered with git).
            new_path: Destination worktree path (must not exist).
            old_branch: Branch name without the ``branch_prefix``.
            new_branch: Destination branch name without the ``branch_prefix``.

        Raises:
            RuntimeError: if ``git worktree move`` or ``git branch -m``
                returns a non-zero exit code. The caller is expected to
                surface this as an auto-eval failure and fall back to the
                normal block path.
        """
        full_old = f'{self.config.branch_prefix}{old_branch}'
        full_new = f'{self.config.branch_prefix}{new_branch}'

        new_path.parent.mkdir(parents=True, exist_ok=True)

        rc, _, err = await _run(
            ['git', 'worktree', 'move', str(old_path), str(new_path)],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(
                f'rename_worktree: git worktree move {old_path} -> '
                f'{new_path} failed (rc={rc}): {err}'
            )

        rc, _, err = await _run(
            ['git', 'branch', '-m', full_old, full_new],
            cwd=self.project_root,
        )
        if rc != 0:
            # Best-effort rollback of the worktree move so the caller can
            # retry. The directory rename is the half that actually
            # surfaces conflicts; the branch rename rarely fails alone.
            await _run(
                ['git', 'worktree', 'move', str(new_path), str(old_path)],
                cwd=self.project_root,
            )
            raise RuntimeError(
                f'rename_worktree: git branch -m {full_old} -> {full_new} '
                f'failed (rc={rc}): {err}'
            )

        logger.info(
            'Renamed worktree %s -> %s and branch %s -> %s',
            old_path, new_path, full_old, full_new,
        )

    async def cleanup_worktree(self, worktree: Path, branch: str) -> None:
        """Remove worktree and delete branch.

        **Pool-aware**: if *worktree* is a warm lane, routes to
        :meth:`release_warm_lane` (retain worktree + ``target/``, flip FREE)
        instead of removing.  Mirrors :meth:`cleanup_merge_worktree`'s
        persistent-path no-op pattern.  Covers the workflow done-gate and all
        harness reconcile call sites without touching each individually.
        """
        if self.warm_lane_pool is not None and self.warm_lane_pool.is_lane(worktree):
            await self.release_warm_lane(worktree, branch)
            return

        # Symmetric for the merge-speculation pool: a '_spec-' lane must be
        # RELEASED back to its pool (retain worktree + target/, flip FREE),
        # never removed.  Without this, a spec lane routed here (e.g. by the
        # crash-recovery sweep) would be git-worktree-removed and lose its
        # pool slot.  warm=True selects the pool-retain path in
        # release_spec_lane.
        if (
            self.spec_warm_lane_pool is not None
            and self.spec_warm_lane_pool.is_lane(worktree)
        ):
            await self.release_spec_lane(worktree, warm=True)
            return

        full_branch = f'{self.config.branch_prefix}{branch}'

        # Remove worktree
        rc, _, err = await _run(
            ['git', 'worktree', 'remove', str(worktree), '--force'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to remove worktree {worktree}: {err}')

        # Delete branch
        rc, _, err = await _run(
            ['git', 'branch', '-D', full_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to delete branch {full_branch}: {err}')

        logger.info(f'Cleaned up worktree {worktree} and branch {full_branch}')

    async def reclaim_worktree_build_artifacts(
        self,
        worktree: Path,
        dir_names: list[str] | None = None,
    ) -> list[Path]:
        """Remove regenerable build-artifact directories from a done worktree.

        Drops only the named build-output subdirectories (e.g. ``target/``)
        and never touches git refs, the worktree admin entry, or any other
        content.  This is appropriate when the task's merge commit is
        confirmed on main but the branch tip is a pre-rebase duplicate —
        the forensic history is preserved while the large regenerable cache
        is reclaimed.

        *dir_names* overrides which subdirectory names to reap.  When
        ``None``, falls back to ``self.config.reap_build_artifact_dirs``
        (default ``['target']``).

        Best-effort: each removal is wrapped in try/except; failures are
        logged as warnings but never propagated.  Mirrors the
        never-raise contract of ``cleanup_merge_worktree`` and
        ``prune_worktrees``.

        Returns the list of directory paths that were successfully removed.
        Returns ``[]`` when nothing was reaped (dirs absent or worktree
        path does not exist).
        """
        names = dir_names if dir_names is not None else self.config.reap_build_artifact_dirs
        removed: list[Path] = []

        for name in names:
            candidate = worktree / name
            if not candidate.is_dir():
                continue
            try:
                shutil.rmtree(candidate)
                removed.append(candidate)
            except Exception:
                logger.warning(
                    'reclaim_worktree_build_artifacts: failed to remove %s',
                    candidate, exc_info=True,
                )

        if removed:
            logger.info(
                'reclaim_worktree_build_artifacts: removed %d dir(s) from %s: %s',
                len(removed), worktree, [str(p) for p in removed],
            )
        return removed

    # ── Orphan-worktree hygiene (Fix B/C) ─────────────────────────────

    @property
    def quarantine_base(self) -> Path:
        """Sibling base for quarantined worktrees — OUTSIDE ``worktree_base``.

        A direct sibling (``<worktree_dir>-orphaned``) rather than a child, so
        a quarantined worktree is never re-scanned by crash-recovery or the
        orphan reaper (both iterate ``worktree_base`` only).
        """
        return self.worktree_base.parent / f'{self.worktree_base.name}-orphaned'

    async def worktree_has_unsaved_work(self, worktree: Path, branch: str) -> bool:
        """Whether a worktree holds work that must be preserved before removal.

        ``True`` if EITHER the branch carries commits beyond main
        (``rev-list --count main..task/<branch> > 0``) OR the working tree is
        dirty (``git status --porcelain`` non-empty).  **Fail-safe ``True``**
        on any git error (including a missing branch) — never report a worktree
        as safe-to-reap when we cannot prove it is empty and clean.
        """
        full_branch = f'{self.config.branch_prefix}{branch}'
        try:
            # Commits beyond main.  A missing branch makes rev-list fail → True.
            rc, out, _ = await _run(
                ['git', 'rev-list', '--count',
                 f'{self.config.main_branch}..{full_branch}'],
                cwd=self.project_root,
            )
            if rc != 0:
                return True
            if int(out.strip()) > 0:
                return True
            # No commits beyond main — check for uncommitted WIP in the tree.
            rc, status_out, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=worktree,
            )
            if rc != 0:
                return True
            return bool(status_out.strip())
        except (WorktreeMissing, ValueError, OSError) as e:
            logger.warning(
                'worktree_has_unsaved_work: error inspecting %s (%s) — '
                'treating as unsaved (fail-safe)', worktree, e,
            )
            return True

    async def quarantine_worktree(
        self, worktree: Path, branch: str, reason: str,
    ) -> Path | None:
        """Relocate a worktree (and its branch) into the quarantine base.

        Best-effort: commits any uncommitted WIP first (so it is preserved on
        the renamed branch), then moves the worktree to
        ``quarantine_base/<branch>-<UTC-ts>`` and renames the branch to
        ``task/<branch>-<ts>``.  Logs a WARNING and returns the destination
        path, or ``None`` if the relocation could not complete.  **Never
        raises** — callers treat a ``None`` return as "left in place".
        """
        ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        dest_name = f'{branch}-{ts}'
        dest_path = self.quarantine_base / dest_name
        try:
            # Preserve uncommitted WIP on the branch before relocating.
            try:
                await self.commit(worktree, f'chore: quarantine WIP ({reason})')
            except Exception as e:
                logger.warning(
                    'quarantine_worktree: WIP commit failed for %s (%s) — '
                    'continuing with relocation: %s', worktree, reason, e,
                )
            await self.rename_worktree(worktree, dest_path, branch, dest_name)
            logger.warning(
                'QUARANTINED worktree %s -> %s (reason=%s)',
                worktree, dest_path, reason,
            )
            return dest_path
        except Exception as e:
            logger.warning(
                'quarantine_worktree: failed to relocate %s (reason=%s): %s',
                worktree, reason, e,
            )
            return None

    async def prune_worktrees(self) -> None:
        """Best-effort ``git worktree prune`` — clears stale admin entries.

        Clears the ``.git/worktrees`` administrative records left behind by
        worktrees removed off-band (manual ``rm -rf``, quarantine, reap).
        Never raises.
        """
        try:
            rc, _, err = await _run(
                ['git', 'worktree', 'prune'], cwd=self.project_root,
            )
            if rc != 0:
                logger.warning('prune_worktrees: git worktree prune failed: %s', err)
        except Exception as e:
            logger.warning('prune_worktrees: git worktree prune raised: %s', e)
