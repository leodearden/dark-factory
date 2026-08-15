"""Tests for ``TaskWorkflow._reconcile_metadata_files_for_done``.

Refactored 2026-05-10 (Stage 2 of stuck-done recovery): truth source is
the merge-diff, not the architect's ``plan.files``.  The architect's plan
can include files that get squashed or refactored away before merge; the
merge-diff is the actual record of what landed.

Refactored again 2026-07-01 (Task 1965): the merge-diff anchor changed
from the stale two-dot ``base_commit..merge_sha`` (``base_commit`` is
captured once at worktree creation and never refreshed after a rebase, so
it can union in every sibling task that merged into main during the
window between this task's branch point and its own merge — the reported
cross-task ``metadata.files`` contamination) to the merge commit's OWN
first parent (``get_merge_commit_diff_files(merge_sha)``, i.e.
``merge_sha^1..merge_sha``) — a diff base captured atomically at merge
time that yields exactly this task's own branch changes.  ``_base_commit``
is no longer a diff input at all.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler, same_module_siblings
from shared.locking import directory_locks

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import files_to_modules
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# The depth these tests pin on their MagicMock config, in ONE place.
#
# The constant alone only deduplicates a literal — it CANNOT by itself stop
# the premise and the code under test from drifting apart, because it is
# itself the re-typed value: swap a `config.lock_depth` pin for a real config
# (which resolves the LIVE operational depth via the autouse
# `_isolate_orch_config` fixture) and `_LOCK_DEPTH` still reads 2, so every
# `files_to_modules(..., _LOCK_DEPTH)` precondition below would keep asserting
# 2 and keep PASSING against a workflow running at 12.  That silently-wrong
# outcome — rather than a loud precondition failure — is the failure mode task
# 3866 exists to document, so it is `_assert_depth_pin_holds` below, not the
# constant, that actually forecloses it.
_LOCK_DEPTH = 2


def _assert_depth_pin_holds(wf: TaskWorkflow) -> None:
    """Fail LOUDLY if the workflow's real depth diverges from `_LOCK_DEPTH`.

    Read through `wf.config` — the object the code under test actually
    consults — not through the local pin, so this compares the premise the
    preconditions were computed at against the depth the workflow will really
    run at.  Both factories call it at construction time, which is what makes
    the two impossible to diverge silently.
    """
    actual = wf.config.lock_depth
    assert actual == _LOCK_DEPTH, (
        f'this module computes its same-module/cross-module preconditions at '
        f'lock_depth={_LOCK_DEPTH}, but the workflow under test resolves '
        f'{actual}. Those preconditions are now false (or vacuous) for the code '
        'being exercised. Re-point _LOCK_DEPTH at the depth the workflow really '
        'uses — do NOT silence this by editing the preconditions alone (task '
        '3866).'
    )


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    backend_metadata: dict | None = None,
) -> tuple[TaskWorkflow, AsyncMock, AsyncMock, AsyncMock]:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = _LOCK_DEPTH
    config.project_root = project_root

    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.get_task = AsyncMock(
        return_value={'id': task_id, 'metadata': backend_metadata or {}}
    )

    git_ops = MagicMock()
    get_merge_diff_files = AsyncMock(return_value=([], None))
    git_ops.get_merge_diff_files = get_merge_diff_files
    get_merge_commit_diff_files = AsyncMock(return_value=([], None))
    git_ops.get_merge_commit_diff_files = get_merge_commit_diff_files

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    _assert_depth_pin_holds(wf)
    return wf, update_task, get_merge_diff_files, get_merge_commit_diff_files


def _persisted_payload(update_task: AsyncMock) -> dict:
    """Return the metadata dict passed to the most recent update_task call."""
    assert update_task.await_args is not None
    args, _ = update_task.await_args
    return args[1]


@pytest.mark.asyncio
async def test_writes_merge_diff_files_when_merge_sha_and_base_commit_set(
    tmp_path: Path,
):
    """Happy path: writes the first-parent merge-commit diff (not plan.files).

    The diff is anchored to the merge commit's OWN first parent via
    ``get_merge_commit_diff_files(merge_sha)`` — NOT
    ``base_commit..merge_sha``.  The old two-dot ``get_merge_diff_files``
    is no longer awaited by the reconcile at all.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    # plan.files names what the architect said it would touch — but the
    # merge actually landed a different set of paths.
    wf.plan = {'files': ['old/path.py'], 'steps': []}
    get_merge_commit_diff_files.return_value = (
        ['src/landed_a.py', 'src/landed_b.py'], None,
    )

    await wf._reconcile_metadata_files_for_done()

    get_merge_commit_diff_files.assert_awaited_once_with('b' * 40)
    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with(
        '101', {'files': ['src/landed_a.py', 'src/landed_b.py']},
    )
    # Anchor the read half of the RMW — prevents silent regression where
    # get_task is skipped and only the bare {'files': ...} payload is sent.
    wf.scheduler.get_task.assert_awaited_once_with('101')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reconcile_ignores_stale_base_commit_uses_merge_first_parent(
    tmp_path: Path,
):
    """Contamination regression guard: a stale ``_base_commit`` must never
    leak into the diff.

    ``_base_commit`` is captured once at worktree creation and never
    refreshed after a rebase, so by done-time it can be far behind the
    actual merge window — diffing it against ``merge_sha`` would union in
    every sibling task that merged in between (the reported cross-task
    ``metadata.files`` contamination).  The fix drops ``_base_commit`` as a
    diff input entirely: only ``merge_sha`` (via its first parent) feeds
    the diff.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    stale_base = 'c' * 40  # distinct from merge_sha — must never be diffed against
    wf._base_commit = stale_base
    wf._merge_sha = 'b' * 40
    # Sibling-free: what the merge commit's first parent actually yields.
    get_merge_commit_diff_files.return_value = (['src/only_mine.py'], None)

    await wf._reconcile_metadata_files_for_done()

    get_merge_commit_diff_files.assert_awaited_once_with('b' * 40)
    # The old two-dot method (the only one that could ever accept a base
    # SHA) must never be called — so stale_base can never reach a diff.
    get_merge_diff_files.assert_not_awaited()
    for call in get_merge_commit_diff_files.await_args_list:
        assert stale_base not in call.args, (
            f'stale _base_commit must never be passed to a diff call; got {call}'
        )
    update_task.assert_awaited_once_with('101', {'files': ['src/only_mine.py']})


@pytest.mark.asyncio
async def test_clears_when_merge_sha_missing(tmp_path: Path):
    """No merge_sha (already-on-main shortcuts) → write empty list.

    The gate-skip-when-verified-provenance branch in fused-memory's
    task_interceptor.py handles the missing-files case.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = None
    wf.plan = {'files': ['anything.py']}

    await wf._reconcile_metadata_files_for_done()

    get_merge_commit_diff_files.assert_not_awaited()
    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_computes_diff_when_base_commit_missing_but_merge_sha_set(
    tmp_path: Path,
):
    """``base_commit`` is no longer a diff input — its absence is irrelevant.

    Pre-fix, a missing ``_base_commit`` short-circuited to an empty list
    (``if self._merge_sha and self._base_commit``).  Post-fix the gate is
    ``if self._merge_sha`` alone: the first-parent diff is computed from
    ``merge_sha`` regardless of whether ``_base_commit`` was ever captured
    (e.g. eval mode without ``create_worktree``).
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    wf._base_commit = None
    wf._merge_sha = 'b' * 40
    get_merge_commit_diff_files.return_value = (['src/landed.py'], None)

    await wf._reconcile_metadata_files_for_done()

    get_merge_commit_diff_files.assert_awaited_once_with('b' * 40)
    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': ['src/landed.py']})


@pytest.mark.asyncio
async def test_writes_empty_list_when_diff_returns_empty(tmp_path: Path):
    """Empty diff (e.g. revert merge) → empty list, no error."""
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_commit_diff_files.return_value = ([], None)

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_reconcile_preserves_memory_hints_from_backend(tmp_path: Path):
    """Sibling keys added by Stage-2 reconciliation survive the files write.

    The in-memory task dict has no memory_hints; the backend (get_task)
    has memory_hints + _causation_id attached after the workflow loaded.
    The persisted payload must contain BOTH the recomputed files AND the
    preserved sibling keys.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(
            project_root=tmp_path,
            backend_metadata={
                'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
                '_causation_id': 'C1',
            },
        )
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_commit_diff_files.return_value = (['src/landed.py'], None)

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload == {
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
        '_causation_id': 'C1',
        'files': ['src/landed.py'],
    }, f'Sibling keys from backend must survive the files write; got {payload}'


@pytest.mark.asyncio
async def test_reconcile_explicit_files_override_backend_files(tmp_path: Path):
    """New 'files' value always wins over any pre-existing backend 'files'.

    Even when the backend already has a stale 'files' list, the freshly
    computed merge-diff replaces it.  Sibling keys are still preserved.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(
            project_root=tmp_path,
            backend_metadata={
                'files': ['old/stale.py'],
                'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
            },
        )
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_commit_diff_files.return_value = (['src/new.py'], None)

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload.get('files') == ['src/new.py'], (
        f"New files must override stale backend files; got {payload.get('files')!r}"
    )
    assert payload.get('memory_hints') == {'entities': ['E1'], 'queries': ['q1']}, (
        f"memory_hints must survive when files key is overridden; got {payload!r}"
    )


@pytest.mark.asyncio
async def test_reconcile_writes_empty_files_on_git_error_fail_open(tmp_path: Path):
    """When get_merge_commit_diff_files returns ([], error), files=[] is written (fail-open).

    Sibling keys from the backend are still preserved — the read-modify-write
    structure is unchanged even on the error path.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(
            project_root=tmp_path,
            backend_metadata={
                'memory_hints': {'entities': ['E2'], 'queries': ['q2']},
            },
        )
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_commit_diff_files.return_value = ([], OSError('diff failed'))

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload.get('files') == [], (
        f'files must be [] (fail-open) on git error; got {payload.get("files")!r}'
    )
    assert payload.get('memory_hints') == {'entities': ['E2'], 'queries': ['q2']}, (
        f'memory_hints must survive even on git error path; got {payload!r}'
    )


@pytest.mark.asyncio
async def test_reconcile_strips_directory_shaped_merge_diff_files(tmp_path: Path):
    """Directory-shaped entries from git diff are stripped before persisting.

    The strip itself (``strip_directory_locks`` applied to diff output before
    writing, mirroring the change-#1 ``scheduler._persist_files_metadata`` fix)
    is unchanged and still pinned below.  What changed is WHICH entries are
    directory-shaped.

    AMENDED by dark_factory #3248 — this test previously encoded the bug.  Its
    fixture used ``fused-memory/docker/Dockerfile`` and ``hooks/pre-commit`` as
    the entries that "must be stripped", and its own docstring explained why
    that was wrong without drawing the conclusion: it argued that the only
    directory-shaped output ``git diff --name-only`` can realistically produce
    is an **extension-less tracked FILE**, because "git diff --name-only lists
    *files*, never bare directories".  Both fixture entries were, by that same
    reasoning, real ``git ls-files`` paths.  Asserting they be stripped was
    therefore asserting that real tracked files be silently erased from
    ``metadata.files`` on every reconcile — the exact data-loss defect #3248
    fixes, not a property worth pinning.

    Since #3248 those names are in ``EXTENSIONLESS_FILENAMES`` and classify as
    files, so they now SURVIVE, and this test pins that.  A genuinely
    directory-shaped entry is kept in the fixture so the strip is still proven
    to function — it cannot arise from ``--name-only`` output, but the strip is
    defensive and the coverage is worth keeping.

    Note the classification rule is per-segment-extension (plus the enumerated
    extension-less name list), NOT "dotfiles as a class" — ``.gitignore`` IS
    file-shaped (``gitignore`` is allowlisted).  The converse case (a leading-dot
    segment that stays directory-shaped, e.g. ``.worktrees``) is pinned in the
    predicate suites themselves (``test_leading_dot_directories_stay_directories``
    in both ``shared/tests/test_locking.py`` and
    ``fused-memory/tests/test_lock_charter_guard.py``) and deliberately is NOT
    re-asserted here, because such a path cannot appear in a merge diff.
    """
    wf, update_task, get_merge_diff_files, get_merge_commit_diff_files = (
        _make_workflow(project_root=tmp_path)
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    # Mixed: the two extension-less entries are real tracked FILES and must
    # SURVIVE (#3248); 'crates/reify-eval/src' is a real directory and must be
    # stripped; the .py files are ordinary file-level entries.
    get_merge_commit_diff_files.return_value = (
        [
            'fused-memory/docker/Dockerfile',
            'hooks/pre-commit',
            'crates/reify-eval/src',
            'src/landed.py',
            'orchestrator/src/orchestrator/workflow.py',
        ],
        None,
    )

    await wf._reconcile_metadata_files_for_done()

    persisted = _persisted_payload(update_task)['files']
    # Only file-level entries must survive (order preserved).
    assert persisted == [
        'fused-memory/docker/Dockerfile',
        'hooks/pre-commit',
        'src/landed.py',
        'orchestrator/src/orchestrator/workflow.py',
    ], (
        f'directory-shaped entries must be stripped and real tracked files '
        f'must survive; got {persisted!r}'
    )
    # The persisted set passes the update_task lock-charter guard.
    assert directory_locks(persisted) == [], (
        f'persisted files must all be file-level (no directory locks); got {directory_locks(persisted)!r}'
    )
    # Optimistic in-memory copy must match the persisted (stripped) list.
    assert wf.task['metadata']['files'] == persisted, (
        f'in-memory task metadata.files must equal persisted files; got {wf.task["metadata"]["files"]!r}'
    )


# ---------------------------------------------------------------------------
# TestReplanScopeReconcile — post-``_replan`` scope reconcile (task 2874)
#
# On a BLOCKING review verdict, ``_execute_verify_review_loop`` calls
# ``_replan()``, which lets the architect rewrite plan.json ("you may add
# new steps") — the architect naturally widens ``plan.files`` for the files
# those new steps touch.  The post-replan block previously only re-read the
# plan and validated steps/prerequisites — it never routed the possibly-
# widened ``plan.files`` through the scope-reconciliation choke point
# (``TaskWorkflow._set_task_scope``), so ``metadata.files``/module-locks
# could lag ``plan.files`` for an entire ``reviewer_comprehensive`` pass —
# the window the orphan-reaper sweeps in and fires the
# ``plan.files ⊋ metadata.files`` divergence (esc-2865-11 cluster).
#
# These tests drive one EXECUTE→VERIFY→REVIEW cycle with a BLOCKING verdict
# (forcing a replan) and assert the reconcile happened BEFORE the loop
# re-enters EXECUTE for the second time — the orphan-reaper sweep point.
# ---------------------------------------------------------------------------


def _make_replan_workflow(
    *,
    tmp_path: Path,
    initial_files: list[str],
    modules: list[str],
    task_id: str = '2874',
    max_review_cycles: int = 2,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Loop-driving harness for ``_execute_verify_review_loop``.

    Modeled on ``test_workflow_review_persistence.py::_make_workflow`` but
    wired to ``_workflow_helpers.FakeScheduler`` (instead of a bare
    MagicMock scheduler) so ``blast_radius_calls``/``update_task_calls``/
    ``blast_radius_result`` are trackable — the scope-reconcile assertion
    seam this task's tests need.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = list(modules)

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = max_review_cycles
    config.max_amendment_rounds = 1
    config.lock_depth = _LOCK_DEPTH
    config.project_root = tmp_path / 'proj'
    # Real reviewer model string so _reviewer_config_fingerprint() (task 2749)
    # yields a deterministic, non-None digest — mirrors
    # test_workflow_review_persistence.py::_make_workflow.
    config.models = MagicMock()
    config.models.reviewer = 'sonnet'

    scheduler = FakeScheduler()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    _assert_depth_pin_holds(wf)
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd')
    wf.artifacts = artifacts
    wf.worktree = worktree
    wf.modules = list(modules)

    plan = {
        'task_id': task_id,
        'title': 'T',
        'files': list(initial_files),
        'analysis': 'a',
        'prerequisites': [],
        'steps': [
            {
                'id': 'step-1',
                'type': 'test',
                'description': 'x',
                'status': 'pending',
                'commit': None,
            },
        ],
        'design_decisions': [],
        'reuse': [],
    }
    artifacts.write_plan(plan)
    artifacts.stamp_plan_provenance(wf.session_id)
    wf.plan = artifacts.read_plan()

    # Loop sub-steps default to success; the REVIEW/replan branch is what
    # each test exercises, so stub VERIFY to DONE and the git_ops seams
    # _execute_verify_review_loop reads directly.
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf.git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')
    wf.git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value={})
    wf.git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    wf._amend = AsyncMock(return_value=True)

    return wf, scheduler


def _snapshotting_execute_iterations(
    scheduler: FakeScheduler, snapshots: list[dict],
) -> AsyncMock:
    """AsyncMock for ``wf._execute_iterations`` that snapshots the
    scheduler's blast-radius/update_task call lists on EVERY invocation, so
    ``snapshots[1]`` captures state at the entry of the 2nd EXECUTE pass —
    the orphan-reaper sweep point (the reviewer_comprehensive pass runs
    between the replan and this 2nd EXECUTE)."""

    async def _side_effect(*_args, **_kwargs) -> WorkflowOutcome:
        snapshots.append({
            'blast': list(scheduler.blast_radius_calls),
            'update': list(scheduler.update_task_calls),
        })
        return WorkflowOutcome.DONE

    return AsyncMock(side_effect=_side_effect)


def _widening_replan(wf: TaskWorkflow, new_file: str) -> AsyncMock:
    """AsyncMock for ``wf._replan`` that mimics an architect replan which
    WIDENS plan.files — appends *new_file*, keeps the existing steps, and
    resets ``prerequisites`` to ``[]`` (mirrors what a real replanning
    architect leaves in plan.json for a single-file widen)."""

    async def _side_effect(_reviews) -> None:
        assert wf.artifacts is not None
        plan = wf.artifacts.read_plan()
        plan['files'] = [*plan.get('files', []), new_file]
        plan['prerequisites'] = []
        wf.artifacts.write_plan(plan)
        wf.plan = plan

    return AsyncMock(side_effect=_side_effect)


def _blocking_then_pass_reviews() -> AsyncMock:
    """AsyncMock for ``wf._review``: cycle 1 BLOCKING (forces a replan),
    cycle 2 PASS (drives the loop to DONE after the reconcile — only
    reached if the fix does not short-circuit on a lock conflict)."""
    return AsyncMock(side_effect=[
        ReviewAggregation(
            has_blocking_issues=True,
            blocking_issues=[{
                'reviewer': 'reviewer_comprehensive',
                'severity': 'blocking',
                'location': 'x.py:1',
                'category': 'missing_edge_case',
                'description': 'needs a fix',
            }],
            suggestions=[],
            reviews={},
        ),
        ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[],
            reviews={},
        ),
    ])


@pytest.mark.asyncio
class TestReplanScopeReconcile:
    """Acceptance tests for task 2874: the post-``_replan`` block must
    reconcile a possibly-widened ``plan.files`` through ``_set_task_scope``
    BEFORE the loop re-enters EXECUTE — closing the window where
    ``metadata.files``/module-locks could lag ``plan.files`` for an entire
    ``reviewer_comprehensive`` pass.
    """

    async def test_new_module_widen_persists_and_locks_before_next_execute(
        self, tmp_path: Path,
    ):
        initial_files = ['a/b/c/d/e.py']
        new_file = 'x/y/z/w/v.py'
        modules = files_to_modules(initial_files, _LOCK_DEPTH)
        assert files_to_modules(initial_files, _LOCK_DEPTH) != files_to_modules(
            [*initial_files, new_file], _LOCK_DEPTH,
        ), 'precondition: the new file must widen the MODULE set'

        wf, scheduler = _make_replan_workflow(
            tmp_path=tmp_path, initial_files=initial_files, modules=modules,
        )
        snapshots: list[dict] = []
        wf._execute_iterations = _snapshotting_execute_iterations(scheduler, snapshots)
        wf._review = _blocking_then_pass_reviews()
        wf._replan = _widening_replan(wf, new_file)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        assert len(snapshots) == 2, (
            f'expected exactly 2 _execute_iterations calls; got {len(snapshots)}'
        )
        assert any(
            persist_files is not None and new_file in persist_files
            for _current, _needed, persist_files in snapshots[1]['blast']
        ), (
            'expected a handle_blast_radius_expansion call persisting the '
            f'widened files (including {new_file!r}) before the 2nd EXECUTE; '
            f'got {snapshots[1]["blast"]!r}'
        )
        widened_files = [*initial_files, new_file]
        assert wf.modules == files_to_modules(widened_files, _LOCK_DEPTH), (
            'wf.modules must cover the new file module before the 2nd '
            f'EXECUTE; got {wf.modules!r}'
        )

    async def test_same_module_widen_persists_via_update_task_before_next_execute(
        self, tmp_path: Path,
    ):
        # Derived, not a literal — see same_module_siblings' docstring for why.
        f1, f2 = same_module_siblings(_LOCK_DEPTH)
        assert files_to_modules([f1], _LOCK_DEPTH) == files_to_modules(
            [f1, f2], _LOCK_DEPTH,
        ), 'precondition: f1 and f2 must map to the SAME module set'
        modules = files_to_modules([f1], _LOCK_DEPTH)

        wf, scheduler = _make_replan_workflow(
            tmp_path=tmp_path, initial_files=[f1], modules=modules,
        )
        snapshots: list[dict] = []
        wf._execute_iterations = _snapshotting_execute_iterations(scheduler, snapshots)
        wf._review = _blocking_then_pass_reviews()
        wf._replan = _widening_replan(wf, f2)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        assert len(snapshots) == 2, (
            f'expected exactly 2 _execute_iterations calls; got {len(snapshots)}'
        )
        assert not any(
            persist_files is not None and f2 in persist_files
            for _current, _needed, persist_files in snapshots[1]['blast']
        ), (
            'a same-module widen must NOT call handle_blast_radius_expansion; '
            f'got {snapshots[1]["blast"]!r}'
        )
        assert any(
            f2 in (metadata.get('files') or [])
            for _tid, metadata in snapshots[1]['update']
            if isinstance(metadata, dict)
        ), (
            'expected metadata.files to be persisted directly (via update_task) '
            f'including {f2!r} before the 2nd EXECUTE; got {snapshots[1]["update"]!r}'
        )

    async def test_cross_module_lock_conflict_requeues(self, tmp_path: Path):
        """A genuine cross-module lock conflict on the replan reconcile must
        REQUEUE — not silently proceed into the 2nd EXECUTE under a foreign
        lock."""
        initial_files = ['a/b/c/d/e.py']
        new_file = 'x/y/z/w/v.py'
        modules = files_to_modules(initial_files, _LOCK_DEPTH)
        assert files_to_modules(initial_files, _LOCK_DEPTH) != files_to_modules(
            [*initial_files, new_file], _LOCK_DEPTH,
        ), 'precondition: the new file must widen the MODULE set'

        wf, scheduler = _make_replan_workflow(
            tmp_path=tmp_path, initial_files=initial_files, modules=modules,
        )
        scheduler.blast_radius_result = False  # a sibling holds the additional lock
        snapshots: list[dict] = []
        wf._execute_iterations = _snapshotting_execute_iterations(scheduler, snapshots)
        wf._review = _blocking_then_pass_reviews()
        wf._replan = _widening_replan(wf, new_file)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.REQUEUED
        report = wf._terminal_report
        assert report is not None
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.reason == 'plan_blast_radius_lock_conflict'
        assert len(snapshots) == 1, (
            'the 2nd EXECUTE must never run under a foreign lock; '
            f'got {len(snapshots)} _execute_iterations call(s)'
        )
