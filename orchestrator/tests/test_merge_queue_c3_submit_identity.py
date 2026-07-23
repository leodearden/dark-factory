"""C3 submit-path request-identity gate tests (PRD merge-worktree-lifecycle-integrity §4 C3 / §5 D1-D3).

Covers the SHA-sensitive merge_request submit gate enforced at
``coalesce_or_enqueue_merge_request`` and surfaced through the escalation
``merge_request`` MCP tool:

- same branch + same SHA → COALESCE (never a second work item, even mid-verify — D1)
- same branch + new SHA + QUEUED (verify not started) → REPLACE iff descendant (D2)
- same branch + new SHA + IN VERIFY → structured REJECT ``duplicate_in_verify`` (D3)

Imports of ``orchestrator.merge_queue`` symbols are done LOCALLY inside each
test so a not-yet-defined symbol never breaks collection during RED.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import (
    InFlightMergeRegistry,
    MergeRequest,
    QueuedBranch,
)

# ---------------------------------------------------------------------------
# Real-git fixtures + helpers (per-file duplication convention — adapted from
# test_merge_queue_duplicate_submission.py:68-90 and
# test_remove_merge_worktree_guarded.py:47-84; merge_queue symbols imported
# LOCALLY inside each test so a not-yet-defined symbol never breaks collection).
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


async def _commit(repo: Path, name: str) -> str:
    """Add a commit on the current branch/HEAD and return its SHA.

    Successive calls build a linear history, so an earlier SHA is a strict
    ancestor (SUPERSET when it is the in-flight tip and a later SHA is
    submitted) of a later one.
    """
    (repo / f'{name}.txt').write_text(f'{name}\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', name], cwd=repo)
    return await _head_sha(repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_event_store(tmp_path: Path) -> EventStore:
    return EventStore(db_path=tmp_path / 'c3_events.db', run_id='c3-test')


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    request_id: str | None = None,
    snapshot_tip: str | None = None,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future (adapted from
    test_merge_queue_duplicate_submission.py:148)."""
    kwargs: dict[str, Any] = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    if snapshot_tip is not None:
        kwargs['snapshot_tip'] = snapshot_tip
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **kwargs,
    )


def _verify_snapshot(request_id: str, branch: str, *, verify_age_secs: float = 42.0):
    """Fake ``live_snapshot`` reporting *request_id* as IN VERIFY.

    Matches the SpeculativeMergeWorker.snapshot() entry schema (merge_queue.py
    :9251) — ``verify_started_at`` non-None ⟺ verify has started.
    """
    def _snap() -> dict:
        return {'entries': [{
            'request_id': request_id,
            'branch': branch,
            'state': 'verifying',
            'verify_started_at': 1000.0,
            'verify_age_secs': verify_age_secs,
        }]}
    return _snap


def _queued_snapshot(request_id: str, branch: str):
    """Fake ``live_snapshot`` reporting *request_id* as QUEUED (verify NOT started)."""
    def _snap() -> dict:
        return {'entries': [{
            'request_id': request_id,
            'branch': branch,
            'state': 'queued',
            'verify_started_at': None,
            'verify_age_secs': None,
        }]}
    return _snap


class _ScratchSpy:
    """Spy ``_FindInflightWorktreeP``: returns a planted scratch path and records
    each guarded-removal call (actually deleting the dir), so a REPLACE test can
    assert the gate cleaned the dropped queued entry's scratch via the C1
    primitive without depending on find_inflight_merge_worktree's subject-match
    internals (those have their own coverage in
    test_remove_merge_worktree_guarded.py).  Ancestry classification is driven
    by a REAL GitOps passed as ``classifier_git_ops`` — this spy is only the
    disk-scan/removal surface.
    """

    def __init__(self, scratch: Path | None) -> None:
        self._scratch = scratch
        self.removed: list[tuple[Path, str]] = []
        self.find_calls = 0

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:  # noqa: ARG002
        self.find_calls += 1
        return self._scratch

    async def remove_merge_worktree_guarded(self, path: Path, *, reason: str) -> str:
        self.removed.append((path, reason))
        if path.exists():
            shutil.rmtree(path)
        return 'removed'

    async def cleanup_merge_worktree(self, merge_wt: Path) -> None:  # noqa: ARG002
        return None


async def _make_patch_equivalent_twin(repo: Path) -> tuple[str, str]:
    """Return ``(tip_old, tip_new)`` that are topologically DIVERGENT but
    patch-id EQUIVALENT (a rebased twin — same diff, different commit).

    ``tip_old`` (on main) and ``tip_new`` (on a sibling branch off the same
    base) apply the SAME diff with DIFFERENT commit messages, so their SHAs
    differ (guaranteeing divergence, not SAME) while ``git patch-id`` — which
    ignores the message — matches, so resolve_divergent folds them to SUBSET.
    """
    base = await _head_sha(repo)
    (repo / 'feat.txt').write_text('hello\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'feat old'], cwd=repo)
    tip_old = await _head_sha(repo)
    await _run(['git', 'checkout', '-b', 'twin', base], cwd=repo)
    (repo / 'feat.txt').write_text('hello\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'feat new (rebased)'], cwd=repo)
    tip_new = await _head_sha(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return tip_old, tip_new


async def _make_divergent_forcepush(repo: Path) -> tuple[str, str]:
    """Return ``(tip_old, tip_new)`` that are DIVERGENT with genuinely NEW
    content (a real force-push, not a rebased twin) — different diff, so
    resolve_divergent classifies them SUPERSET (patch NOT contained)."""
    base = await _head_sha(repo)
    (repo / 'feat.txt').write_text('hello\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'feat old'], cwd=repo)
    tip_old = await _head_sha(repo)
    await _run(['git', 'checkout', '-b', 'forcepush', base], cwd=repo)
    (repo / 'feat.txt').write_text('DIFFERENT content\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'feat forcepush'], cwd=repo)
    tip_new = await _head_sha(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return tip_old, tip_new


# ---------------------------------------------------------------------------
# TestDecideC3SubmitAction — step-1 RED: pure decision function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDecideC3SubmitAction:
    """step-1 RED: decide_c3_submit_action pure mapping (PRD §4 C3 / §5 D1-D3).

    RED until step-2 impl: C3SubmitAction/decide_c3_submit_action don't exist.
    Mirrors the decide_attach_action test conventions (local imports, pure,
    no git I/O — resolved TipRelation + verifying flag → C3SubmitAction).

    Table:

    +------------------+------------------+------------------+
    | relation         | verifying=False  | verifying=True   |
    +==================+==================+==================+
    | SAME             | COALESCE         | COALESCE         |  (D1: same SHA never rejects)
    +------------------+------------------+------------------+
    | SUBSET           | COALESCE         | COALESCE         |  (stale retry / patch-id twin)
    +------------------+------------------+------------------+
    | SUPERSET         | REPLACE          | REJECT           |  (D2 queued / D3 in-verify)
    +------------------+------------------+------------------+
    | DIVERGENT        | ValueError       | ValueError       |  (resolve_divergent first)
    +------------------+------------------+------------------+
    """

    async def test_same_not_verifying_coalesces(self):
        """SAME + not verifying → COALESCE (D1)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SAME, verifying=False) is C3SubmitAction.COALESCE

    async def test_same_verifying_coalesces(self):
        """SAME + verifying → COALESCE (D1: same SHA never rejects/replaces even during verify)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SAME, verifying=True) is C3SubmitAction.COALESCE

    async def test_subset_not_verifying_coalesces(self):
        """SUBSET + not verifying → COALESCE (stale retry / patch-id-equivalent twin)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUBSET, verifying=False) is C3SubmitAction.COALESCE

    async def test_subset_verifying_coalesces(self):
        """SUBSET + verifying → COALESCE (containment coalesces regardless of verify state)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUBSET, verifying=True) is C3SubmitAction.COALESCE

    async def test_superset_not_verifying_replaces(self):
        """SUPERSET + not verifying (QUEUED) → REPLACE (D2: drop old queued, dispatch fresh)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUPERSET, verifying=False) is C3SubmitAction.REPLACE

    async def test_superset_verifying_rejects(self):
        """SUPERSET + verifying (IN VERIFY) → REJECT (D3: structured duplicate_in_verify)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUPERSET, verifying=True) is C3SubmitAction.REJECT

    async def test_divergent_raises_value_error(self):
        """DIVERGENT → ValueError (must resolve via resolve_divergent first)."""
        from orchestrator.merge_queue import TipRelation, decide_c3_submit_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_c3_submit_action(TipRelation.DIVERGENT, verifying=False)

    async def test_divergent_verifying_also_raises(self):
        """DIVERGENT + verifying → ValueError (same resolve-first constraint)."""
        from orchestrator.merge_queue import TipRelation, decide_c3_submit_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_c3_submit_action(TipRelation.DIVERGENT, verifying=True)


# ---------------------------------------------------------------------------
# TestC3SubmitGateInVerify — step-3 RED: coalesce_or_enqueue in-verify reject
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestC3SubmitGateInVerify:
    """step-3 RED: coalesce_or_enqueue_merge_request IN-VERIFY reject (D3) and
    the same-SHA-while-verifying coalesce contrast (D1).

    Real-git fixture so classify_tip_relation runs against real commits
    (SHA1 is a strict ancestor of the descendant SHA2 → SUPERSET). The
    in-verify state is signalled via a fake ``live_snapshot`` reporting the
    in-flight entry's request_id with ``verify_started_at`` set — the exact
    production signal the gate consumes (entry.verifying is left False so this
    proves the snapshot-based detection, not the flag).

    RED until step-4 wires the C3 route: current γ2 wiring RESNAPSHOTs a
    SUPERSET (verifying=entry.verifying=False) and independent-enqueues, so
    ``rejected`` stays False.
    """

    async def test_in_verify_newer_sha_rejects(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """(a) NEW descendant SHA while the earlier SHA is IN VERIFY → structured REJECT (D3).

        Negative assertion: a coalesce or dispatch (rejected=False) FAILS the
        test — the reject must actually fire.  The live in-flight entry must be
        left UNDISTURBED (still registered, primary future not cancelled).
        """
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        sha1 = await _commit(git_repo, 'c1')
        sha2 = await _commit(git_repo, 'c2')  # strict descendant of sha1 → SUPERSET

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '900', 'task-old', old_fut, request_id='mr-old', snapshot_tip=sha1,
        )

        req = _make_request('task-new', '900', tmp_path, config, request_id='mr-new', snapshot_tip=sha2)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=None,
            live_snapshot=_verify_snapshot('mr-old', '900', verify_age_secs=42.0),
            classifier_git_ops=git_ops,
        )

        assert result.rejected is True, (
            f'a newer SHA submitted while the earlier SHA is in verify must REJECT; got {result}'
        )
        assert result.reject_code == 'duplicate_in_verify'
        assert result.dispatched is False
        assert result.in_flight is False
        assert result.existing_sha == sha1
        assert result.inflight_request_id == 'mr-old'
        assert result.verify_age_secs is not None

        # Live entry UNDISTURBED — not dropped, not cancelled, nothing enqueued.
        assert registry.is_inflight('900')
        entry = registry.entry('900')
        assert entry is not None and entry.request_id == 'mr-old'
        assert not old_fut.cancelled()
        assert queue.qsize() == 0

    async def test_same_sha_while_verifying_coalesces(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """(b) SAME SHA while IN VERIFY → coalesce, NOT reject (D1)."""
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        sha1 = await _commit(git_repo, 'c1')

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '901', 'task-old', old_fut, request_id='mr-old', snapshot_tip=sha1,
        )

        req = _make_request('task-new', '901', tmp_path, config, request_id='mr-new', snapshot_tip=sha1)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=None,
            live_snapshot=_verify_snapshot('mr-old', '901'),
            classifier_git_ops=git_ops,
        )

        assert result.in_flight is True, (
            f'same SHA must coalesce even mid-verify (D1); got {result}'
        )
        assert result.rejected is False
        assert queue.qsize() == 0
        entry = registry.entry('901')
        assert entry is not None and entry.request_id == 'mr-old'


# ---------------------------------------------------------------------------
# TestC3SubmitGateQueued — step-5 RED: coalesce_or_enqueue queued replace/coalesce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestC3SubmitGateQueued:
    """step-5 RED: the QUEUED (verify-not-started) C3 arms with real commits —
    REPLACE (D2), SUBSET stale-retry coalesce, patch-id-equivalent-twin
    coalesce (amendment), and genuine-force-push REPLACE + divergence WARNING.

    Ancestry runs against real git (classifier_git_ops); the dropped queued
    entry's scratch is a planted dir returned by a spy _FindInflightWorktreeP.

    RED until step-6 swaps the interim independent-enqueue for the atomic
    drop-and-reacquire + C1 scratch clean: (a) the slot still holds the OLD
    request and the scratch is not cleaned, and (d) no divergence WARNING is
    logged.  (b)/(c) already coalesce (pinned here).
    """

    async def test_queued_newer_sha_replaces_and_cleans_scratch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """(a) REPLACE: NEW descendant SHA while QUEUED → drop old entry, dispatch
        fresh, clean the dropped scratch via the C1 guarded primitive (D2)."""
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        sha1 = await _commit(git_repo, 'c1')
        sha2 = await _commit(git_repo, 'c2')  # strict descendant → SUPERSET

        scratch = tmp_path / '_merge-scratch'
        scratch.mkdir()
        spy = _ScratchSpy(scratch)

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '910', 'task-old', old_fut, request_id='mr-old', snapshot_tip=sha1,
        )

        req = _make_request('task-new', '910', tmp_path, config, request_id='mr-new', snapshot_tip=sha2)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=spy,
            live_snapshot=_queued_snapshot('mr-old', '910'),  # verify NOT started
            classifier_git_ops=git_ops,
        )

        assert result.dispatched is True, (
            f'SUPERSET while queued must REPLACE-dispatch; got {result}'
        )
        assert result.in_flight is False
        assert result.rejected is False
        # Old entry DROPPED: the slot now holds the NEW request (one item/branch).
        entry = registry.entry('910')
        assert entry is not None and entry.request_id == 'mr-new', (
            f'REPLACE must drop the old entry and hold mr-new; got '
            f'{entry.request_id if entry else None}'
        )
        # Dropped scratch cleaned via the C1 guarded primitive.
        assert spy.removed == [(scratch, 'c3_replace_drop_queued')], (
            f'REPLACE must clean the dropped queued scratch; got {spy.removed}'
        )
        assert not scratch.exists()
        assert queue.qsize() == 1

    async def test_queued_older_sha_coalesces_no_scratch_clean(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """(b) SUBSET stale retry: an ancestor SHA → coalesce, old entry retained,
        scratch NOT cleaned."""
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        sha1 = await _commit(git_repo, 'c1')
        sha2 = await _commit(git_repo, 'c2')  # sha1 is ancestor of sha2

        scratch = tmp_path / '_merge-scratch'
        scratch.mkdir()
        spy = _ScratchSpy(scratch)

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '911', 'task-old', old_fut, request_id='mr-old', snapshot_tip=sha2,
        )

        # Submit the OLDER SHA1 (ancestor of in-flight sha2) → SUBSET.
        req = _make_request('task-new', '911', tmp_path, config, request_id='mr-new', snapshot_tip=sha1)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=spy,
            live_snapshot=_queued_snapshot('mr-old', '911'),
            classifier_git_ops=git_ops,
        )

        assert result.in_flight is True, f'SUBSET must coalesce; got {result}'
        assert result.rejected is False
        entry = registry.entry('911')
        assert entry is not None and entry.request_id == 'mr-old', 'old entry retained'
        assert spy.removed == [], 'coalesce must NOT clean scratch'
        assert scratch.exists()
        assert queue.qsize() == 0

    async def test_queued_patch_equivalent_twin_coalesces(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """(c) Patch-id-equivalent DIVERGENT (rebased twin) → coalesce (amendment):
        resolve_divergent folds it to SUBSET, so genuine content is never lost."""
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        tip_old, tip_new = await _make_patch_equivalent_twin(git_repo)

        scratch = tmp_path / '_merge-scratch'
        scratch.mkdir()
        spy = _ScratchSpy(scratch)

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '912', 'task-old', old_fut, request_id='mr-old', snapshot_tip=tip_old,
        )

        req = _make_request('task-new', '912', tmp_path, config, request_id='mr-new', snapshot_tip=tip_new)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=spy,
            live_snapshot=_queued_snapshot('mr-old', '912'),
            classifier_git_ops=git_ops,
        )

        assert result.in_flight is True, (
            f'patch-id-equivalent rebased twin must coalesce, not replace; got {result}'
        )
        assert result.rejected is False
        entry = registry.entry('912')
        assert entry is not None and entry.request_id == 'mr-old', 'old entry retained'
        assert spy.removed == [], 'coalesce must NOT clean scratch'
        assert queue.qsize() == 0

    async def test_queued_genuine_divergent_replaces_and_warns(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
        caplog,
    ) -> None:
        """(d) Genuine force-push DIVERGENT (new content) while QUEUED → REPLACE
        and log a divergence WARNING (D2)."""
        from orchestrator.merge_queue import coalesce_or_enqueue_merge_request

        tip_old, tip_new = await _make_divergent_forcepush(git_repo)

        scratch = tmp_path / '_merge-scratch'
        scratch.mkdir()
        spy = _ScratchSpy(scratch)

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = _make_event_store(tmp_path)

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '913', 'task-old', old_fut, request_id='mr-old', snapshot_tip=tip_old,
        )

        req = _make_request('task-new', '913', tmp_path, config, request_id='mr-new', snapshot_tip=tip_new)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await coalesce_or_enqueue_merge_request(
                queue, req, event_store, registry,
                git_ops=spy,
                live_snapshot=_queued_snapshot('mr-old', '913'),
                classifier_git_ops=git_ops,
            )

        assert result.dispatched is True, (
            f'genuine divergence while queued must REPLACE-dispatch; got {result}'
        )
        assert result.rejected is False
        entry = registry.entry('913')
        assert entry is not None and entry.request_id == 'mr-new', 'REPLACE holds mr-new'
        # A divergence (force-push) WARNING must be emitted (D2).
        assert any(
            'diverg' in r.message.lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), f'expected a divergence WARNING; got {[r.message for r in caplog.records]}'
        assert spy.removed == [(scratch, 'c3_replace_drop_queued')]
