"""Tests for MergeQueueStore and related utilities (task 1772).

Covers:
  step-1:  record() + load() round-trips a single MergeRequest's identity.
  step-3:  remove() removes one entry; unknown id is a no-op.
  step-5:  load() robustness (non-existent path, empty file, corrupt file,
           idempotent record by request_id).
  step-7:  record() skips GroupMergeRequest.
  step-9:  reconstruct_merge_request() returns a fresh MergeRequest with
           correct identity.
  step-11: recover_pending_merges() selects survivors and drops already-landed
           / branch-gone records.
  task-1808: journal_corrupt flag and corrupt-signal propagation into recover report.
  task-2037: record() strips a leading branch_prefix so the journal only ever
             holds the bare canonical branch shape; recover_pending_merges
             tolerates legacy already-prefixed journal entries via the shared
             canonical_queued_branch_name helper (double-prefix branch-drop fix).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from _orch_helpers import make_placeholder_future
from test_verify_merge_flake_suppression import _module_config

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.merge_queue import GroupMergeRequest, MergeRequest

# Import the module under test — will fail (ImportError) until step-2 creates it.
from orchestrator.merge_queue_store import (
    MergeQueueStore,
    PersistedMergeRequest,
    reconstruct_merge_request,
    recover_pending_merges,
)
from orchestrator.merge_types import InFlightMergeRegistry, QueuedBranch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _real_config(
    tmp_path: Path,
    *,
    merge_verify_breadth: Literal['scoped', 'full'] = 'scoped',
) -> OrchestratorConfig:
    """Minimal real OrchestratorConfig pointing at tmp_path.

    *merge_verify_breadth* mirrors ``test_merge_queue_main_health._make_config``'s
    knob of the same name; the default preserves every pre-existing call site.
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        merge_verify_breadth=merge_verify_breadth,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
    )


def _make_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    snapshot_tip: str | None = 'abc123',
    generation: int = 1,
    lane: str = 'normal',
    task_files: list[str] | None = None,
    pre_rebased: bool = False,
    module_configs: list[ModuleConfig] | None = None,
) -> MergeRequest:
    """Build a MergeRequest with a placeholder future (safe outside a running loop).

    *module_configs* defaults to ``None`` -> ``[]``, so every pre-existing call
    site is byte-identical to the old hardcoded empty list.
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=task_files,
        module_configs=list(module_configs or []),
        config=config,
        result=make_placeholder_future(),
        snapshot_tip=snapshot_tip,
        generation=generation,
        lane=lane,  # type: ignore[arg-type]
    )


NINE_PREFIXES = [
    'cockpit', 'dashboard', 'escalation', 'fused-memory', 'orchestrator',
    'sampler', 'scripts', 'shared', 'tests/scripts',
]
"""dark_factory's live nine-module registry shape.

THE single definition, cross-imported by
``test_merge_boundary_effective_module_configs`` so the two task-5063 suites
cannot silently disagree about what "the whole registry" means.  Exported
without the leading underscore precisely because it is cross-module.

``tests/scripts`` is included deliberately: it is multi-segment and so
exercises ``OrchestratorConfig.for_module``'s inward walk.
"""


def _config_with_modules(
    tmp_path: Path,
    prefixes: list[str],
    *,
    merge_verify_breadth: Literal['scoped', 'full'] = 'full',
) -> OrchestratorConfig:
    """A real OrchestratorConfig whose module registry holds *prefixes*.

    Populates the ``_module_configs`` PrivateAttr directly — the repo-wide
    blessed idiom for a module registry in tests (``conftest.mock_orch_config``
    structurally cannot carry a PrivateAttr, and a bare MagicMock config is
    rejected by orchestrator's ``check_bare_magicmock_config.py`` lint).

    Each entry is built by ``test_verify_merge_flake_suppression._module_config``
    — THE single definition of the fixture's command shape, per that helper's
    own docstring — rather than hand-copied here, matching the sibling
    ``test_merge_boundary_effective_module_configs``.  All three commands are
    non-None there, so ``verify_plan._derive_full_suite_runs`` emits real
    FULL_SUITE runs rather than reasoned SKIPPED ones.
    """
    config = _real_config(tmp_path, merge_verify_breadth=merge_verify_breadth)
    config._module_configs = {prefix: _module_config(prefix) for prefix in prefixes}
    return config



def _make_group_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> GroupMergeRequest:
    """Build a GroupMergeRequest with dummy train fields and async callbacks."""
    async def _status_check(ids: list[str]) -> dict[str, str]:
        return {}

    async def _mark_done(tid: str, sha: str) -> None:
        pass

    return GroupMergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        train_id='train-001',
        member_task_ids=[task_id],
        tip_branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        tip_task_id=task_id,
        status_check=_status_check,
        mark_member_done=_mark_done,
    )


# ---------------------------------------------------------------------------
# step-1 — record() + load() round-trip
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRecordLoad:
    """record() + load() round-trips a single-task MergeRequest's serializable identity."""

    def test_record_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify all serializable identity fields survive record + load."""
        store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        worktree = tmp_path / '.worktrees' / 'task-42'
        worktree.mkdir(parents=True, exist_ok=True)

        req = _make_req(
            task_id='42',
            branch='42',
            worktree=worktree,
            config=config,
            snapshot_tip='deadbeef',
            generation=2,
            lane='high',
            task_files=['src/foo.py', 'tests/test_foo.py'],
            pre_rebased=True,
            module_configs=[ModuleConfig(prefix='orchestrator')],
        )

        store.record(req)
        records = store.load()

        assert len(records) == 1, f'Expected 1 record, got {len(records)}'
        p = records[0]
        assert isinstance(p, PersistedMergeRequest)
        assert p.request_id == req.request_id
        assert p.task_id == '42'
        assert p.branch == '42'
        assert p.worktree == str(worktree)
        assert p.pre_rebased is True
        assert p.task_files == ['src/foo.py', 'tests/test_foo.py']
        assert p.snapshot_tip == 'deadbeef'
        assert p.generation == 2
        assert p.lane == 'high'
        assert p.enqueued_at == pytest.approx(req.enqueued_at, rel=1e-6)
        assert p.module_prefixes == ['orchestrator']


# ---------------------------------------------------------------------------
# step-3 — remove()
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRemove:
    """remove() deletes a single entry; unknown id is a no-op."""

    def test_remove_one_keeps_other(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req_a = _make_req('a', 'a', wt, config)
        req_b = _make_req('b', 'b', wt, config)

        store.record(req_a)
        store.record(req_b)

        store.remove(req_a.request_id)
        records = store.load()

        assert len(records) == 1
        assert records[0].request_id == req_b.request_id

    def test_remove_unknown_is_noop(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req('x', 'x', wt, config)
        store.record(req)

        # Removing a nonexistent id must not raise or remove the existing entry.
        store.remove('mr-does-not-exist')
        records = store.load()
        assert len(records) == 1
        assert records[0].request_id == req.request_id


# ---------------------------------------------------------------------------
# step-5 — robustness: fail-open loads, idempotent record
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRobustness:
    """load() is fail-open; record() is idempotent by request_id."""

    def test_load_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        store = MergeQueueStore(tmp_path / 'nonexistent' / 'merge_queue.json')
        assert store.load() == []

    def test_load_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / 'merge_queue.json'
        p.write_text('', encoding='utf-8')
        store = MergeQueueStore(p)
        assert store.load() == []

    def test_load_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / 'merge_queue.json'
        p.write_text('not json {{{{', encoding='utf-8')
        store = MergeQueueStore(p)
        assert store.load() == []

    def test_record_idempotent_by_request_id(self, tmp_path: Path) -> None:
        """Recording the same request_id twice yields exactly one entry."""
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req('t', 't', wt, config, generation=1)
        store.record(req)

        # Mutate generation on the SAME request object (same request_id) and record again.
        object.__setattr__(req, 'generation', 3) if hasattr(req, '__dataclass_fields__') else None
        req2 = MergeRequest(
            task_id=req.task_id,
            branch=req.branch,
            worktree=req.worktree,
            pre_rebased=req.pre_rebased,
            task_files=req.task_files,
            module_configs=[],
            config=config,
            result=make_placeholder_future(),
            request_id=req.request_id,  # SAME id
            snapshot_tip=req.snapshot_tip,
            generation=3,             # updated value
            lane=req.lane,
            enqueued_at=req.enqueued_at,
        )
        store.record(req2)

        records = store.load()
        assert len(records) == 1, f'Expected 1 (idempotent), got {len(records)}'
        assert records[0].generation == 3, 'Latest value should overwrite the previous'


# ---------------------------------------------------------------------------
# step-7 — GroupMergeRequest is NOT journaled
# ---------------------------------------------------------------------------


class TestMergeQueueStoreSkipsGroupMergeRequest:
    """record() returns early for GroupMergeRequest; load() stays empty."""

    def test_group_request_not_recorded(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        group_req = _make_group_req('train-tip', 'train-tip', wt, config)
        store.record(group_req)

        assert store.load() == [], 'GroupMergeRequest must not be persisted'


# ---------------------------------------------------------------------------
# step-9 — reconstruct_merge_request()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconstructMergeRequest:
    """reconstruct_merge_request returns a MergeRequest with correct identity."""

    async def test_reconstruct_preserves_identity(self, tmp_path: Path) -> None:
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        # Build a PersistedMergeRequest directly.
        original = _make_req(
            '99',
            '99',
            wt,
            config,
            snapshot_tip='cafebabe',
            generation=2,
            lane='high',
            task_files=['a.py'],
        )
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        store.record(original)
        [persisted] = store.load()

        reconstructed = reconstruct_merge_request(persisted, config)

        # Must be a MergeRequest (not a GroupMergeRequest).
        assert type(reconstructed) is MergeRequest

        # Fresh unresolved future.
        assert isinstance(reconstructed.result, asyncio.Future)
        assert not reconstructed.result.done(), 'Future should NOT be resolved yet'

        # Identity fields preserved.
        assert reconstructed.request_id == original.request_id
        assert reconstructed.task_id == '99'
        assert reconstructed.branch.bare_id == '99'
        assert reconstructed.snapshot_tip == 'cafebabe'
        assert reconstructed.generation == 2
        assert reconstructed.lane == 'high'
        assert reconstructed.task_files == ['a.py']
        assert reconstructed.worktree == wt
        assert isinstance(reconstructed.worktree, Path)

        # Defaults for non-serializable fields.
        assert reconstructed.module_configs == []
        assert reconstructed.pre_rebased is False
        assert reconstructed.config is config


# ---------------------------------------------------------------------------
# step-11 — recover_pending_merges() with fake git_ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMerges:
    """recover_pending_merges selects survivors and drops already-landed or gone branches."""

    async def test_recover_selects_survivors(self, tmp_path: Path) -> None:
        """
        Three records:
          A — branch exists and NOT an ancestor of main → re-enqueued (recovered)
          B — is_ancestor True → already merged, dropped
          C — resolve_branch_sha None → branch gone, dropped
        """
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)

        req_a = _make_req('a', 'a', wt, config)
        req_b = _make_req('b', 'b', wt, config)
        req_c = _make_req('c', 'c', wt, config)
        store.record(req_a)
        store.record(req_b)
        store.record(req_c)

        main_branch = 'main'
        branch_prefix = 'task/'

        # Fake git_ops:
        # - A: branch exists (resolve returns sha), NOT ancestor
        # - B: branch exists, IS ancestor
        # - C: branch gone (resolve returns None)
        async def fake_resolve(branch: str) -> str | None:
            mapping = {
                f'{branch_prefix}a': 'sha-a',
                f'{branch_prefix}b': 'sha-b',
                f'{branch_prefix}c': None,
            }
            return mapping.get(branch)

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            return ancestor == f'{branch_prefix}b'

        fake_git_ops = MagicMock()
        fake_git_ops.resolve_branch_sha = fake_resolve
        fake_git_ops.is_ancestor = fake_is_ancestor

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store,
            queue,
            fake_git_ops,
            config,
            event_store=None,
            main_branch=main_branch,
            branch_prefix=branch_prefix,
        )

        # --- queue assertions ---
        assert queue.qsize() == 1, f'Expected exactly 1 re-enqueued item; got {queue.qsize()}'
        recovered_req = queue.get_nowait()
        assert recovered_req.request_id == req_a.request_id, (
            f'Expected req_a to be recovered; got {recovered_req.request_id!r}'
        )
        assert not recovered_req.result.done(), 'Recovered request must have unresolved future'

        # --- drop assertions ---
        remaining = store.load()
        remaining_ids = {r.request_id for r in remaining}
        assert req_b.request_id not in remaining_ids, 'Already-merged B must be removed'
        assert req_c.request_id not in remaining_ids, 'Branch-gone C must be removed'

        # --- report ---
        assert report['recovered'] == 1
        assert report['dropped'] == 2
        assert len(report['requests']) == 1
        assert report['requests'][0].request_id == req_a.request_id


# ---------------------------------------------------------------------------
# task 2926 (C3 γ) step-3 — recover_pending_merges registry-gated dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMergesRegistryDedup:
    """task 2926 (C3 γ) step-3 RED: recover_pending_merges collapses per-branch
    duplicates through the InFlightMergeRegistry BEFORE enqueue.

    RED until step-4 adds the ``registry`` kwarg + the Phase-2 collapse. Reuses
    the ``_make_req`` + fake-git_ops resolve_branch_sha/is_ancestor scaffolding
    from :class:`TestRecoverPendingMerges`; constructs a REAL
    ``InFlightMergeRegistry`` so the winner+peer future-mirror (attach) is
    OBSERVED, not inferred.
    """

    def _make_git_ops(
        self,
        *,
        full_branch: str,
        branch_sha: str = 'sha-live',
        ancestor_pairs: set[tuple[str, str]] | None = None,
    ) -> MagicMock:
        """Fake git_ops for the dedup tests.

        * ``resolve_branch_sha(full_branch)`` → *branch_sha* (survives Phase 1),
          None for any other ref.
        * ``is_ancestor(a, b)`` → True iff ``(a, b)`` in *ancestor_pairs*.  The
          survival check ``is_ancestor(full_branch, 'main')`` is therefore False
          (branch not yet landed) unless that pair is explicitly supplied, and
          the Phase-2 tip classification is driven by the snapshot-tip pairs.
        """
        pairs = ancestor_pairs if ancestor_pairs is not None else set()

        async def fake_resolve(branch: str) -> str | None:
            return branch_sha if branch == full_branch else None

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            return (ancestor, descendant) in pairs

        git_ops = MagicMock()
        git_ops.resolve_branch_sha = fake_resolve
        git_ops.is_ancestor = fake_is_ancestor
        return git_ops

    async def test_same_sha_coalesces_to_one_with_peer_mirror(
        self, tmp_path: Path
    ) -> None:
        """Two same-SHA records for one branch → exactly ONE enqueued winner;
        the loser attaches as a peer whose future mirrors the winner's outcome."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req1 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        req2 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        store.record(req1)
        store.record(req2)

        registry = InFlightMergeRegistry()
        git_ops = self._make_git_ops(full_branch='task/5326')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        # Exactly ONE enqueued winner; the duplicate coalesced.
        assert queue.qsize() == 1
        assert report['recovered'] == 1
        assert report['coalesced'] == 1

        # Registry entry holds the primary + one attached peer waiter.
        entry = registry.entry('5326')
        assert entry is not None
        assert len(entry.waiters) == 2, (
            f'Expected primary+peer waiters; got {len(entry.waiters)}'
        )

        # First-seen wins the SAME tie → req1 is the enqueued winner.
        assert len(report['requests']) == 1
        winner_req = report['requests'][0]
        assert winner_req.request_id == req1.request_id

        # Grab the PEER future BEFORE resolving the winner.
        peer_futures = [
            w.future for w in entry.waiters if w.future is not winner_req.result
        ]
        assert len(peer_futures) == 1
        peer = peer_futures[0]
        assert not peer.done()

        # Resolving the winner mirrors the terminal outcome onto the peer:
        # both requesters resolve — the coalesce attach is OBSERVED.
        sentinel = object()
        winner_req.result.set_result(sentinel)
        await asyncio.sleep(0)
        assert peer.done()
        assert peer.result() is sentinel

        # The loser's journal entry is removed; the winner stays journaled.
        remaining_ids = {r.request_id for r in store.load()}
        assert req2.request_id not in remaining_ids, 'loser must be store.remove()d'
        assert req1.request_id in remaining_ids, 'winner stays journaled'

    async def test_descendant_wins_ancestor_first(self, tmp_path: Path) -> None:
        """Journal order [ancestor, descendant] → the DESCENDANT is enqueued (REPLACE)."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req_anc = _make_req('5326', '5326', wt, config, snapshot_tip='anc')
        req_desc = _make_req('5326', '5326', wt, config, snapshot_tip='desc')
        store.record(req_anc)   # ancestor first in the journal
        store.record(req_desc)

        registry = InFlightMergeRegistry()
        git_ops = self._make_git_ops(
            full_branch='task/5326', ancestor_pairs={('anc', 'desc')},
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        assert queue.qsize() == 1
        assert report['coalesced'] == 1
        enqueued = queue.get_nowait()
        assert enqueued.request_id == req_desc.request_id, (
            'the DESCENDANT record must be the single enqueued winner'
        )

    async def test_descendant_wins_descendant_first(self, tmp_path: Path) -> None:
        """Journal order [descendant, ancestor] → still the DESCENDANT is enqueued
        (order-independence: the pre-grouping picks the descendant-most tip)."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req_desc = _make_req('5326', '5326', wt, config, snapshot_tip='desc')
        req_anc = _make_req('5326', '5326', wt, config, snapshot_tip='anc')
        store.record(req_desc)   # descendant first in the journal
        store.record(req_anc)

        registry = InFlightMergeRegistry()
        git_ops = self._make_git_ops(
            full_branch='task/5326', ancestor_pairs={('anc', 'desc')},
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        assert queue.qsize() == 1
        assert report['coalesced'] == 1
        enqueued = queue.get_nowait()
        assert enqueued.request_id == req_desc.request_id, (
            'order-independent: the descendant wins regardless of journal order'
        )

    async def test_divergence_replaces_and_warns(
        self, tmp_path: Path, caplog, monkeypatch
    ) -> None:
        """Divergent pair (is_ancestor False both ways) + patch NOT contained →
        resolve_divergent SUPERSET → REPLACE to the later record + a D2 WARNING
        naming the branch."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req_x = _make_req('5326', '5326', wt, config, snapshot_tip='X')
        req_y = _make_req('5326', '5326', wt, config, snapshot_tip='Y')
        store.record(req_x)
        store.record(req_y)

        async def _pcc(head: str, upstream: str, git_ops: object) -> bool:
            return False

        monkeypatch.setattr(
            'orchestrator.merge_queue.patch_content_contained', _pcc,
        )

        registry = InFlightMergeRegistry()
        git_ops = self._make_git_ops(full_branch='task/5326')  # DIVERGENT
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        with caplog.at_level(logging.WARNING):
            report = await recover_pending_merges(
                store, queue, git_ops, config, event_store=None,
                main_branch='main', branch_prefix='task/', registry=registry,
            )

        assert queue.qsize() == 1
        assert report['coalesced'] == 1
        enqueued = queue.get_nowait()
        assert enqueued.request_id == req_y.request_id, (
            'the later divergent (SUPERSET) record wins (D2)'
        )

        warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'diverg' in r.message.lower()
        ]
        assert warns, (
            f'Expected a divergence WARNING; got {[r.message for r in caplog.records]}'
        )
        assert any('5326' in r.message for r in warns), (
            f'WARNING must name the branch; got {[r.message for r in warns]}'
        )

    async def test_registry_none_preserves_double_enqueue(
        self, tmp_path: Path
    ) -> None:
        """registry omitted (default None) → current behavior preserved: two
        duplicate records both enqueue (qsize()==2), report['coalesced']==0."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req1 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        req2 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        store.record(req1)
        store.record(req2)

        git_ops = self._make_git_ops(full_branch='task/5326')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/',
        )

        assert queue.qsize() == 2, (
            'registry=None must preserve direct-enqueue-per-survivor'
        )
        assert report['recovered'] == 2
        assert report['coalesced'] == 0

    async def test_preexisting_inflight_coalesces_winner_as_peer(
        self, tmp_path: Path
    ) -> None:
        """A concurrent live merge_request already holds the branch slot at
        recovery → acquire returns False → the recovered winner attaches as a
        PEER (never a second enqueued work item), its journal record is removed,
        and it is counted as coalesced.

        Also locks suggestion-2 alias correctness: on the acquire-False path
        BOTH the winner and the loser alias onto the ACTUAL in-flight primary's
        request_id — not the winner's — matching
        ``coalesce_or_enqueue_merge_request`` (a durable poll on any coalesced id
        must resolve to the real primary outcome)."""
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        req1 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        req2 = _make_req('5326', '5326', wt, config, snapshot_tip='sha-same')
        store.record(req1)
        store.record(req2)

        # A concurrent live merge_request holds the slot BEFORE recovery runs.
        registry = InFlightMergeRegistry()
        primary_future: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '5326', '5326', primary_future,
            request_id='mr-preexisting', source='mcp', snapshot_tip='sha-same',
        )

        # Capture retention.record_alias(alias_id, primary_request_id) calls.
        aliases: list[tuple[str, str]] = []

        class _FakeRetention:
            def record_alias(self, alias_id: str, primary_request_id: str) -> None:
                aliases.append((alias_id, primary_request_id))

        git_ops = self._make_git_ops(full_branch='task/5326')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/',
            registry=registry, retention=_FakeRetention(),
        )

        # Nothing enqueued — both recovered records coalesced onto the entry.
        assert queue.qsize() == 0
        assert report['recovered'] == 0
        assert report['coalesced'] == 2
        assert report['requests'] == []

        # Pre-existing primary + two attached recovery peers.
        entry = registry.entry('5326')
        assert entry is not None
        assert len(entry.waiters) == 3, (
            f'Expected primary+2 peers; got {len(entry.waiters)}'
        )

        # Both recovered journal records were removed (winner + loser).
        remaining_ids = {r.request_id for r in store.load()}
        assert req1.request_id not in remaining_ids, 'winner must be store.remove()d'
        assert req2.request_id not in remaining_ids, 'loser must be store.remove()d'

        # Both recovery peers mirror the pre-existing primary's terminal outcome.
        peer_futures = [
            w.future for w in entry.waiters if w.future is not primary_future
        ]
        assert len(peer_futures) == 2
        assert all(not f.done() for f in peer_futures)
        sentinel = object()
        primary_future.set_result(sentinel)
        await asyncio.sleep(0)
        assert all(f.done() and f.result() is sentinel for f in peer_futures), (
            'both coalesced requesters must resolve with the primary outcome'
        )

        # Suggestion-2: winner AND loser alias onto the ACTUAL primary
        # ('mr-preexisting'), never onto the winner's own request_id.
        assert set(aliases) == {
            (req1.request_id, 'mr-preexisting'),
            (req2.request_id, 'mr-preexisting'),
        }, f'both recovered ids must alias onto the pre-existing primary; got {aliases}'


# ---------------------------------------------------------------------------
# task-1808 step-1 — MergeQueueStore.journal_corrupt flag
# ---------------------------------------------------------------------------


class TestMergeQueueStoreCorruptSignal:
    """MergeQueueStore.journal_corrupt is set (and WARNING emitted) on a corrupt
    journal; stays clear (and silent) on absent or valid-empty-dict journal."""

    def test_corrupt_journal_sets_flag_and_warns(
        self, tmp_path: Path, caplog
    ) -> None:
        """Constructing on a corrupt file sets journal_corrupt=True + emits WARNING."""
        p = tmp_path / 'merge_queue.json'
        p.write_text('not json {{{{', encoding='utf-8')

        with caplog.at_level(logging.WARNING):
            store = MergeQueueStore(p)

        assert store.journal_corrupt is True, (
            'Expected journal_corrupt=True for a corrupt JSON file'
        )
        warn_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warn_records) >= 1, (
            f'Expected at least one WARNING; got: {[r.message for r in caplog.records]}'
        )
        # The WARNING must mention the path
        path_str = str(p)
        assert any(path_str in r.message for r in warn_records), (
            f'Expected WARNING to name the path {path_str!r}; '
            f'got: {[r.message for r in warn_records]}'
        )

    def test_nonexistent_journal_no_flag_no_warn(
        self, tmp_path: Path, caplog
    ) -> None:
        """Constructing on a nonexistent path leaves journal_corrupt=False, no WARNING."""
        p = tmp_path / 'nonexistent' / 'merge_queue.json'

        with caplog.at_level(logging.WARNING):
            store = MergeQueueStore(p)

        assert store.journal_corrupt is False, (
            'Expected journal_corrupt=False for a nonexistent file'
        )
        warn_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'corrupt' in r.message.lower()
        ]
        assert len(warn_records) == 0, (
            f'Expected no corrupt WARNING for missing file; '
            f'got: {[r.message for r in warn_records]}'
        )

    def test_valid_empty_dict_journal_no_flag(
        self, tmp_path: Path, caplog
    ) -> None:
        """Constructing on a valid empty-dict journal leaves journal_corrupt=False."""
        import json as _json
        p = tmp_path / 'merge_queue.json'
        p.write_text(_json.dumps({}), encoding='utf-8')

        with caplog.at_level(logging.WARNING):
            store = MergeQueueStore(p)

        assert store.journal_corrupt is False, (
            'Expected journal_corrupt=False for a valid empty-dict journal'
        )
        warn_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'corrupt' in r.message.lower()
        ]
        assert len(warn_records) == 0, (
            f'Expected no corrupt WARNING for valid journal; '
            f'got: {[r.message for r in warn_records]}'
        )


# ---------------------------------------------------------------------------
# task-1808 step-3 — recover_pending_merges journal_corrupt propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMergesCorruptSignal:
    """recover_pending_merges propagates journal_corrupt into the report dict
    and emits a distinct WARNING when the journal was corrupt."""

    async def test_corrupt_journal_report_flag_and_warn(
        self, tmp_path: Path, caplog
    ) -> None:
        """Corrupt journal → report['journal_corrupt']=True + WARNING about pending merges."""
        p = tmp_path / 'merge_queue.json'
        p.write_text('not json {{{{', encoding='utf-8')
        store = MergeQueueStore(p)

        # Minimal fake git_ops that never gets called (journal is corrupt → no records)
        fake_git_ops = MagicMock()

        queue: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
        config = _real_config(tmp_path)

        with caplog.at_level(logging.WARNING):
            report = await recover_pending_merges(
                store,
                queue,
                fake_git_ops,
                config,
                event_store=None,
                main_branch='main',
                branch_prefix='task/',
            )

        assert report.get('journal_corrupt') is True, (
            f'Expected report["journal_corrupt"]=True; got report={report}'
        )
        # A distinct WARNING about pending merges being lost must be emitted
        corrupt_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'journal' in r.message.lower()
            and ('corrupt' in r.message.lower() or 'pending' in r.message.lower())
        ]
        assert len(corrupt_warns) >= 1, (
            f'Expected a WARNING about corrupt journal / pending merges; '
            f'got: {[r.message for r in caplog.records]}'
        )

    async def test_fresh_journal_no_corrupt_flag(
        self, tmp_path: Path, caplog
    ) -> None:
        """Nonexistent/fresh journal → report['journal_corrupt']=False, no corrupt WARNING."""
        p = tmp_path / 'nonexistent' / 'merge_queue.json'
        store = MergeQueueStore(p)

        fake_git_ops = MagicMock()
        queue: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
        config = _real_config(tmp_path)

        with caplog.at_level(logging.WARNING):
            report = await recover_pending_merges(
                store,
                queue,
                fake_git_ops,
                config,
                event_store=None,
                main_branch='main',
                branch_prefix='task/',
            )

        assert report.get('journal_corrupt') is False, (
            f'Expected report["journal_corrupt"]=False for fresh journal; got {report}'
        )
        corrupt_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and 'corrupt' in r.message.lower()
            and 'journal' in r.message.lower()
        ]
        assert len(corrupt_warns) == 0, (
            f'Expected no corrupt-journal WARNING for fresh journal; '
            f'got: {[r.message for r in corrupt_warns]}'
        )


# ---------------------------------------------------------------------------
# task-2037 step-3 — MergeQueueStore.record() branch-prefix normalization
# ---------------------------------------------------------------------------


class TestMergeQueueStoreNormalizesBranch:
    """record() strips a leading branch_prefix so the journal only ever holds
    the bare canonical branch shape (task 2037 fix 2 — enqueue normalization).
    """

    def test_record_strips_leading_prefix(self, tmp_path: Path) -> None:
        """A MergeRequest submitted with an already-prefixed branch is persisted bare."""
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req('4959', 'task/4959', wt, config)
        store.record(req)

        [persisted] = store.load()
        assert persisted.branch == '4959', (
            f"Expected leading branch_prefix stripped to bare '4959'; "
            f'got {persisted.branch!r}'
        )

    @pytest.mark.parametrize('branch', ['4959', '42'])
    def test_record_bare_branch_is_idempotent(self, tmp_path: Path, branch: str) -> None:
        """A MergeRequest submitted with an already-bare branch is unchanged.

        Parametrized over a couple of bare-branch values (was previously two
        near-identical tests differing only in the literal id).
        """
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req(branch, branch, wt, config)
        store.record(req)

        [persisted] = store.load()
        assert persisted.branch == branch


# ---------------------------------------------------------------------------
# task-2037 step-5 — recover_pending_merges tolerates a legacy prefixed journal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMergesPrefixedBranch:
    """MANDATED regression test (task 2037).

    Reproduces the pre-fix on-disk journal shape directly (bypassing the
    now-normalizing ``MergeQueueStore.record()`` from step-4) by writing the
    journal JSON file to disk verbatim, so the regression is exercised
    independent of the enqueue-normalization fix — legacy journals written
    before this change may still hold already-prefixed branch values (the
    observed mr-dd1014fe / branch 'task/4959' shape).

    Without the fix, ``recover_pending_merges`` builds
    ``f'{branch_prefix}{record.branch}'`` == ``'task/task/4959'`` (double
    prefix), which never resolves — silently dropping a live in-flight merge.
    """

    async def test_prefixed_branch_re_enqueues_live_drops_deleted(
        self, tmp_path: Path,
    ) -> None:
        import json
        from dataclasses import asdict

        config = _real_config(tmp_path)
        branch_prefix = 'task/'
        main_branch = 'main'

        wt_live = tmp_path / 'wt-4959'
        wt_live.mkdir()
        wt_deleted = tmp_path / 'wt-9999'
        wt_deleted.mkdir()

        live = PersistedMergeRequest(
            request_id='mr-4959',
            task_id='4959',
            branch='task/4959',  # already-prefixed: the legacy/pre-fix on-disk shape
            worktree=str(wt_live),
            pre_rebased=False,
            task_files=None,
            snapshot_tip=None,
            generation=1,
            lane='normal',
            enqueued_at=1000.0,
        )
        deleted = PersistedMergeRequest(
            request_id='mr-9999',
            task_id='9999',
            branch='task/9999',  # already-prefixed; branch no longer exists
            worktree=str(wt_deleted),
            pre_rebased=False,
            task_files=None,
            snapshot_tip=None,
            generation=1,
            lane='normal',
            enqueued_at=1000.0,
        )

        # Write the journal file DIRECTLY — reproduces a pre-fix journal
        # independent of the new record() normalization (step-4).
        store_path = tmp_path / 'merge_queue.json'
        store_path.write_text(
            json.dumps({
                live.request_id: asdict(live),
                deleted.request_id: asdict(deleted),
            }),
            encoding='utf-8',
        )

        store = MergeQueueStore(store_path)
        assert store.journal_corrupt is False, 'Hand-written journal must parse cleanly'

        # Fake git_ops: resolves ONLY the exact key 'task/4959'. A naive
        # double-concat (f'{branch_prefix}{record.branch}' == 'task/task/4959')
        # would resolve to None here — reproducing the pre-fix bug.
        async def fake_resolve(branch: str) -> str | None:
            return 'sha-4959' if branch == 'task/4959' else None

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            return False

        fake_git_ops = MagicMock()
        fake_git_ops.resolve_branch_sha = fake_resolve
        fake_git_ops.is_ancestor = fake_is_ancestor

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store,
            queue,
            fake_git_ops,
            config,
            event_store=None,
            main_branch=main_branch,
            branch_prefix=branch_prefix,
        )

        # --- live branch: re-enqueued, NOT dropped ---
        assert report['recovered'] == 1, f'Expected 1 recovered; got {report}'
        assert queue.qsize() == 1, f'Expected exactly 1 re-enqueued item; got {queue.qsize()}'
        recovered_req = queue.get_nowait()
        assert recovered_req.request_id == 'mr-4959'
        assert recovered_req.branch.full_name == 'task/4959', (
            f'Expected reconstructed branch to resolve via the already-prefixed '
            f"shape 'task/4959'; got {recovered_req.branch!r}"
        )
        assert report['requests'][0].request_id == 'mr-4959'

        # --- deleted branch: dropped ---
        assert report['dropped'] == 1, f'Expected 1 dropped; got {report}'

        remaining_ids = {r.request_id for r in store.load()}
        assert 'mr-9999' not in remaining_ids, 'Deleted-branch record must be removed'
        assert 'mr-4959' in remaining_ids, (
            'Recovered (re-enqueued) record must REMAIN in the store — removal '
            'happens elsewhere, on merge completion'
        )

    async def test_bare_branch_re_enqueues_via_canonical_prefix(
        self, tmp_path: Path,
    ) -> None:
        """Sibling case: the post-fix normal journal shape (bare branch).

        A journal written after the step-4 enqueue normalization holds the
        bare branch ('4959') rather than the legacy prefixed shape exercised
        above.  Recovery must still resolve it by prepending branch_prefix
        (canonical_queued_branch_name is shape-tolerant either way), while
        the reconstructed request's ``branch`` stays bare — reconstruct_merge_
        request passes ``persisted.branch`` through verbatim, so the shape on
        the re-enqueued MergeRequest differs by journal generation even though
        resolution succeeds identically for both.
        """
        import json
        from dataclasses import asdict

        config = _real_config(tmp_path)
        branch_prefix = 'task/'
        main_branch = 'main'

        wt_live = tmp_path / 'wt-4959-bare'
        wt_live.mkdir()

        live = PersistedMergeRequest(
            request_id='mr-4959',
            task_id='4959',
            branch='4959',  # bare: the post-fix (step-4 normalized) on-disk shape
            worktree=str(wt_live),
            pre_rebased=False,
            task_files=None,
            snapshot_tip=None,
            generation=1,
            lane='normal',
            enqueued_at=1000.0,
        )

        store_path = tmp_path / 'merge_queue.json'
        store_path.write_text(
            json.dumps({live.request_id: asdict(live)}),
            encoding='utf-8',
        )

        store = MergeQueueStore(store_path)
        assert store.journal_corrupt is False, 'Hand-written journal must parse cleanly'

        # Fake git_ops: resolves ONLY the canonically-prefixed key 'task/4959'
        # — the bare persisted shape 'task/4959' -> unchanged by canonical_
        # queued_branch_name is NOT what's stored; the bare '4959' itself
        # must be prepended by the fix (mirrors the prefixed-branch case).
        async def fake_resolve(branch: str) -> str | None:
            return 'sha-4959' if branch == 'task/4959' else None

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            return False

        fake_git_ops = MagicMock()
        fake_git_ops.resolve_branch_sha = fake_resolve
        fake_git_ops.is_ancestor = fake_is_ancestor

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store,
            queue,
            fake_git_ops,
            config,
            event_store=None,
            main_branch=main_branch,
            branch_prefix=branch_prefix,
        )

        assert report['recovered'] == 1, f'Expected 1 recovered; got {report}'
        assert queue.qsize() == 1, f'Expected exactly 1 re-enqueued item; got {queue.qsize()}'
        recovered_req = queue.get_nowait()
        assert recovered_req.request_id == 'mr-4959'
        assert recovered_req.branch.bare_id == '4959', (
            f"Expected the reconstructed request's branch to stay bare "
            f"('4959', passed through verbatim from the persisted record) "
            f"even though resolution went through the canonically prefixed "
            f"'task/4959' ref; got {recovered_req.branch!r}"
        )


# ---------------------------------------------------------------------------
# task-2798 (ν) — MergeRequest.branch is a typed QueuedBranch at the recovery /
# reconstruct construction boundary.
#
# These pin the typed-branch invariant at the cleanest unit seam (the journal
# recovery path), representative of the invariant across all construction
# boundaries: a MergeRequest reconstituted from a bare-persisted journal record
# carries a QueuedBranch whose .full_name is the canonically prefixed ref
# ('task/591') and whose .bare_id is the bare task id ('591'), and the recovery
# path drives its git-existence checks off that .full_name.
#
# isinstance-narrowing pattern: assert isinstance(..., QueuedBranch) BEFORE any
# .full_name/.bare_id access, so the tests are pyright-clean pre-flip (branch is
# declared str; isinstance narrows to QueuedBranch) and RED at runtime today
# (reconstruct currently returns the bare str, so the isinstance assert fails).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconstructMergeRequestTypedBranch:
    """reconstruct_merge_request yields a MergeRequest whose .branch is a QueuedBranch."""

    async def test_reconstruct_merge_request_branch_is_queuedbranch(
        self, tmp_path: Path
    ) -> None:
        config = _real_config(tmp_path)  # git.branch_prefix == 'task/'
        wt = tmp_path / 'wt'
        wt.mkdir()

        # Persisted bare, exactly as record() normalizes it on disk.
        persisted = PersistedMergeRequest(
            request_id='mr-591',
            task_id='591',
            branch='591',
            worktree=str(wt),
            pre_rebased=False,
            task_files=None,
            snapshot_tip='abc123',
            generation=1,
            lane='normal',
            enqueued_at=0.0,
        )

        req = reconstruct_merge_request(persisted, config)

        # Narrow FIRST (pyright-clean pre-flip; RED at runtime today).
        assert isinstance(req.branch, QueuedBranch), (
            f'reconstruct_merge_request must build a typed QueuedBranch branch; '
            f'got {type(req.branch).__name__}: {req.branch!r}'
        )
        assert req.branch.full_name == 'task/591'
        assert req.branch.bare_id == '591'


@pytest.mark.asyncio
class TestRecoverPendingMergesTypedBranch:
    """recover_pending_merges re-enqueues a MergeRequest with a QueuedBranch branch
    and drives its git-existence checks off .full_name."""

    async def test_recover_pending_merges_enqueues_queuedbranch_branch(
        self, tmp_path: Path
    ) -> None:
        config = _real_config(tmp_path)  # git.branch_prefix == 'task/'
        wt = tmp_path / 'wt'
        wt.mkdir()  # worktree must exist or the record is dropped

        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        # record() strips the prefix, so this persists bare '591'.
        store.record(_make_req('591', '591', wt, config))

        main_branch = 'main'
        branch_prefix = 'task/'

        # Capture the branch refs the recovery path resolves against, to prove
        # the git-existence checks are driven by the canonical .full_name.
        resolve_calls: list[str] = []
        ancestor_calls: list[tuple[str, str]] = []

        async def fake_resolve(branch: str) -> str | None:
            resolve_calls.append(branch)
            return 'sha-591'  # branch exists

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            ancestor_calls.append((ancestor, descendant))
            return False  # not yet on main → survivor

        fake_git_ops = MagicMock()
        fake_git_ops.resolve_branch_sha = fake_resolve
        fake_git_ops.is_ancestor = fake_is_ancestor

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store,
            queue,
            fake_git_ops,
            config,
            event_store=None,
            main_branch=main_branch,
            branch_prefix=branch_prefix,
        )

        assert report['recovered'] == 1, f'Expected 1 recovered; got {report}'
        assert queue.qsize() == 1, f'Expected exactly 1 re-enqueued item; got {queue.qsize()}'
        recovered_req = queue.get_nowait()

        # Narrow FIRST (pyright-clean pre-flip; RED at runtime today).
        assert isinstance(recovered_req.branch, QueuedBranch), (
            f'recovery must re-enqueue a typed QueuedBranch branch; '
            f'got {type(recovered_req.branch).__name__}: {recovered_req.branch!r}'
        )
        assert recovered_req.branch.full_name == 'task/591'

        # The existence checks were driven by the canonical full_name ref.
        assert resolve_calls == ['task/591'], (
            f'resolve_branch_sha must be driven by the full_name ref; '
            f'got {resolve_calls!r}'
        )
        assert ancestor_calls == [('task/591', 'main')], (
            f'is_ancestor must be driven by the full_name ref; '
            f'got {ancestor_calls!r}'
        )


class TestDelegatesToSharedAtomicWriter:
    """``MergeQueueStore._save_raw`` delegates to ``shared.safe_io.atomic_write_text``.

    Task 3223 consolidated the repo's tmp+rename writers into ``shared.safe_io``,
    which also gives this site a unique-per-writer temp name in place of the old
    fixed ``<dest>.json.tmp`` (two concurrent writers used to share it).
    ``mode`` must stay at the umask default: this file is read by other
    processes (the dashboard, the gamma/epsilon watchers, scripts/drain_check.py),
    so narrowing it to 0o600 is the specific silent regression this task avoids.
    """

    @staticmethod
    def _recorder(monkeypatch):
        import shared.safe_io as _safe_io

        calls = []
        real = _safe_io.atomic_write_text

        def recorder(path, text, **kwargs):
            calls.append((path, text, kwargs))
            return real(path, text, **kwargs)

        monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
        return calls

    @staticmethod
    def _assert_common(kwargs):
        assert kwargs.get('mkdir') is True, 'this site created its parent dir'
        assert kwargs.get('encoding') == 'utf-8'
        assert not kwargs.get('fsync'), 'this site never fsynced'
        assert kwargs.get('mode') is None, (
            'umask default, NOT 0o600 — this file is read by other processes'
        )

    def test_delegates_with_preserved_semantics(self, tmp_path: Path, monkeypatch) -> None:
        calls = self._recorder(monkeypatch)
        store = MergeQueueStore(tmp_path / 'data' / 'merge_queue.json')
        store.remove('nope')  # no-op; drive _save_raw directly instead
        store._save_raw({})

        assert len(calls) == 1, f'expected exactly one delegated call, got {calls}'
        self._assert_common(calls[0][2])

    def test_on_disk_mode_matches_write_text_reference(self, tmp_path: Path) -> None:
        reference = tmp_path / 'reference.json'
        reference.write_text('ref', encoding='utf-8')
        path = tmp_path / 'merge_queue.json'
        MergeQueueStore(path)._save_raw({})
        assert path.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777

    def test_oserror_still_swallowed_with_warning(self, tmp_path: Path, monkeypatch, caplog) -> None:
        """The fail-open boundary stays at the call site: swallow + WARNING."""
        import logging

        import shared.safe_io as _safe_io

        def boom(*_a, **_kw):
            raise OSError('disk full')

        monkeypatch.setattr(_safe_io, 'atomic_write_text', boom)
        store = MergeQueueStore(tmp_path / 'merge_queue.json')

        with caplog.at_level(logging.WARNING):
            store._save_raw({})  # must NOT raise

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected one WARNING, got {caplog.records}'
        assert 'disk full' in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# task-5063 — the journal persists the request's module PREFIXES.
#
# reconstruct_merge_request used to hardcode ``module_configs=[]`` in the belief
# that [] meant "all configured modules".  It does not: merge_queue.
# _merge_boundary_module_configs deliberately never widens an empty set, so a
# restart-rehydrated request was verified FILE-SCOPED.  Persisting the prefixes
# is what lets reconstruction restore a truthful set.
#
# ``None`` and ``[]`` are NOT interchangeable on the persisted field:
#   None -> written before the field existed; the module set is UNKNOWN.
#   []   -> the task genuinely had no assigned modules.
# ---------------------------------------------------------------------------


def _legacy_journal_entry(
    request_id: str,
    *,
    task_id: str = '5063',
    branch: str = '5063',
    worktree: str = '/tmp/wt',
    task_files: list[str] | None = None,
) -> dict[str, object]:
    """A journal entry in the PRE-task-5063 ten-field shape.

    Deliberately a dict LITERAL rather than ``asdict(PersistedMergeRequest(...))``:
    asdict generates its keys from the CURRENT dataclass, so it could never
    produce an entry that is genuinely MISSING ``module_prefixes`` — which is
    the whole point of these fixtures.
    """
    return {
        'request_id': request_id,
        'task_id': task_id,
        'branch': branch,
        'worktree': worktree,
        'pre_rebased': False,
        'task_files': task_files,
        'snapshot_tip': 'abc123',
        'generation': 1,
        'lane': 'normal',
        'enqueued_at': 1000.0,
    }


class TestJournalPersistsModulePrefixes:
    """record() persists the request's module prefixes; load() round-trips them."""

    def test_record_persists_the_requests_module_prefixes(self, tmp_path: Path) -> None:
        """A request scoped to two modules journals both prefixes, in order."""
        store = MergeQueueStore(tmp_path / 'merge_queue.json')
        config = _real_config(tmp_path)
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        req = _make_req(
            task_id='5063',
            branch='5063',
            worktree=worktree,
            config=config,
            module_configs=[
                ModuleConfig(prefix='orchestrator'),
                ModuleConfig(prefix='shared'),
            ],
        )
        store.record(req)

        [p] = store.load()
        assert p.module_prefixes == ['orchestrator', 'shared'], (
            f'the journal must persist the module prefixes in order; '
            f'got {p.module_prefixes!r}'
        )

    def test_zero_module_request_persists_an_explicit_empty_list(
        self, tmp_path: Path,
    ) -> None:
        """A genuinely zero-module task journals ``[]``, NOT ``None``.

        Load-bearing: an explicit ``[]`` is the ONLY thing distinguishing "this
        task has no modules" from "this record predates the field", and
        reconstruct_merge_request branches on exactly that difference.
        """
        store = MergeQueueStore(tmp_path / 'merge_queue.json')
        config = _real_config(tmp_path)
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        store.record(_make_req('5063', '5063', worktree, config))

        [p] = store.load()
        assert p.module_prefixes == []
        assert p.module_prefixes is not None, (
            'a zero-module task must persist an EXPLICIT empty list; None is '
            'reserved for records written before the field existed'
        )

    def test_legacy_journal_entry_without_the_key_loads_with_the_none_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """A pre-task-5063 entry still loads, with ``module_prefixes is None``."""
        import json

        store_path = tmp_path / 'merge_queue.json'
        store_path.write_text(
            json.dumps({'mr-legacy': _legacy_journal_entry('mr-legacy')}),
            encoding='utf-8',
        )

        store = MergeQueueStore(store_path)
        assert store.journal_corrupt is False, 'Hand-written journal must parse cleanly'

        records = store.load()
        assert len(records) == 1, (
            f'a legacy entry must still load, not be skipped; got {records!r}'
        )
        assert records[0].module_prefixes is None, (
            'an entry written before the field existed must carry the None '
            'sentinel ("module set UNKNOWN"), never an empty list'
        )


# ---------------------------------------------------------------------------
# task-5063 — a LEGACY journal record (written before module_prefixes existed)
# re-derives its module set from the preserved task_files.
#
# The exposure is transient by construction: the journal holds only in-flight
# requests, so it turns over completely on the first restart after this lands.
# It is closed anyway because that one window is exactly the restart the
# recovery path exists to survive.
# ---------------------------------------------------------------------------


def _write_journal(store_path: Path, entry: dict[str, object]) -> MergeQueueStore:
    """Hand-write a one-entry journal and return the store reading it."""
    import json

    store_path.write_text(
        json.dumps({str(entry['request_id']): entry}), encoding='utf-8',
    )
    store = MergeQueueStore(store_path)
    assert store.journal_corrupt is False, 'Hand-written journal must parse cleanly'
    return store


@pytest.mark.asyncio
class TestLegacyJournalRecordRederivesModules:
    """A record with no ``module_prefixes`` re-derives its modules from
    ``task_files`` rather than silently reconstructing an empty (file-scoped)
    set."""

    async def test_legacy_record_rederives_modules_from_task_files(
        self, tmp_path: Path,
    ) -> None:
        """The re-derivation runs the same pipeline that produced the original
        set: derive_modules -> for_module -> dedupe by prefix.

        Measured stable at BOTH the test default ``lock_depth=2`` and the live
        ``lock_depth=12``, so the expectation is not depth-fragile.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store = _write_journal(
            tmp_path / 'merge_queue.json',
            _legacy_journal_entry(
                'mr-legacy',
                task_files=[
                    'orchestrator/src/orchestrator/merge_queue_store.py',
                    'tests/scripts/test_x.py',
                    'README.md',
                ],
            ),
        )
        [record] = store.load()

        req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator', 'tests/scripts'], (
            f'a legacy record must re-derive its modules from the preserved '
            f'task_files; got {[mc.prefix for mc in req.module_configs]!r}'
        )

    async def test_legacy_record_with_docs_only_task_files_rederives_nothing(
        self, tmp_path: Path,
    ) -> None:
        """A docs-only legacy record keeps the global-fallback path.

        Guards the re-derivation against widening indiscriminately: neither
        derived key resolves through ``for_module``, so the correct answer is
        the empty set — the same gate that task already had.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store = _write_journal(
            tmp_path / 'merge_queue.json',
            _legacy_journal_entry(
                'mr-docs', task_files=['README.md', 'docs/foo.md'],
            ),
        )
        [record] = store.load()

        req = reconstruct_merge_request(record, config)

        assert req.module_configs == [], (
            f'a docs-only legacy record must NOT be widened; got '
            f'{[mc.prefix for mc in req.module_configs]!r}'
        )

    async def test_legacy_record_without_task_files_logs_and_reconstructs_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The one genuinely unrecoverable case must be LOUD, not silent.

        With neither ``module_prefixes`` nor ``task_files`` the module set
        cannot be recovered at all, so this request WILL verify narrower than
        its original.  Widening to the whole registry instead would put every
        genuinely docs-only legacy record onto a nine-module full-suite gate,
        so the empty set stays — but a WARNING names the record.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store = _write_journal(
            tmp_path / 'merge_queue.json',
            _legacy_journal_entry('mr-blind', task_files=None),
        )
        [record] = store.load()

        with caplog.at_level(logging.WARNING):
            req = reconstruct_merge_request(record, config)

        assert req.module_configs == []
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('mr-blind' in m for m in warnings), (
            f'the unrecoverable case must be diagnosed by request_id, not '
            f'silently degraded; warnings were {warnings!r}'
        )

    async def test_persisted_prefixes_that_no_longer_resolve_fall_back_to_task_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-empty persisted list that resolves to nothing is POSITIVE
        evidence the task had modules — so it must not silently produce the
        narrow gate this fix exists to close."""
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        entry = _legacy_journal_entry(
            'mr-stale',
            task_files=['orchestrator/src/orchestrator/merge_queue.py'],
        )
        entry['module_prefixes'] = ['deleted-module']
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        with caplog.at_level(logging.WARNING):
            req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator'], (
            f'an unresolvable persisted prefix must fall back to the '
            f'task_files re-derivation, not to the empty (file-scoped) set; '
            f'got {[mc.prefix for mc in req.module_configs]!r}'
        )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('deleted-module' in m for m in warnings), (
            f'the unresolved prefix must be named; warnings were {warnings!r}'
        )


# ---------------------------------------------------------------------------
# task-5063 amendment — the three properties the original suite left implicit:
#   * dedupe-by-mc.prefix, the stated reason _resolve_prefixes exists at all;
#   * VALUE-shape validation of the persisted module_prefixes payload;
#   * the UNKNOWN encoding surviving a re-record, so an unrecoverable module
#     set stays LOUD across every restart instead of only the first.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPersistedPrefixesAreDedupedByResolvedConfig:
    """Several persisted prefixes that resolve to ONE ModuleConfig yield one
    entry, in first-seen order.

    This is the whole reason ``_resolve_prefixes`` keys a ``seen`` dict by
    ``mc.prefix`` rather than returning a plain list: ``for_module`` walks
    INWARD, so ``orchestrator/src`` and ``orchestrator/tests`` both land on the
    registered ``orchestrator`` config.  Reproducing
    ``TaskWorkflow._resolve_module_configs``' grouping is what makes the
    reconstructed set equal the original's by construction — and a regression
    that dropped the dedupe would leave every other test in this file green
    while emitting duplicate ModuleConfigs into the merge-role verify plan.
    """

    async def test_prefixes_collapsing_to_one_config_reconstruct_to_one_entry(
        self, tmp_path: Path,
    ) -> None:
        config = _config_with_modules(tmp_path, ['orchestrator'])
        entry = _legacy_journal_entry('mr-dedupe', task_files=['a.py'])
        entry['module_prefixes'] = [
            'orchestrator/src', 'orchestrator/tests', 'orchestrator',
        ]
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator'], (
            f'three prefixes resolving to one registered config must dedupe to '
            f'a single ModuleConfig; got '
            f'{[mc.prefix for mc in req.module_configs]!r}'
        )

    async def test_dedupe_preserves_first_seen_order(self, tmp_path: Path) -> None:
        """First-seen order, not registry order — that is what the producer of
        the ORIGINAL module_configs preserves."""
        config = _config_with_modules(tmp_path, ['orchestrator', 'shared'])
        entry = _legacy_journal_entry('mr-order', task_files=['a.py'])
        entry['module_prefixes'] = [
            'shared/src', 'orchestrator/src', 'shared', 'orchestrator/tests',
        ]
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['shared', 'orchestrator']


@pytest.mark.asyncio
class TestMalformedModulePrefixesValueIsTreatedAsUnknown:
    """A ``module_prefixes`` VALUE of the wrong shape degrades to UNKNOWN.

    ``load()`` type-checks the ENTRY (``isinstance(entry, dict)``) but applies
    no checking to field VALUES, so a hand-edited or foreign journal can put
    anything here.  Both untreated shapes fail badly rather than loudly: a bare
    ``str`` iterates as CHARACTERS, and a non-str element reaches
    ``config.for_module(123)`` -> ``123.strip('/')`` -> ``AttributeError``,
    which is not in ``load()``'s ``except (TypeError, KeyError)`` and escapes
    ``reconstruct_merge_request``.
    """

    async def test_a_string_value_does_not_iterate_as_characters(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        entry = _legacy_journal_entry(
            'mr-strval',
            task_files=['orchestrator/src/orchestrator/merge_queue.py'],
        )
        entry['module_prefixes'] = 'orchestrator'
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        with caplog.at_level(logging.WARNING):
            req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator'], (
            f'a malformed value must fall through to the task_files '
            f're-derivation, not be iterated as characters; got '
            f'{[mc.prefix for mc in req.module_configs]!r}'
        )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('not a list of strings' in m for m in warnings), (
            f'the malformed value must be named as such; warnings were {warnings!r}'
        )

    async def test_a_non_string_element_does_not_raise_out_of_reconstruct(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``[123]`` used to reach ``123.strip('/')``.

        ``recover_pending_merges`` bounds that blast radius to one record, but
        it is the same class of defect the entry-shape guard was added for.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        entry = _legacy_journal_entry(
            'mr-intval',
            task_files=['orchestrator/src/orchestrator/merge_queue.py'],
        )
        entry['module_prefixes'] = [123]
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        with caplog.at_level(logging.WARNING):
            req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator']

    async def test_an_empty_string_value_is_unknown_not_an_empty_set(
        self, tmp_path: Path,
    ) -> None:
        """``''`` is the shape that slips past BOTH untreated branches.

        ``'' == []`` is False so the explicit-empty early-return misses it, and
        ``if ''`` is falsy so it reaches the re-derivation — by accident rather
        than by decision.  Pinned so the guard, not the coincidence, is what
        produces the answer.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        entry = _legacy_journal_entry(
            'mr-emptystr',
            task_files=['orchestrator/src/orchestrator/merge_queue.py'],
        )
        entry['module_prefixes'] = ''
        store = _write_journal(tmp_path / 'merge_queue.json', entry)
        [record] = store.load()

        req = reconstruct_merge_request(record, config)

        assert [mc.prefix for mc in req.module_configs] == ['orchestrator']


@pytest.mark.asyncio
class TestUnrecoverableModuleSetStaysLoudAcrossRestarts:
    """Re-journaling a recovered request must not rewrite an UNKNOWN module set
    as an authoritative EMPTY one.

    THE FAILURE THIS PINS.  A legacy record with nothing to re-derive from
    reconstructs to ``[]`` with the loud "unrecoverable" WARNING.  The
    recovered request is then re-journaled by
    ``SpeculativeMergeWorker._buffer_owned_request``.  If that write persisted
    a plain ``[]``, the NEXT crash+recovery would take the "task genuinely had
    no assigned modules" early-return and say NOTHING — so a request still
    verifying narrower than its original goes silent from the second restart
    onward, against the repo's loud-over-silent-degradation norm.
    """

    @staticmethod
    def _recover_once(
        store_path: Path, config: OrchestratorConfig,
    ) -> tuple[MergeRequest, MergeQueueStore]:
        """One full crash-recovery cycle: read the journal, reconstruct, and
        re-journal the recovered request exactly as the worker's
        ``_buffer_owned_request`` does."""
        store = MergeQueueStore(store_path)
        [record] = store.load()
        req = reconstruct_merge_request(record, config)
        store.record(req)
        return req, store

    async def test_second_recovery_of_an_unrecoverable_record_still_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store_path = tmp_path / 'merge_queue.json'
        _write_journal(store_path, _legacy_journal_entry('mr-blind2', task_files=None))

        # Restart #1 — the loud one.
        with caplog.at_level(logging.WARNING):
            first, _ = self._recover_once(store_path, config)
        assert first.module_configs == []
        assert any(
            'mr-blind2' in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        ), 'the FIRST recovery must diagnose the unrecoverable set'

        # Restart #2 — reads the journal the first recovery just rewrote.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            second, _ = self._recover_once(store_path, config)

        assert second.module_configs == []
        assert any(
            'mr-blind2' in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        ), (
            'a request that is STILL verifying narrower than its original must '
            'be diagnosed on EVERY restart, not only the first — a degradation '
            'that goes silent is the exact failure this task exists to close'
        )

    async def test_the_re_record_keeps_the_unknown_sentinel_not_an_empty_list(
        self, tmp_path: Path,
    ) -> None:
        """The mechanism behind the test above, asserted directly on disk.

        ``None`` and ``[]`` are the whole design; collapsing UNKNOWN into
        EMPTY at re-record time is what would silence the diagnostic.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store_path = tmp_path / 'merge_queue.json'
        _write_journal(store_path, _legacy_journal_entry('mr-blind3', task_files=None))

        self._recover_once(store_path, config)

        [reloaded] = MergeQueueStore(store_path).load()
        assert reloaded.module_prefixes is None, (
            f'an UNKNOWN module set must stay UNKNOWN across the re-record; '
            f'got {reloaded.module_prefixes!r}, which reads as "this task '
            f'genuinely has no modules" on the next recovery'
        )

    async def test_a_genuinely_zero_module_request_still_re_records_as_empty(
        self, tmp_path: Path,
    ) -> None:
        """CONTROL — the preservation above must not swallow the ``[]`` case.

        A task that genuinely has no assigned modules was journaled with an
        EXPLICIT ``[]``, which is not ``None``, so the re-record leaves it
        alone and the global-fallback gate is preserved.
        """
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        store.record(_make_req('5063', '5063', worktree, config))

        [record] = store.load()
        assert record.module_prefixes == []
        store.record(reconstruct_merge_request(record, config))

        [reloaded] = MergeQueueStore(store_path).load()
        assert reloaded.module_prefixes == [], (
            f'an explicitly-empty module set must survive the re-record as [], '
            f'not be demoted to the UNKNOWN sentinel; got '
            f'{reloaded.module_prefixes!r}'
        )

    async def test_a_recovered_request_that_regains_modules_overwrites_unknown(
        self, tmp_path: Path,
    ) -> None:
        """A NON-empty live set always wins: the module set is known again."""
        config = _config_with_modules(tmp_path, NINE_PREFIXES)
        store_path = tmp_path / 'merge_queue.json'
        store = _write_journal(
            store_path, _legacy_journal_entry('mr-regained', task_files=None),
        )
        [record] = store.load()
        req = reconstruct_merge_request(record, config)
        req.module_configs = [config.for_module('orchestrator')]  # type: ignore[list-item]

        store.record(req)

        [reloaded] = MergeQueueStore(store_path).load()
        assert reloaded.module_prefixes == ['orchestrator']


# ---------------------------------------------------------------------------
# task-5063 — load() tolerates an UNKNOWN key instead of dropping the record.
#
# The forward-compat half of the module_prefixes schema change: load() builds
# records with PersistedMergeRequest(**entry) inside `except (TypeError,
# KeyError)`, so one unexpected key made the whole entry vanish with only a
# `skipping malformed entry` warning — and a vanished entry is an in-flight
# merge request that is never recovered.
#
# READ THE DIRECTION PRECISELY. The tolerance lives in the READER, so what
# these tests pin is THIS binary and later ones surviving a journal written by
# a NEWER one — the payoff is for FUTURE schema additions. They do NOT, and
# cannot, make a rollback PAST this commit safe: a revert removes the tolerance
# along with the writer, so a pre-5063 binary reading a journal carrying
# `module_prefixes` still hits TypeError and still skips every such entry.
#
# What bounds THAT case is separate and weaker, and is pinned by
# TestSkippedEntriesSurviveOnDisk below: load() never mutates _cache, so a
# skipped entry stays in the mirror and is rewritten verbatim by the next
# _save_raw. The in-flight merges it names are STALLED for the rollback's
# duration, not erased — recoverable again once a tolerant binary reads them.
# ---------------------------------------------------------------------------


def _current_journal_entry(request_id: str, **overrides: object) -> dict[str, object]:
    """A journal entry in the CURRENT (post-module_prefixes) shape."""
    entry = _legacy_journal_entry(request_id, task_files=['a.py'])
    entry['module_prefixes'] = ['orchestrator']
    entry.update(overrides)
    return entry


class TestJournalLoadToleratesUnknownKeys:
    """An entry carrying a key this binary does not know still loads."""

    def test_entry_with_an_unknown_key_still_loads_with_its_known_fields(
        self, tmp_path: Path,
    ) -> None:
        entry = _current_journal_entry('mr-future', future_field='x')
        store = _write_journal(tmp_path / 'merge_queue.json', entry)

        records = store.load()

        assert len(records) == 1, (
            f'an entry with an unknown key must still load — dropping it '
            f'silently loses an in-flight merge request; got {records!r}'
        )
        [p] = records
        assert p.request_id == 'mr-future'
        assert p.task_id == '5063'
        assert p.branch == '5063'
        assert p.task_files == ['a.py']
        assert p.module_prefixes == ['orchestrator']

    def test_unknown_key_is_reported_not_silently_dropped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Tolerance must not trade one silent failure for another."""
        entry = _current_journal_entry('mr-future', future_field='x')
        store_path = tmp_path / 'merge_queue.json'
        store = _write_journal(store_path, entry)

        with caplog.at_level(logging.WARNING):
            store.load()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('mr-future' in m and 'future_field' in m for m in warnings), (
            f'the dropped key must be reported by request_id and by name; '
            f'warnings were {warnings!r}'
        )

    def test_entry_missing_a_required_field_is_still_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only the EXTRA-key direction is relaxed.

        A genuinely malformed record — one missing a required field — must
        still be skipped rather than half-constructed.
        """
        entry = _current_journal_entry('mr-broken')
        del entry['task_id']
        store = _write_journal(tmp_path / 'merge_queue.json', entry)

        with caplog.at_level(logging.WARNING):
            records = store.load()

        assert records == [], (
            f'an entry missing a required field must still be skipped; '
            f'got {records!r}'
        )
        assert [r for r in caplog.records if r.levelno == logging.WARNING], (
            'the skip must still be warned about'
        )


class TestSkippedEntriesSurviveOnDisk:
    """A skipped entry is DEFERRED, not destroyed.

    This is the honest, weaker mitigation for the one case the unknown-key
    tolerance above genuinely cannot reach — a binary OLDER than that tolerance
    reading a journal that carries ``module_prefixes``.  Such a binary skips the
    entry, so the merge is not recovered while the rollback lasts; but ``load()``
    never mutates ``_cache``, so the entry is still there for the next tolerant
    reader.  Pinned here so the claim in the section header above is checked
    rather than asserted.

    A record MISSING a required field is used as the stand-in: it is the shape
    THIS binary still skips, and it exercises the identical
    skip-but-do-not-evict path an older binary would take on an unknown key.
    """

    def test_a_skipped_entry_is_rewritten_by_a_later_save(
        self, tmp_path: Path,
    ) -> None:
        import json

        store_path = tmp_path / 'merge_queue.json'
        broken = _current_journal_entry('mr-unreadable')
        del broken['task_id']
        store_path.write_text(
            json.dumps({'mr-unreadable': broken}), encoding='utf-8',
        )

        store = MergeQueueStore(store_path)
        assert store.load() == [], 'precondition: this binary skips the entry'

        # Any unrelated journal write flushes the whole cache back out.
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        store.record(_make_req('9999', '9999', worktree, _real_config(tmp_path)))

        on_disk = json.loads(store_path.read_text(encoding='utf-8'))
        assert 'mr-unreadable' in on_disk, (
            'a skipped entry must survive a subsequent save — an unreadable '
            'in-flight merge request is deferred until a binary that can parse '
            'it reads the journal, not silently evicted'
        )
        assert on_disk['mr-unreadable'] == broken, (
            'and it must survive VERBATIM, so the newer binary that wrote it '
            'can still read every field it wrote'
        )


# ---------------------------------------------------------------------------
# step-9: a journal entry whose VALUE is not a mapping is skipped PER-ENTRY.
#
# step-8's unknown-key filter dereferences `entry.items()` / `set(entry)`
# OUTSIDE the try, so a journal whose top level is a dict but whose value for
# one key is a string / list / number / None raises an uncaught AttributeError
# out of load().  recover_pending_merges calls store.load() unguarded and
# Harness.run catches it only at the outermost `except Exception` around
# _recover_pending_merges — so ONE bad value aborts the ENTIRE recovery pass
# and every in-flight merge request in the journal is lost.  That is precisely
# the failure mode steps 7-8 were written to prevent, and it contradicts this
# module's documented per-entry fail-open contract.
# ---------------------------------------------------------------------------


def _write_journal_entries(
    store_path: Path, entries: dict[str, object],
) -> MergeQueueStore:
    """Hand-write a multi-entry journal VERBATIM and return the store reading it.

    Sibling of ``_write_journal``, which cannot express these fixtures: it takes
    exactly ONE entry and keys the journal by ``entry['request_id']``, and a
    non-dict entry has no ``['request_id']`` to subscript.

    ``_load_raw``'s ``isinstance(data, dict)`` check inspects only the TOP
    level, so a dict-with-a-non-dict-value parses cleanly, ``journal_corrupt``
    stays False, and the bad value genuinely reaches ``load()``.
    """
    import json

    store_path.write_text(json.dumps(entries), encoding='utf-8')
    store = MergeQueueStore(store_path)
    assert store.journal_corrupt is False, 'Hand-written journal must parse cleanly'
    return store


class TestJournalLoadToleratesNonDictEntries:
    """A non-mapping entry value loses only ITSELF, never its siblings."""

    def test_non_dict_entry_is_skipped_while_its_siblings_still_load(
        self, tmp_path: Path,
    ) -> None:
        """The headline regression: one bad value must not abort the pass.

        ``recover_pending_merges`` calls ``load()`` outside any try, so an
        exception here drops every in-flight merge request in the journal —
        not just the malformed one.
        """
        store = _write_journal_entries(
            tmp_path / 'merge_queue.json',
            {'mr-bad': 'not-a-dict', 'mr-ok': _current_journal_entry('mr-ok')},
        )

        records = store.load()

        assert len(records) == 1, (
            f'a non-dict entry must cost only itself — its siblings are '
            f'in-flight merge requests; got {records!r}'
        )
        [p] = records
        assert p.request_id == 'mr-ok'
        assert p.task_files == ['a.py']
        assert p.module_prefixes == ['orchestrator'], (
            'the surviving sibling must be fully intact, not half-constructed'
        )

    @pytest.mark.parametrize(
        'bad_value',
        ['not-a-dict', ['a', 'b'], 42, None],
        ids=['str', 'list', 'int', 'null'],
    )
    def test_every_non_mapping_entry_shape_is_skipped_not_raised(
        self, tmp_path: Path, bad_value: object,
    ) -> None:
        """The guard keys on "is it a mapping", not on one incidental type.

        These are the shapes a hand-edited journal, a foreign writer, or a
        half-migrated schema can actually produce.
        """
        store = _write_journal_entries(
            tmp_path / 'merge_queue.json',
            {'mr-bad': bad_value, 'mr-ok': _current_journal_entry('mr-ok')},
        )

        records = store.load()

        assert [p.request_id for p in records] == ['mr-ok'], (
            f'a {type(bad_value).__name__} entry value must be skipped, not '
            f'raised through; got {records!r}'
        )

    def test_non_dict_entry_is_reported_by_its_journal_key(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The journal KEY is the only id a non-dict entry has.

        There is no ``request_id`` field to read defensively from, so the
        warning must come from the key (loud-over-silent, as already pinned by
        ``test_unknown_key_is_reported_not_silently_dropped``).
        """
        store = _write_journal_entries(
            tmp_path / 'merge_queue.json',
            {'mr-bad': 'not-a-dict', 'mr-ok': _current_journal_entry('mr-ok')},
        )

        with caplog.at_level(logging.WARNING):
            store.load()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any('mr-bad' in m for m in warnings), (
            f'the skipped entry must be named by its journal key; '
            f'warnings were {warnings!r}'
        )

    def test_a_non_dict_entry_does_not_mark_the_whole_journal_corrupt(
        self, tmp_path: Path,
    ) -> None:
        """A per-entry defect must not escalate to the journal-wide signal.

        ``recover_pending_merges`` emits a much louder, operator-facing
        "journal was corrupt at startup — pending merges may have been lost"
        warning for the corrupt case.  This also forecloses "just mark it
        corrupt" as a fix.
        """
        store = _write_journal_entries(
            tmp_path / 'merge_queue.json',
            {'mr-bad': 'not-a-dict', 'mr-ok': _current_journal_entry('mr-ok')},
        )

        store.load()

        assert store.journal_corrupt is False, (
            'one malformed entry value is a per-entry skip, not a corrupt journal'
        )
