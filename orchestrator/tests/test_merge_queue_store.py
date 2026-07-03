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
from unittest.mock import MagicMock

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.merge_queue import GroupMergeRequest, MergeRequest

# Import the module under test — will fail (ImportError) until step-2 creates it.
from orchestrator.merge_queue_store import (
    MergeQueueStore,
    PersistedMergeRequest,
    reconstruct_merge_request,
    recover_pending_merges,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _real_config(tmp_path: Path) -> OrchestratorConfig:
    """Minimal real OrchestratorConfig pointing at tmp_path."""
    return OrchestratorConfig(
        project_root=tmp_path,
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
) -> MergeRequest:
    """Build a MergeRequest with a placeholder future (safe outside a running loop)."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=task_files,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        snapshot_tip=snapshot_tip,
        generation=generation,
        lane=lane,  # type: ignore[arg-type]
    )


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
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        train_id='train-001',
        member_task_ids=[task_id],
        tip_branch=branch,
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
        assert reconstructed.branch == '99'
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
        assert recovered_req.branch == 'task/4959', (
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
        assert recovered_req.branch == '4959', (
            f"Expected the reconstructed request's branch to stay bare "
            f"('4959', passed through verbatim from the persisted record) "
            f"even though resolution went through the canonically prefixed "
            f"'task/4959' ref; got {recovered_req.branch!r}"
        )
