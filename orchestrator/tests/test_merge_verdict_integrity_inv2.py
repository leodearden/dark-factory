"""PRD §7 two-way boundary capstone for INV-2 (task 2884, leaf β).

INV-2: a RemoteRunner's merge-verify verdict is adoptable ONLY if the runner
executed CURRENT gate logic — the remote Dark-Factory *code* checkout that runs
``orchestrator verify-merge`` over ssh must be at the dispatcher's HEAD.  Incident
bb834dd42a: the laptop's DF *code* checkout sat frozen for ~5 weeks and so
rubber-stamped trivial PASSes (966f23a6).  Separately the best-effort
project-*main* push silently failed rc=1 forever after the laptop reify main
diverged 07-20 (non-fast-forward), so the remote never saw fresh main.

This capstone composes the REAL substrate end-to-end — RemoteRunner.sync_if_stale
+ RemoteRunner.run_merge_verify driven through the REAL VerifyRunnerPool.dispatch
— with ONLY the subprocess/ssh transport faked (a single recording fake_run),
mirroring the inv1 capstone's ``_RecordingEventStore`` + command-spy doubles.  It
pins the runner→pipeline boundary from BOTH sides (PRD §1 INV-2 / §3.1 / §7 / §8β):

  (1) STALE + sync FAILS  → fail-closed bench.  ``runner_stale`` is recorded and
      NO ``runner_synced``; the remote verdict is NEVER taken (no merge-sha push).
      A single-runner remote pool raises ``RunnerUnavailable`` (the production
      bench signal → merge_queue._run_inflight_verify quarantine_and_release +
      local re-dispatch); a ``[remote, local]`` pool quarantines the remote and
      adopts the LOCAL verdict.
  (2) STALE + sync SUCCEEDS → ``runner_stale`` → ``runner_synced``(df_checkout)
      recorded IN ORDER, and the REMOTE verdict IS adopted (remote not benched).
  (3) mirror-push arm → a diverged project main (FF push rejected, force accepted)
      records ``runner_synced``(project_main_mirror, forced=True) and the verify
      still proceeds to the remote verdict.

Assertions are on event ORDER and PAYLOADS, not merely presence.
"""
from __future__ import annotations

from typing import Any, ClassVar

import pytest

from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    MergeVerifySpec,
    RemoteRunner,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    VerifyRunnerPool,
    result_to_json,
)

_TASK_ID = 't-2884'


# ---------------------------------------------------------------------------
# Doubles — recording event store + fake local anchor + real-RemoteRunner-with-
# faked-transport factory (single fake_run services sync_if_stale AND
# run_merge_verify, since injecting run= makes _ssh_run == run).
# ---------------------------------------------------------------------------


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory.

    Mirrors test_merge_verdict_integrity_inv1._RecordingEventStore.
    """

    def __init__(self) -> None:
        self.events: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
        **kw: Any,
    ) -> None:
        self.events.append((event_type, task_id, dict(data or {})))

    def events_of(self, event_type: Any) -> list[dict[str, Any]]:
        return [data for (et, _tid, data) in self.events if et == event_type]

    def types(self) -> list[Any]:
        return [et for (et, _tid, _data) in self.events]


class _CapstoneLocal:
    """Minimal is_local trust-anchor / fallback runner (mirrors _PoolFakeLocal)."""

    is_local: ClassVar[bool] = True

    def __init__(self, name: str = 'local') -> None:
        self.name = name
        self.calls: list[tuple[str, Any]] = []

    async def health(self) -> bool:
        return True

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        self.calls.append((merge_sha, spec))
        return VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='local-ok',
        )


def _make_spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


def _make_capstone_remote(
    *,
    local_head: str = 'DISPATCHER_HEAD',
    remote_head: str = 'DISPATCHER_HEAD',  # equal => DF checkout already current
    post_sync_head: str | None = None,
    pull_rc: int = 0,
    uv_rc: int = 0,
    ff_rc: int = 0,
    force_rc: int = 0,
    resolved_main: str = 'MAINSHA',
    name: str = 'laptop',
    verify_result: VerifyResult | None = None,
) -> tuple[RemoteRunner, list[tuple[list[str], Any]], VerifyResult]:
    """Build a REAL RemoteRunner (INV-2 df paths + main_branch configured) whose
    subprocess transport is one recording fake_run servicing BOTH sync_if_stale
    and run_merge_verify.  Returns (runner, calls, expected_remote_result).

    fake_run routes canned (rc, stdout, stderr) by argv shape:
      sync_if_stale:
        * ``git rev-parse HEAD`` (cwd=df_local)          -> local_head
        * ssh ``git -C <df> rev-parse HEAD``             -> remote_head, then
                                                            post_sync_head once
                                                            a pull has fired
        * ssh ``git -C <df> pull --ff-only``             -> pull_rc
        * ssh ``cd <df> && uv sync``                     -> uv_rc
      run_merge_verify:
        * ``git rev-parse main`` (cwd=cwd, dedup probe)  -> resolved_main
        * ``git push origin main:refs/heads/main`` (FF)     -> ff_rc
        * ``git push origin +main:refs/heads/main`` (force) -> force_rc
        * ``git push origin <sha>:refs/merge-verify/..`` (load-bearing) -> 0
        * ``git push origin --delete <ref>`` (cleanup)      -> 0
        * ssh ``orchestrator verify-merge ..``              -> JSON verify_result
    """
    calls: list[tuple[list[str], Any]] = []
    state = {'pulled': False}
    settled = post_sync_head if post_sync_head is not None else local_head
    expected = verify_result or VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='', summary='remote-ok',
    )

    async def fake_run(argv, *, cwd=None):
        calls.append((list(argv), cwd))
        # sync_if_stale: local DF HEAD (network-free rev-parse)
        if argv[:3] == ['git', 'rev-parse', 'HEAD']:
            return (0, local_head, '')
        # run_merge_verify Step-0 dedup: local main tip
        if argv[:3] == ['git', 'rev-parse', 'main']:
            return (0, resolved_main, '')
        # git push refspecs (main FF / force / merge-sha / --delete cleanup)
        if argv[:2] == ['git', 'push'] and len(argv) > 3:
            refspec = argv[3]
            if refspec == 'main:refs/heads/main':
                return (ff_rc, '', '' if ff_rc == 0 else 'rejected: non-fast-forward')
            if refspec == '+main:refs/heads/main':
                return (force_rc, '', '' if force_rc == 0 else 'rejected: hook declined')
            if 'refs/merge-verify/' in refspec:
                return (0, '', '')
            if refspec == '--delete':
                return (0, '', '')
            return (0, '', '')
        # ssh: DF-head probe / pull / uv sync / verify-dispatch
        if argv and argv[0] == 'ssh':
            remote_cmd = argv[-1]
            if 'pull --ff-only' in remote_cmd:
                state['pulled'] = True
                return (pull_rc, '', '' if pull_rc == 0 else 'pull rejected: non-fast-forward')
            if 'uv sync' in remote_cmd:
                return (uv_rc, '', '' if uv_rc == 0 else 'uv sync failed')
            if 'rev-parse HEAD' in remote_cmd:
                return (0, settled if state['pulled'] else remote_head, '')
            if 'verify-merge' in remote_cmd:
                return (0, result_to_json(expected), '')
            return (0, '', '')
        return (0, '', '')

    runner = RemoteRunner(
        name=name,
        ssh_host='laptop.local',
        git_remote='origin',
        cwd='/repo',
        main_branch='main',
        df_remote_checkout='/remote/dark-factory',
        df_local_checkout='/local/dark-factory',
        run=fake_run,
        id_factory=lambda: 'fixed-id',
    )
    return runner, calls, expected


def _push_refspecs(calls) -> list[str]:
    """Every `git push` refspec argument seen, in order."""
    return [
        argv[3] for (argv, _cwd) in calls
        if argv[:2] == ['git', 'push'] and len(argv) > 3
    ]


def _merge_sha_pushed(calls) -> bool:
    """True iff the load-bearing merge-sha push (Step 1 of run_merge_verify) fired
    — i.e. run_merge_verify was actually ENTERED past the sync gate."""
    return any('refs/merge-verify/' in rs for rs in _push_refspecs(calls))


# ---------------------------------------------------------------------------
# (1) STALE + sync FAILS → fail-closed bench, remote verdict NEVER taken.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInv2StaleSyncFailsBenchesFailClosed:
    """A stale remote whose DF-checkout sync FAILS is benched fail-closed:
    runner_stale (no runner_synced), the remote verdict is never adopted."""

    async def test_single_runner_pool_raises_runner_unavailable_no_verdict(self):
        """Single-runner production pool: pool.dispatch raises RunnerUnavailable
        (the bench signal), runner_stale recorded, NO runner_synced, and the
        remote's run_merge_verify is NEVER entered (no merge-sha push, no
        merge_verify verdict event)."""
        remote, calls, _expected = _make_capstone_remote(
            local_head='DISPATCHER_HEAD', remote_head='FROZEN_5W_OLD', pull_rc=1,
        )
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        with pytest.raises(RunnerUnavailable):
            await pool.dispatch('mergesha', _make_spec())

        # runner_stale announced the divergence BEFORE the bench, with the heads.
        stales = store.events_of(EventType.runner_stale)
        assert len(stales) == 1
        assert stales[0] == {
            'runner': 'laptop',
            'local_head': 'DISPATCHER_HEAD',
            'remote_head': 'FROZEN_5W_OLD',
        }
        # Fail-closed: NO sync-success and NO adopted verdict.
        assert store.events_of(EventType.runner_synced) == []
        assert store.events_of(EventType.merge_verify) == []
        # The remote verdict was never taken — run_merge_verify not entered.
        assert not _merge_sha_pushed(calls), (
            f'merge-sha push must NOT fire on a benched runner; got {_push_refspecs(calls)!r}'
        )

    async def test_two_runner_pool_benches_remote_and_adopts_local(self):
        """[remote, local] pool: the stale-sync-fail remote is quarantined and the
        LOCAL trust-anchor verdict is adopted; runner_stale recorded, no
        runner_synced, remote verdict never taken."""
        remote, calls, _expected = _make_capstone_remote(
            local_head='DISPATCHER_HEAD', remote_head='FROZEN_5W_OLD', pull_rc=1,
        )
        local = _CapstoneLocal()
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote, local], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        # The LOCAL verdict is adopted (never the stale remote's).
        assert result.summary == 'local-ok'
        assert len(local.calls) == 1
        assert pool.is_quarantined('laptop') is True
        # runner_stale fired; no sync success.
        assert len(store.events_of(EventType.runner_stale)) == 1
        assert store.events_of(EventType.runner_synced) == []
        # merge_verify event attributes the verdict to the local anchor.
        mv = store.events_of(EventType.merge_verify)
        assert len(mv) == 1
        assert mv[0]['runner'] == 'local'
        assert mv[0]['merge_sha'] == 'mergesha'
        # The remote never produced a verdict — no merge-sha push on its transport.
        assert not _merge_sha_pushed(calls)


# ---------------------------------------------------------------------------
# (2) STALE + sync SUCCEEDS → runner_stale → runner_synced, remote adopted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInv2StaleSyncSucceedsAdoptsRemote:
    """A stale remote whose DF-checkout sync SUCCEEDS emits runner_stale then
    runner_synced(df_checkout) IN ORDER and its verdict IS adopted."""

    async def test_stale_then_synced_in_order_then_remote_verdict_adopted(self):
        remote, calls, expected = _make_capstone_remote(
            local_head='NEW_DISPATCHER_HEAD',
            remote_head='OLD_REMOTE_HEAD',
            post_sync_head='NEW_DISPATCHER_HEAD',
            pull_rc=0, uv_rc=0, ff_rc=0,  # main push FF-clean → no mirror event
        )
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        # The REMOTE verdict is adopted (sync brought it current first).
        assert result == expected
        assert result.summary == 'remote-ok'
        assert pool.is_quarantined('laptop') is False
        assert _merge_sha_pushed(calls), 'remote verdict adopted → merge-sha push must fire'

        # Exactly one runner_stale then one runner_synced(df_checkout), IN ORDER.
        types = store.types()
        assert types.index(EventType.runner_stale) < types.index(EventType.runner_synced), (
            f'runner_stale must precede runner_synced; got {types!r}'
        )
        stales = store.events_of(EventType.runner_stale)
        assert len(stales) == 1
        assert stales[0]['local_head'] == 'NEW_DISPATCHER_HEAD'
        assert stales[0]['remote_head'] == 'OLD_REMOTE_HEAD'

        synced = store.events_of(EventType.runner_synced)
        assert len(synced) == 1
        assert synced[0]['kind'] == 'df_checkout'
        assert synced[0]['forced'] is False
        assert synced[0]['from_head'] == 'OLD_REMOTE_HEAD'
        assert synced[0]['to_head'] == 'NEW_DISPATCHER_HEAD'

        # Events carried the dispatch task_id through the pool.
        assert all(tid == _TASK_ID for (_et, tid, _d) in store.events)

        # The verdict was adopted from the remote.
        mv = store.events_of(EventType.merge_verify)
        assert len(mv) == 1
        assert mv[0]['runner'] == 'laptop'
        assert mv[0]['passed'] is True


# ---------------------------------------------------------------------------
# (3) Mirror-push arm — diverged project main is force-mirrored, verify proceeds.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInv2MirrorPushDivergedProjectMain:
    """A diverged remote PROJECT main (FF push rejected) is force-mirrored;
    runner_synced(project_main_mirror, forced=True) is recorded and the verify
    still proceeds to the remote verdict.  DF *code* checkout is current here, so
    NO df_checkout sync events fire — isolating the project-main mirror arm."""

    async def test_ff_rejected_forces_mirror_and_verify_proceeds(self):
        remote, calls, expected = _make_capstone_remote(
            local_head='CURR', remote_head='CURR',  # DF code current → no df sync
            ff_rc=1, force_rc=0, resolved_main='MAINSHA',
        )
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        # Verify proceeded to the remote verdict despite the divergent main.
        assert result == expected
        assert pool.is_quarantined('laptop') is False
        assert _merge_sha_pushed(calls)

        # The FF main push preceded the force refspec (fast path tried first).
        refspecs = _push_refspecs(calls)
        assert 'main:refs/heads/main' in refspecs
        assert '+main:refs/heads/main' in refspecs
        assert refspecs.index('main:refs/heads/main') < refspecs.index('+main:refs/heads/main')

        # Exactly one runner_synced, of kind project_main_mirror, forced.
        synced = store.events_of(EventType.runner_synced)
        assert len(synced) == 1
        assert synced[0]['kind'] == 'project_main_mirror'
        assert synced[0]['forced'] is True
        assert synced[0]['to_head'] == 'MAINSHA'

        # DF code checkout was current → NO df_checkout staleness churn.
        assert store.events_of(EventType.runner_stale) == []
        assert all(e['kind'] == 'project_main_mirror' for e in synced)

    async def test_ff_and_force_both_fail_is_non_fatal_verify_still_returns(self):
        """Both the FF and the force main push fail → best-effort swallow: no
        runner_synced, no raise, and the load-bearing merge-sha verify STILL
        returns the remote verdict (the main push is never load-bearing)."""
        remote, calls, expected = _make_capstone_remote(
            local_head='CURR', remote_head='CURR',
            ff_rc=1, force_rc=1,  # both rejected
        )
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        assert result == expected
        assert _merge_sha_pushed(calls), 'merge-sha push stays load-bearing even when main push fails'
        # No mirror event when the force also failed.
        assert store.events_of(EventType.runner_synced) == []
