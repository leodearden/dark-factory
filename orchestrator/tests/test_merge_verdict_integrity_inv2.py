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


# ---------------------------------------------------------------------------
# (4) Task 4539 — the auto-sync must not DESTROY the host it is syncing.
#
# dark-factory's root pyproject.toml declares a uv WORKSPACE
# ([tool.uv.workspace].members).  In a workspace a bare `uv sync` syncs only the
# ROOT project's environment and PRUNES what the root does not declare —
# including the workspace MEMBERS' console-script entry points.  MEASURED on the
# real second host (leo-laptop, /home/leo/src/dark-factory at main): before,
# `.venv/bin/orchestrator` existed and `orchestrator verify-merge --help`
# returned rc=0; after a bare `uv sync` the entry point was GONE and the ssh
# entry-point wrapper failed rc=127; `uv sync --all-packages` restored it.
#
# So wiring df_checkout_path for a remote runner meant the FIRST stale-sync
# deleted the very CLI the runner then invokes over ssh, and every later
# dispatch raised RunnerUnavailable — fail-SAFE (the pool falls back to local)
# but a SILENT disabling of the remote host that presents as transport
# flakiness.
#
# `_FakeWorkspaceHost` below models exactly that measured behaviour, including
# the load-bearing detail that the destructive bare sync EXITS 0.
# ---------------------------------------------------------------------------


class _FakeWorkspaceHost:
    """A remote host holding a uv-WORKSPACE dark-factory checkout.

    Models the measured uv semantics:

      * ``uv sync --all-packages`` installs every workspace member, so the
        ``orchestrator`` console script exists  -> entry point ANSWERS (rc 0).
      * a bare ``uv sync`` syncs the workspace ROOT only and prunes the members'
        console scripts                          -> entry point GONE (rc 127).
      * BOTH return 0.  The sync's own exit code cannot distinguish them; that
        is precisely why keying success on the return codes alone reproduces
        this defect.

    ``restore_on_all_packages=False`` models a host whose install is broken for
    some other reason — even the correct sync command leaves no working CLI —
    so the liveness assertion is exercised independently of the flag fix.
    """

    def __init__(self, *, entry_point_present: bool = True,
                 restore_on_all_packages: bool = True) -> None:
        self.entry_point_present = entry_point_present
        self.restore_on_all_packages = restore_on_all_packages
        self.sync_commands: list[str] = []

    def uv_sync(self, remote_cmd: str) -> tuple[int, str, str]:
        self.sync_commands.append(remote_cmd)
        if '--all-packages' in remote_cmd:
            # A healthy host reinstalls every member's console script; a host
            # whose install is broken for some OTHER reason still ends the sync
            # with no runnable CLI, and the sync still exits 0.
            self.entry_point_present = self.restore_on_all_packages
        else:
            # Workspace-root-only sync: the members' entry points are pruned.
            self.entry_point_present = False
        return (0, '', '')  # rc 0 EITHER WAY — the whole point.

    def run_entry_point(self) -> tuple[int, str, str]:
        """What the ssh entry-point wrapper does for `orchestrator ...`."""
        if not self.entry_point_present:
            return (127, '', 'bash: line 1: orchestrator: command not found')
        return (0, '', '')


def _make_workspace_remote(
    *,
    host: _FakeWorkspaceHost,
    local_head: str = 'NEW_DISPATCHER_HEAD',
    remote_head: str = 'ONE_COMMIT_STALE',
    name: str = 'laptop',
) -> tuple[RemoteRunner, list[tuple[list[str], Any]], VerifyResult]:
    """REAL RemoteRunner whose ssh transport is serviced by *host*.

    Differs from ``_make_capstone_remote`` only in that the ``uv sync`` and
    ``orchestrator ...`` legs are answered by the stateful host rather than by
    canned return codes — so a sync that breaks the CLI actually breaks the
    subsequent dispatch, exactly as it did on the real second host.
    """
    calls: list[tuple[list[str], Any]] = []
    state = {'pulled': False}
    expected = VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='', summary='remote-ok',
    )

    async def fake_run(argv, *, cwd=None):
        calls.append((list(argv), cwd))
        if argv[:3] == ['git', 'rev-parse', 'HEAD']:
            return (0, local_head, '')
        if argv[:3] == ['git', 'rev-parse', 'main']:
            return (0, 'MAINSHA', '')
        if argv[:3] == ['git', 'rev-parse', '@{upstream}']:
            return (128, '', 'fatal: no upstream configured')
        if argv[:2] == ['git', 'push']:
            return (0, '', '')
        if argv and argv[0] == 'ssh':
            remote_cmd = argv[-1]
            if 'pull --ff-only' in remote_cmd:
                state['pulled'] = True
                return (0, '', '')
            if 'uv sync' in remote_cmd:
                return host.uv_sync(remote_cmd)
            if 'rev-parse HEAD' in remote_cmd:
                return (0, local_head if state['pulled'] else remote_head, '')
            if remote_cmd.startswith('orchestrator '):
                rc, _out, err = host.run_entry_point()
                if rc != 0:
                    return (rc, '', err)
                if '--help' in remote_cmd:
                    return (0, 'Usage: orchestrator verify-merge [OPTIONS]', '')
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


@pytest.mark.asyncio
class TestInv2SyncDoesNotBreakTheWorkspaceHost:
    """The user-observable signal for task 4539."""

    async def test_stale_workspace_host_still_answers_after_sync_and_verdict_parses(self):
        """Against a workspace-layout checkout deliberately set ONE COMMIT STALE,
        a dispatch through sync_if_stale leaves the remote's
        ``orchestrator verify-merge --help`` returning rc=0, and the verify
        dispatch returns a parseable VerifyResult rather than RunnerUnavailable.

        Asserting rc=0 AFTER the sync is the load-bearing half: a test that only
        checked the sync's own exit code would pass against the very defect this
        exists to fix (the destructive bare ``uv sync`` exits 0).
        """
        host = _FakeWorkspaceHost(entry_point_present=True)
        remote, calls, expected = _make_workspace_remote(host=host)
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        # The sync ran (the checkout WAS stale) ...
        assert len(store.events_of(EventType.runner_stale)) == 1
        assert host.sync_commands, 'the stale checkout must have been synced'

        # ... and the host's CLI still answers rc=0 afterwards.
        assert host.entry_point_present is True, (
            f'the sync deleted the remote orchestrator entry point; '
            f'sync commands issued: {host.sync_commands!r}'
        )
        assert host.run_entry_point()[0] == 0, (
            '`orchestrator verify-merge --help` must return rc=0 on the remote '
            'AFTER the sync'
        )

        # ... and the dispatch produced a parseable VerifyResult, not a bench.
        assert result == expected
        assert result.summary == 'remote-ok'
        assert pool.is_quarantined('laptop') is False
        assert _merge_sha_pushed(calls)
        assert len(store.events_of(EventType.runner_synced)) == 1
        assert store.events_of(EventType.runner_synced)[0]['kind'] == 'df_checkout'

    async def test_second_dispatch_is_not_disabled_by_the_first_sync(self):
        """The silent-disabling shape: the FIRST stale-sync deleted the CLI, so
        every LATER dispatch raised RunnerUnavailable and the remote host was
        effectively off — read as transport flakiness, not as a broken sync.

        Both dispatches must adopt the remote verdict.
        """
        host = _FakeWorkspaceHost(entry_point_present=True)
        remote, calls, expected = _make_workspace_remote(host=host)
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote], event_store=store, task_id=_TASK_ID)

        first = await pool.dispatch('mergesha-1', _make_spec())
        second = await pool.dispatch('mergesha-2', _make_spec())

        assert first == expected
        assert second == expected, (
            'the first sync disabled the remote host for every later dispatch'
        )
        assert pool.is_quarantined('laptop') is False

    async def test_a_sync_that_breaks_the_host_benches_it_loudly_before_dispatch(self):
        """When the sync leaves no working CLI for ANY reason, the runner is
        benched at the gate — not dispatched to and rediscovered as rc=127.

        With a local trust anchor present the pool adopts the LOCAL verdict; the
        remote's merge-sha push must never fire, which is what distinguishes
        "benched by the liveness assertion" from "dispatched, then failed".
        """
        host = _FakeWorkspaceHost(
            entry_point_present=True, restore_on_all_packages=False,
        )
        remote, calls, _expected = _make_workspace_remote(host=host)
        local = _CapstoneLocal()
        store = _RecordingEventStore()
        pool = VerifyRunnerPool([remote, local], event_store=store, task_id=_TASK_ID)

        result = await pool.dispatch('mergesha', _make_spec())

        assert result.summary == 'local-ok'
        assert pool.is_quarantined('laptop') is True
        # Benched BEFORE the wasted dispatch: no merge-sha push on the remote.
        assert not _merge_sha_pushed(calls), (
            'the liveness assertion must bench the runner at the gate, not let '
            f'the dispatch proceed to an rc=127; pushes: {_push_refspecs(calls)!r}'
        )
        # A broken sync is not a sync success.
        assert len(store.events_of(EventType.runner_stale)) == 1
        assert store.events_of(EventType.runner_synced) == []
        # The verdict is attributed to the local anchor.
        mv = store.events_of(EventType.merge_verify)
        assert len(mv) == 1
        assert mv[0]['runner'] == 'local'
