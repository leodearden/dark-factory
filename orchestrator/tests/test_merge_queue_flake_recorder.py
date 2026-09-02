"""Boundary row B3, end-to-end on the DISPATCHER (PRD task ε).

`plans/flake-ledger-prd.md` §5.8.  The discriminator runs wherever the
worktree is; the recorder runs in ``merge_queue._run_post_merge_verify``, the one
scope where a local OR remote ``VerifyResult`` sits alongside ``event_store``,
``escalation_queue``, ``project_root``, ``merge_sha`` and ``task_id`` at once.

That co-location is the whole point: it makes the three side-effects unconditional
BY CONSTRUCTION rather than dependent on which host ran the verify.  Before ε a
REMOTE merge verify landed ZERO of the three — no ledger row (there was no ledger
call anywhere), no ``merge_flake_suppressed`` fact (the remote host has no event
store), and a storm streak that reset with the remote process.  These tests drive
the REAL ``_run_post_merge_verify`` over both hosts and assert the outcome is
identical.

Modelled on test_merge_boundary_effective_module_configs.py, which already drives
the same funnel with a real config and a real ``LocalRunner``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_flake_recorder import _FakeLedgerTaskClient
from test_merge_boundary_effective_module_configs import (
    _ALPHA_TEST,
    _BETA_FAILING_ID,
    _BETA_TEST,
    _DELTA_TEST,
    _failing_scoped_result,
    _passing_result,
)
from test_merge_queue_main_health import _make_config, _make_git_ops, _make_req
from test_verify_merge_flake_suppression import (
    _FakeEscalationQueue,
    _FakeEventStore,
    _materialize,
    _module_config,
)

from orchestrator import chronic_flake, flake_recorder, verify
from orchestrator.event_store import EventType
from orchestrator.flake_ledger import (
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    ledger_db_path,
    list_open_debt,
    read_occurrences,
)
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_queue import (
    GroupMergeRequest,
    MergeOutcome,
    QueuedBranch,
    RealMergeItem,
    SpeculativeMergeWorker,
    _do_train_merge,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import VerifyRunner, result_from_json, result_to_json

_MERGE_SHA = 'd' * 40
_REMOTE = 'remote-lab-1'


@pytest.fixture(autouse=True)
def _reset_suppression_streak():
    """The INV-4 streak is a module global on ``flake_recorder``; reset it around
    every test so a count assertion measures THIS test's suppressions only."""
    flake_recorder._merge_flake_suppression_streak = 0
    yield
    flake_recorder._merge_flake_suppression_streak = 0


def _suppression(
    verdict: FlakeVerdict = FlakeVerdict.passes_in_isolation,
    *,
    test_ids: tuple[str, ...] = (_BETA_FAILING_ID,),
    reason: str | None = None,
) -> FlakeSuppression:
    """What the DISCRIMINATOR produced, wherever the worktree was.

    ``runner='local'`` is what it always stamps — host-relative and therefore
    wrong once read on the dispatcher; ``VerifyRunnerPool.dispatch`` corrects it.
    """
    return FlakeSuppression(
        verdict=verdict,
        test_ids=test_ids,
        observed_at='2026-08-22T12:00:00+00:00',
        call_site=FlakeCallSite.merge_gate,
        runner='local',
        psi_cpu_some10=88.0,
        unconfirmable_reason=reason,
    )


def _wired(result: VerifyResult) -> VerifyResult:
    """Round-trip through the REAL runner codec, as the remote path does.

    Not decoration: this is what makes the recorder's ``==``-not-``is`` verdict
    comparison load-bearing, since JSON hands ``verdict`` back as a plain ``str``.
    """
    return result_from_json(result_to_json(result))


def _suppressed_pass(s: FlakeSuppression) -> VerifyResult:
    """The shape ``apply_merge_flake_suppression`` returns on a suppression."""
    return VerifyResult(
        passed=True,
        test_output=f'FAILED {_BETA_FAILING_ID}\n',
        lint_output='',
        type_output='',
        summary='merge-verify flake suppressed (isolated re-run passed)',
        category='merge_flake_suppressed',
        flake_suppression=s,
    )


def _still_failing(s: FlakeSuppression) -> VerifyResult:
    """The shape it returns on a NON-suppressing verdict: the red, observed."""
    from dataclasses import replace

    return replace(_failing_scoped_result(_BETA_FAILING_ID), flake_suppression=s)


def _remote_runner(*results: VerifyResult) -> MagicMock:
    """A fake REMOTE runner returning *results* in order (one per dispatch)."""
    r = MagicMock(spec=VerifyRunner)
    r.name = _REMOTE
    r.is_local = False
    r.run_merge_verify = AsyncMock(side_effect=list(results))
    return r


async def _drive(
    tmp_path: Path,
    *,
    task_id: str,
    runner: MagicMock | None,
    event_store=None,
    escalation_queue=None,
    cross_check: bool = False,
    rerun_passes: bool = True,
    max_enospc: int = 1,
    task_client=None,
):
    """Drive the REAL ``_run_post_merge_verify``.

    On the LOCAL path (*runner* None) the boundary builds its own ``LocalRunner``,
    so ``run_scoped_verification`` is patched to the red and
    ``verify.run_verification`` to the isolated re-run — the production gate then
    produces the ``FlakeSuppression`` itself, rather than the test hand-feeding one.
    On the REMOTE path the injected runner returns its own queued results directly,
    which is exactly how a real remote's already-suppressed verdict arrives.

    ``verify_cross_check_remote_green`` is OFF by default so the dispatched
    verdict's recording is measured on its own; the one case that needs the
    cross-check turns it on explicitly.
    """
    config = _make_config(tmp_path, merge_verify_breadth='full')
    mc_alpha, mc_beta = _module_config('alpha'), _module_config('beta')
    config._module_configs = {'alpha': mc_alpha, 'beta': mc_beta}
    config.verify_cross_check_remote_green = cross_check

    git_ops = _make_git_ops(tmp_path)
    task_wt = tmp_path / f'task-wt-{task_id}'
    task_wt.mkdir(parents=True, exist_ok=True)
    merge_wt = tmp_path / f'merge-wt-{task_id}'
    merge_wt.mkdir(parents=True, exist_ok=True)
    _materialize(merge_wt, _ALPHA_TEST, _BETA_TEST, _DELTA_TEST)

    req = _make_req(task_id, task_wt, config)
    req.module_configs = [mc_alpha]

    with (
        patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=_failing_scoped_result(_BETA_FAILING_ID)),
        ),
        patch(
            'orchestrator.merge_queue._run_unscoped_typechecks',
            new=AsyncMock(return_value=PostMergePyrightResult()),
        ),
        patch.object(
            verify, 'run_verification',
            new=AsyncMock(
                return_value=_passing_result() if rerun_passes
                else _failing_scoped_result(_BETA_FAILING_ID),
            ),
        ),
    ):
        return await _run_post_merge_verify(
            git_ops, req, merge_wt,
            timeouts={},
            enospc_retries={},
            max_timeouts=3,
            max_enospc=max_enospc,
            event_store=event_store,
            escalation_queue=escalation_queue,
            merge_sha=_MERGE_SHA,
            runner=runner,
            task_client=task_client,
        )


def _rows(tmp_path: Path):
    return read_occurrences(ledger_db_path(tmp_path))


def _suppression_events(store: _FakeEventStore) -> list[tuple]:
    return [e for e in store.emits if e[0] is EventType.merge_flake_suppressed]


@pytest.mark.asyncio
class TestDispatcherRecordsTheFlakeObservation:
    """B3: ledger row + structured fact + storm streak, on BOTH hosts."""

    # -- (a) B3 on the REMOTE path — the case that landed zero of three -------

    async def test_b3_remote_verdict_lands_all_three(self, tmp_path: Path) -> None:
        """B3 — the headline, asserted in ONE test.

        The observation is genuinely wire-deserialized, so this also pins that
        the recorder's ``==`` verdict comparison survives JSON, and that
        ``dispatch`` re-stamped ``runner`` from the remote's honest-but-relative
        ``'local'`` to the remote's NAME — the column θ's class-3 check reads to
        tell a bad HOST from a bad SUITE.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()

        outcome = await _drive(
            tmp_path, task_id='b3-remote',
            runner=_remote_runner(_wired(_suppressed_pass(_suppression()))),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None, f'the suppressed red must still land; got {outcome!r}'

        # (1) the structured fact
        events = _suppression_events(store)
        assert len(events) == 1, store.emits
        assert events[0][2]['node_ids'] == [_BETA_FAILING_ID]
        assert events[0][2]['merge_sha'] == _MERGE_SHA

        # (2) the durable row, attributed to the host that actually ran
        rows = _rows(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].test_id == _BETA_FAILING_ID
        assert rows[0].verdict == 'passes_in_isolation'
        assert rows[0].call_site == 'merge_gate'
        assert rows[0].runner == _REMOTE
        assert rows[0].merge_sha == _MERGE_SHA
        assert rows[0].task_id == 'b3-remote'

        # (3) the INV-4 storm streak
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (b) the LOCAL path records the identical three ----------------------

    async def test_local_path_records_the_identical_three(self, tmp_path: Path) -> None:
        """The recorder is unconditional BY CONSTRUCTION, not by which host ran.

        Here the production gate inside the boundary's own ``LocalRunner``
        produces the observation — nothing is hand-fed — and the same three
        land, with ``runner`` reading ``'local'`` because that is where it ran.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()

        outcome = await _drive(
            tmp_path, task_id='b3-local', runner=None,
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None
        assert len(_suppression_events(store)) == 1, store.emits
        rows = _rows(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].verdict == 'passes_in_isolation'
        assert rows[0].test_id == _BETA_FAILING_ID
        assert rows[0].runner == 'local'
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (c) B13: an un-upgraded remote must change nothing -------------------

    async def test_old_remote_without_the_field_is_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """B13 — new dispatcher, OLD remote: the wire payload has no
        ``flake_suppression`` key at all.

        None of the three fire, nothing raises, and the MERGE OUTCOME is exactly
        what a suppression-less run produces.  A version skew must cost the
        observation and nothing else.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        green = _passing_result()
        assert green.flake_suppression is None

        outcome = await _drive(
            tmp_path, task_id='b13', runner=_remote_runner(_wired(green)),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None
        assert _suppression_events(store) == []
        assert _rows(tmp_path) == []
        assert queue.submitted == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    # -- (d) a NON-suppressing verdict: the row, and only the row -------------

    async def test_unconfirmable_records_the_row_and_the_merge_stays_red(
        self, tmp_path: Path,
    ) -> None:
        """§5.5 — record the OBSERVATION, not the remedy.  An unconfirmable
        verdict changes no verdict, so the merge stays red and no fact is
        emitted; but the row IS written, because θ's class-1 health check is an
        unconfirmable RATE and a dropped row makes it uncomputable."""
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        s = _suppression(
            FlakeVerdict.unconfirmable, test_ids=(_BETA_FAILING_ID,),
            reason='no recoverable node-id',
        )

        outcome = await _drive(
            tmp_path, task_id='unconf', runner=_remote_runner(_wired(_still_failing(s))),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is not None, 'a non-suppressed red must NOT land'
        rows = _rows(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].verdict == 'unconfirmable'
        assert _suppression_events(store) == []
        assert flake_recorder._merge_flake_suppression_streak == 0

    # -- (e) the cross-check's own observation is recorded too ---------------

    async def test_cross_check_local_verify_is_recorded(self, tmp_path: Path) -> None:
        """The remote-green cross-check ``LocalRunner`` had ``event_store`` and
        ``escalation_queue`` wired BEFORE ε, so its suppressions emitted and
        bumped.  ε must not regress that detective-control path: its own
        ``local_verify`` observation is recorded on the dispatcher too.

        The remote returns a plain green carrying NO observation, so the only
        observation in play is the cross-check's — which the boundary's own
        ``LocalRunner`` produces through the real gate.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()

        outcome = await _drive(
            tmp_path, task_id='xcheck',
            runner=_remote_runner(_wired(_passing_result())),
            event_store=store, escalation_queue=queue,
            cross_check=True,
        )

        assert outcome is None, f'the cross-check agreed; expected a land, got {outcome!r}'
        rows = _rows(tmp_path)
        assert len(rows) == 1, (
            f'the cross-check LocalRunner suppressed a red and that observation '
            f'must reach the ledger; got {rows}'
        )
        assert rows[0].verdict == 'passes_in_isolation'
        assert rows[0].runner == 'local'
        assert len(_suppression_events(store)) == 1, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 1

    async def test_a_remote_suppression_and_a_cross_check_suppression_both_count(
        self, tmp_path: Path,
    ) -> None:
        """The compound case: the dispatched (remote) verdict suppressed AND the
        cross-check's own gate suppressed — TWO observations for ONE merge SHA.

        Both are recorded, deliberately.  The unit of the ledger and of the INV-4
        window is a SUPPRESSION (one red masked), never a merge: two independent
        gate runs on two different hosts really did mask two separate reds, and the
        differing ``runner`` column is precisely what lets θ's class-3 check tell a
        bad HOST from a bad SUITE (the same tests suppressed on both ⇒ the suite).

        Pinned rather than deduped because the alternative reading — "one merge, one
        unit" — would silently discard the local trust anchor's independent evidence.
        Before ε the remote leg contributed nothing at all here, so this is the count
        becoming COMPLETE, not double-counting; ``_bump_suppression_streak_and_maybe_
        escalate``'s docstring says so where an operator reading the counter will
        find it.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()

        outcome = await _drive(
            tmp_path, task_id='xcheck2',
            runner=_remote_runner(_wired(_suppressed_pass(_suppression()))),
            event_store=store, escalation_queue=queue,
            cross_check=True,
        )

        assert outcome is None, f'both legs passed; expected a land, got {outcome!r}'
        rows = _rows(tmp_path)
        assert len(rows) == 2, f'one row per OBSERVATION, not per merge; got {rows}'
        assert {r.verdict for r in rows} == {'passes_in_isolation'}
        # The remote's row is re-stamped by dispatch; the cross-check ran locally.
        assert sorted(r.runner or '' for r in rows) == ['local', _REMOTE], rows
        assert len(_suppression_events(store)) == 2, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 2

    # -- (f) every ATTEMPT's observation, not only the settled verdict's ------

    async def test_infra_transient_retry_records_only_the_settled_verdict(
        self, tmp_path: Path,
    ) -> None:
        """An attempt that carried NO observation contributes nothing when it is
        superseded — the retry's verdict is the only thing recorded.

        Attempt 0 returns an infra-transient red with no observation (the gate does
        not suppress an infra category); the retry returns the suppressed pass.
        Exactly one logical observation is recorded, and the streak advances by
        exactly one.  The companion test below covers the case where the superseded
        attempt DID carry one.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        transient = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='disk full', category='disk_full',
        )

        outcome = await _drive(
            tmp_path, task_id='retry',
            runner=_remote_runner(
                _wired(transient), _wired(_suppressed_pass(_suppression())),
            ),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None
        assert len(_rows(tmp_path)) == 1, _rows(tmp_path)
        assert len(_suppression_events(store)) == 1, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 1

    async def test_a_superseded_attempts_observation_is_still_recorded(
        self, tmp_path: Path,
    ) -> None:
        """A SUPERSEDED verdict is dead; its OBSERVATION is not (§5.5 — record the
        observation, not the remedy).

        The compound shape this covers is real and load-correlated, which is exactly
        the regime the PRD exists to measure: ``apply_merge_flake_suppression`` runs
        on the SCOPED leg, so an attempt can suppress a scoped red — a genuine
        ``passes_in_isolation`` observation, a real masked red — and STILL come back
        failing because the unscoped gate then broke or the host ran out of disk.
        That result is infra-transient, so it is retried and ``verify`` is rebound.
        Recording only the settled verdict would drop the observation entirely,
        where the pre-ε inline emit reported it at the moment it happened.

        Attempt 0: infra-transient red CARRYING a suppression.  Retry: a clean green
        with none.  Both attempts' observations reach the recorder — here that is
        one, from the attempt whose verdict was thrown away.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        from dataclasses import replace as _replace

        suppressed_then_broke = _replace(
            _suppressed_pass(_suppression()),
            passed=False,
            summary='disk full during the unscoped typecheck gate',
            category='disk_full',
        )

        outcome = await _drive(
            tmp_path, task_id='superseded',
            runner=_remote_runner(
                _wired(suppressed_then_broke), _wired(_passing_result()),
            ),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None, f'the retry was green; expected a land, got {outcome!r}'
        rows = _rows(tmp_path)
        assert len(rows) == 1, (
            f'the superseded attempt observed a real masked red; got {rows}'
        )
        assert rows[0].verdict == 'passes_in_isolation'
        assert rows[0].test_id == _BETA_FAILING_ID
        assert len(_suppression_events(store)) == 1, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 1


class _ExplodingTaskClient:
    """A task client whose every method raises — a wedged MCP dispatch, a closed
    scheduler, a partial adapter.  B12's worst case at this boundary."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def submit_task(self, arguments: dict) -> str:
        self.calls.append('submit_task')
        raise RuntimeError('mcp dispatch failed')

    async def get_statuses(self, ids: list[str]) -> tuple[dict[str, str], Exception | None]:
        # The PAIR, matching `flake_ledger.FlakeLedgerTaskClient`.  It raises rather than
        # returning, so the annotation is unreachable at runtime and was previously wrong
        # (`dict[str, str]`) without any test noticing.  That is a latent silent-pass, not
        # a cosmetic slip: the Protocol is structural, so nothing catches the drift, and
        # the first test to seed an OWNER here would have unpacked a bare dict, raised
        # ValueError inside `_ensure_owner_task`'s guard, and routed into the degrade
        # branch — a green test proving nothing, which is the exact hazard
        # test_flake_recorder.py's `_FakeLedgerTaskClient` docstring documents.
        self.calls.append('get_statuses')
        raise RuntimeError('mcp dispatch failed')

    async def commit_planning(self, task_ids: list[str]) -> None:
        self.calls.append('commit_planning')
        raise RuntimeError('mcp dispatch failed')


def _debt(tmp_path: Path):
    return list_open_debt(ledger_db_path(tmp_path))


@pytest.mark.asyncio
class TestDispatcherOpensDebt:
    """Boundary row B7 end-to-end (PRD task ζ): a suppressed merge red leaves an
    OWNED ``flake_debt`` row.

    ε proved the observation reaches the dispatcher; this proves the dispatcher
    turns it into §5.9's enforced invariant — a test in the ledger has a
    non-terminal de-flake task responsible both for fixing the root defect and for
    removing the test from the ledger.  That filed task is this task's whole
    user-observable signal, so it is asserted through the REAL
    ``_run_post_merge_verify`` rather than at the recorder's own seam.
    """

    # -- (a) the LOCAL path: the user-observable signal ----------------------

    async def test_local_suppression_files_the_task_and_owns_the_row(
        self, tmp_path: Path,
    ) -> None:
        """A merge whose scoped red was suppressed leaves the failing test in the
        ledger, OWNED, and exactly one de-flake task filed under ``planning_mode``.

        The observation here is produced by the production gate inside the
        boundary's own ``LocalRunner`` — nothing is hand-fed — so this is the real
        path from "a red was masked" to "somebody owns it".
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        client = _FakeLedgerTaskClient()

        outcome = await _drive(
            tmp_path, task_id='debt-local', runner=None,
            event_store=store, escalation_queue=queue, task_client=client,
        )

        assert outcome is None, f'the suppressed red must still land; got {outcome!r}'

        rows = _debt(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].test_id == _BETA_FAILING_ID
        assert rows[0].owner_task_id, f'§5.9 breach — the row has no owner: {rows[0]}'

        assert len(client.submit_calls) == 1, client.submit_calls
        args = client.submit_calls[0]
        assert args['planning_mode'] is True, args
        assert _BETA_FAILING_ID in args['title'] or _BETA_FAILING_ID in args['description']
        assert args['metadata']['flake_debt_test'] == _BETA_FAILING_ID
        # planning_mode files DEFERRED; the second phase is what releases it.
        assert client.commit_calls == [[rows[0].owner_task_id]], client.commit_calls

    # -- (b) the REMOTE path: identical, because the host must not matter -----

    async def test_remote_wire_deserialized_suppression_opens_the_same_debt(
        self, tmp_path: Path,
    ) -> None:
        """The same observation, genuinely wire-serialized through the runner codec.

        The invariant must not be a property of WHICH HOST ran the verify — that
        host-dependence is the entire defect ε fixed for the other three effects,
        and a debt row that only appears on local merges would reintroduce it for
        the fourth.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        client = _FakeLedgerTaskClient()

        outcome = await _drive(
            tmp_path, task_id='debt-remote',
            runner=_remote_runner(_wired(_suppressed_pass(_suppression()))),
            event_store=store, escalation_queue=queue, task_client=client,
        )

        assert outcome is None
        rows = _debt(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].test_id == _BETA_FAILING_ID
        assert rows[0].owner_task_id
        assert len(client.submit_calls) == 1, client.submit_calls
        assert client.submit_calls[0]['metadata']['flake_debt_test'] == _BETA_FAILING_ID

    # -- (c) no client wired: byte-identical to ε -----------------------------

    async def test_no_task_client_is_byte_identical_to_the_epsilon_behaviour(
        self, tmp_path: Path,
    ) -> None:
        """The default, and the two ``_run_post_merge_verify`` callers that thread
        nothing.

        Occurrence row, event and streak exactly as ε left them; NO debt row,
        because a row nobody can own would be rendered by ι as an invariant breach
        and would swamp the surface that exists to show a filing which was
        ATTEMPTED and failed.  This is what makes those two callers provably
        unaffected by ζ.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()

        outcome = await _drive(
            tmp_path, task_id='debt-none',
            runner=_remote_runner(_wired(_suppressed_pass(_suppression()))),
            event_store=store, escalation_queue=queue,
        )

        assert outcome is None
        assert _debt(tmp_path) == []
        assert len(_rows(tmp_path)) == 1, _rows(tmp_path)
        assert len(_suppression_events(store)) == 1, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 1

    # -- (d) B12: a wedged client costs the filing and nothing else -----------

    async def test_a_raising_task_client_leaves_the_merge_outcome_unchanged(
        self, tmp_path: Path,
    ) -> None:
        """B12 at the boundary that matters: bookkeeping must never stall a merge.

        Every client method raises, so the filing is lost — and the MERGE OUTCOME,
        the occurrence row, the structured fact and the INV-4 streak are all exactly
        what a ledger-less run produces.  The breach is left VISIBLE (a row with no
        owner) rather than papered over, which is precisely what ι renders and what
        distinguishes a failed filing from an unwired one.
        """
        store, queue = _FakeEventStore(), _FakeEscalationQueue()
        client = _ExplodingTaskClient()

        outcome = await _drive(
            tmp_path, task_id='debt-boom',
            runner=_remote_runner(_wired(_suppressed_pass(_suppression()))),
            event_store=store, escalation_queue=queue, task_client=client,
        )

        assert outcome is None, f'a wedged task client must not stall the merge: {outcome!r}'
        assert client.calls == ['submit_task'], client.calls
        rows = _debt(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0].owner_task_id is None, 'a failed filing must stay visible'
        assert len(_rows(tmp_path)) == 1
        assert len(_suppression_events(store)) == 1, store.emits
        assert flake_recorder._merge_flake_suppression_streak == 1


# ---------------------------------------------------------------------------
# The CONSTRUCTION seam (PRD task ζ, §5.9): the worker builds the task client,
# and both of its readers actually hand it to `_run_post_merge_verify`.
#
# Cited from `merge_queue.py::SpeculativeMergeWorker.__init__` at the
# `self._flake_task_client` assignment.
# ---------------------------------------------------------------------------


def _worker(git_ops, *, scheduler=None) -> SpeculativeMergeWorker:
    """A bare worker over *git_ops*, wired with *scheduler* (or not).

    Only the two ctor arguments that decide ``_flake_task_client`` are varied;
    everything else is the same bare-harness construction the rest of the merge
    suite uses (``git_ops`` + an empty queue).
    """
    return SpeculativeMergeWorker(
        git_ops=git_ops, queue=asyncio.Queue(), scheduler=scheduler,
    )


def _group_req(tmp_path: Path, config, *, task_id: str = 'train-tip'):
    """A minimal ``GroupMergeRequest`` whose members all report merge-deferred.

    Shaped after ``test_atomic_train_merge.build_group_merge_request`` — a
    single-member train, which is all ``_do_train_merge`` needs to reach its
    ``_run_post_merge_verify`` call: the status pre-check passes, the rebase and
    the merge are stubbed green on ``git_ops``.
    """
    return GroupMergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=tmp_path,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        train_id='train-1',
        member_task_ids=[task_id],
        tip_branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        tip_task_id=task_id,
        status_check=AsyncMock(return_value={task_id: 'merge-deferred'}),
        mark_member_done=AsyncMock(),
    )


@pytest.mark.asyncio
class TestWorkerBuildsTheFlakeTaskClient:
    """The worker CONSTRUCTS the ζ flake task client, and BOTH readers supply it.

    ``TestDispatcherOpensDebt`` above proves the *parameter* is honoured: hand
    ``_run_post_merge_verify`` a client and a suppressed red becomes an owned debt
    row.  That is a different claim from "the production callers actually hand it
    one", and this class pins the second.

    The distinction matters because this seam's failure mode is SILENT.  A caller
    that stops supplying the client — a typo'd ``getattr`` name in
    ``_do_train_merge``, a ctor refactor that drops the ``scheduler`` argument, a
    ``project_root`` that stops resolving — degrades exactly into the well-tested
    "nothing wired" configuration: the merge still lands, the occurrence row and
    the structured fact and the INV-4 streak are all still written, and §5.9 simply
    stops being enforced with nothing anywhere going red.  Only an assertion that
    the handle ARRIVES at the boundary can catch that, so each forwarding test also
    asserts the handle is not None — ``None is None`` would pass for the exact
    regression these tests exist to detect.

    Construction is conditional on both a scheduler and a project_root because the
    alternative is an UNOWNED debt row (see
    ``flake_recorder.py::record_merge_flake_suppression``), so the None cases are
    pinned as deliberate behaviour rather than tolerated slack.
    """

    # -- (a) construction ----------------------------------------------------

    async def test_scheduler_and_project_root_build_a_bound_client(
        self, tmp_path: Path,
    ) -> None:
        """Both ingredients present → a ``SchedulerChronicFlakeTaskClient`` bound to
        THIS scheduler and THIS project root.

        The binding is asserted, not just the type: a client built over some other
        root would file its de-flake tasks against the wrong project, which is
        indistinguishable from "filed correctly" at every other seam.
        """
        scheduler = MagicMock()
        worker = _worker(_make_git_ops(tmp_path), scheduler=scheduler)

        client = worker._flake_task_client
        assert isinstance(client, chronic_flake.SchedulerChronicFlakeTaskClient), (
            f'§5.9 needs a real adapter here, got {client!r}'
        )
        assert client._scheduler is scheduler, (
            'the client must dispatch through the worker\'s own scheduler'
        )
        assert client._project_root == str(tmp_path), (
            f'client bound to {client._project_root!r}, expected {str(tmp_path)!r}'
        )

    async def test_no_scheduler_yields_no_client(self, tmp_path: Path) -> None:
        """A bare-harness worker (no scheduler) gets None — the recorder then opens
        no debt row at all, rather than one nobody can own."""
        worker = _worker(_make_git_ops(tmp_path))

        assert worker._flake_task_client is None, (
            f'no scheduler must mean no client, got {worker._flake_task_client!r}'
        )

    async def test_git_ops_without_project_root_yields_no_client(self) -> None:
        """A mock ``git_ops`` carrying no usable ``project_root`` gets None too.

        ``MagicMock(spec=GitOps)`` has no ``project_root`` attribute at all (it is
        an instance attribute, not a class one), which is precisely the bare-mock
        shape most of the merge suite constructs — the same None-safety
        ``_shadow_state_path`` is built with one line above.
        """
        worker = _worker(MagicMock(spec=GitOps), scheduler=MagicMock())

        assert worker._flake_task_client is None, (
            f'no project_root must mean no client, got {worker._flake_task_client!r}'
        )

    # -- (b) read site 1: SpeculativeMergeWorker._run_inflight_verify --------

    async def test_run_inflight_verify_supplies_the_client(
        self, tmp_path: Path,
    ) -> None:
        """The speculative worker's own verify dispatch passes ``_flake_task_client``.

        Driven exactly as ``test_merge_queue_depth_telemetry.py``'s
        ``TestRunInflightVerifyDepthWiring`` drives the same method — a real
        ``RealMergeItem`` + ``HostLease`` with ``_run_post_merge_verify`` patched to
        capture its kwargs.
        """
        from orchestrator.verify_runner import HostLease

        config = _make_config(tmp_path)
        req = _make_req('inflight-tip', tmp_path, config)
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit=_MERGE_SHA, merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='a' * 40,
            speculative=True,
        )
        lease = HostLease(name='remote-1', runner=MagicMock(), is_local=False)

        captured: dict = {}

        async def _capture(*_args, **kwargs):
            captured.update(kwargs)
            return None  # verify passed; nothing else in the method matters here

        worker = _worker(_make_git_ops(tmp_path), scheduler=MagicMock())

        with patch('orchestrator.merge_queue._run_post_merge_verify', _capture):
            await worker._run_inflight_verify(item, lease, depth=0)

        assert 'task_client' in captured, (
            f'_run_inflight_verify supplied no task_client at all: {sorted(captured)}'
        )
        assert worker._flake_task_client is not None, (
            'guard: the worker was built WITH a scheduler and a project_root, so a '
            'None here would make the identity assertion below vacuous'
        )
        assert captured['task_client'] is worker._flake_task_client, (
            f'expected the worker\'s own client, got {captured["task_client"]!r}'
        )

    # -- (c) read site 2: module-level _do_train_merge ------------------------

    async def test_do_train_merge_supplies_the_client(self, tmp_path: Path) -> None:
        """The train pipeline passes the same handle off the worker it is given.

        This site reads the attribute by NAME through ``getattr`` (the narrow
        ``_TrainMergeHost`` Protocol declares neither flake handle), so a typo in
        that string silently degrades to the ``None`` default with every other
        train assertion in the suite still green.  Only an identity check here
        catches it.

        A train's masked red is no less debt than a single-branch one, which is why
        the train path must supply a client at all.
        """
        config = _make_config(tmp_path)
        req = _group_req(tmp_path, config)

        git_ops = _make_git_ops(tmp_path)
        git_ops.rebase_onto_main = AsyncMock(return_value=True)
        git_ops.merge_to_main = AsyncMock(return_value=MergeResult(
            success=True, merge_commit=_MERGE_SHA, merge_worktree=tmp_path,
        ))

        worker = _worker(git_ops, scheduler=MagicMock())

        captured: dict = {}

        async def _capture(*_args, **kwargs):
            captured.update(kwargs)
            # A non-None outcome short-circuits _do_train_merge right here, so the
            # test never reaches the CAS-advance / finalize machinery it is not
            # about.  failure_category='' keeps it off the interaction-candidate
            # tagging branch.
            return MergeOutcome('blocked', reason='verify red (test short-circuit)')

        with patch('orchestrator.merge_queue._run_post_merge_verify', _capture):
            outcome = await _do_train_merge(worker, req)

        assert outcome.status == 'blocked', (
            f'guard: the patched verify must have been reached; got {outcome!r}'
        )
        assert 'task_client' in captured, (
            f'_do_train_merge supplied no task_client at all: {sorted(captured)}'
        )
        assert worker._flake_task_client is not None, (
            'guard: built WITH a scheduler and a project_root, so the identity '
            'assertion below is not None-is-None'
        )
        assert captured['task_client'] is worker._flake_task_client, (
            f'expected the worker\'s own client (is the getattr name right?), '
            f'got {captured["task_client"]!r}'
        )
