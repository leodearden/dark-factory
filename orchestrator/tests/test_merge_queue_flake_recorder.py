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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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

from orchestrator import flake_recorder, verify
from orchestrator.event_store import EventType
from orchestrator.flake_ledger import (
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    ledger_db_path,
    read_occurrences,
)
from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_queue import _run_post_merge_verify
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
