"""Task 5471: a confirmed pre-existing main break always ends owned.

Leo's ruling: the non-auto-heal-eligible branches FILE a normal fix-main task
alongside the deduplicated escalation, one task per sha-independent failure
signature.  The invariant pinned here: no classification outcome terminates
with neither a fix task nor an owned escalation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue
from shared.task_submit import SubmitFailed, SubmittedTask
from test_merge_queue_main_health import (
    COMPILE_ERROR_RESULT,
    MAIN_SHA,
    _make_config,
    _make_git_ops,
    _make_req,
)

from orchestrator.artifacts import TaskArtifacts
from orchestrator.git_ops import GitOps
from orchestrator.main_health_actions import MainHealthNextAction
from orchestrator.main_health_fix_owner import (
    REPRODUCE_FIRST,
    SIGNATURE_KEY,
    MainBreak,
    MainFixTaskFiler,
    ensure_main_fix_owner,
)
from orchestrator.merge_lane.worker import (
    MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS,
    SpeculativeMergeWorker,
    _build_main_health_outcome,
)
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

OTHER_MAIN_SHA = 'cafef00d0987654321'

TEST_FAILURE_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED tests/test_foo.py::test_bar',
    lint_output='',
    type_output='',
    summary='pytest failed',
    cause_hint='tests/test_foo.py::test_bar',
    category='test_failure',
)

OWNED_ACTIONS = frozenset({
    MainHealthNextAction.FIX_TASK_FILED,
    MainHealthNextAction.AUTO_HEAL_IN_FLIGHT,
})


def _envelope(payload: dict) -> dict:
    return {'result': {'content': [{'type': 'text', 'text': json.dumps(payload)}]}}


class FakeTaskStore:
    """An in-memory task store speaking the scheduler's dispatch_tool/submit_task seam."""

    def __init__(
        self, *, lookup_fails: bool = False, submit_fails: bool = False, commit_fails: bool = False,
    ) -> None:
        self.tasks: dict[str, dict] = {}
        self.submits: list[dict] = []
        self.lookup_fails = lookup_fails
        self.submit_fails = submit_fails
        self.commit_fails = commit_fails

    def add(self, *, status: str, metadata: dict) -> str:
        task_id = str(9000 + len(self.tasks))
        self.tasks[task_id] = {'status': status, 'metadata': dict(metadata)}
        return task_id

    def open_owners(self, signature: str) -> list[str]:
        return [
            tid for tid, task in self.tasks.items()
            if task['metadata'].get(SIGNATURE_KEY) == signature
            and task['status'] not in ('done', 'cancelled')
        ]

    async def dispatch_tool(self, name: str, arguments: dict, *, timeout: float = 15) -> dict:
        await asyncio.sleep(0)
        if name == 'find_tasks_by_metadata':
            if self.lookup_fails:
                return _envelope({'error': 'store unavailable', 'error_type': 'BackendError'})
            rows = [
                {'task_id': int(tid), 'status': task['status']}
                for tid, task in sorted(self.tasks.items())
                if task['metadata'].get(arguments['key']) == arguments['value']
                and task['status'] in arguments['statuses']
            ]
            return _envelope({'results': rows})
        if name == 'get_task':
            task = self.tasks[arguments['id']]
            return _envelope({'data': {'id': arguments['id'], **task}})
        if name == 'commit_planning':
            return _envelope(self._commit_planning(arguments))
        raise AssertionError(f'unexpected tool {name}')

    def _commit_planning(self, arguments: dict) -> dict:
        """Like fused-memory: only a planning hold (or none) is released."""
        if self.commit_fails:
            return {'error': 'timed out', 'error_type': 'TimeoutError'}
        for tid in arguments['task_ids'].split(','):
            task = self.tasks[tid]
            if task['metadata'].get('deferral', {'kind': 'planning'})['kind'] != 'planning':
                return {'error': 'not a planning hold', 'error_type': 'DeferralInvalid'}
            if task['status'] == 'deferred':
                task['status'] = arguments['target_status']
                task['metadata'].pop('deferral', None)
        return {'success': True}

    async def submit_task(self, arguments: dict, *, timeout: float = 30) -> SubmittedTask | SubmitFailed:
        await asyncio.sleep(0)
        self.submits.append(arguments)
        if self.submit_fails:
            return SubmitFailed('submit refused')
        if arguments.get('planning_mode'):
            metadata = {**arguments['metadata'], 'deferral': {'kind': 'planning'}}
            return SubmittedTask(self.add(status='deferred', metadata=metadata), 'deferred')
        return SubmittedTask(self.add(status='pending', metadata=arguments['metadata']), 'pending')

    async def post_submit_tasks(self, arguments_list: list[dict]) -> None:
        for arguments in arguments_list:
            await self.submit_task(arguments)


def _break_of(outcome) -> MainBreak:
    return MainBreak(outcome.failure_category or '', outcome.failure_cause_hint or '', outcome.reason)


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _ensure(store: FakeTaskStore, brk: MainBreak, filer: MainFixTaskFiler | None = None):
    filer = filer or MainFixTaskFiler(store)
    return asyncio.run(filer.ensure(brk, project_root='/proj'))


TEST_BREAK = MainBreak('test_failure', 'tests/test_foo.py::test_bar', 'main red')


class TestMainFixTaskFiler:
    def test_files_one_high_lane_task_and_releases_it_from_planning(self) -> None:
        store = FakeTaskStore()

        owner = _ensure(store, TEST_BREAK)

        assert owner.task_id is not None and owner.filed
        [arguments] = store.submits
        assert arguments['planning_mode'] is True
        assert arguments['priority'] == 'high'
        assert arguments['title'].startswith('fix main:')
        assert REPRODUCE_FIRST in arguments['description']
        assert arguments['metadata'] == {
            'merge_lane': 'high',
            'spawn_context': 'main_health_fix_filed',
            SIGNATURE_KEY: TEST_BREAK.signature,
        }
        assert store.tasks[owner.task_id]['status'] == 'pending'
        assert owner.next_action(MainHealthNextAction.NO_FIX_OWNER) is MainHealthNextAction.FIX_TASK_FILED

    @pytest.mark.parametrize(
        'status', ['pending', 'in-progress', 'blocked', 'deferred', 'merge-deferred', 'infra-hold'],
    )
    def test_adopts_the_open_task_for_the_signature(self, status: str) -> None:
        store = FakeTaskStore()
        existing = store.add(status=status, metadata={SIGNATURE_KEY: TEST_BREAK.signature})

        owner = _ensure(store, TEST_BREAK)

        assert store.submits == []
        assert owner.task_id == existing and not owner.filed

    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_a_closed_owner_means_the_break_recurred_and_files_anew(self, status: str) -> None:
        store = FakeTaskStore()
        closed = store.add(status=status, metadata={SIGNATURE_KEY: TEST_BREAK.signature})

        owner = _ensure(store, TEST_BREAK)

        assert len(store.submits) == 1
        assert owner.task_id not in (None, closed)

    def test_a_half_filed_owner_is_released_by_the_next_break(self) -> None:
        store = FakeTaskStore(commit_fails=True)
        filer = MainFixTaskFiler(store)
        first = _ensure(store, TEST_BREAK, filer).task_id
        assert first is not None
        assert store.tasks[first]['status'] == 'deferred'
        store.commit_fails = False

        second = _ensure(store, TEST_BREAK, filer)

        assert second.task_id == first
        assert len(store.submits) == 1
        assert store.tasks[first]['status'] == 'pending'

    def test_a_hand_carried_deferral_is_left_alone(self) -> None:
        store = FakeTaskStore()
        held = store.add(status='deferred', metadata={
            SIGNATURE_KEY: TEST_BREAK.signature,
            'spawn_context': 'main_health_fix_filed',
            'deferral': {'kind': 'held_by_session', 'reason': 'hand carry'},
        })

        owner = _ensure(store, TEST_BREAK)

        assert owner.task_id == held
        assert store.submits == []
        assert store.tasks[held]['status'] == 'deferred'

    def test_concurrent_breaks_of_one_signature_file_one_task(self) -> None:
        store = FakeTaskStore()
        filer = MainFixTaskFiler(store)

        async def _five_at_once():
            return await asyncio.gather(
                *(filer.ensure(TEST_BREAK, project_root='/proj') for _ in range(5)),
            )

        owners = asyncio.run(_five_at_once())

        assert len(store.submits) == 1
        assert {owner.task_id for owner in owners} == {store.open_owners(TEST_BREAK.signature)[0]}

    def test_a_failed_lookup_files_nothing(self) -> None:
        store = FakeTaskStore(lookup_fails=True)

        owner = _ensure(store, TEST_BREAK)

        assert store.submits == []
        assert owner.task_id is None
        assert owner.next_action(MainHealthNextAction.NO_FIX_OWNER) is MainHealthNextAction.NO_FIX_OWNER

    def test_a_failed_submit_leaves_no_owner(self) -> None:
        owner = _ensure(FakeTaskStore(submit_fails=True), TEST_BREAK)

        assert owner.task_id is None

    def test_no_task_store_wired_means_no_owner(self) -> None:
        owner = asyncio.run(ensure_main_fix_owner(None, TEST_BREAK, project_root='/proj'))

        assert owner.task_id is None
        assert owner.escalation_detail('why') == 'why'

    def test_the_escalation_detail_names_the_owner(self) -> None:
        store = FakeTaskStore()

        owner = _ensure(store, TEST_BREAK)

        detail = owner.escalation_detail('main is red')
        assert detail.startswith('main is red')
        assert f'task {owner.task_id}' in detail
        assert TEST_BREAK.signature in detail


# ---------------------------------------------------------------------------
# The three classification sites, driven with one shared task store
# ---------------------------------------------------------------------------


def _make_worker(
    tmp_path: Path, queue: EscalationQueue, store: FakeTaskStore | None,
    monkeypatch: pytest.MonkeyPatch,
):
    worker = SpeculativeMergeWorker(
        _make_git_ops(tmp_path), asyncio.Queue(),
        escalation_queue=queue, mcp=MagicMock(), scheduler=store,
    )
    monkeypatch.setattr(worker, '_post_submit_tasks', (store or FakeTaskStore()).post_submit_tasks)
    return worker


def _make_workflow(
    tmp_path: Path, worker: SpeculativeMergeWorker, queue: EscalationQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> TaskWorkflow:
    config = _make_config(tmp_path)
    worktree = tmp_path / 'task-wt'
    worktree.mkdir(exist_ok=True)
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    workflow = TaskWorkflow(
        assignment=TaskAssignment(
            task_id='42',
            task={
                'id': '42', 'title': 'X', 'description': '', 'status': 'pending',
                'metadata': {'files': ['lib']}, 'dependencies': [],
            },
            modules=['lib'],
        ),
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.merge_worker = worker
    workflow.escalation_queue = queue
    monkeypatch.setattr(workflow, '_post_submit_tasks', worker._post_submit_tasks)
    monkeypatch.setattr(workflow, '_mark_blocked', AsyncMock(return_value=WorkflowOutcome.BLOCKED))
    return workflow


FOREIGN_BREAK = MainBreak('compile_error', 'error TS1005: other.tsx', 'another break')


def _arrange(case: str, worker: SpeculativeMergeWorker, store: FakeTaskStore | None, brk: MainBreak) -> None:
    if case == 'attempt_cap':
        for _ in range(MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS):
            worker.auto_heal_registry.record_attempt(brk.signature)
    if case == 'foreign_halt_rebreak':
        worker.auto_heal_registry.record_attempt(brk.signature)
        if store is not None:
            store.add(
                status='done',
                metadata={SIGNATURE_KEY: brk.signature, 'spawn_context': 'main_health_auto_heal'},
            )
    in_flight = {
        'in_flight': brk, 'foreign_halt': FOREIGN_BREAK, 'foreign_halt_rebreak': FOREIGN_BREAK,
    }.get(case)
    if in_flight is not None:
        worker.auto_heal_registry.record_attempt(in_flight.signature)
        worker.halt_lane('normal', 'auto-heal already in flight')
        if store is not None:
            store.add(
                status='in-progress',
                metadata={SIGNATURE_KEY: in_flight.signature, 'spawn_context': 'main_health_auto_heal'},
            )


MERGE_LANE_CASES = {
    'non_eligible': TEST_FAILURE_RESULT,
    'attempt_cap': COMPILE_ERROR_RESULT,
    'happy': COMPILE_ERROR_RESULT,
    'in_flight': COMPILE_ERROR_RESULT,
    'foreign_halt': COMPILE_ERROR_RESULT,
    'foreign_halt_rebreak': COMPILE_ERROR_RESULT,
}

VERIFY_ARM_CASES = {
    'inherited_test_failure': TEST_FAILURE_RESULT,
    'inherited_compile_error': COMPILE_ERROR_RESULT,
}


def _inherited_info(verify: VerifyResult) -> dict:
    brk = MainBreak(verify.category or '', verify.cause_hint, 'Verify failure is preexisting on main')
    return {
        'reason': brk.reason, 'detail': verify.failure_report(),
        'category': 'preexisting_main_break', 'fingerprint': 'fp-verify', 'main_break': brk,
    }


def _run_site(
    site: str, case: str, tmp_path: Path, store: FakeTaskStore | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MainBreak, list[str]]:
    """Drive one classification site; return the break and every next action it filed."""
    queue = EscalationQueue(tmp_path / 'escalations')
    worker = _make_worker(tmp_path, queue, store, monkeypatch)
    verify = {**MERGE_LANE_CASES, **VERIFY_ARM_CASES}[case]
    outcome = _build_main_health_outcome(verify, MAIN_SHA)
    brk = _break_of(outcome)
    _arrange(case, worker, store, brk)

    async def _drive() -> list[str]:
        if site == 'deferred_worker':
            await worker._auto_heal_main_health_deferred(
                outcome, _make_req('42', tmp_path / 'task-wt', _make_config(tmp_path)),
            )
            await _settle()
            return []
        workflow = _make_workflow(tmp_path, worker, queue, monkeypatch)
        if site == 'sync_workflow':
            await workflow._auto_heal_main_health(outcome, merge_phase=True)
            await _settle()
            return []
        await workflow._block_on_inherited_break(_inherited_info(verify))
        mark_blocked = workflow._mark_blocked
        assert isinstance(mark_blocked, AsyncMock)
        return [mark_blocked.call_args.kwargs['suggested_action']]

    filed_by_mark_blocked = asyncio.run(_drive())
    actions = [esc.suggested_action for esc in queue.get_pending()] + filed_by_mark_blocked
    return brk, actions


SITE_CASES = [
    *((site, case) for site in ('deferred_worker', 'sync_workflow') for case in MERGE_LANE_CASES),
    *(('task_verify', case) for case in VERIFY_ARM_CASES),
]


class TestNoOutcomeTerminatesOwnerless:
    """Every (site, outcome) ends with an open fix-main task for its signature, and
    every escalation it files names an owned next action."""

    @pytest.mark.parametrize(('site', 'case'), SITE_CASES)
    def test_every_outcome_has_an_open_fix_task_and_an_owned_escalation(
        self, tmp_path: Path, site: str, case: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = FakeTaskStore()

        brk, actions = _run_site(site, case, tmp_path, store, monkeypatch)

        assert len(store.open_owners(brk.signature)) == 1, store.tasks
        assert actions, f'{site}/{case} filed no escalation'
        assert set(actions) <= OWNED_ACTIONS, actions

    @pytest.mark.parametrize(('site', 'case'), SITE_CASES)
    def test_without_a_task_store_every_outcome_still_files_an_escalation(
        self, tmp_path: Path, site: str, case: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        brk, actions = _run_site(site, case, tmp_path, None, monkeypatch)

        assert actions, f'{site}/{case} ended with neither a task nor an escalation'
        assert all(action in set(MainHealthNextAction) for action in actions), actions


class TestOneTaskPerSignature:
    def test_every_site_meeting_one_break_files_one_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = FakeTaskStore()
        queue = EscalationQueue(tmp_path / 'escalations')
        worker = _make_worker(tmp_path, queue, store, monkeypatch)
        workflow = _make_workflow(tmp_path, worker, queue, monkeypatch)
        req = _make_req('42', tmp_path / 'task-wt', _make_config(tmp_path))

        async def _three_sites() -> None:
            first = _build_main_health_outcome(TEST_FAILURE_RESULT, MAIN_SHA)
            await worker._auto_heal_main_health_deferred(first, req)
            await workflow._auto_heal_main_health(first, merge_phase=True)
            await workflow._block_on_inherited_break({
                'reason': 'inherited', 'detail': '', 'category': 'preexisting_main_break',
                'fingerprint': 'fp', 'main_break': MainBreak(
                    TEST_FAILURE_RESULT.category or '', TEST_FAILURE_RESULT.cause_hint, 'inherited',
                ),
            })

        asyncio.run(_three_sites())

        assert len(store.submits) == 1

    @pytest.mark.parametrize('heal_site', ['deferred_worker', 'sync_workflow'])
    def test_a_verify_arm_owner_is_adopted_by_the_auto_heal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, heal_site: str,
    ) -> None:
        store = FakeTaskStore()
        queue = EscalationQueue(tmp_path / 'escalations')
        worker = _make_worker(tmp_path, queue, store, monkeypatch)
        workflow = _make_workflow(tmp_path, worker, queue, monkeypatch)
        outcome = _build_main_health_outcome(COMPILE_ERROR_RESULT, MAIN_SHA)

        async def _verify_arm_then_heal() -> None:
            await workflow._block_on_inherited_break(_inherited_info(COMPILE_ERROR_RESULT))
            if heal_site == 'deferred_worker':
                await worker._auto_heal_main_health_deferred(
                    outcome, _make_req('42', tmp_path / 'task-wt', _make_config(tmp_path)),
                )
            else:
                await workflow._auto_heal_main_health(outcome, merge_phase=True)
            await _settle()

        asyncio.run(_verify_arm_then_heal())

        assert len(store.open_owners(_break_of(outcome).signature)) == 1, store.tasks
        assert len(store.submits) == 1
        assert worker.is_lane_halted('normal')

    def test_a_commit_to_a_red_main_does_not_file_a_second_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = FakeTaskStore()
        queue = EscalationQueue(tmp_path / 'escalations')
        worker = _make_worker(tmp_path, queue, store, monkeypatch)
        req = _make_req('42', tmp_path / 'task-wt', _make_config(tmp_path))
        before = _build_main_health_outcome(TEST_FAILURE_RESULT, MAIN_SHA)
        after = _build_main_health_outcome(TEST_FAILURE_RESULT, OTHER_MAIN_SHA)
        assert before.dedupe_fingerprint != after.dedupe_fingerprint

        async def _two_probes() -> None:
            await worker._auto_heal_main_health_deferred(before, req)
            await worker._auto_heal_main_health_deferred(after, req)

        asyncio.run(_two_probes())

        assert len(store.submits) == 1
        assert {esc.suggested_action for esc in queue.get_pending()} == {
            MainHealthNextAction.FIX_TASK_FILED,
        }
