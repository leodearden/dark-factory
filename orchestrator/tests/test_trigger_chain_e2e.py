"""The completion-driven trigger chain, end to end (plans/completion-driven-triggers-prd.md §4 boundary rows 1-16).

Every hop is production code: the snippet's own submit shapes, the fused-memory
submit guards behind a live MCP server over a real SqliteTaskBackend, a real
Scheduler time gate, a real DeterministicRunner executing the real factory
script, and a real EscalationQueue.  Only the host processes are stood in for.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

pytest.importorskip('fused_memory')

from _git_fixtures import seed_repo
from _trigger_chain_doc_shapes import (
    DocShapeError,
    authoring_metadata_exemplars,
    concretize,
    fenced_blocks,
    snippet_placeholders,
    snippet_submit_calls,
    submit_calls,
)
from _trigger_chain_rig import PROJECT_ID, Chain, TriggerChainRig, open_rig, review_due
from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES
from escalation.models import Escalation
from fused_memory.middleware.deterministic_task_guard import (
    deterministic_task_error,
    inject_task_kind,
)
from shared.repo_paths import DARK_FACTORY_ROOT_ENV
from shared.task_metadata import TASK_METADATA_CAPABILITIES, parse_metadata

import orchestrator.mcp_lifecycle
from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.event_store import EventType
from orchestrator.workflow import WorkflowOutcome

pytestmark = pytest.mark.usefixtures('code_default_config')

RUN = 'review-lambda-20261112-1'
# An open finding is held by another lane, so this rig's scheduler never dispatches it.
OPEN = 'in-progress'


@pytest.fixture
def factory_root(repo_root: Path | None, monkeypatch: pytest.MonkeyPatch) -> Path:
    if repo_root is None:
        pytest.skip('not running inside a dark-factory checkout')
    monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(repo_root))
    return repo_root


@pytest_asyncio.fixture
async def rig(
    tmp_path: Path, factory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[TriggerChainRig]:
    # A test that ran McpLifecycle.start() under a patched McpSession leaves a mock here,
    # and mcp_call would route the Scheduler's calls to it instead of the live server.
    monkeypatch.setattr(orchestrator.mcp_lifecycle, '_session', None)
    async with open_rig(tmp_path, factory_root) as opened:
        monkeypatch.setenv('FUSED_MEMORY_MCP_URL', opened.url)
        monkeypatch.setenv(
            'PATH', f'{Path(sys.executable).parent}{os.pathsep}{os.environ.get("PATH", "")}'
        )
        yield opened


def _doc_placeholders(project_root: Path) -> Mapping[str, str]:
    return snippet_placeholders(
        skill='review', run_id=RUN, project_root=project_root, project_id=PROJECT_ID, now=datetime.now(UTC)
    )


def _assert_accepted(task_kind: str, metadata: dict, project_root: Path) -> None:
    assert deterministic_task_error(task_kind, metadata, str(project_root)) is None
    _, warnings = parse_metadata(
        inject_task_kind(metadata, task_kind), direction='write', enforce=True
    )
    assert warnings == []


class TestDocShapes:
    def test_every_snippet_submit_call_passes_the_guard_and_the_model(
        self, factory_root: Path, tmp_path: Path
    ) -> None:
        project_root = seed_repo(tmp_path / 'project')
        placeholders = _doc_placeholders(project_root)
        calls = [concretize(call, placeholders) for call in snippet_submit_calls(factory_root)]

        assert {call['metadata']['trigger_chain']['role'] for call in calls} == {
            'completion_gate',
            'human_gate',
        }
        for call in calls:
            _assert_accepted(call['task_kind'], call['metadata'], project_root)

    def test_every_authoring_section_6_json_exemplar_passes(
        self, factory_root: Path, tmp_path: Path
    ) -> None:
        project_root = seed_repo(tmp_path / 'project')
        exemplars = authoring_metadata_exemplars(factory_root)

        assert exemplars
        for exemplar in exemplars:
            concrete = concretize(exemplar, _doc_placeholders(project_root))
            _assert_accepted(concrete['task_kind'], concrete, project_root)

    def test_a_block_that_cannot_be_extracted_fails_loudly(self) -> None:
        non_literal = "```python\nsubmit_task(project_root='x', metadata=some_name)\n```\n"
        with pytest.raises(DocShapeError, match=r'submit_task.*metadata'):
            submit_calls(non_literal)

        no_call = "```python\nadd_dependency(id='1', depends_on='2')\n```\n"
        with pytest.raises(DocShapeError, match='submit_task'):
            submit_calls(no_call)

        with pytest.raises(DocShapeError, match='<mystery>'):
            concretize({'title': 'about <mystery>'}, {'skill': 'review'})

    @pytest.mark.parametrize(('tag', 'lang'), [('py', 'python'), ('python3', 'python'), ('jsonc', 'json')])
    def test_a_fence_retagged_with_another_spelling_fails_loudly(self, tag: str, lang: str) -> None:
        with pytest.raises(DocShapeError, match=f'```{tag} fence; tag it ```{lang}'):
            fenced_blocks(f'```{tag}\n{{}}\n```\n', lang)

    def test_a_fence_is_read_by_the_first_word_of_its_info_string(self) -> None:
        text = '```JSON title=exemplar\n{"a": 1}\n```\n```js\nlet a\n```\n```\nprose\n```\n'

        assert fenced_blocks(text, 'json') == ['{"a": 1}']


async def _titled(rig: TriggerChainRig, title: str) -> list[dict]:
    listing = await rig.call('get_tasks', project_root=str(rig.root))
    return [task for task in listing['tasks'] if task.get('title') == title]


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestSubmitBoundary:
    async def test_an_unknown_before_done_key_is_refused_and_nothing_persists(
        self, rig: TriggerChainRig
    ) -> None:
        call = rig.snippet_call('completion_gate', RUN)
        call['metadata']['before_done']['recheck_mins'] = 86400

        reply = await rig.call('submit_task', **call)

        assert reply['error_type'] == 'UnsupportedTaskMetadata'
        assert reply['unsupported_keys'] == ['recheck_mins']
        assert 'task_id' not in reply
        assert await _titled(rig, call['title']) == []

    async def test_get_status_reports_the_running_capabilities(
        self, rig: TriggerChainRig
    ) -> None:
        status = await rig.call('get_status')

        assert status['task_metadata_capabilities'] == sorted(TASK_METADATA_CAPABILITIES)

    async def test_a_slashed_skill_is_refused_naming_the_bare_form(
        self, rig: TriggerChainRig
    ) -> None:
        call = rig.snippet_call('human_gate', RUN)
        call['metadata']['trigger_chain']['skill'] = '/review'

        reply = await rig.call('submit_task', **call)

        assert reply['error_type'] == 'ValidationError'
        assert "trigger_chain.skill='/review' must be the bare skill name 'review'" in reply['error']
        assert await _titled(rig, call['title']) == []

    async def test_a_factory_rooted_gate_is_accepted_on_a_root_without_scripts(
        self, rig: TriggerChainRig
    ) -> None:
        assert not (rig.root / 'scripts').exists()
        call = rig.snippet_call('completion_gate', RUN)
        project_rooted = rig.snippet_call('completion_gate', RUN)
        del project_rooted['metadata']['before_done']['script_root']

        refused = await rig.call('submit_task', **project_rooted)
        accepted = await rig.call('submit_task', **call)

        assert refused['error_type'] == 'ValidationError'
        assert 'scripts/check_run_completion.py' in refused['error']
        assert 'does not exist under project_root' in refused['error']
        assert 'task_id' not in refused
        assert accepted.get('task_id')


def _metadata(task: dict) -> dict:
    return task['metadata']


def _at(task: dict, key: str = 'at') -> datetime:
    return datetime.fromisoformat(_metadata(task)['milestone'][key])


def _rearm_events(rig: TriggerChainRig, task_id: str) -> list[dict]:
    return [e for e in rig.events(EventType.predicate_rearmed) if e['task_id'] == task_id]


@dataclass(frozen=True)
class _FirstCheck:
    chain: Chain
    findings: tuple[str, ...]
    first_at: datetime


async def _first_check_not_yet(
    rig: TriggerChainRig, runner: DeterministicRunner | None = None
) -> _FirstCheck:
    """Run R = (high open, medium done, low done) behind a final-form chain, checked once at its `at`."""
    findings = await rig.seed_run(RUN, ('high', OPEN), ('medium', 'done'), ('low', 'done'))
    chain = await rig.file_final_chain(RUN)
    first_at = _at(await rig.task(chain.completion_id))

    assert await rig.dispatch() == (None, None)
    rig.clock.set(first_at)
    assignment, outcome = await rig.dispatch(runner=runner)

    assert assignment is not None and assignment.task_id == chain.completion_id
    assert outcome == WorkflowOutcome.REQUEUED
    return _FirstCheck(chain, findings, first_at)


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestCompletionGate:
    async def test_not_yet_re_arms_without_escalating(self, rig: TriggerChainRig) -> None:
        first = await _first_check_not_yet(rig)

        assert first.chain.completion_id not in rig.scheduler.get_state_snapshot().requeue_cooldowns
        assert await rig.dispatch() == (None, None)
        gate = await rig.task(first.chain.completion_id)
        meta = _metadata(gate)
        assert gate['status'] == 'pending'
        assert _at(gate) == first.first_at + timedelta(seconds=86400)
        assert _at(gate, 'scheduled_fire_at') == first.first_at
        assert meta['predicate_recheck']['consecutive'] == 1
        assert datetime.fromisoformat(meta['predicate_recheck']['first_rearmed_at']) == first.first_at
        assert meta['predicate_verdict']['verdict'] == 'not_yet'
        assert meta['predicate_verdict']['run'] == RUN
        (event,) = _rearm_events(rig, first.chain.completion_id)
        assert (event['data']['rc'], event['data']['consecutive'], event['data']['recheck_secs']) == (
            75,
            1,
            86400,
        )
        assert rig.own_escalations(first.chain.completion_id) == []
        assert rig.own_escalations(first.chain.human_id) == []
        assert (await rig.task(first.chain.human_id))['status'] == 'pending'

    async def test_seven_tenths_at_point_seven_passes(self, rig: TriggerChainRig) -> None:
        await rig.seed_run(RUN, ('critical', 'done'), ('high', 'done'), ('high', OPEN))
        chain = await rig.file_final_chain(RUN)
        rig.clock.set(_at(await rig.task(chain.completion_id)))

        assignment, outcome = await rig.dispatch()

        assert assignment is not None and assignment.task_id == chain.completion_id
        assert outcome == WorkflowOutcome.DONE
        gate = await rig.task(chain.completion_id)
        verdict = _metadata(gate)['predicate_verdict']
        assert gate['status'] == 'done'
        assert _metadata(gate)['done_provenance']['kind'] == 'deterministic-milestone'
        assert '"verdict":"pass"' in _metadata(gate)['done_provenance']['note']
        assert verdict['share'] == 0.7
        assert (verdict['landed'], verdict['pending']) == (7, 3)

    async def test_the_scripts_own_transport_pages_the_live_lookup(
        self, rig: TriggerChainRig
    ) -> None:
        seed = [('high', OPEN)] * 50 + [('low', 'done')] * 51
        await rig.seed_run(RUN, *seed)
        chain = await rig.file_final_chain(RUN)
        rig.clock.set(_at(await rig.task(chain.completion_id)))

        _, outcome = await rig.dispatch()

        assert outcome == WorkflowOutcome.REQUEUED
        verdict = _metadata(await rig.task(chain.completion_id))['predicate_verdict']
        assert verdict['total'] == 101
        assert verdict['by_priority'] == {'high': [0, 50, 0], 'low': [51, 0, 0]}

    async def test_an_undeclared_seventy_five_is_a_failed_check(
        self, rig: TriggerChainRig
    ) -> None:
        await rig.seed_run(RUN, ('high', OPEN))
        call = rig.snippet_call('completion_gate', RUN)
        del call['metadata']['trigger_chain']
        del call['metadata']['before_done']['recheck_secs']
        gate_id = await rig.submit(call)
        await rig.call('commit_planning', project_root=str(rig.root), task_ids=gate_id)
        rig.clock.set(_at(await rig.task(gate_id)))

        _, outcome = await rig.dispatch()

        assert outcome == WorkflowOutcome.BLOCKED
        (escalation,) = rig.own_escalations(gate_id, status='pending')
        assert (escalation.category, escalation.level) == ('milestone_check_failed', 2)
        assert 'before_done.recheck_secs is not declared' in escalation.detail
        assert (await rig.task(gate_id))['status'] == 'blocked'
        assert _rearm_events(rig, gate_id) == []


@dataclass(frozen=True)
class _Fired:
    first: _FirstCheck
    escalation: Escalation


async def _human_gate_fired(rig: TriggerChainRig) -> _Fired:
    """Row 1's re-armed chain; its high finding lands, the gate passes, the human gate files review_due."""
    first = await _first_check_not_yet(rig)
    await rig.set_finding_status(first.findings[0], 'done')
    rig.clock.set(_at(await rig.task(first.chain.completion_id)))

    gate_assignment, gate_outcome = await rig.dispatch()
    human_assignment, human_outcome = await rig.dispatch()

    assert gate_assignment is not None and gate_assignment.task_id == first.chain.completion_id
    assert gate_outcome == WorkflowOutcome.DONE
    assert human_assignment is not None and human_assignment.task_id == first.chain.human_id
    assert human_outcome == WorkflowOutcome.BLOCKED
    (escalation,) = rig.own_escalations(first.chain.human_id, status='pending')
    return _Fired(first, escalation)


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestHumanGateFires:
    async def test_a_passed_gate_fires_review_due(self, rig: TriggerChainRig) -> None:
        fired = await _human_gate_fired(rig)
        chain = fired.first.chain

        gate = await rig.task(chain.completion_id)
        assert gate['status'] == 'done'
        assert _metadata(gate)['done_provenance']['kind'] == 'deterministic-milestone'
        assert '"verdict":"pass"' in _metadata(gate)['done_provenance']['note']
        escalation = fired.escalation
        assert (escalation.category, escalation.level) == ('review_due', 2)
        due = review_due(escalation)
        assert due['run_id'] == RUN
        assert due['skill'] == 'review'
        verdict = due['verdict']
        assert verdict is not None
        share = verdict['share']
        assert isinstance(share, float) and share >= 0.7
        assert due['completion_gate_task_id'] == chain.completion_id
        assert due['human_gate_task_id'] == chain.human_id
        assert due['launched_from'] == escalation.id
        assert due['project_root'] == str(rig.root)
        assert (await rig.task(chain.human_id))['status'] == 'blocked'

    async def test_a_bare_skill_is_stored_bare_and_rendered_with_a_slash(
        self, rig: TriggerChainRig
    ) -> None:
        human_id = await rig.submit(rig.snippet_call('human_gate', RUN))
        await rig.call_ok('commit_planning', project_root=str(rig.root), task_ids=human_id)

        _, outcome = await rig.dispatch()

        assert outcome == WorkflowOutcome.BLOCKED
        (escalation,) = rig.own_escalations(human_id, status='pending')
        assert review_due(escalation)['skill'] == 'review'
        assert escalation.summary == f'Run /review on {rig.project_id} ({RUN} landed)'

    async def test_the_interim_gate_fires_259200_seconds_after_its_last_dependency(
        self, rig: TriggerChainRig
    ) -> None:
        first_dep, last_dep, _ = await rig.seed_run(RUN, ('high', OPEN), ('high', OPEN), ('low', OPEN))
        human_id = await rig.file_interim_gate(RUN, deps=(first_dep, last_dep))

        await rig.set_finding_status(first_dep, 'cancelled')
        assert await rig.dispatch() == (None, None)
        assert 'milestone_deps_satisfied_at' not in _metadata(await rig.task(human_id))

        rig.clock.advance(3600)
        last_terminal = rig.clock()
        await rig.set_finding_status(last_dep, 'done')
        assert await rig.dispatch() == (None, None)
        anchor = _metadata(await rig.task(human_id))['milestone_deps_satisfied_at']
        assert datetime.fromisoformat(anchor) == last_terminal

        rig.clock.set(last_terminal + timedelta(seconds=259199))
        assert await rig.dispatch() == (None, None)
        rig.clock.set(last_terminal + timedelta(seconds=259200))
        assignment, outcome = await rig.dispatch()

        assert assignment is not None and assignment.task_id == human_id
        assert outcome == WorkflowOutcome.BLOCKED
        (escalation,) = rig.own_escalations(human_id, status='pending')
        assert escalation.category == 'review_due'
        assert 'trigger_chain' in _metadata(await rig.task(human_id))
        assert review_due(escalation)['completion_gate_task_id'] is None
        assert review_due(escalation)['verdict'] is None


def _recheck(task: dict) -> dict:
    return _metadata(task)['predicate_recheck']


@dataclass(frozen=True)
class _Stalled:
    first: _FirstCheck
    escalation: Escalation


async def _stalled_at_cap_two(rig: TriggerChainRig) -> _Stalled:
    """Two consecutive 75s against a cap-2 runner: the second files the stall."""
    runner = rig.runner(stall_cap=2)
    first = await _first_check_not_yet(rig, runner)
    rig.clock.set(_at(await rig.task(first.chain.completion_id)))

    _, outcome = await rig.dispatch(runner=runner)

    assert outcome == WorkflowOutcome.BLOCKED
    (escalation,) = rig.queue.get_by_task(first.chain.completion_id, status='pending')
    return _Stalled(first, escalation)


async def _resumed_after_stall(rig: TriggerChainRig) -> _Stalled:
    stalled = await _stalled_at_cap_two(rig)
    await rig.resolve(stalled.escalation, 'resume', 'keep waiting')
    assert (await rig.task(stalled.first.chain.completion_id))['status'] == 'pending'

    _, outcome = await rig.dispatch(runner=rig.runner(stall_cap=2))

    assert outcome == WorkflowOutcome.REQUEUED
    return stalled


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestStall:
    async def test_the_cap_files_one_stalled_l2(self, rig: TriggerChainRig) -> None:
        stalled = await _stalled_at_cap_two(rig)
        gate_id = stalled.first.chain.completion_id

        escalation = stalled.escalation
        assert (escalation.category, escalation.level, escalation.severity) == (
            'milestone_check_stalled',
            2,
            'critical',
        )
        assert escalation.agent_role == 'orchestrator-deterministic'
        assert escalation.category in L2_AUTO_CLOSE_DENY_CATEGORIES
        gate = await rig.task(gate_id)
        assert gate['status'] == 'blocked'
        assert _recheck(gate)['consecutive'] == 0
        assert datetime.fromisoformat(_recheck(gate)['first_rearmed_at']) == stalled.first.first_at
        assert _metadata(gate).get('gate_escalated_at')
        (event,) = [
            e for e in rig.events(EventType.predicate_recheck_stalled) if e['task_id'] == gate_id
        ]
        assert (event['data']['consecutive'], event['data']['stall_cap']) == (2, 2)

    async def test_resume_after_a_stall_re_arms_with_a_fresh_streak(
        self, rig: TriggerChainRig
    ) -> None:
        stalled = await _resumed_after_stall(rig)
        gate_id = stalled.first.chain.completion_id

        recheck = _recheck(await rig.task(gate_id))
        assert recheck['consecutive'] == 1
        assert datetime.fromisoformat(recheck['first_rearmed_at']) == stalled.first.first_at
        assert [e.status for e in rig.own_escalations(gate_id)] == ['resolved']

    async def test_a_stale_gate_escalated_at_re_runs_and_re_arms(
        self, rig: TriggerChainRig
    ) -> None:
        stalled = await _resumed_after_stall(rig)
        gate_id = stalled.first.chain.completion_id
        rig.clock.set(_at(await rig.task(gate_id)))

        _, outcome = await rig.dispatch(runner=rig.runner(stall_cap=30))

        assert outcome == WorkflowOutcome.REQUEUED
        gate = await rig.task(gate_id)
        assert _metadata(gate).get('gate_escalated_at')
        assert _recheck(gate)['consecutive'] == 2
        assert datetime.fromisoformat(_recheck(gate)['first_rearmed_at']) == stalled.first.first_at
        assert datetime.fromisoformat(_recheck(gate)['last_checked_at']) == rig.clock()
        assert rig.own_escalations(gate_id, status='pending') == []


RUN_2 = 'review-lambda-20261112-2'


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestSupersede:
    async def test_an_open_chain_refuses_a_rival_until_superseded(
        self, rig: TriggerChainRig
    ) -> None:
        chain = await rig.file_final_chain(RUN)
        original_at = _at(await rig.task(chain.completion_id))
        rival = rig.snippet_call('completion_gate', RUN_2)

        refused = await rig.call('submit_task', **rival)

        assert refused['error_type'] == 'TriggerChainOpen'
        assert {task['id'] for task in refused['open_chain']} == {
            chain.completion_id,
            chain.human_id,
        }
        assert 'task_id' not in refused
        assert await _titled(rig, rival['title']) == []

        await rig.supersede(chain, RUN_2, launched_from=None)

        for task_id in (chain.completion_id, chain.human_id):
            task = await rig.task(task_id)
            assert task['status'] == 'cancelled'
            assert _metadata(task)['trigger_chain']['superseded_by'] == RUN_2
        rig.clock.set(original_at + timedelta(seconds=1))
        assert await rig.dispatch() == (None, None)
        assert rig.queue.get_by_task(chain.human_id) == []
        await rig.file_final_chain(RUN_2)

    async def test_cancelling_the_completion_gate_first_would_fire_the_human_gate(
        self, rig: TriggerChainRig
    ) -> None:
        chain = await rig.file_final_chain(RUN)
        await rig.call_ok(
            'set_task_status', id=chain.completion_id, status='cancelled', project_root=str(rig.root)
        )

        assignment, _ = await rig.dispatch()

        assert assignment is not None and assignment.task_id == chain.human_id

    async def test_the_launched_run_answers_and_supersedes_in_one_step(
        self, rig: TriggerChainRig
    ) -> None:
        fired = await _human_gate_fired(rig)
        chain = fired.first.chain
        launching = fired.escalation
        gate_before = _metadata(await rig.task(chain.completion_id))

        await rig.supersede(chain, RUN_2, launched_from=launching.id)

        (answered,) = rig.own_escalations(chain.human_id)
        assert answered.id == launching.id
        assert (answered.status, answered.resolution, answered.resolution_action) == (
            'resolved',
            f'ran as {RUN_2}',
            'resume',
        )
        human = await rig.task(chain.human_id)
        assert human['status'] == 'done'
        assert _metadata(human)['done_provenance']['kind'] == 'deterministic-gate'
        assert _metadata(human)['done_provenance']['escalation_id'] == launching.id
        gate = await rig.task(chain.completion_id)
        assert gate['status'] == 'done'
        assert _metadata(gate) == {
            **gate_before,
            'trigger_chain': {**gate_before['trigger_chain'], 'superseded_by': RUN_2},
        }
        await rig.file_final_chain(RUN_2)

    async def test_a_stalled_gate_is_abandoned_and_a_parked_human_gate_cancelled(
        self, rig: TriggerChainRig
    ) -> None:
        stalled = await _stalled_at_cap_two(rig)
        chain = stalled.first.chain
        await rig.call_ok(
            'set_task_status',
            id=chain.human_id,
            status='deferred',
            deferral={'kind': 'until_condition', 'condition': 'the operator un-parks this chain'},
            project_root=str(rig.root),
        )

        await rig.supersede(chain, RUN_2, launched_from=None)

        for task_id in (chain.completion_id, chain.human_id):
            assert (await rig.task(task_id))['status'] == 'cancelled'
        (abandoned,) = rig.own_escalations(chain.completion_id)
        assert abandoned.id == stalled.escalation.id
        assert (abandoned.status, abandoned.resolution, abandoned.resolution_action) == (
            'resolved',
            f'superseded by {RUN_2}',
            'abandon',
        )
        assert rig.queue.get_by_task(chain.human_id) == []
        assert await rig.dispatch() == (None, None)
        await rig.file_final_chain(RUN_2)


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestLostStatusWrite:
    async def test_a_lost_pending_write_still_withholds_until_at(
        self, rig: TriggerChainRig
    ) -> None:
        await rig.seed_run(RUN, ('high', OPEN))
        chain = await rig.file_final_chain(RUN)
        first_at = _at(await rig.task(chain.completion_id))
        rig.clock.set(first_at)
        failing = rig.status_write_failing('pending')

        _, outcome = await rig.dispatch(runner=rig.runner(scheduler=failing))

        assert outcome == WorkflowOutcome.REQUEUED
        assert failing.refused == [(chain.completion_id, 'pending')]
        gate = await rig.task(chain.completion_id)
        # A deterministic slot writes no dispatch-time in-progress stamp, so nothing is stranded.
        assert gate['status'] == 'pending'
        assert _at(gate) == first_at + timedelta(seconds=86400)
        assert _recheck(gate)['consecutive'] == 1
        assert rig.own_escalations(chain.completion_id) == []

        rig.clock.set(_at(gate) - timedelta(seconds=1))
        assert await rig.dispatch() == (None, None)
        rig.clock.set(_at(gate))
        assignment, _ = await rig.dispatch()
        assert assignment is not None and assignment.task_id == chain.completion_id


@pytest.mark.asyncio
@pytest.mark.real_mcp_session
class TestPendingReport:
    async def test_a_project_with_no_chain_says_none_was_ever_filed_distinct_from_none_open(
        self, rig: TriggerChainRig
    ) -> None:
        never_filed = f'no trigger chain has been filed in {rig.root}'
        none_open = f'no open trigger chain in {rig.root}'

        before = rig.pending_report()

        assert before.rc == 0, before.stderr
        assert any(never_filed in line for line in before.lines)
        assert not any(none_open in line for line in before.lines)

        chain = await rig.file_final_chain(RUN)
        await rig.set_status(chain.human_id, 'cancelled')
        await rig.set_status(chain.completion_id, 'cancelled')
        after = rig.pending_report()

        assert after.rc == 0, after.stderr
        assert any(none_open in line for line in after.lines)
        assert not any(never_filed in line for line in after.lines)

    async def test_an_open_chain_reports_its_streak_and_live_share(
        self, rig: TriggerChainRig
    ) -> None:
        first = await _first_check_not_yet(rig)
        stamped = _metadata(await rig.task(first.chain.completion_id))['predicate_verdict']

        report = rig.pending_report('--json')

        assert report.rc == 0, report.stderr
        (project,) = json.loads(report.lines[-1])['projects']
        (chain,) = project['chains']
        assert (chain['skill'], chain['run_id']) == ('review', RUN)
        assert chain['completion_gate']['streak'] == 1
        assert chain['verdict']['share'] == stamped['share']
