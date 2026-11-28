"""C5 of plans/live-task-facts-at-recall-prd.md for the orchestrator's memory writers.

Each test drives a real ``TaskWorkflow._write_*_to_memory`` method over its
real MCP POST into ``orchestrator.evals.runner.RecordingMemorySink`` and reads
back the arguments that would have been written. Fixture titles, descriptions
and analyses use no task-status word, so any hit comes from a template.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from shared.testing_code_writer_contract import task_status_words_in

from orchestrator.completion_record import completion_record_metadata
from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.runner import (
    RecordingMemorySink,
    build_eval_orch_config,
    eval_mcp_lifecycle,
)
from orchestrator.plan_decisions import supersede_decision
from orchestrator.workflow import TaskWorkflow

_TASK_ID = '4102'
_PLAN_STEPS = [{'id': 'step-1', 'status': 'done'}, {'id': 'step-2', 'status': 'pending'}]
_DECISIONS = [
    {'decision': 'Reuse the task 4103 helper for the sweep', 'rationale': 'It already pages the store'},
    {'decision': 'Key the sweep on metadata', 'rationale': 'Text keys drift'},
]


def _workflow(sink: RecordingMemorySink, tmp_path, *, task_id: str = _TASK_ID) -> TaskWorkflow:
    """A bare TaskWorkflow wired only with the attrs the ``_write_*_to_memory`` methods read."""
    eval_config = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')
    orch_config = build_eval_orch_config(
        eval_config,
        {'id': task_id, 'project_root': str(tmp_path)},
        OrchestratorConfig(project_root=tmp_path),
        memory_endpoint=sink.url,
    )
    wf = object.__new__(TaskWorkflow)
    wf.mcp = eval_mcp_lifecycle(orch_config)
    wf.config = orch_config
    wf.task = {'id': task_id, 'title': 'Widen the audit', 'description': 'adds a sweep'}
    wf.plan = {
        'analysis': 'the sweep reads metadata',
        'design_decisions': _DECISIONS,
        'steps': _PLAN_STEPS,
    }
    wf.modules = ['orchestrator/src/orchestrator']
    wf.task_id = task_id
    return wf


def _add_memory_arguments(sink: RecordingMemorySink) -> list[dict]:
    assert {tool for tool, _ in sink.writes} == {'add_memory'}, sink.writes
    return [arguments for _, arguments in sink.writes]


class TestCompletionRecord:
    @pytest.mark.asyncio
    async def test_it_carries_the_registered_marker_and_names_no_status(self, tmp_path):
        assert task_status_words_in(' '.join(step['status'] for step in _PLAN_STEPS)), (
            'premise: the plan steps carry statuses the template could leak'
        )
        with RecordingMemorySink() as sink:
            await _workflow(sink, tmp_path)._write_completion_to_memory()

        [arguments] = _add_memory_arguments(sink)
        assert arguments['metadata'] == completion_record_metadata(_TASK_ID)
        assert task_status_words_in(arguments['content']) == (), arguments['content']


class TestDecisionNotes:
    @pytest.mark.asyncio
    async def test_each_note_declares_its_task_and_names_it(self, tmp_path):
        with RecordingMemorySink() as sink:
            await _workflow(sink, tmp_path)._write_decisions_to_memory()

        written = _add_memory_arguments(sink)
        assert len(written) == len(_DECISIONS)
        for arguments, decision in zip(written, _DECISIONS, strict=True):
            assert arguments['category'] == 'decisions_and_rationale'
            assert arguments['entities'] == [{'kind': 'task', 'id': _TASK_ID}]
            assert f'task {_TASK_ID}' in arguments['content']
            assert decision['decision'] in arguments['content']
            assert decision['rationale'] in arguments['content']

    @pytest.mark.asyncio
    async def test_a_non_numeric_task_id_declares_nothing(self, tmp_path):
        """An eval task id such as 't' is no task number, and a declaration of it would fail the gate."""
        with RecordingMemorySink() as sink:
            await _workflow(sink, tmp_path, task_id='t')._write_decisions_to_memory()

        written = _add_memory_arguments(sink)
        assert len(written) == len(_DECISIONS)
        assert all('entities' not in arguments for arguments in written), written

    @pytest.mark.asyncio
    async def test_a_superseded_rationale_never_reaches_the_corpus(self, tmp_path):
        superseded_rationale = _DECISIONS[0]['rationale']
        decisions = supersede_decision(
            _DECISIONS, 0,
            decision=_DECISIONS[0]['decision'],
            rationale='Its paging was measured to be complete',
            reason='the store-paging claim was never checked',
        )
        with RecordingMemorySink() as sink:
            wf = _workflow(sink, tmp_path)
            wf.plan = {**wf.plan, 'design_decisions': decisions}
            await wf._write_decisions_to_memory()

        written = _add_memory_arguments(sink)
        assert 'Its paging was measured to be complete' in written[0]['content']
        assert all(superseded_rationale not in arguments['content'] for arguments in written), written


class TestSuggestionNotes:
    @pytest.mark.asyncio
    async def test_they_forward_the_reviewers_prose_and_declare_no_task(self, tmp_path):
        reviews = SimpleNamespace(suggestions=[{'category': 'perf', 'description': 'batch the reads'}])
        with RecordingMemorySink() as sink:
            await _workflow(sink, tmp_path)._write_suggestions_to_memory(reviews)

        [arguments] = _add_memory_arguments(sink)
        assert arguments['category'] == 'preferences_and_norms'
        assert 'batch the reads' in arguments['content']
        assert 'entities' not in arguments
