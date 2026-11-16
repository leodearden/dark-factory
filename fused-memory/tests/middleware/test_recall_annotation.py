"""``bind_tasks`` and ``render_task_facts`` (plans/live-task-facts-at-recall-prd.md C2, C4) over result fixtures."""

from __future__ import annotations

import pytest
from fused_memory.middleware.recall_annotation import (
    Binding,
    BoundTask,
    TaskBindings,
    bind_tasks,
)
from orchestrator.completion_record import completion_record_metadata

from fused_memory.memory_metadata import (
    ORCHESTRATOR_COMPLETION_SOURCE,
    TARGETED_RECONCILIATION_SOURCE,
)
from fused_memory.middleware.task_facts import TaskRef
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult

PROJECT = 'dark_factory'
KNOWN = {'dark_factory': '/src/df', 'reify': '/src/reify'}
RECONCILIATION_ECHO = {
    'source': TARGETED_RECONCILIATION_SOURCE, 'task_id': '3651', 'transition': 'done',
}


def _note(id: str, content: str, metadata: dict | None = None) -> MemoryResult:
    return MemoryResult(
        id=id, content=content, source_store=SourceStore.mem0, metadata=metadata or {},
    )


def _edge(id: str, fact: str, entities: list[str]) -> MemoryResult:
    return MemoryResult(
        id=id, content=fact, source_store=SourceStore.graphiti, entities=entities,
    )


def _bind(results: list[MemoryResult], *, project_id: str = PROJECT) -> TaskBindings:
    return bind_tasks(results, project_id=project_id, known_projects=KNOWN)


def _local(task_id: str) -> TaskRef:
    return TaskRef(PROJECT, task_id)


def _refs(bindings: TaskBindings) -> list[TaskRef]:
    return [bound.ref for bound in bindings.tasks]


class TestPrdFixtures:
    def test_b1_a_text_mention_is_mentioned(self):
        bindings = _bind([_note('n1', 'Task 3651 remains pending')])
        assert bindings.tasks == (BoundTask(_local('3651'), Binding.MENTIONED, ('n1',)),)

    def test_b2_a_graph_endpoint_is_structural_though_the_fact_never_names_it(self):
        bindings = _bind([_edge('e1', 'Leo reviewed the fix', ['Task 3651', 'Leo'])])
        assert bindings.tasks == (BoundTask(_local('3651'), Binding.STRUCTURAL, ('e1',)),)

    @pytest.mark.parametrize(
        'metadata',
        [completion_record_metadata('3651'), RECONCILIATION_ECHO],
        ids=['orchestrator-completion-record', 'reconciliation-echo'],
    )
    def test_b3_a_completion_record_is_structural(self, metadata):
        bindings = _bind([_note('n1', 'Merged the fix.', metadata)])
        assert bindings.tasks == (BoundTask(_local('3651'), Binding.STRUCTURAL, ('n1',)),)

    def test_b3_a_bare_task_id_is_authorship_not_a_link(self):
        assert _bind([_note('n1', 'Merged the fix.', {'task_id': '3651'})]).tasks == ()

    def test_b4_a_registered_project_name_qualifies_the_mention(self):
        bindings = _bind([_note('n1', 'Reify task 3674 was cancelled')])
        assert bindings.tasks == (BoundTask(TaskRef('reify', '3674'), Binding.MENTIONED, ('n1',)),)
        assert _local('3674') not in _refs(bindings)

    def test_b5_an_ambiguous_number_is_never_bound(self):
        bindings = _bind([_note('n1', 'reify:3674 blocks task 3674')])
        assert bindings.tasks == ()
        assert bindings.ambiguous == ('3674',)


class TestStructuralLinks:
    def test_a_looked_up_node_presented_as_a_graph_result_is_structural(self):
        bindings = _bind([_edge('node', '', ['Task 77'])])
        assert bindings.tasks == (BoundTask(_local('77'), Binding.STRUCTURAL, ('node',)),)

    def test_a_foreign_endpoint_binds_for_the_reader_to_answer(self):
        bindings = _bind([_edge('e1', 'mirrors a task', ['nowhere:5'])])
        assert bindings.tasks == (BoundTask(TaskRef('nowhere', '5'), Binding.STRUCTURAL, ('e1',)),)

    def test_a_self_qualified_endpoint_is_local(self):
        bindings = _bind([_edge('e1', 'a fact', ['dark_factory:12'])])
        assert _refs(bindings) == [_local('12')]

    def test_structural_wins_and_keeps_every_result_in_rank_order(self):
        bindings = _bind([
            _note('n1', 'Task 3651 is pending'),
            _edge('e2', 'Leo reviewed the fix', ['Task 3651']),
        ])
        assert bindings.tasks == (
            BoundTask(_local('3651'), Binding.STRUCTURAL, ('n1', 'e2')),
        )

    def test_an_int_completion_task_id_is_read_as_its_digits(self):
        metadata = {'source': ORCHESTRATOR_COMPLETION_SOURCE, 'task_id': 3651}
        assert _refs(_bind([_note('n1', 'Merged.', metadata)])) == [_local('3651')]

    @pytest.mark.parametrize(
        'task_id', ['', None, True, 3.5, ['3651']], ids=['empty', 'none', 'bool', 'float', 'list'],
    )
    def test_an_unusable_completion_task_id_binds_nothing(self, task_id):
        metadata = {**completion_record_metadata('3651'), 'task_id': task_id}
        assert _bind([_note('n1', 'Merged.', metadata)]).tasks == ()


class TestOrdering:
    def test_structural_first_then_first_rank_then_text_order(self):
        bindings = _bind([
            _note('n1', 'task 11 landed before task 10'),
            _edge('e2', 'a fact', ['Task 21']),
            _note('n3', 'task 12 waits'),
            _edge('e4', 'another fact', ['Task 20']),
        ])
        assert _refs(bindings) == [
            _local('21'), _local('20'), _local('11'), _local('10'), _local('12'),
        ]
        assert [bound.binding for bound in bindings.tasks] == [
            Binding.STRUCTURAL, Binding.STRUCTURAL,
            Binding.MENTIONED, Binding.MENTIONED, Binding.MENTIONED,
        ]

    def test_the_project_is_canonicalized(self):
        bindings = _bind([_note('n1', 'task 5 is open')], project_id='Dark-Factory')
        assert _refs(bindings) == [TaskRef('dark_factory', '5')]


class TestPurity:
    def test_the_results_are_not_mutated_and_the_answer_is_repeatable(self):
        results = [
            _note('n1', 'Task 3651 remains pending', {'task_id': '9'}),
            _edge('e2', 'Leo reviewed the fix', ['Task 3651', 'Leo']),
        ]
        before = [result.model_copy(deep=True) for result in results]
        first = _bind(results)
        assert results == before
        assert _bind(results) == first
