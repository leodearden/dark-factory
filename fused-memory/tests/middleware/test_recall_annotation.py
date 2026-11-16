"""``bind_tasks`` and ``render_task_facts`` (plans/live-task-facts-at-recall-prd.md C2, C4) over result fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from orchestrator.completion_record import completion_record_metadata

from fused_memory.memory_metadata import (
    ORCHESTRATOR_COMPLETION_SOURCE,
    TARGETED_RECONCILIATION_SOURCE,
)
from fused_memory.middleware.recall_annotation import (
    MAX_RESULTS_SCANNED,
    MAX_TASKS,
    Binding,
    BoundTask,
    FactsUnavailable,
    TaskBindings,
    bind_tasks,
    render_task_facts,
)
from fused_memory.middleware.task_facts import TaskFacts, TaskRef
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.utils.clip_disclosure import fit_at_word

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

    def test_a_number_bound_by_another_result_is_not_also_ambiguous(self):
        bindings = _bind([
            _note('n1', 'reify:3674 blocks task 3674'),
            _note('n2', 'Task 3674 is done'),
        ])
        assert bindings.tasks == (BoundTask(_local('3674'), Binding.MENTIONED, ('n2',)),)
        assert bindings.ambiguous == ()


class TestRegistry:
    @pytest.mark.parametrize(
        'known_projects', [{}, {'/src/df': 'dark_factory'}], ids=['empty', 'path-shaped-keys'],
    )
    def test_without_a_usable_registry_a_text_qualifier_binds_nothing(self, known_projects):
        bindings = bind_tasks(
            [_note('n1', 'Total: 42 and localhost:6379')],
            project_id=PROJECT, known_projects=known_projects,
        )
        assert bindings.tasks == ()

    def test_without_a_registry_the_own_project_still_qualifies(self):
        bindings = bind_tasks(
            [_note('n1', 'see dark_factory: 12')], project_id=PROJECT, known_projects={},
        )
        assert _refs(bindings) == [_local('12')]


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


class TestBounds:
    def test_b7_many_tasks_keep_the_late_structural_hits_and_count_the_rest(self):
        notes = [_note(f'n{i}', f'task {1000 + i} is open') for i in range(50)]
        edges = [_edge(f'e{i}', 'a fact', [f'Task {2000 + i}']) for i in range(10)]
        bindings = _bind([*notes, *edges])
        assert len(bindings.tasks) == MAX_TASKS
        assert bindings.omitted_tasks == 20
        assert _refs(bindings) == [
            *(_local(str(2000 + i)) for i in range(10)),
            *(_local(str(1000 + i)) for i in range(30)),
        ]
        assert [bound.binding for bound in bindings.tasks] == (
            [Binding.STRUCTURAL] * 10 + [Binding.MENTIONED] * 30
        )

    def test_results_beyond_the_scan_cap_are_counted(self):
        bindings = _bind([_note(f'n{i}', 'nothing here') for i in range(MAX_RESULTS_SCANNED + 5)])
        assert bindings.results_not_scanned == 5

    def test_a_task_named_only_beyond_the_scan_cap_is_absent_and_not_omitted(self):
        results = [_note(f'n{i}', 'nothing here') for i in range(MAX_RESULTS_SCANNED)]
        results += [
            _note('late-note', 'task 7 is open'),
            _edge('late-edge', 'a fact', ['Task 8']),
            _note('late-record', 'Merged.', completion_record_metadata('9')),
        ]
        bindings = _bind(results)
        assert bindings.tasks == ()
        assert bindings.omitted_tasks == 0
        assert bindings.results_not_scanned == 3

    @pytest.mark.parametrize('count', [0, 1, MAX_RESULTS_SCANNED])
    def test_at_or_under_the_scan_cap_nothing_is_unscanned(self, count):
        bindings = _bind([_note(f'n{i}', 'task 5') for i in range(count)])
        assert bindings.results_not_scanned == 0


def _mentioned(task_id: str) -> BoundTask:
    return BoundTask(_local(task_id), Binding.MENTIONED, ('n1',))


def _bindings(tasks, *, omitted_tasks=0, results_not_scanned=0, ambiguous=()) -> TaskBindings:
    return TaskBindings(
        tasks=tuple(tasks), ambiguous=ambiguous,
        omitted_tasks=omitted_tasks, results_not_scanned=results_not_scanned,
    )


class TestInvariants:
    def test_more_than_max_tasks_is_refused(self):
        with pytest.raises(ValueError, match=rf'MAX_TASKS.*{MAX_TASKS + 1}'):
            _bindings(_mentioned(str(i)) for i in range(MAX_TASKS + 1))

    @pytest.mark.parametrize('field', ['omitted_tasks', 'results_not_scanned'])
    def test_a_negative_count_is_refused(self, field):
        with pytest.raises(ValueError, match=rf'{field}.*-1'):
            _bindings([], **{field: -1})

    def test_omitted_tasks_needs_a_full_task_list(self):
        with pytest.raises(ValueError, match=r'omitted_tasks.*3'):
            _bindings([_mentioned('1')], omitted_tasks=3)

    def test_a_duplicate_ref_is_refused(self):
        with pytest.raises(ValueError, match=r"once.*task_id='1'"):
            _bindings([_mentioned('1'), _mentioned('1')])

    def test_a_duplicate_ambiguous_number_is_refused(self):
        with pytest.raises(ValueError, match=r"ambiguous.*once.*'7', '7'"):
            _bindings([], ambiguous=('7', '7'))

    def test_an_ambiguous_number_that_is_also_bound_is_refused(self):
        with pytest.raises(ValueError, match=r"ambiguous.*bound.*'1'"):
            _bindings([_mentioned('1')], ambiguous=('1',))

    def test_a_mentioned_entry_before_a_structural_one_is_refused(self):
        structural = BoundTask(_local('2'), Binding.STRUCTURAL, ('e1',))
        with pytest.raises(ValueError, match=r'STRUCTURAL.*before.*MENTIONED'):
            _bindings([_mentioned('1'), structural])

    def test_a_bound_task_needs_a_result(self):
        with pytest.raises(ValueError, match=r'result_ids.*\(\)'):
            BoundTask(_local('1'), Binding.MENTIONED, ())


READ_AT = datetime(2026, 11, 13, 15, 4, 5, tzinfo=UTC)
HEADER = (
    '**Live task status** (read from the task store at 15:04Z; '
    'status words inside the memory text below are history)'
)


def _render(bindings: TaskBindings, facts, *, read_at: datetime = READ_AT) -> str:
    return render_task_facts(bindings, facts, project_id=PROJECT, read_at=read_at)


def _found(task_id: str, status: str, title: str | None = 'T', **finished) -> TaskFacts:
    return TaskFacts.found(_local(task_id), status=status, title=title, **finished)


def _by_ref(*facts: TaskFacts) -> dict[TaskRef, TaskFacts]:
    return {fact.ref: fact for fact in facts}


def _structural(*task_ids: str) -> TaskBindings:
    return _bind([_edge(f'e{task_id}', 'a fact', [f'Task {task_id}']) for task_id in task_ids])


class TestRenderTaskFacts:
    def test_the_block_groups_and_labels_every_kind_of_fact(self):
        bindings = _bind([
            _edge('e1', 'a fact', ['Task 3651']),
            _edge('e2', 'a fact', ['Task 9261']),
            _edge('e3', 'a fact', ['Task 4725']),
            _note('n4', 'task 3127 is pending'),
            _edge('e5', 'a fact', ['reify:132']),
            _note('n6', 'task 77 is open'),
            _note('n7', 'reify:3674 blocks task 3674'),
        ])
        facts = _by_ref(
            _found('3651', 'done', 'Short title', finished_how='merged', finished_commit='8ca12d33a1b2c3d4'),
            _found('9261', 'pending', waits_on=((_local('3057'), 'pending'),)),
            _found('4725', 'done', 'Other', finished_how='found_on_main'),
            _found('3127', 'done'),
            TaskFacts.could_not_check(TaskRef('reify', '132'), 'project not registered'),
            TaskFacts.no_such_task(_local('77')),
        )
        assert _render(bindings, facts).splitlines() == [
            HEADER,
            '- done: 3651 "Short title" (merged 8ca12d33a1) · 4725 "Other" (found on main)',
            '- pending: 9261 "T", waits on 3057 (pending)',
            '- mentioned in the text only, check the title: 3127 "T" (done)',
            '- no such task: 77',
            '- could not check: reify:132 (project not registered)',
            '- ambiguous, not looked up: 3674',
        ]

    def test_a_done_task_without_provenance_has_no_parenthesis(self):
        facts = _by_ref(_found('3651', 'done', 'Short title'))
        assert _render(_structural('3651'), facts).splitlines()[1] == '- done: 3651 "Short title"'

    def test_no_such_task_and_could_not_check_ignore_the_binding(self):
        bindings = _bind([_edge('e1', 'a fact', ['Task 77']), _note('n2', 'task 78 and task 79')])
        facts = _by_ref(
            TaskFacts.no_such_task(_local('77')),
            TaskFacts.no_such_task(_local('78')),
            TaskFacts.could_not_check(_local('79'), 'task store read failed: OSError: gone'),
        )
        assert _render(bindings, facts).splitlines()[1:] == [
            '- no such task: 77 · 78',
            '- could not check: 79 (task store read failed: OSError: gone)',
        ]

    def test_a_long_title_is_cut_at_a_word_and_a_missing_one_has_no_quotes(self):
        title = 'a title long enough that it must be cut somewhere past sixty characters'
        facts = _by_ref(_found('1', 'pending', title), _found('2', 'pending', None))
        assert _render(_structural('1', '2'), facts).splitlines()[1] == (
            f'- pending: 1 "{fit_at_word(title, 60)}" · 2'
        )

    def test_a_title_renders_on_one_line_inside_its_own_quotes(self):
        facts = _by_ref(_found('1', 'pending', 'Line one\n  line "two"'))
        assert _render(_structural('1'), facts).splitlines()[1:] == [
            '- pending: 1 "Line one line \'two\'"',
        ]

    @pytest.mark.parametrize(
        ('task_count', 'result_count', 'expected'),
        [
            (60, 105, '- not shown: 20 more tasks; 5 results not scanned'),
            (45, 100, '- not shown: 5 more tasks'),
            (1, 105, '- not shown: 5 results not scanned'),
        ],
    )
    def test_overflow_is_a_trailing_line_with_only_its_non_zero_parts(
        self, task_count, result_count, expected,
    ):
        results = [_note(f'n{i}', f'task {1000 + i} is open') for i in range(task_count)]
        results += [_note(f'x{i}', 'nothing here') for i in range(result_count - task_count)]
        bindings = _bind(results)
        facts = _by_ref(*(_found(bound.ref.task_id, 'pending') for bound in bindings.tasks))
        assert _render(bindings, facts).splitlines()[-1] == expected

    def test_the_read_time_is_rendered_in_utc(self):
        plus_two = datetime(2026, 11, 13, 17, 4, 5, tzinfo=timezone(timedelta(hours=2)))
        facts = _by_ref(_found('1', 'pending'))
        assert _render(_structural('1'), facts, read_at=plus_two).splitlines()[0] == HEADER

    def test_the_project_is_canonicalized(self):
        facts = _by_ref(_found('1', 'pending'))
        rendered = render_task_facts(
            _structural('1'), facts, project_id='Dark-Factory', read_at=READ_AT,
        )
        assert rendered.splitlines()[1] == '- pending: 1 "T"'


class TestRenderTaskFactsOtherForms:
    @pytest.mark.parametrize('task_ids', [(), ('1',)], ids=['no-tasks', 'tasks'])
    def test_unavailable_facts_render_one_line_with_the_reason(self, task_ids):
        rendered = _render(_structural(*task_ids), FactsUnavailable('no task store configured'))
        assert rendered.splitlines() == [rendered]
        assert 'could not be read' in rendered
        assert 'no task store configured' in rendered

    def test_an_empty_reason_is_refused(self):
        with pytest.raises(ValueError, match='reason'):
            FactsUnavailable('')

    def test_nothing_to_report_renders_nothing(self):
        assert _render(_bind([_note('n1', 'nothing here')]), {}) == ''

    def test_a_naive_read_time_is_refused(self):
        with pytest.raises(ValueError, match='read_at'):
            _render(_structural('1'), _by_ref(_found('1', 'pending')), read_at=datetime(2026, 11, 13))

    def test_a_bound_ref_without_facts_is_refused_by_name(self):
        with pytest.raises(ValueError, match="task_id='3651'"):
            _render(_structural('3651'), {})
