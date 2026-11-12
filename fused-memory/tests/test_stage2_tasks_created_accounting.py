"""Wiring tests for the Stage 2 task-accounting prompt contract (tasks 3046, 4873).

Stage 2 self-reports ``tasks_created``, ``tasks_combined`` and
``tasks_hints_updated``; ``derive_stage_stats`` does not recompute them.
Run 507bc25b filed task 3045 yet self-reported ``tasks_created: 0``. The
framework now grounds these counters in the code-side creation journal
(``middleware.task_creation_journal``, keyed on the ``_causation_id`` in
``submit_task`` metadata) and in the ``task_created_records`` /
``task_hints_updated_records`` action records the prompt asks for.

This module pins the WIRING of the shared renderers
``render_task_creation_accounting_section()`` and
``render_task_hints_update_accounting_section()`` (``recon_self_model.py``) —
each exactly once per assembled Stage 2 prompt, on both
``build_stage2_system_prompt`` branches — and the machine-readable contract
tokens the Python side reads (``tasks_created``, ``tasks_combined``,
``task_created_records``, ``task_hints_updated_records`` and the two
``"action"`` record shapes). Per the ``test_recon_gate_closure_guidance.py``
house rule, prose wording beyond those tokens is intentionally NOT pinned.

It also pins the code-side layer: ``stage2_counter_grounding.coerce_count``
and the upward-only repair ``TaskKnowledgeSync._apply_post_flight_guards``
performs when the self-reported ``tasks_created`` undercounts its confirmed
evidence — the exact shape of the run-507bc25b regression.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.recon_self_model import (
    render_task_creation_accounting_section,
    render_task_hints_update_accounting_section,
)
from fused_memory.reconciliation.stage2_counter_grounding import coerce_count
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync


class TestTaskCreationAccountingWiring:
    """render_task_creation_accounting_section() is wired into Stage 2 exactly once."""

    def test_embedded_verbatim_exactly_once_in_stage2_prompt(self):
        section = render_task_creation_accounting_section()

        # prompt.count(section) == 1 subsumes non-emptiness: str.count('')
        # over a non-empty STAGE2_SYSTEM_PROMPT is len(prompt) + 1, never 1.
        assert STAGE2_SYSTEM_PROMPT.count(section) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both runtime builder branches keep the section, exactly once.

        `autopilot_video` splices its contamination guardrail at the
        `'## Available Tools'` sentinel; that injection must not drop or
        duplicate this section.
        """
        section = render_task_creation_accounting_section()
        built = build_stage2_system_prompt(project_id)

        assert built.count(section) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_assembled_prompt_carries_machine_readable_contract_tokens(self, project_id: str):
        """The Python side reads tasks_created / task_created_records / the
        "action": "task_created" record shape — pin those tokens, not prose.

        This also covers the '## Available Tools' sentinel and the
        mcp__recon-report__ run_id= guard by construction: if the rendered
        section carried either, `build_stage2_system_prompt('dark_factory')`
        would either raise (duplicated sentinel — see
        test_survives_build_stage2_system_prompt's autopilot_video branch)
        or would be caught separately by test_recon_report_guidance_drift.py,
        which scans this exact assembled prompt for run_id=-less call
        examples. No need to re-pin either as a standalone negative-substring
        check here (task-3046 amendment)."""
        built = build_stage2_system_prompt(project_id)

        assert 'tasks_created' in built
        assert 'task_created_records' in built
        assert '"action": "task_created"' in built
        assert 'tasks_combined' in built
        assert 'task_hints_updated_records' in built
        assert '"action": "task_hints_updated"' in built


class TestHintsUpdateAccountingWiring:
    """render_task_hints_update_accounting_section() is wired into Stage 2
    exactly once (task 4018)."""

    def test_embedded_verbatim_exactly_once_in_stage2_prompt(self):
        section = render_task_hints_update_accounting_section()

        assert STAGE2_SYSTEM_PROMPT.count(section) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_task_hints_update_accounting_section() verbatim, exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        section = render_task_hints_update_accounting_section()
        built = build_stage2_system_prompt(project_id)

        assert built.count(section) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry '
            'render_task_hints_update_accounting_section() verbatim, exactly once.'
        )


class TestCoerceCount:
    """Pins coerce_count: best-effort int coercion for a self-reported Stage 2
    counter, so a non-int self-report (plausible free-form LLM JSON) doesn't
    collapse to 0 and look like an undercount relative to smaller evidence —
    which would silently move a legitimately larger self-report DOWN, breaking
    the upward-only repair contract (task 2230 / W5-mu). Never raises (task
    4140: NaN / infinity once escaped as ValueError / OverflowError)."""

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (0, 0),
            (3, 3),
            ('0', 0),
            ('3', 3),
            (' 3 ', 3),
            (3.0, 3),
            ('3.0', 3),
        ],
        ids=['int-0', 'int-3', 'str-0', 'str-3', 'str-3-whitespace', 'float-3', 'str-float-3'],
    )
    def test_coerces_legitimate_numeric_forms(self, value, expected):
        assert coerce_count(value) == expected

    @pytest.mark.parametrize(
        'value',
        [None, True, False, 'not-a-number', [], {}],
        ids=['none', 'true', 'false', 'non-numeric-str', 'list', 'dict'],
    )
    def test_uncoercible_or_bool_falls_back_to_zero(self, value):
        """A bool is REJECTED even though isinstance(True, int) holds in
        Python — never trusted as a legitimate self-reported count."""
        assert coerce_count(value) == 0

    @pytest.mark.parametrize(
        'value',
        [float('nan'), float('inf'), float('-inf'), 'inf', '-inf', '1e400', 'nan', ' Infinity '],
        ids=['nan', 'inf', '-inf', 'str-inf', 'str--inf', 'str-1e400', 'str-nan', 'str-Infinity'],
    )
    def test_non_finite_values_fall_back_to_zero_without_raising(self, value):
        assert coerce_count(value) == 0


# ── Task-3046 step-5: TaskKnowledgeSync._apply_post_flight_guards repair ────


class _AllPresentTaskmaster:
    """Corroborating taskmaster double (task 3051).

    Every ``get_task`` returns a task record, so the corroboration pass task
    3051 inserted ahead of the repair confirms every structurally-valid
    record. That keeps the pins BELOW exercising the task-3046 repair itself
    — which records count, how they dedup, how a non-int self-report coerces
    — rather than accidentally re-testing the corroboration gate, whose own
    contract (what happens when a record can NOT be confirmed) is owned by
    ``tests/test_stage2_stats_audit.py``.
    """

    async def get_task(self, task_id, project_root):
        return {'id': str(task_id), 'title': f'task {task_id}', 'status': 'pending'}


#: project_id -> project_root for every project the records below name.
#: Required as of task 3051: an unresolvable project_id issues no get_task
#: call and therefore corroborates nothing, which would withhold the repair.
_KNOWN_PROJECTS = {
    'dark_factory': '/repo/dark-factory',
    'reify': '/repo/reify',
    'test_project': '/tmp/test',
}


def _mock_deps() -> dict:
    """Kwargs for constructing a TaskKnowledgeSync with mocked deps (task 3046).

    Replicates the ~12-line construction from
    ``tests/test_stages.py::stage2_guard_mock_deps`` (10083-10098) rather than
    cross-importing that module-private fixture — the same choice
    ``test_recon_gate_closure_guidance.py::_make_consolidator`` documents and
    makes for ``MemoryConsolidator``.

    Supplies a corroborating taskmaster and a populated ``known_projects``
    (task 3051): a bare ``AsyncMock()`` taskmaster confirms nothing and an
    empty ``known_projects`` issues no lookups at all, either of which now
    withholds the upward repair these pins exercise.
    """
    config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
    write_journal_mock = MagicMock()
    write_journal_mock.get_ops_by_causation = AsyncMock(return_value=[])
    journal_mock = MagicMock()
    journal_mock.write_journal = write_journal_mock
    journal_mock.record_run_session = AsyncMock()
    journal_mock.clear_run_session = AsyncMock()
    return {
        'memory_service': AsyncMock(),
        'taskmaster': _AllPresentTaskmaster(),
        'journal': journal_mock,
        'config': config,
        'scope': ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
        'known_projects': dict(_KNOWN_PROJECTS),
    }


def _make_stage() -> TaskKnowledgeSync:
    return TaskKnowledgeSync(StageId.task_knowledge_sync, **_mock_deps())


def _make_report(stats: dict) -> StageReport:
    now = datetime.now(tz=UTC)
    return StageReport(
        stage=StageId.task_knowledge_sync,
        started_at=now,
        completed_at=now,
        stats=stats,
    )


_REPAIR_LOGGER = 'fused_memory.reconciliation.stages.task_knowledge_sync'


class TestPostFlightTasksCreatedRepair:
    """TaskKnowledgeSync._apply_post_flight_guards' tasks_created repair (task 3046).

    Exercises the post-flight guard directly (not via ``stage.run()``): with
    ``flag_deleted_records`` absent from stats,
    ``_acknowledge_resolved_stage1_markers`` short-circuits to 0 without
    touching the mocked memory service, isolating these assertions to the
    tasks_created normalization + upward-only repair added by step-6.
    """

    @pytest.mark.asyncio
    async def test_normalization_on_empty_stats(self):
        """No counter, no records: both new keys default to 0, no repair key."""
        stage = _make_stage()
        report = _make_report({})

        await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_run_507bc25b_regression_undercount_is_repaired(self, caplog):
        """The exact run-507bc25b shape: tasks_created=0 self-reported while
        one task_created_records entry confirms task 3045 was filed via the
        proactive-sample surface — repaired upward to 1."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                    'source_path': 'proactive_sample',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 1

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'tasks_created=0' in message
        assert 'to 1' in message

    @pytest.mark.asyncio
    async def test_missing_counter_key_derives_from_records(self):
        """tasks_created absent entirely; two distinct confirmed records ⇒ 2."""
        stage = _make_stage()
        report = _make_report({
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
                {
                    'action': 'task_created',
                    'task_id': '3046',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 2
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 2

    @pytest.mark.asyncio
    async def test_agreement_leaves_counter_untouched_no_repair_key(self, caplog):
        """Self-report already matches the record-derived count: no repair,
        no spurious tasks_created_reported key, no WARNING."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 2,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
                {
                    'action': 'task_created',
                    'task_id': '3046',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 2
        assert 'tasks_created_reported' not in report.stats
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_no_downward_clamp_on_over_report(self, caplog):
        """Mirrors test_stages.py::TestPostFlightGuardsNoFlagCounterClamp
        (task 2230): an over-reported tasks_created survives untouched even
        though ground truth is lower. task_created_records_valid still
        publishes the lower count so the divergence stays observable, and no
        WARNING fires (this guard warns only on an UNDER-count repair)."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 3,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 3
        assert report.stats['task_created_records_valid'] == 1
        assert 'tasks_created_reported' not in report.stats
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_non_list_records_are_inert(self):
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': 'not-a-list',
        })

        await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_non_dict_and_failed_status_entries_are_inert(self):
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                None,
                'x',
                ['a'],
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'failed',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    # -- non-int self-reports never repair downward (task-3046 amendment) ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('raw_tasks_created', 'expected', 'repaired'),
        [
            ('0', 1, True),    # numeric str below ground truth -> repaired up
            ('3', 3, False),   # numeric str at/above ground truth -> normalized only
            (3.0, 3, False),   # float at/above ground truth -> normalized only
            (True, 1, True),   # bool rejected outright -> treated as unreported -> repaired up
            (None, 1, True),   # explicit None value -> treated as unreported -> repaired up
        ],
        ids=[
            'str-zero-undercounts',
            'str-three-agrees',
            'float-three-agrees',
            'bool-rejected',
            'none-value',
        ],
    )
    async def test_non_int_self_report_never_repaired_downward(
        self, raw_tasks_created, expected, repaired,
    ):
        """A non-int self-report must coerce to its real numeric value
        before the upward-only comparison, never collapse to 0 and get
        spuriously "repaired" DOWN from a legitimately larger self-report
        (review suggestion #1). The post-guard value is always a plain
        ``int`` (never a residual ``str``/``float``/``bool``)."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': raw_tasks_created,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, 'test-run-3046')

        assert report.stats['tasks_created'] == expected
        assert isinstance(report.stats['tasks_created'], int)
        assert not isinstance(report.stats['tasks_created'], bool)
        assert ('tasks_created_reported' in report.stats) is repaired
        if repaired:
            assert report.stats['tasks_created_reported'] == raw_tasks_created
