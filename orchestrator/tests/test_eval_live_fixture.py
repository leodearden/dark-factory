"""Tests for orchestrator.evals.live_fixture — the live shadow-eval fixture builder.

Hermetic throughout, mirroring ``test_eval_task_sampler``'s style: every task
record and plan is an inline synthetic dict, every path is under ``tmp_path``,
and nothing here reads the live repo, the task store, the network or an LLM.
That is not merely test hygiene — C3's purity invariant (the builder reads
nothing but its arguments) is itself one of the properties under test, so the
suite would be unable to state it if it needed a real checkout to run.

Step map:
  step-01/02  ShadowShape — the four-value cell vocabulary and its plan rule
  step-03/04  build_live_fixture — the emitted key surface (and the omissions)
  step-05/06  build_live_fixture — the refusal table
  step-07/08  build_live_fixture — argument purity: no repo reads, no aliasing
  step-09/10  round trip through runner.load_task + build_eval_orch_config
  step-11/12  live_verify_commands / live_stratum — the task_sampler hookups
"""

from __future__ import annotations

import json

import pytest

from orchestrator.evals.live_fixture import ShadowShape, build_live_fixture
from orchestrator.evals.task_sampler import default_verify_commands


class TestShadowShape:
    """The closed cell-shape vocabulary (PRD C1) and the shape→plan rule."""

    def test_exactly_the_four_c1_values(self):
        assert [s.value for s in ShadowShape] == [
            'implementer',
            'architect',
            'architect-consequence',
            'end-to-end',
        ]

    def test_member_is_a_plain_str_and_json_round_trips(self):
        # C3 types `shape: str`; a StrEnum member satisfies that verbatim and
        # serialises as the bare value (no 'ShadowShape.IMPLEMENTER' leaking
        # into the shadow_cells.shape TEXT column or a persisted result).
        assert isinstance(ShadowShape.IMPLEMENTER, str)
        assert json.dumps(ShadowShape.IMPLEMENTER) == '"implementer"'
        assert json.loads(json.dumps(ShadowShape.IMPLEMENTER)) == 'implementer'

    @pytest.mark.parametrize(
        'raw,member',
        [
            ('implementer', ShadowShape.IMPLEMENTER),
            ('architect', ShadowShape.ARCHITECT),
            ('architect-consequence', ShadowShape.ARCHITECT_CONSEQUENCE),
            ('end-to-end', ShadowShape.END_TO_END),
        ],
    )
    def test_string_coercion_recovers_the_member(self, raw, member):
        # A caller holding a value read back from the shadow_cells TEXT column
        # coerces it without a lookup table of its own.
        assert ShadowShape(raw) is member

    @pytest.mark.parametrize(
        'shape,requires_plan',
        [
            (ShadowShape.IMPLEMENTER, True),
            (ShadowShape.ARCHITECT_CONSEQUENCE, True),
            (ShadowShape.ARCHITECT, False),
            (ShadowShape.END_TO_END, False),
        ],
    )
    def test_requires_plan_is_total_over_the_vocabulary(self, shape, requires_plan):
        assert shape.requires_plan is requires_plan


# ---------------------------------------------------------------------------
# Synthetic live task records + plans (inline, hermetic — no live data)
# ---------------------------------------------------------------------------

BASE_SHA = 'a' * 40


def live_task(**overrides) -> dict:
    """A synthetic LIVE production task record, shaped as fused-memory stores it."""
    task = {
        'id': '5383',
        'title': 'Live fixture builder',
        'description': 'Render a production task as an eval fixture.',
        'details': 'Implement build_live_fixture per PRD contract C3.',
        'metadata': {
            'modules': ['orchestrator/evals'],
            'complexity': 'simple',
            'files': ['orchestrator/src/orchestrator/evals/live_fixture.py'],
            'branch_base_sha': BASE_SHA,
        },
    }
    task.update(overrides)
    return task


def live_plan() -> dict:
    return {
        'task_id': '5383',
        'steps': [
            {'id': 'step-1', 'description': 'RED', 'status': 'pending'},
            {'id': 'step-2', 'description': 'GREEN', 'status': 'pending'},
        ],
    }


DF_VERIFY = {
    'test': 'cd orchestrator && uv run pytest tests/ -x',
    'lint': 'cd orchestrator && uv run ruff check src/',
    'typecheck': 'cd orchestrator && uv run pyright src/',
}

# Every key load_task's consumers read that this builder deliberately does NOT
# emit, so the runner's documented default stands. Pinned as an executable
# absence assertion rather than left to the module docstring's prose.
OMITTED_KEYS = (
    'reference',
    'post_task_commit',
    'setup_commands',
    'timeout_minutes',
    'max_execute_iterations',
    'max_review_cycles',
    'judge_after_each_iteration',
    'max_architect_turns',
    'adversarial',
    'complexity',
    'project',
    'cohort',
    'provenance',
    'verify_outcome',
)


class TestBuildLiveFixtureKeySurface:
    """Exactly the eight emitted keys, key by key — and the omissions."""

    def _build(self, tmp_path, task=None, plan=None):
        return build_live_fixture(
            task if task is not None else live_task(),
            base_sha=BASE_SHA,
            project_root=tmp_path,
            plan=plan if plan is not None else live_plan(),
            verify_commands=default_verify_commands('df'),
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )

    def test_id_is_the_shadow_cell_id(self, tmp_path):
        assert self._build(tmp_path)['id'] == 'shadow_5383_01JCELL'

    def test_pre_task_commit_is_the_callers_base_sha_byte_equal(self, tmp_path):
        # Never "HEAD now", and never normalised — create_eval_worktree
        # compares it against `git rev-parse HEAD` literally.
        assert self._build(tmp_path)['pre_task_commit'] == BASE_SHA

    def test_name_is_the_task_title(self, tmp_path):
        task = live_task()
        assert self._build(tmp_path, task=task)['name'] == task['title']

    def test_task_definition_carries_title_description_details(self, tmp_path):
        task = live_task()
        assert self._build(tmp_path, task=task)['task_definition'] == {
            'title': task['title'],
            'description': task['description'],
            'details': task['details'],
        }

    def test_task_definition_withholds_the_live_task_id_and_metadata(self, tmp_path):
        # briefing.py::_format_task renders task_definition into the agent's
        # brief. An `id` line would hand the shadow agent the key to look up
        # the very task it is shadowing — a contamination path into the
        # measurement — and metadata.files is queue-time guesswork, not task
        # statement.
        task_def = self._build(tmp_path)['task_definition']
        assert 'id' not in task_def
        assert 'metadata' not in task_def
        assert set(task_def) == {'title', 'description', 'details'}

    def test_verify_commands_are_the_callers_gates(self, tmp_path):
        assert self._build(tmp_path)['verify_commands'] == default_verify_commands('df')

    def test_modules_come_from_the_record_metadata(self, tmp_path):
        task = live_task()
        assert self._build(tmp_path, task=task)['modules'] == task['metadata']['modules']

    def test_plan_is_the_plan_as_given(self, tmp_path):
        plan = live_plan()
        assert self._build(tmp_path, plan=plan)['plan'] == plan

    def test_emits_exactly_the_eight_keys(self, tmp_path):
        assert set(self._build(tmp_path)) == {
            'id',
            'name',
            'project_root',
            'pre_task_commit',
            'task_definition',
            'verify_commands',
            'modules',
            'plan',
        }

    @pytest.mark.parametrize('key', OMITTED_KEYS)
    def test_deliberately_omits(self, tmp_path, key):
        assert key not in self._build(tmp_path)

    def test_missing_details_becomes_an_empty_string(self, tmp_path):
        task = live_task()
        del task['details']
        assert self._build(tmp_path, task=task)['task_definition'] == {
            'title': task['title'],
            'description': task['description'],
            'details': '',
        }

    def test_missing_metadata_entirely_yields_no_modules(self, tmp_path):
        task = live_task()
        del task['metadata']
        fixture = self._build(tmp_path, task=task)
        assert fixture['modules'] == []
        # The rest of the record still lands — an absent metadata block is a
        # thin task, not a malformed one.
        assert fixture['id'] == 'shadow_5383_01JCELL'

    def test_null_description_becomes_an_empty_string(self, tmp_path):
        fixture = self._build(tmp_path, task=live_task(description=None))
        assert fixture['task_definition']['description'] == ''
