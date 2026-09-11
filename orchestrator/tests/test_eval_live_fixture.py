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


class TestBuildLiveFixtureRefuses:
    """The guard block: every bad input is a ValueError naming what was wrong.

    Refusing at BUILD time is the point. `runner.py`'s own plan check fires at
    :529, AFTER `create_eval_worktree` has already run at :495 — so a plan-less
    implementer fixture leaks an eval worktree before it fails, and a bad
    base_sha dies as a late HEAD-mismatch RuntimeError with a worktree already
    on disk. Refusing here keeps a data-plumbing bug from becoming a leaked
    worktree and a scored candidate decline (INV-11, no-silent-fail-soft).
    """

    def _build(self, tmp_path, **overrides):
        kwargs = {
            'base_sha': BASE_SHA,
            'project_root': tmp_path,
            'plan': live_plan(),
            'verify_commands': default_verify_commands('df'),
            'shape': ShadowShape.IMPLEMENTER,
            'cell_id': '01JCELL',
        }
        task = overrides.pop('task', None)
        kwargs.update(overrides)
        return build_live_fixture(task if task is not None else live_task(), **kwargs)

    @pytest.mark.parametrize(
        'shape', [ShadowShape.IMPLEMENTER, ShadowShape.ARCHITECT_CONSEQUENCE]
    )
    @pytest.mark.parametrize('plan', [None, {}], ids=['none', 'empty'])
    def test_plan_requiring_shape_without_a_plan(self, tmp_path, shape, plan):
        # {} is falsy, which is exactly what runner.py:529 also rejects.
        with pytest.raises(ValueError, match=str(shape.value)):
            self._build(tmp_path, shape=shape, plan=plan)

    @pytest.mark.parametrize('shape', [ShadowShape.ARCHITECT, ShadowShape.END_TO_END])
    def test_plan_forbidden_shape_given_a_plan(self, tmp_path, shape):
        # The same invariant in the other direction, so shape→plan is a total
        # function. Neither runner reads `plan`, so a stray plan here would be
        # silently ignored — leaving the consequence measurement comparing
        # legs that were not configured as intended, with nothing in the
        # record to show it.
        with pytest.raises(ValueError, match=str(shape.value)):
            self._build(tmp_path, shape=shape, plan=live_plan())

    @pytest.mark.parametrize('shape', [ShadowShape.ARCHITECT, ShadowShape.END_TO_END])
    def test_plan_forbidden_shape_with_no_plan_is_accepted(self, tmp_path, shape):
        assert self._build(tmp_path, shape=shape, plan=None)['plan'] is None

    @pytest.mark.parametrize(
        'shape', ['Implementer', 'consequence', '', 'plan', 'end_to_end']
    )
    def test_unrecognised_shape_names_the_four_valid_values(self, tmp_path, shape):
        with pytest.raises(ValueError) as excinfo:
            self._build(tmp_path, shape=shape)
        message = str(excinfo.value)
        assert repr(shape) in message
        for valid in ShadowShape:
            assert valid.value in message

    @pytest.mark.parametrize(
        'base_sha',
        ['', 'abc123def456', 'main', 'z' * 40, 'A' * 40, None],
        ids=['empty', 'short', 'refname', 'non-hex', 'uppercase', 'none'],
    )
    def test_base_sha_must_be_a_40_hex_sha(self, tmp_path, base_sha):
        with pytest.raises(ValueError, match='base_sha'):
            self._build(tmp_path, base_sha=base_sha)

    @pytest.mark.parametrize('cell_id', ['', '   ', '\t\n'], ids=['empty', 'sp', 'ws'])
    def test_cell_id_must_be_non_empty(self, tmp_path, cell_id):
        with pytest.raises(ValueError, match='cell_id'):
            self._build(tmp_path, cell_id=cell_id)

    @pytest.mark.parametrize('task_id', ['', '   ', None], ids=['empty', 'ws', 'none'])
    def test_task_id_must_be_non_empty(self, tmp_path, task_id):
        # The id is load-bearing: it is half the fixture id, so an empty one
        # silently produces 'shadow__<cell>' and two tasks collide.
        task = live_task(id=task_id)
        with pytest.raises(ValueError, match='id'):
            self._build(tmp_path, task=task)

    def test_missing_id_key_entirely_is_refused(self, tmp_path):
        task = live_task()
        del task['id']
        with pytest.raises(ValueError, match='id'):
            self._build(tmp_path, task=task)

    @pytest.mark.parametrize(
        'overrides',
        [
            {'plan': None},
            {'shape': 'nonsense'},
            {'base_sha': 'main'},
            {'cell_id': ''},
        ],
        ids=['no-plan', 'bad-shape', 'bad-sha', 'no-cell-id'],
    )
    def test_the_raise_is_the_only_exit(self, tmp_path, overrides):
        # No half-built fixture is ever returned alongside a logged complaint.
        with pytest.raises(ValueError):
            result = self._build(tmp_path, **overrides)
            pytest.fail(f'expected ValueError, got a fixture: {result!r}')
