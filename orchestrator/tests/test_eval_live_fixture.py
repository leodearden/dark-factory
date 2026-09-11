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

import asyncio
import copy
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from orchestrator.config import load_config
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.live_fixture import (
    ShadowShape,
    build_live_fixture,
    live_stratum,
    live_verify_commands,
)
from orchestrator.evals.runner import build_eval_orch_config, load_task
from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    cell_of,
    default_verify_commands,
)


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


class TestBuildLiveFixtureIsArgumentPure:
    """C3's purity invariant: reads only its arguments, and shares none of them.

    Two halves. (1) The builder never reaches the live repo — so a cell's
    inputs are reproducible from its `shadow_cells` row rather than bound to
    whenever the builder happened to run. (2) It neither mutates its arguments
    nor hands back aliases of them — the plan dict the coordinator passes in is
    production's freshly-read `.task-meta/<worktree>/plan.json`, and `run_eval`
    hands the fixture's plan to a real workflow that flips step status in place.
    """

    def _forbid_subprocesses(self, monkeypatch):
        def _fail(*args, **kwargs):
            raise AssertionError('live repo read')

        for attr in ('run', 'Popen', 'check_output', 'check_call', 'call'):
            monkeypatch.setattr(subprocess, attr, _fail)
        for attr in ('create_subprocess_exec', 'create_subprocess_shell'):
            monkeypatch.setattr(asyncio, attr, _fail)

    def test_builds_with_no_git_repo_and_a_nonexistent_project_root(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)          # a temp dir containing no git repo
        self._forbid_subprocesses(monkeypatch)
        absent = tmp_path / 'no-such-checkout'
        assert not absent.exists()

        fixture = build_live_fixture(
            live_task(),
            base_sha=BASE_SHA,
            project_root=absent,
            plan=live_plan(),
            verify_commands=default_verify_commands('df'),
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )

        # The base is the caller's, never "HEAD now" — there is no HEAD here.
        assert fixture['pre_task_commit'] == BASE_SHA
        assert Path(fixture['project_root']) == absent

    def test_does_not_mutate_its_arguments(self, tmp_path):
        task, plan = live_task(), live_plan()
        verify_commands = default_verify_commands('df')
        before = copy.deepcopy((task, plan, verify_commands))

        build_live_fixture(
            task,
            base_sha=BASE_SHA,
            project_root=tmp_path,
            plan=plan,
            verify_commands=verify_commands,
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )

        assert (task, plan, verify_commands) == before

    def test_the_returned_fixture_shares_nothing_with_the_caller(self, tmp_path):
        task, plan = live_task(), live_plan()
        verify_commands = default_verify_commands('df')
        fixture = build_live_fixture(
            task,
            base_sha=BASE_SHA,
            project_root=tmp_path,
            plan=plan,
            verify_commands=verify_commands,
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )
        before = copy.deepcopy((task, plan, verify_commands))

        # Exactly what run_eval's workflow does to the plan it is handed.
        fixture['plan']['steps'].append({'id': 'step-3', 'status': 'pending'})
        fixture['plan']['steps'][0]['status'] = 'done'
        fixture['verify_commands']['test'] = 'echo pwned'
        fixture['modules'].append('orchestrator/agents')

        assert (task, plan, verify_commands) == before
        assert plan['steps'][0]['status'] == 'pending'
        assert task['metadata']['modules'] == ['orchestrator/evals']

    def test_fixture_plan_still_equals_the_caller_plan(self, tmp_path):
        # C3 says "plan as given": equal by value is all it requires, and a
        # copy satisfies it.
        plan = live_plan()
        fixture = build_live_fixture(
            live_task(),
            base_sha=BASE_SHA,
            project_root=tmp_path,
            plan=plan,
            verify_commands=default_verify_commands('df'),
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )
        assert fixture['plan'] == plan
        assert fixture['plan'] is not plan


def _base_config(tmp_path: Path):
    """A deterministic pure-code-default base config via the REAL load_config().

    Copied from ``test_eval_driver._base_config``: a minimal YAML setting only
    project_root, layered over the packaged defaults.yaml by the real
    production config-load entry point — never a hand-built OrchestratorConfig,
    so a leaf that regresses to a pydantic default is visible here.
    """
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    return load_config(cfg_path)


class TestLiveFixtureRoundTrip:
    """The PRD's β signal: the emitted dict survives the REAL consumers.

    Exercises ``runner.load_task`` and ``build_eval_orch_config`` rather than
    re-pinning a key list — a fixture that satisfies the key surface but dies
    at ``json.dump`` or at ``load_task``'s ``raw_root.startswith(...)`` would
    pass the former and fail the campaign.
    """

    def _round_trip(self, tmp_path) -> dict:
        checkout = tmp_path / 'checkout'
        checkout.mkdir()
        fixture = build_live_fixture(
            live_task(),
            base_sha=BASE_SHA,
            project_root=checkout,
            plan=live_plan(),
            verify_commands=default_verify_commands('df'),
            shape=ShadowShape.IMPLEMENTER,
            cell_id='01JCELL',
        )
        fixture_path = tmp_path / 'shadow.json'
        fixture_path.write_text(json.dumps(fixture))
        return load_task(fixture_path)

    def test_survives_json_dump_and_load_task_untouched(self, tmp_path):
        loaded = self._round_trip(tmp_path)
        assert loaded['id'] == 'shadow_5383_01JCELL'
        assert loaded['pre_task_commit'] == BASE_SHA
        # The path exists, so neither load_task rewrite branch fires and the
        # cell's own checkout stands.
        assert loaded['project_root'] == str(tmp_path / 'checkout')

    def test_project_root_is_a_str(self, tmp_path):
        # load_task:234 calls raw_root.startswith(...) — a Path or a null is an
        # AttributeError there, so the string form is load-bearing.
        loaded = self._round_trip(tmp_path)
        assert isinstance(loaded['project_root'], str)

    def test_derived_orch_config_runs_the_fixtures_gates(self, tmp_path):
        loaded = self._round_trip(tmp_path)
        orch = build_eval_orch_config(
            EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
            loaded,
            base_config=_base_config(tmp_path),
        )
        gates = default_verify_commands('df')
        assert orch.test_command == gates['test']
        assert orch.lint_command == gates['lint']
        assert orch.type_check_command == gates['typecheck']
        assert orch.project_root == tmp_path / 'checkout'

    def test_omitted_knobs_land_on_the_documented_runner_defaults(self, tmp_path):
        # The deliberate omissions resolve to the eval standard the INCUMBENT
        # is also measured under — not to something surprising.
        orch = build_eval_orch_config(
            EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
            self._round_trip(tmp_path),
            base_config=_base_config(tmp_path),
        )
        assert orch.max_execute_iterations == 20
        assert orch.max_review_cycles == 1
        assert orch.judge_after_each_iteration is True

    def test_task_assignment_inputs_resolve_to_the_emitted_values(self, tmp_path):
        # The exact reads run_eval performs to build its TaskAssignment. The
        # `name`-synthesised fallback must NOT fire: it would drop `details`
        # from the brief and hand the shadow agent less than production got.
        loaded = self._round_trip(tmp_path)
        task = live_task()
        fallback = {
            'title': loaded.get('name', loaded['id']),
            'description': loaded.get('name', ''),
        }
        task_def = loaded.get('task_definition', fallback)
        assert task_def != fallback
        assert task_def == {
            'title': task['title'],
            'description': task['description'],
            'details': task['details'],
        }
        assert loaded.get('modules', []) == task['metadata']['modules']

    def test_the_plan_survives_the_round_trip(self, tmp_path):
        # run_eval refuses a falsy plan (runner.py:529) AFTER creating the
        # worktree, so the plan reaching it intact is the whole point.
        loaded = self._round_trip(tmp_path)
        assert loaded.get('plan') == live_plan()


class TestLiveVerifyCommands:
    """The gates a live shadow cell runs — one definition, task_sampler's."""

    @pytest.mark.parametrize(
        'project,repo',
        [
            ('dark_factory', 'df'),
            ('dark-factory', 'df'),
            ('reify', 'reify'),
            ('know_live', 'kl'),
        ],
    )
    def test_resolves_the_repo_gate_set(self, project, repo):
        assert live_verify_commands(project) == default_verify_commands(repo)

    def test_returns_a_fresh_copy_the_caller_may_mutate(self):
        first = live_verify_commands('dark_factory')
        first['test'] = 'echo pwned'
        assert live_verify_commands('dark_factory') == default_verify_commands('df')

    def test_unrecognised_project_propagates_rather_than_defaulting(self):
        # Silently defaulting to 'df' would run a Rust task's cell under
        # pytest and score the resulting red gate as a candidate failure.
        with pytest.raises(ValueError) as excinfo:
            live_verify_commands('nope')
        assert repr('nope') in str(excinfo.value)


class TestLiveStratum:
    """The (repo, kind, path) cell for a LIVE record — the sampler's own axes."""

    def test_returns_the_structured_triple_not_a_delimited_string(self):
        stratum = live_stratum(live_task(), project='dark_factory')
        assert isinstance(stratum, tuple)
        assert len(stratum) == 3
        assert all(isinstance(axis, str) for axis in stratum)

    def test_bugfix_title_without_complexity(self):
        task = live_task(
            title='Fix the settle deadline comparison',
            description='The deadline is compared the wrong way round.',
            metadata={},
        )
        assert live_stratum(task, project='dark_factory') == ('df', 'bugfix', 'full')

    def test_feature_title_declared_simple(self):
        task = live_task(
            title='Add a shadow cell store',
            description='Store rows and read them back.',
            metadata={'complexity': 'simple'},
        )
        assert live_stratum(task, project='dark_factory') == ('df', 'feature', 'simple')

    def test_declared_simple_is_vetoed_by_a_blocker_token(self):
        # classify_path reuses production's has_simple_task_blocker veto, so
        # the stratum matches how the orchestrator would actually have routed
        # the task — not merely what its author declared.
        task = live_task(
            title='Refactor the shadow coordinator',
            description='Touches the architecture of the merge lane.',
            metadata={'complexity': 'simple'},
        )
        assert live_stratum(task, project='dark_factory') == ('df', 'refactor', 'full')

    def test_project_selects_the_repo_axis(self):
        task = live_task(title='Add a new subcommand', description='', metadata={})
        assert live_stratum(task, project='reify')[0] == 'reify'
        assert live_stratum(task, project='dark_factory')[0] == 'df'

    def test_unrecognised_project_propagates(self):
        with pytest.raises(ValueError) as excinfo:
            live_stratum(live_task(), project='nope')
        assert repr('nope') in str(excinfo.value)

    def test_reads_only_the_record(self, tmp_path, monkeypatch):
        # No project_root parameter to pass, and no subprocess to run: the
        # stratum of a live cell is a property of the task record alone.
        parameters = inspect.signature(live_stratum).parameters
        assert list(parameters) == ['task', 'project']
        assert parameters['project'].kind is inspect.Parameter.KEYWORD_ONLY

        monkeypatch.chdir(tmp_path)

        def _fail(*args, **kwargs):
            raise AssertionError('live repo read')

        for attr in ('run', 'Popen', 'check_output', 'check_call', 'call'):
            monkeypatch.setattr(subprocess, attr, _fail)
        assert live_stratum(live_task(), project='dark_factory')[0] == 'df'

    def test_agrees_with_the_sampler_over_the_same_facts(self):
        # SPOT: live_stratum must be the sampler's classifiers applied to a
        # live record, not a second implementation that can drift from them.
        task = live_task(
            title='Fix the settle deadline comparison',
            description='The deadline is compared the wrong way round.',
            metadata={'complexity': 'simple'},
        )
        equivalent = CompletedTaskCandidate(
            task_id=task['id'],
            project='dark_factory',
            project_root='',
            title=task['title'],
            description=task['description'],
            complexity='simple',
        )
        assert live_stratum(task, project='dark_factory') == cell_of(equivalent)
