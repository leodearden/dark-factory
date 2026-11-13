"""Verify-cache unit keys and the plan/evidence seams they build on (task 5652)."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling

from orchestrator import verify_evidence, verify_plan
from orchestrator.verify import VerifyResult
from orchestrator.verify_cache import keys, modes, toolchain
from orchestrator.verify_cmd import parse_config_command, scope_to
from orchestrator.verify_evidence import Leg, UnitTier
from orchestrator.verify_plan import PlannedRun, ScopeKind, VerifyPlan
from orchestrator.verify_spec_shared import canonical_json

_TEST_CMD = parse_config_command('uv run --project alpha pytest alpha/tests -q')
_LINT_CMD = parse_config_command('uv run --project alpha ruff check alpha')
_TYPE_CMD = parse_config_command('uv run --project alpha pyright alpha')


def _full_suite_test_run(reason: str = 'pytest: conftest touched') -> PlannedRun:
    return PlannedRun('alpha', _TEST_CMD, ScopeKind.FULL_SUITE, reason)


class TestRunSpecHash:
    def test_identical_runs_hash_equal(self):
        assert verify_evidence.run_spec_hash(_full_suite_test_run().to_dict()) == (
            verify_evidence.run_spec_hash(_full_suite_test_run().to_dict())
        )

    def test_prose_reason_is_not_identity(self):
        first = _full_suite_test_run('pytest: conftest touched').to_dict()
        second = _full_suite_test_run('pytest: structural change').to_dict()

        assert verify_evidence.run_spec_hash(first) == verify_evidence.run_spec_hash(second)

    def test_scope_kind_is_identity(self):
        full = _full_suite_test_run().to_dict()
        scoped = PlannedRun(
            'alpha', _TEST_CMD, ScopeKind.FILE_SCOPED, 'pytest: scoped',
            ('alpha/tests/test_one.py',),
        ).to_dict()

        assert verify_evidence.run_spec_hash(full) != verify_evidence.run_spec_hash(scoped)

    def test_plan_hash_keeps_its_alpha_definition(self):
        run = _full_suite_test_run().to_dict()
        execution_spec = {key: value for key, value in run.items() if key != 'reason'}

        assert verify_evidence.plan_hash({'runs': [run]}) == (
            hashlib.sha256(canonical_json([execution_spec]).encode()).hexdigest()
        )


class TestPlannedRunLeg:
    def test_each_reason_prefix_names_its_leg(self):
        lint = PlannedRun('alpha', _LINT_CMD, ScopeKind.FULL_SUITE, 'lint: structural')
        type_check = PlannedRun('alpha', _TYPE_CMD, ScopeKind.FULL_SUITE, 'pyright: structural')
        test = _full_suite_test_run()

        assert verify_plan.planned_run_leg(lint) is Leg.LINT
        assert verify_plan.planned_run_leg(type_check) is Leg.TYPE
        assert verify_plan.planned_run_leg(test) is Leg.TEST

    def test_a_reason_without_a_tool_prefix_has_no_leg(self):
        skipped = PlannedRun('__fallback__', None, ScopeKind.SKIPPED, 'no .py files touched')

        assert verify_plan.planned_run_leg(skipped) is None

    def test_leg_values_are_the_check_run_labels(self):
        assert {str(leg) for leg in Leg} == {'test', 'lint', 'type'}
        assert verify_evidence.module_leg_name('alpha', Leg.TYPE) == 'module:alpha:type'

    def test_module_of_a_registered_module_run_is_its_prefix(self):
        assert verify_plan.module_of(_full_suite_test_run()) == 'alpha'

    def test_module_of_the_sentinel_prefixes_is_none(self):
        fallback = PlannedRun('__fallback__', _TEST_CMD, ScopeKind.FULL_SUITE, 'pytest: conftest')
        trivial = PlannedRun('', None, ScopeKind.TRIVIAL, 'No source files changed')

        assert verify_plan.module_of(fallback) is None
        assert verify_plan.module_of(trivial) is None

    def test_verify_result_cache_mode_is_the_modes_enum(self):
        result = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='')

        assert result.cache_mode is modes.CacheMode.OFF


_KEYS_SEED = RepoSeed(
    files=(
        ('pyproject.toml', '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'),
        ('uv.lock', 'version = 1\n'),
        ('conftest.py', '# root conftest\n'),
        ('alpha/pyproject.toml', '[project]\nname = "alpha"\nversion = "0"\n'),
        ('alpha/src/alpha/__init__.py', 'A = 1\n'),
        ('alpha/tests/test_a.py', 'def test_a():\n    pass\n'),
        ('beta/pyproject.toml', '[project]\nname = "beta"\nversion = "0"\ndependencies = ["alpha>=0"]\n'),
        ('beta/src/beta/__init__.py', 'B = 1\n'),
        ('beta/scripts/check.py', 'print("beta check")\n'),
        ('tests/scripts/test_s.py', 'def test_s():\n    pass\n'),
        ('tools/check.py', 'print("check")\n'),
        ('docs/notes.md', '# notes\n'),
    ),
    message='keys fixture',
)


def _module_runs(prefix: str, *, lint: str, type_check: str, test: str) -> tuple[PlannedRun, ...]:
    return (
        PlannedRun(prefix, parse_config_command(lint), ScopeKind.FULL_SUITE, 'lint: structural change'),
        PlannedRun(prefix, parse_config_command(type_check), ScopeKind.FULL_SUITE, 'pyright: structural change'),
        PlannedRun(prefix, parse_config_command(test), ScopeKind.FULL_SUITE, 'pytest: conftest touched'),
    )


_ALPHA_RUNS = _module_runs(
    'alpha',
    lint='uv run --project alpha ruff check alpha/ && python3 tools/check.py alpha',
    type_check='uv run --project alpha pyright alpha',
    test='uv run --project alpha pytest alpha/tests -q',
)
_BETA_RUNS = _module_runs(
    'beta',
    lint='uv run --project beta ruff check beta',
    type_check='uv run --project beta pyright beta',
    test='uv run --project beta pytest beta -q',
)
_ALPHA_TEST_RUN = _ALPHA_RUNS[2]
_PLAN = VerifyPlan(runs=_ALPHA_RUNS + _BETA_RUNS)
_TOOLCHAIN = toolchain.ToolchainFingerprint(
    role='merge',
    interpreter='CPython 3.13.9',
    uv_lock_blob=None,
    tool_versions=(('ruff', 'uv.lock', '0.15.9'),),
    verify_env=(('DF_VERIFY_ROLE', 'merge'),),
    preprovision_command='uv sync --all-packages',
    orchestrator_code='c' * 40,
)
_ALPHA = {verify_evidence.module_leg_name('alpha', leg) for leg in Leg}
_BETA = {verify_evidence.module_leg_name('beta', leg) for leg in Leg}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _edit(repo: Path, path: str, text: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, 'add', '-A')
    return _git(repo, 'write-tree')


async def _unit_keys(
    repo: Path, tree: str, plan: VerifyPlan = _PLAN, fingerprint: toolchain.ToolchainFingerprint = _TOOLCHAIN,
) -> dict[str, keys.UnitKey]:
    inputs = await keys.read_tree_inputs(repo, tree, plan)
    return {unit.unit_id: unit for unit in keys.module_unit_keys(inputs, plan, fingerprint)}


async def _keys_for(repo: Path, tree: str, **kwargs) -> dict[str, str]:
    return {unit_id: unit.key for unit_id, unit in (await _unit_keys(repo, tree, **kwargs)).items()}


def _moved(before: dict[str, str], after: dict[str, str]) -> set[str]:
    assert set(before) == set(after)
    return {unit_id for unit_id in before if before[unit_id] != after[unit_id]}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', _KEYS_SEED)


@pytest.fixture
def tree(repo: Path) -> str:
    return _git(repo, 'rev-parse', 'HEAD^{tree}')


class TestModuleUnitKeys:
    @pytest.mark.asyncio
    async def test_unit_ids_are_the_module_leg_names(self, repo, tree):
        assert set(await _keys_for(repo, tree)) == _ALPHA | _BETA

    @pytest.mark.asyncio
    async def test_a_change_outside_every_module_moves_no_key(self, repo, tree):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, _edit(repo, 'docs/notes.md', '# edited\n'))

        assert _moved(before, after) == set()

    @pytest.mark.asyncio
    async def test_a_dependent_module_change_moves_only_its_own_keys(self, repo, tree):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, _edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n'))

        assert _moved(before, after) == _BETA

    @pytest.mark.asyncio
    async def test_a_dependency_change_moves_its_dependents_keys_too(self, repo, tree):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, _edit(repo, 'alpha/src/alpha/__init__.py', 'A = 2\n'))

        assert _moved(before, after) == _ALPHA | _BETA

    @pytest.mark.asyncio
    @pytest.mark.parametrize('root_file', ['uv.lock', 'conftest.py'])
    async def test_a_root_file_change_moves_every_key(self, repo, tree, root_file):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, _edit(repo, root_file, '# edited\n'))

        assert _moved(before, after) == _ALPHA | _BETA

    @pytest.mark.asyncio
    async def test_a_path_the_command_names_moves_only_that_leg(self, repo, tree):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, _edit(repo, 'tools/check.py', 'print("edited")\n'))

        assert _moved(before, after) == {'module:alpha:lint'}

    @pytest.mark.asyncio
    async def test_a_helper_named_inside_another_member_keys_that_file_not_its_member(self, repo, tree):
        plan = VerifyPlan(runs=(PlannedRun(
            'alpha', parse_config_command('uv run --project alpha ruff check alpha && python3 beta/scripts/check.py'),
            ScopeKind.FULL_SUITE, 'lint: structural change',
        ),))
        before = await _keys_for(repo, tree, plan=plan)

        unrelated = await _keys_for(repo, _edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n'), plan=plan)
        helper = await _keys_for(repo, _edit(repo, 'beta/scripts/check.py', 'print("edited")\n'), plan=plan)

        assert _moved(before, unrelated) == set()
        assert _moved(before, helper) == {'module:alpha:lint'}

    @pytest.mark.asyncio
    async def test_a_clause_with_a_cwd_reads_its_words_in_that_cwd_only(self, repo, tree):
        plan = VerifyPlan(runs=_module_runs(
            'alpha',
            lint='uv run --directory alpha ruff check src/ tests/ && python3 tools/check.py alpha',
            type_check='uv run --directory alpha pyright src/ tests/',
            test='uv run --directory alpha pytest tests/ -q',
        ))
        before = await _unit_keys(repo, tree, plan=plan)

        after = await _keys_for(repo, _edit(repo, 'tests/scripts/test_s.py', '# edited\n'), plan=plan)

        assert _moved({unit_id: unit.key for unit_id, unit in before.items()}, after) == set()
        assert dict(before['module:alpha:test'].inputs)['tree:alpha/tests'] == (
            _git(repo, 'rev-parse', f'{tree}:alpha/tests')
        )

    @pytest.mark.asyncio
    async def test_a_module_in_no_workspace_member_is_keyed_on_the_whole_tree(self, repo, tree):
        plan = VerifyPlan(runs=(PlannedRun(
            'tools', parse_config_command('python3 tools/check.py'), ScopeKind.FULL_SUITE, 'lint: structural change',
        ),))
        before = (await _unit_keys(repo, tree, plan=plan))['module:tools:lint']

        after = await _keys_for(repo, _edit(repo, 'docs/notes.md', '# edited\n'), plan=plan)

        assert dict(before.inputs)['tree:.'] == tree
        assert after['module:tools:lint'] != before.key

    @pytest.mark.asyncio
    async def test_a_plan_with_two_runs_for_one_unit_is_rejected_naming_it(self, repo, tree):
        plan = VerifyPlan(runs=(_ALPHA_TEST_RUN, _full_suite_test_run('pytest: structural change')))
        inputs = await keys.read_tree_inputs(repo, tree, plan)

        with pytest.raises(ValueError, match="'module:alpha:test'"):
            keys.module_unit_keys(inputs, plan, _TOOLCHAIN)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('override', [
        {'role': 'task'},
        {'preprovision_command': ''},
        {'interpreter': 'CPython 3.14.0'},
        {'verify_env': (('DF_VERIFY_ROLE', 'merge'), ('EXTRA', '1'))},
        {'orchestrator_code': 'd' * 40},
    ])
    async def test_a_toolchain_change_moves_every_key(self, repo, tree, override):
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, tree, fingerprint=dataclasses.replace(_TOOLCHAIN, **override))

        assert _moved(before, after) == _ALPHA | _BETA

    @pytest.mark.asyncio
    async def test_a_narrower_scope_of_the_same_command_moves_only_that_leg(self, repo, tree):
        assert _ALPHA_TEST_RUN.cmd is not None
        scoped = PlannedRun(
            'alpha', scope_to(_ALPHA_TEST_RUN.cmd, ['alpha/tests/test_a.py']), ScopeKind.FILE_SCOPED,
            'pytest: scoped to touched tests', ('alpha/tests/test_a.py',),
        )
        scoped_plan = VerifyPlan(runs=tuple(scoped if run is _ALPHA_TEST_RUN else run for run in _PLAN.runs))
        before = await _keys_for(repo, tree)

        after = await _keys_for(repo, tree, plan=scoped_plan)

        assert _moved(before, after) == {'module:alpha:test'}

    @pytest.mark.asyncio
    async def test_a_key_records_its_inputs_for_audit(self, repo, tree):
        unit = (await _unit_keys(repo, tree))['module:alpha:test']
        inputs = dict(unit.inputs)

        assert inputs['tree:alpha'] == _git(repo, 'rev-parse', f'{tree}:alpha')
        assert inputs['spec'] == verify_evidence.run_spec_hash(_ALPHA_TEST_RUN.to_dict())
        assert unit.spec_hash == inputs['spec']
        for name in ('conftest.py', 'pyproject.toml', 'uv.lock'):
            assert inputs[f'root:{name}'] == _git(repo, 'rev-parse', f'{tree}:{name}')
        assert inputs['toolchain.role'] == 'merge'
        assert unit.tier is UnitTier.MODULE

    @pytest.mark.asyncio
    async def test_runs_that_name_no_module_leg_get_no_key(self, repo, tree):
        assert _ALPHA_TEST_RUN.cmd is not None
        unkeyed = VerifyPlan(runs=(
            PlannedRun('alpha', None, ScopeKind.SKIPPED, 'pytest: no test_command configured'),
            PlannedRun('__fallback__', _ALPHA_TEST_RUN.cmd, ScopeKind.FULL_SUITE, 'pytest: conftest touched'),
            PlannedRun('alpha', _ALPHA_TEST_RUN.cmd, ScopeKind.FULL_SUITE, 'no tool prefix'),
        ))

        inputs = await keys.read_tree_inputs(repo, tree, unkeyed)

        assert keys.module_unit_keys(inputs, unkeyed, _TOOLCHAIN) == ()

    @pytest.mark.asyncio
    async def test_an_absent_named_path_is_keyed_and_its_creation_moves_the_key(self, repo, tree):
        naming_absent = VerifyPlan(runs=(PlannedRun(
            'alpha', parse_config_command('uv run --project alpha pytest missing/x.py'),
            ScopeKind.FULL_SUITE, 'pytest: conftest touched',
        ),))
        before = await _unit_keys(repo, tree, plan=naming_absent)

        after = await _unit_keys(repo, _edit(repo, 'missing/x.py', 'X = 1\n'), plan=naming_absent)

        assert dict(before['module:alpha:test'].inputs)['tree:missing/x.py'] == 'absent'
        assert before['module:alpha:test'].key != after['module:alpha:test'].key

    @pytest.mark.asyncio
    async def test_scoped_targets_holding_unicode_whitespace_are_keyed_present_and_absent(self, repo):
        assert _ALPHA_TEST_RUN.cmd is not None
        present, absent = 'alpha/tests/test\xa0a.py', 'alpha/tests/test\x0ba.py'
        tree = _edit(repo, present, 'def test_a():\n    pass\n')
        scoped = VerifyPlan(runs=(PlannedRun(
            'alpha', scope_to(_ALPHA_TEST_RUN.cmd, [present, absent]), ScopeKind.FILE_SCOPED,
            'pytest: scoped to touched tests', (present, absent),
        ),))

        inputs = dict((await _unit_keys(repo, tree, plan=scoped))['module:alpha:test'].inputs)

        assert inputs[f'tree:{present}'] == _git(repo, 'rev-parse', f'{tree}:{present}')
        assert inputs[f'tree:{absent}'] == 'absent'


class TestCommandPathCandidates:
    def test_a_shell_payload_is_split_and_never_itself_a_path(self):
        run = PlannedRun(
            'alpha', parse_config_command("bash -c 'pytest tests/x.py -q'"), ScopeKind.FULL_SUITE,
            'pytest: conftest touched',
        )

        paths = {candidate.path for candidate in keys.command_path_candidates(run)}

        assert 'tests/x.py' in paths
        assert 'pytest tests/x.py -q' not in paths
        assert not [path for path in paths if set(path) & set(' \t\r\n')]

    @pytest.mark.parametrize('ch', ['\xa0', '\x0b', '\x0c', '\x85', '\u2028', '\u3000'])
    def test_a_scoped_target_holding_non_lexer_whitespace_is_one_path(self, ch):
        assert _TEST_CMD is not None
        target = f'alpha/tests/test{ch}a.py'
        run = PlannedRun(
            'alpha', scope_to(_TEST_CMD, [target]), ScopeKind.FILE_SCOPED,
            'pytest: scoped to touched tests', (target,),
        )

        assert keys.CommandPath(target, True) in keys.command_path_candidates(run)

    def test_a_payload_naming_a_path_with_non_lexer_whitespace_keeps_that_path(self):
        run = PlannedRun(
            'alpha', parse_config_command('bash -c \'pytest "alpha/tests/a\xa0b.py" -q\''), ScopeKind.FULL_SUITE,
            'pytest: conftest touched',
        )

        assert keys.CommandPath('alpha/tests/a\xa0b.py', True) in keys.command_path_candidates(run)


class TestUnitKeyInvariants:
    def test_a_key_that_is_not_the_digest_of_its_inputs_is_rejected(self):
        with pytest.raises(ValueError, match='0{64}.*[0-9a-f]{64}'):
            keys.UnitKey(
                unit_id='module:alpha:test', tier=UnitTier.MODULE, key='0' * 64,
                spec_hash='s', inputs=(('spec', 's'),),
            )

    def test_a_unit_id_of_another_tier_is_rejected(self):
        with pytest.raises(ValueError, match="'tree'.*module"):
            keys.UnitKey.of('tree', UnitTier.MODULE, 's', (('spec', 's'),))

    @pytest.mark.parametrize('inputs', [(('tree:alpha', 'a'),), (('spec', 't'), ('tree:alpha', 'a'))])
    def test_a_spec_hash_that_is_not_its_spec_input_is_rejected(self, inputs):
        with pytest.raises(ValueError, match="spec_hash 's'"):
            keys.UnitKey.of('module:alpha:test', UnitTier.MODULE, 's', inputs)

    @pytest.mark.parametrize('inputs', [
        (('tree:b', 'x'), ('spec', 's'), ('tree:a', 'y')),
        (('spec', 's'), ('tree:a', 'x'), ('tree:a', 'y')),
    ])
    def test_inputs_out_of_order_or_with_a_repeated_name_are_rejected(self, inputs):
        with pytest.raises(ValueError, match="not sorted with unique names.*'tree:a'"):
            keys.UnitKey('module:alpha:test', UnitTier.MODULE, keys.digest(inputs), 's', inputs)


_TEST_FILE = 'alpha/tests/test_a.py'
_READ = 'alpha/src/alpha/__init__.py'
_RECORDED = keys.RecordedInputs(
    test_file=(_TEST_FILE, 'f' * 40),
    conftest_chain=(('conftest.py', 'c' * 40),),
    reads=(_READ,),
)
_CURRENT = {_TEST_FILE: 'f' * 40, 'conftest.py': 'c' * 40, _READ: 'a' * 40}


def _test_key(current: Mapping[str, str | None]) -> keys.UnitKey | None:
    return keys.test_unit_key(f'{_TEST_FILE}::test_a', _RECORDED, current, _ALPHA_TEST_RUN, _TOOLCHAIN)


class TestTreeAndTestKeys:
    def test_the_tree_key_names_the_tree_tier(self):
        unit = keys.tree_unit_key('t' * 40, 'p' * 64, _TOOLCHAIN)

        assert unit.unit_id == 'tree'
        assert unit.tier is UnitTier.TREE

    @pytest.mark.parametrize('changed', [
        {'tree_oid': 'u' * 40},
        {'plan_hash': 'q' * 64},
        {'fingerprint': dataclasses.replace(_TOOLCHAIN, role='task')},
    ])
    def test_the_tree_key_moves_with_tree_plan_and_role(self, changed):
        base = {'tree_oid': 't' * 40, 'plan_hash': 'p' * 64, 'fingerprint': _TOOLCHAIN}
        moved = {**base, **changed}

        assert keys.tree_unit_key(base['tree_oid'], base['plan_hash'], base['fingerprint']).key != (
            keys.tree_unit_key(moved['tree_oid'], moved['plan_hash'], moved['fingerprint']).key
        )

    def test_a_matching_recording_keys_the_test(self):
        unit = _test_key(_CURRENT)

        assert unit is not None
        assert unit.unit_id == f'test:{_TEST_FILE}::test_a'
        assert unit.tier is UnitTier.TEST

    @pytest.mark.parametrize('path', [_TEST_FILE, 'conftest.py'])
    def test_a_moved_test_file_or_conftest_is_no_key(self, path):
        assert _test_key({**_CURRENT, path: 'e' * 40}) is None

    def test_a_moved_recorded_read_moves_the_key(self):
        before = _test_key(_CURRENT)
        after = _test_key({**_CURRENT, _READ: 'b' * 40})

        assert before is not None and after is not None
        assert before.key != after.key
