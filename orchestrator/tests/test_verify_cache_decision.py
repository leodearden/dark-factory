"""The verify-result cache decision: which planned units replay a recorded green (task 5653)."""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import patch

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import git_env_with_ceiling
from _verify_cache_fixtures import (
    LOG_MODES,
    MARKERS_ENV,
    ON_MODES,
    TEST_CODE,
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    clean_unscoped_gate,
    put_pytest_on_path,
)

from orchestrator import deployed_sha, verify_plan
from orchestrator.config import (
    ModuleConfig,
    OrchestratorConfig,
    VerifyCacheConfig,
)
from orchestrator.deployed_sha import DeployedSha
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.verify import (
    MERGE_JUNIT_DIR,
    VerifyResult,
    is_no_verdict_summary,
    run_scoped_verification,
)
from orchestrator.verify_cache.keys import TREE_UNIT_ID, UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import VerifyEvidenceRef, record_greens
from orchestrator.verify_cache.registry import Entry, Registry, registry_db_path
from orchestrator.verify_cache.replay import CacheAccess, ReplayPlan, plan_replay
from orchestrator.verify_evidence import (
    Leg,
    UnitOutcome,
    UnitTier,
    VerifyUnit,
    module_leg_name,
    module_leg_unit,
)
from orchestrator.verify_plan import VerifyPlan
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
    VerifyRunnerPool,
    build_merge_verify_spec,
)


def _units(prefix: str) -> set[str]:
    return {module_leg_name(prefix, leg) for leg in Leg}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args], cwd=repo,
        env=git_env_with_ceiling(repo), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_edit(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, 'add', path)
    _git(repo, 'commit', '-q', '-m', f'edit {path}')


def _full_suite_plan(modules: Sequence[ModuleConfig], role: VerifyRole = 'merge') -> VerifyPlan:
    plan_role: Literal['merge', 'task'] = 'task' if role == 'task' else 'merge'
    return VerifyPlan(runs=tuple(
        run for mc in modules for run in verify_plan._derive_full_suite_runs(mc, plan_role)
    ))


def _no_env(_mc: ModuleConfig | None) -> Mapping[str, str]:
    return {}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', WORKSPACE_SEED)


@pytest.fixture
def markers(tmp_path: Path) -> Path:
    return tmp_path / 'markers.txt'


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


@pytest.fixture
def access(registry: Registry) -> CacheAccess:
    return CacheAccess(registry=registry, orchestrator_code=TEST_CODE)


@pytest.fixture
def modules(markers: Path) -> list[ModuleConfig]:
    return [cache_module('alpha', markers), cache_module('beta', markers)]


async def _replay(
    repo: Path,
    modules: Sequence[ModuleConfig | None],
    access: CacheAccess,
    *,
    modes: VerifyCacheConfig = ON_MODES,
    role: VerifyRole = 'merge',
    plan: VerifyPlan | None = None,
    config: OrchestratorConfig | None = None,
) -> ReplayPlan:
    if plan is None and all(mc is not None for mc in modules):
        plan = _full_suite_plan([mc for mc in modules if mc is not None], role)
    replay = await plan_replay(
        repo, plan, modules, config=config or cache_config(repo, modes), role=role, access=access,
        leg_env=_no_env,
    )
    assert replay is not None
    return replay


async def _record(registry: Registry, replay: ReplayPlan, unit_ids: set[str]) -> dict[str, Entry]:
    recorded = {}
    for unit_id in sorted(unit_ids):
        key = replay.modules[unit_id].key
        assert key is not None
        recorded[unit_id] = await registry.record_green(key, evidence_event_id=7, run_id='r', sha='s')
    return recorded


async def _record_tree(registry: Registry, replay: ReplayPlan) -> Entry:
    assert replay.tree.key is not None
    return await registry.record_green(replay.tree.key, evidence_event_id=9, run_id='r', sha='s')


def _decisions(replay: ReplayPlan, unit_ids: set[str]) -> set[Decision]:
    return {replay.modules[unit_id].decision.decision for unit_id in unit_ids}


class TestPlanReplay:
    @pytest.mark.asyncio
    async def test_row_1_an_empty_registry_misses_every_unit_with_a_key(self, repo, modules, access):
        replay = await _replay(repo, modules, access)

        assert set(replay.modules) == _units('alpha') | _units('beta')
        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.MISS
            assert isinstance(decided.key, UnitKey)
            assert decided.decision.reason
        assert replay.tree.decision.decision is Decision.MISS
        assert replay.tree.key is not None
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_row_2_an_edit_outside_alpha_replays_alpha(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), _units('alpha'))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert _decisions(replay, _units('beta')) == {Decision.MISS}
        assert replay.replayed_legs('alpha') == frozenset(Leg)
        assert replay.replayed_legs('beta') == frozenset()

    @pytest.mark.asyncio
    async def test_row_3_an_edit_to_alpha_misses_alpha_and_its_dependent(
        self, repo, modules, access, registry,
    ):
        first = await _replay(repo, modules, access)
        await _record(registry, first, _units('alpha') | _units('beta'))
        _commit_edit(repo, 'alpha/src/alpha/__init__.py', 'A = 2\n')

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, _units('alpha') | _units('beta')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_6_a_task_role_file_scoped_green_does_not_serve_the_merge_plan(
        self, repo, modules, access, registry,
    ):
        alpha = modules[0]
        task_plan = verify_plan.derive_verify_plan(
            ['alpha/src/alpha/__init__.py'], [alpha], cache_config(repo),
            lambda path: (repo / path).read_text(), role='task',
        )
        task_replay = await _replay(repo, [alpha], access, role='task', plan=task_plan)
        keyed = {unit_id for unit_id, decided in task_replay.modules.items() if decided.key is not None}
        assert keyed
        await _record(registry, task_replay, keyed)

        replay = await _replay(repo, modules, access)

        assert _decisions(replay, _units('alpha')) == {Decision.MISS}

    @pytest.mark.asyncio
    async def test_row_9_log_mode_would_hit_with_the_entry(self, repo, modules, access, registry):
        recorded = await _record(registry, await _replay(repo, modules, access), _units('alpha'))

        replay = await _replay(repo, modules, access, modes=LOG_MODES)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.WOULD_HIT
            assert replay.modules[unit_id].decision.entry == recorded[unit_id]
        assert replay.replayed_legs('alpha') == frozenset()

    @pytest.mark.asyncio
    async def test_row_10_an_unreadable_registry_is_unavailable_naming_it(self, repo, modules, tmp_path):
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=TEST_CODE)

        replay = await _replay(repo, modules, access)

        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert str(db_path) in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_runner_consults_no_registry(self, repo, modules):
        replay = await _replay(repo, modules, await CacheAccess.on_runner())

        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert 'no_registry_on_runner' in decided.decision.reason

    @pytest.mark.asyncio
    async def test_a_working_copy_that_differs_from_head_is_unavailable(self, repo, modules, access):
        (repo / 'stray.py').write_text('x = 1\n')

        replay = await _replay(repo, modules, access)

        assert set(replay.modules) == _units('alpha') | _units('beta')
        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            assert '1 path' in decided.decision.reason

    @pytest.mark.parametrize(('stray', 'decisions'), [
        pytest.param((), {Decision.MISS}, id='only-own-outputs'),
        pytest.param(('stray.py',), {Decision.UNAVAILABLE}, id='own-outputs-and-a-stray'),
    ])
    @pytest.mark.asyncio
    async def test_the_verify_s_own_outputs_are_not_a_working_copy_change(
        self, repo, modules, access, stray, decisions,
    ):
        (repo / MERGE_JUNIT_DIR).mkdir()
        (repo / MERGE_JUNIT_DIR / 'report.alpha.xml').write_text('<testsuites/>')
        for path in stray:
            (repo / path).write_text('x = 1\n')

        replay = await plan_replay(
            repo, _full_suite_plan(modules), modules, config=cache_config(repo), role='merge', access=access,
            leg_env=_no_env, verify_outputs=(MERGE_JUNIT_DIR,),
        )

        assert replay is not None
        decided = (*replay.modules.values(), replay.tree)
        assert {unit.decision.decision for unit in decided} == decisions
        if stray:
            assert {'1 path' in unit.decision.reason for unit in decided} == {True}

    @pytest.mark.asyncio
    async def test_mode_off_decides_nothing_and_touches_no_registry(self, repo, modules, tmp_path):
        db_path = tmp_path / 'never_created.db'
        access = CacheAccess(registry=Registry(db_path), orchestrator_code=TEST_CODE)

        replay = await plan_replay(
            repo, _full_suite_plan(modules), modules, config=cache_config(repo, VerifyCacheConfig()),
            role='merge', access=access, leg_env=_no_env,
        )

        assert replay is None
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_a_tree_green_replays_every_module_unit(self, repo, modules, access, registry):
        tree_entry = await _record_tree(registry, await _replay(repo, modules, access))

        replay = await _replay(repo, modules, access)

        assert replay.tree.decision.decision is Decision.HIT
        assert replay.tree_hit
        for decided in replay.modules.values():
            assert decided.decision.decision is Decision.HIT
            assert decided.decision.entry == tree_entry

    @pytest.mark.asyncio
    async def test_a_never_cache_module_misses_and_keeps_the_tree_from_hitting(
        self, repo, markers, access, registry,
    ):
        never = [cache_module('alpha', markers, verify_cache='never'), cache_module('beta', markers)]
        first = await _replay(repo, never, access)
        await _record(registry, first, _units('alpha') | _units('beta'))
        await _record_tree(registry, first)

        replay = await _replay(repo, never, access)

        for unit_id in _units('alpha'):
            assert replay.modules[unit_id].decision.decision is Decision.MISS
            assert 'never-cache' in replay.modules[unit_id].decision.reason
        assert replay.tree.decision.decision is not Decision.HIT
        assert not replay.tree_hit

    @pytest.mark.asyncio
    async def test_a_plan_less_run_keys_its_tree_on_the_executed_commands(self, repo, access, markers):
        config = cache_config(repo, test_command='pytest -q', lint_command=f'echo lint >> {markers}')
        other = cache_config(repo, test_command='pytest -q -x', lint_command=f'echo lint >> {markers}')

        replay = await _replay(repo, [None], access, config=config)
        changed = await _replay(repo, [None], access, config=other)

        assert all(decided.key is None for decided in replay.modules.values())
        assert all('not keyed' in decided.decision.reason for decided in replay.modules.values())
        assert replay.tree.key is not None
        assert replay.tree.decision.decision is Decision.MISS
        assert changed.tree.key is not None
        assert changed.tree.key.key != replay.tree.key.key

    @pytest.mark.asyncio
    async def test_without_replayed_drops_hit_legs_and_nothing_left_is_none(
        self, repo, modules, access, registry,
    ):
        alpha, beta = modules
        await _record(registry, await _replay(repo, modules, access), _units('alpha'))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        replay = await _replay(repo, modules, access)

        assert replay.without_replayed(alpha) is None
        assert replay.without_replayed(beta) == beta


# ---------------------------------------------------------------------------
# The decision at the real fan-out seam: run_scoped_verification's execution funnel.

_TASK_ON_MERGE_OFF = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.OFF)


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    """Real legs: a bare ``pytest`` whose junit the verify reads, and every leg leaves a marker."""
    put_pytest_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv(MARKERS_ENV, str(markers))


def _legs(prefix: str) -> set[str]:
    return {f'{prefix}:{leg}' for leg in Leg}


def _ran_legs(markers: Path) -> set[str]:
    """The ``<module>:<leg>`` of every leg that executed since *markers* was last cleared."""
    if not markers.exists():
        return set()
    return {line.split()[0] for line in markers.read_text().splitlines() if line.strip()}


def _clear(markers: Path) -> None:
    markers.unlink(missing_ok=True)


async def _gate(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess | None,
) -> VerifyResult:
    return await run_scoped_verification(
        repo, config, list(modules), role='merge', is_merge_verify=True, force_workspace=True,
        verify_cache=access,
    )


async def _task_verify(
    repo: Path, config: OrchestratorConfig, modules: Sequence[ModuleConfig], access: CacheAccess,
    task_files: list[str],
) -> VerifyResult:
    """A task verify as the workflow runs one: its junit cost report is what admits its test leg."""
    return await run_scoped_verification(
        repo, config, list(modules), task_files, role='task', verify_cache=access,
        attempt_id=1, task_id='5653', archive_root=repo.parent / 'verify-archive',
    )


async def _record_run(registry: Registry, result: VerifyResult, *, event_id: int = 41) -> dict[str, Entry]:
    assert result.passed, result.summary
    assert result.cache_record is not None
    outcome = await record_greens(
        registry, result.cache_record, flake_suppressed=False,
        evidence=VerifyEvidenceRef(event_id, 'r', 'sha-first', True),
    )
    return {entry.unit_id: entry for entry in outcome.recorded}


def _by_unit(result: VerifyResult) -> dict[str, VerifyUnit]:
    assert result.units is not None
    return {unit.unit: unit for unit in result.units}


def _outcomes(result: VerifyResult, unit_ids: set[str]) -> set[UnitOutcome]:
    units = _by_unit(result)
    return {units[unit_id].outcome for unit_id in unit_ids}


def _replay_hint(unit_id: str, entry: Entry) -> str:
    return f'replayed {unit_id} from entry {entry.entry_id} (evidence event {entry.evidence_event_id})'


def _tree_key(result: VerifyResult) -> UnitKey:
    assert result.cache_record is not None and result.cache_record.tree is not None
    key = result.cache_record.tree.key
    assert key is not None
    return key


@pytest.mark.usefixtures('seam')
class TestFanOutSeam:
    @pytest.mark.asyncio
    async def test_an_edit_outside_alpha_replays_alpha_and_runs_beta(
        self, repo, modules, access, registry, markers,
    ):
        config = cache_config(repo)
        first = await _gate(repo, config, modules, access)
        recorded = await _record_run(registry, first)
        assert _units('alpha') <= set(recorded)
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == _legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.ON
        assert result.plan == first.plan == _full_suite_plan(modules).to_dict()
        units = _by_unit(result)
        for unit_id in _units('alpha'):
            assert units[unit_id].outcome is UnitOutcome.CACHE_HIT
            assert units[unit_id].evidence_event_id == 41
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert _outcomes(result, _units('beta')) == {UnitOutcome.RAN_GREEN}

    @pytest.mark.asyncio
    async def test_a_partly_replayed_module_runs_only_its_changed_leg_and_names_each_replay(
        self, repo, access, registry, markers,
    ):
        _commit_edit(repo, 'lint/alpha.cfg', 'strict = true\n')
        alpha = dataclasses.replace(
            cache_module('alpha', markers), lint_command=f'echo alpha:lint >> {markers} && cat lint/alpha.cfg',
        )
        modules = [alpha, cache_module('beta', markers)]
        config = cache_config(repo)
        recorded = await _record_run(registry, await _gate(repo, config, modules, access))
        _commit_edit(repo, 'lint/alpha.cfg', 'strict = false\n')
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        alpha_lint = module_leg_name('alpha', Leg.LINT)
        alpha_replayed = _units('alpha') - {alpha_lint}
        assert _ran_legs(markers) == {'alpha:lint'}
        assert result.passed, result.summary
        assert _outcomes(result, {alpha_lint}) == {UnitOutcome.RAN_GREEN}
        assert _outcomes(result, alpha_replayed | _units('beta')) == {UnitOutcome.CACHE_HIT}
        for unit_id in alpha_replayed:
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint

    @pytest.mark.asyncio
    async def test_a_wholly_replayed_module_reads_as_a_cache_hit_not_a_run(
        self, repo, modules, access, registry, markers,
    ):
        config = cache_config(repo)
        recorded = await _record_run(registry, await _gate(repo, config, modules, access))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        result = await _gate(repo, config, modules[:1], access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        assert result.category == 'cache_hit'
        assert result.duration_secs == 0.0
        assert (result.test_output, result.lint_output, result.type_output) == ('', '', '')
        for unit_id in _units('alpha'):
            assert _replay_hint(unit_id, recorded[unit_id]) in result.cause_hint
        assert _outcomes(result, _units('alpha')) == {UnitOutcome.CACHE_HIT}

    @pytest.mark.asyncio
    async def test_a_recorded_tree_green_runs_no_leg_at_all(self, repo, modules, access, registry, markers):
        config = cache_config(repo)
        first = await _gate(repo, config, modules, access)
        assert TREE_UNIT_ID in await _record_run(registry, first)
        tree_entry = await registry.record_green(_tree_key(first), evidence_event_id=99, run_id='r', sha='sha-tree')
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        units = _by_unit(result)
        assert set(units) == _units('alpha') | _units('beta') | {TREE_UNIT_ID}
        for unit in units.values():
            assert unit.outcome is UnitOutcome.CACHE_HIT
            assert unit.evidence_event_id == tree_entry.evidence_event_id == 99

    @pytest.mark.asyncio
    async def test_log_mode_runs_every_leg_and_reads_would_hit(self, repo, modules, access, registry, markers):
        config = cache_config(repo, LOG_MODES)
        await _record_run(registry, await _gate(repo, config, modules, access))
        _clear(markers)

        result = await _gate(repo, config, modules, access)

        assert _ran_legs(markers) == _legs('alpha') | _legs('beta')
        assert result.passed, result.summary
        assert result.cache_mode is CacheMode.LOG
        assert {unit.outcome for unit in _by_unit(result).values()} == {UnitOutcome.WOULD_HIT}

    @pytest.mark.asyncio
    async def test_outside_the_cache_the_verify_is_as_before(self, repo, modules, tmp_path, markers):
        result = await _gate(repo, cache_config(repo, LOG_MODES), modules, None)

        assert _ran_legs(markers) == _legs('alpha') | _legs('beta')
        assert result.passed, result.summary
        expected = sorted(
            (module_leg_unit(prefix, leg, 0) for prefix in ('alpha', 'beta') for leg in Leg),
            key=lambda unit: unit.unit,
        )
        assert sorted(result.units or (), key=lambda unit: unit.unit) == expected
        assert {unit.reason for unit in expected} == {None}
        assert result.cache_mode is CacheMode.OFF
        assert result.cache_record is None
        assert result.plan == _full_suite_plan(modules).to_dict()
        assert not (tmp_path / 'verify_cache.db').exists()

    @pytest.mark.asyncio
    async def test_each_role_reads_its_own_mode(self, repo, modules, access, registry, markers):
        (repo / '.task').mkdir()
        config = cache_config(repo, _TASK_ON_MERGE_OFF)
        task_files = ['alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py']
        recorded = await _record_run(registry, await _task_verify(repo, config, modules, access, task_files))
        alpha_units = {unit_id for unit_id in recorded if unit_id.startswith('module:alpha:')}
        assert alpha_units
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        _clear(markers)

        task = await _task_verify(repo, config, modules, access, task_files)

        assert task.passed, task.summary
        assert task.cache_mode is CacheMode.ON
        assert _outcomes(task, alpha_units) == {UnitOutcome.CACHE_HIT}
        assert not {leg for leg in _ran_legs(markers) if leg.startswith('alpha:')}
        _clear(markers)

        merge = await _gate(repo, config, modules, access)

        assert merge.passed, merge.summary
        assert merge.cache_mode is CacheMode.OFF
        assert _legs('alpha') <= _ran_legs(markers)
        assert UnitOutcome.CACHE_HIT not in {unit.outcome for unit in _by_unit(merge).values()}

    @pytest.mark.asyncio
    async def test_a_plan_less_gate_replays_its_recorded_tree_green(self, repo, access, registry, markers):
        config = cache_config(
            repo, test_command='pytest alpha/tests beta/tests -q',
            lint_command=f'echo global:lint >> {markers}', type_check_command=f'echo global:type >> {markers}',
        )
        first = await _gate(repo, config, [], access)
        assert first.passed, first.summary
        assert first.plan is None
        await registry.record_green(_tree_key(first), evidence_event_id=77, run_id='r', sha='sha-first')
        _clear(markers)

        result = await _gate(repo, config, [], access)

        assert _ran_legs(markers) == set()
        assert result.passed, result.summary
        units = _by_unit(result)
        assert (units[TREE_UNIT_ID].outcome, units[TREE_UNIT_ID].evidence_event_id) == (UnitOutcome.CACHE_HIT, 77)
        assert {unit.outcome for unit in units.values()} == {UnitOutcome.CACHE_HIT}


# ---------------------------------------------------------------------------
# The merge gate as the lane dispatches it: the pool consumes what a verify
# replayed before the merge_verify row, then records its greens citing that row.


def _merge_pool(
    repo: Path,
    config: OrchestratorConfig,
    modules: Sequence[ModuleConfig],
    registry: Registry,
    store: EventStore,
    *,
    run_scoped: Any = run_scoped_verification,
) -> VerifyRunnerPool:
    local = LocalRunner(
        repo, config, list(modules), None,
        run_scoped=run_scoped, run_unscoped=clean_unscoped_gate,
        verify_cache=CacheAccess(registry, TEST_CODE),
    )
    return VerifyRunnerPool([local], event_store=store, task_id='5653', verify_cache_registry=registry)


def _spec(config: OrchestratorConfig, modules: Sequence[ModuleConfig]) -> MergeVerifySpec:
    return build_merge_verify_spec(config, list(modules), None)


def _head(repo: Path) -> str:
    return _git(repo, 'rev-parse', 'HEAD')


def _row_units(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit['unit']: unit for unit in row['data']['units']}


async def _entries_for(registry: Registry, result: VerifyResult) -> dict[str, Entry]:
    """The live entry of every keyed unit *result* settled, by unit id."""
    record = result.cache_record
    assert record is not None
    settled = (*record.units, *(() if record.tree is None else (record.tree,)))
    view = await registry.view([unit.key for unit in settled if unit.key is not None])
    return {entry.unit_id: entry for entry in view.entries.values()}


def _garble(db_path: Path) -> None:
    for sidecar in ('-wal', '-shm'):
        Path(f'{db_path}{sidecar}').unlink(missing_ok=True)
    db_path.write_bytes(b'garbage, not sqlite' * 64)


class _RemoteDouble:
    """A non-local runner that returns *result* as a remote verify would."""

    name = 'remote-double'
    is_local: ClassVar[bool] = False

    def __init__(self, result: VerifyResult) -> None:
        self._result = result

    async def health(self) -> bool:
        return True

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        return self._result


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='r')


@pytest.mark.usefixtures('seam')
class TestMergeGateRecordsAndConsumes:
    @pytest.mark.asyncio
    async def test_row_1_the_first_gate_records_every_admitted_unit_citing_its_row(
        self, repo, modules, registry, store,
    ):
        config = cache_config(repo)
        merge_sha = _head(repo)

        result = await _merge_pool(repo, config, modules, registry, store).dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        (row,) = store.fetch_events_by_type(EventType.merge_verify)
        units = _row_units(row)
        assert set(units) == _units('alpha') | _units('beta') | {TREE_UNIT_ID}
        for unit in units.values():
            assert unit['outcome'] == 'ran_green'
            assert unit['key']
        recorded = await _entries_for(registry, result)
        assert set(recorded) == set(units)
        for entry in recorded.values():
            assert (entry.evidence_event_id, entry.run_id, entry.sha) == (row['id'], 'r', merge_sha)

    @pytest.mark.asyncio
    async def test_row_2_the_second_gate_replays_alpha_and_consumes_it_before_returning(
        self, repo, modules, registry, store,
    ):
        config = cache_config(repo)
        pool = _merge_pool(repo, config, modules, registry, store)
        first = await pool.dispatch(_head(repo), _spec(config, modules))
        first_entries = await _entries_for(registry, first)
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        merge_sha = _head(repo)

        result = await pool.dispatch(merge_sha, _spec(config, modules))
        consumed = await registry.consumed_for(merge_sha)

        assert result.passed, result.summary
        first_row, row = store.fetch_events_by_type(EventType.merge_verify)
        units = _row_units(row)
        for unit_id in _units('alpha'):
            assert units[unit_id]['outcome'] == 'cache_hit'
            assert units[unit_id]['evidence_event_id'] == first_row['id']
        assert {entry.unit_id: entry.entry_id for entry in consumed} == {
            unit_id: first_entries[unit_id].entry_id for unit_id in _units('alpha')
        }

    @pytest.mark.asyncio
    async def test_row_9_log_mode_reads_would_hit_and_consumes_nothing(self, repo, modules, registry, store):
        config = cache_config(repo, LOG_MODES)
        pool = _merge_pool(repo, config, modules, registry, store)
        await pool.dispatch(_head(repo), _spec(config, modules))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        merge_sha = _head(repo)

        result = await pool.dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        units = _row_units(store.fetch_events_by_type(EventType.merge_verify)[-1])
        assert {units[unit_id]['outcome'] for unit_id in _units('alpha')} == {'would_hit'}
        assert await registry.consumed_for(merge_sha) == ()

    @pytest.mark.asyncio
    async def test_a_replay_the_registry_cannot_consume_is_no_verdict(
        self, repo, modules, registry, store, tmp_path,
    ):
        config = cache_config(repo)
        await _merge_pool(repo, config, modules, registry, store).dispatch(_head(repo), _spec(config, modules))
        _commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        async def _decide_then_garble(*args: Any, **kwargs: Any) -> VerifyResult:
            decided = await run_scoped_verification(*args, **kwargs)
            _garble(tmp_path / 'verify_cache.db')
            return decided

        pool = _merge_pool(repo, config, modules, registry, store, run_scoped=_decide_then_garble)
        result = await pool.dispatch(_head(repo), _spec(config, modules))

        assert _outcomes(result, _units('alpha')) == {UnitOutcome.CACHE_HIT}
        assert not result.passed
        assert is_no_verdict_summary(result.summary), result.summary
        assert store.fetch_events_by_type(EventType.merge_verify)[-1]['data']['passed'] is False

    @pytest.mark.asyncio
    async def test_a_red_gate_keeps_its_own_verdict_when_its_replays_cannot_be_consumed(
        self, repo, modules, registry, store, tmp_path,
    ):
        config = cache_config(repo)
        await _merge_pool(repo, config, modules, registry, store).dispatch(_head(repo), _spec(config, modules))
        _commit_edit(repo, 'beta/tests/test_beta.py', 'def test_beta():\n    assert False\n')
        decided: list[VerifyResult] = []

        async def _decide_then_garble(*args: Any, **kwargs: Any) -> VerifyResult:
            decided.append(await run_scoped_verification(*args, **kwargs))
            _garble(tmp_path / 'verify_cache.db')
            return decided[-1]

        pool = _merge_pool(repo, config, modules, registry, store, run_scoped=_decide_then_garble)
        result = await pool.dispatch(_head(repo), _spec(config, modules))

        red = decided[-1]
        assert _outcomes(red, _units('alpha')) == {UnitOutcome.CACHE_HIT}
        assert not red.passed
        assert red.category
        assert not is_no_verdict_summary(red.summary), red.summary
        assert (result.passed, result.category, result.summary) == (False, red.category, red.summary)

    @pytest.mark.asyncio
    async def test_a_non_local_runner_records_nothing(self, repo, modules, access, registry, store):
        config = cache_config(repo)
        remote_result = await _gate(repo, config, modules, access)
        assert remote_result.passed, remote_result.summary
        assert remote_result.cache_record is not None
        merge_sha = _head(repo)
        pool = VerifyRunnerPool(
            [_RemoteDouble(remote_result)], event_store=store, task_id='5653', verify_cache_registry=registry,
        )

        result = await pool.dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        assert await _entries_for(registry, result) == {}
        assert await registry.consumed_for(merge_sha) == ()


# ---------------------------------------------------------------------------
# The key's code identity: the process's DeployedSha, never wherever the code
# checkout's HEAD has moved since. D1 and D2 name no commit of any checkout.

_D1 = 'd' * 40
_D2 = 'e' * 40


def _deployed(sha: str = _D1, *, indeterminate: bool = False, dirty_wire_path: bool = False) -> DeployedSha:
    return DeployedSha(sha=sha, indeterminate=indeterminate, dirty_wire_path=dirty_wire_path)


def _keys(replay: ReplayPlan) -> dict[str, UnitKey | None]:
    return {TREE_UNIT_ID: replay.tree.key, **{unit_id: decided.key for unit_id, decided in replay.modules.items()}}


class TestCodeIdentity:
    def test_the_dispatcher_access_carries_its_deployed_sha(self, repo):
        assert CacheAccess.on_dispatcher(repo, deployed=_deployed(_D1)).orchestrator_code == _D1

    @pytest.mark.asyncio
    async def test_keys_follow_the_injected_identity(self, repo, modules, registry):
        d1 = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=_D1)))
        d1_again = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=_D1)))
        d2 = _keys(await _replay(repo, modules, CacheAccess(registry=registry, orchestrator_code=_D2)))

        assert set(d1) == set(d2) == {TREE_UNIT_ID} | _units('alpha') | _units('beta')
        assert all(key is not None for key in (*d1.values(), *d2.values()))
        assert d1 == d1_again
        for unit_id, key in d1.items():
            assert key != d2[unit_id], unit_id

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('seam')
    async def test_greens_recorded_under_one_deployed_sha_serve_only_that_sha(self, repo, modules, registry, markers):
        config = cache_config(repo)
        d1 = CacheAccess(registry=registry, orchestrator_code=_D1)
        await _record_run(registry, await _gate(repo, config, modules, d1))
        _clear(markers)

        replayed = await _gate(repo, config, modules, d1)

        assert _ran_legs(markers) == set()
        assert _outcomes(replayed, _units('alpha') | _units('beta')) == {UnitOutcome.CACHE_HIT}
        _clear(markers)

        rerun = await _gate(repo, config, modules, CacheAccess(registry=registry, orchestrator_code=_D2))

        assert _ran_legs(markers) == _legs('alpha') | _legs('beta')
        assert _outcomes(rerun, _units('alpha') | _units('beta')) == {UnitOutcome.RAN_GREEN}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('deployed', 'reason'), [
        pytest.param(None, ['no deployed sha'], id='none'),
        pytest.param(_deployed(indeterminate=True), ['indeterminate', _D1], id='indeterminate'),
        pytest.param(_deployed(dirty_wire_path=True), ['uncommitted', _D1], id='dirty-wire-path'),
    ])
    async def test_a_deployed_sha_that_cannot_name_the_code_makes_every_unit_unavailable(
        self, tmp_path, repo, modules, deployed, reason,
    ):
        elsewhere = tmp_path / 'elsewhere'
        access = CacheAccess.on_dispatcher(elsewhere, deployed=deployed)

        replay = await _replay(repo, modules, access)

        for decided in (*replay.modules.values(), replay.tree):
            assert decided.decision.decision is Decision.UNAVAILABLE
            for fragment in reason:
                assert fragment in decided.decision.reason
        assert not registry_db_path(elsewhere).exists()
        assert not elsewhere.exists()

    @pytest.mark.asyncio
    async def test_a_runner_keys_on_its_pinned_gate_checkout(self):
        access = await CacheAccess.on_runner()

        assert access.registry is None
        checkout = deployed_sha.resolve_code_checkout()
        assert checkout is not None
        assert access.orchestrator_code == deployed_sha.checkout_head(checkout)


# ---------------------------------------------------------------------------
# The composition root keys its one dispatcher access on its own DeployedSha.


def _harness_over(config: OrchestratorConfig, deployed: DeployedSha | None) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config, deployed_sha=deployed)


@pytest.mark.usefixtures('seam', 'code_default_config')
class TestHarnessKeysTheCacheOnItsDeployedSha:
    _TASK_FILES: ClassVar[tuple[str, ...]] = ('alpha/src/alpha/__init__.py', 'beta/src/beta/__init__.py')

    @pytest.fixture
    def config(self, repo: Path) -> OrchestratorConfig:
        """The registry lives under the project's ignored ``data/``, so recording keeps the working copy at HEAD."""
        _commit_edit(repo, '.gitignore', (repo / '.gitignore').read_text() + 'data/\n')
        (repo / '.task').mkdir()
        return cache_config(repo, LOG_MODES)

    async def _seed_d1_greens(self, repo: Path, config: OrchestratorConfig, modules: list[ModuleConfig]) -> set[str]:
        registry = Registry(registry_db_path(repo))
        access = CacheAccess(registry=registry, orchestrator_code=_D1)
        recorded = await _record_run(registry, await _task_verify(repo, config, modules, access, list(self._TASK_FILES)))
        return set(recorded) - {TREE_UNIT_ID}

    async def _module_units(
        self, harness: Harness, repo: Path, config: OrchestratorConfig, modules: list[ModuleConfig],
    ) -> dict[str, VerifyUnit]:
        result = await harness._task_verify_dispatcher(
            repo, config, modules, list(self._TASK_FILES), attempt_id=1, task_id='5653', archive_root=None,
        )
        assert result.passed, result.summary
        return {unit.unit: unit for unit in result.units or () if unit.tier is UnitTier.MODULE}

    @pytest.mark.asyncio
    async def test_a_harness_deployed_at_d1_would_replay_the_d1_greens(self, repo, config, modules):
        seeded = await self._seed_d1_greens(repo, config, modules)

        units = await self._module_units(_harness_over(config, _deployed(_D1)), repo, config, modules)

        assert set(units) == seeded == _units('alpha') | _units('beta')
        assert {unit.outcome for unit in units.values()} == {UnitOutcome.WOULD_HIT}

    @pytest.mark.asyncio
    async def test_a_harness_with_no_deployed_sha_keys_no_unit(self, repo, config, modules):
        await self._seed_d1_greens(repo, config, modules)

        units = await self._module_units(_harness_over(config, None), repo, config, modules)

        assert set(units) == _units('alpha') | _units('beta')
        for unit in units.values():
            assert unit.outcome is UnitOutcome.UNAVAILABLE, unit
            assert 'no deployed sha' in (unit.reason or ''), unit
