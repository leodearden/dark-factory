"""The merge gate as the lane dispatches it: the pool consumes what a verify replayed before its merge_verify row, then records its greens citing that row (task 5653)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from _git_fixtures import seed_repo
from _verify_cache_fixtures import (
    LOG_MODES,
    TEST_CODE,
    WORKSPACE_SEED,
    cache_config,
    cache_module,
    commit_edit,
    entries_for,
    head_of,
    merge_gate,
    merge_pool,
    module_units,
    outcomes_of,
    row_units,
    use_real_legs,
)

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.verify import VerifyResult, is_no_verdict_summary, run_scoped_verification
from orchestrator.verify_cache.keys import TREE_UNIT_ID
from orchestrator.verify_cache.registry import Registry
from orchestrator.verify_cache.replay import CacheAccess
from orchestrator.verify_evidence import UnitOutcome
from orchestrator.verify_runner import (
    MergeVerifySpec,
    VerifyRunnerPool,
    build_merge_verify_spec,
)


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


@pytest.fixture
def seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markers: Path) -> None:
    use_real_legs(tmp_path, monkeypatch, markers)


def _spec(config: OrchestratorConfig, modules: Sequence[ModuleConfig]) -> MergeVerifySpec:
    return build_merge_verify_spec(config, list(modules), None)


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
        merge_sha = head_of(repo)

        result = await merge_pool(repo, config, modules, registry, store).dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        (row,) = store.fetch_events_by_type(EventType.merge_verify)
        units = row_units(row)
        assert set(units) == module_units('alpha') | module_units('beta') | {TREE_UNIT_ID}
        for unit in units.values():
            assert unit['outcome'] == 'ran_green'
            assert unit['key']
        recorded = await entries_for(registry, result)
        assert set(recorded) == set(units)
        for entry in recorded.values():
            assert (entry.evidence_event_id, entry.run_id, entry.sha) == (row['id'], 'r', merge_sha)

    @pytest.mark.asyncio
    async def test_row_2_the_second_gate_replays_alpha_and_consumes_it_before_returning(
        self, repo, modules, registry, store,
    ):
        config = cache_config(repo)
        pool = merge_pool(repo, config, modules, registry, store)
        first = await pool.dispatch(head_of(repo), _spec(config, modules))
        first_entries = await entries_for(registry, first)
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        merge_sha = head_of(repo)

        result = await pool.dispatch(merge_sha, _spec(config, modules))
        consumed = await registry.consumed_for(merge_sha)

        assert result.passed, result.summary
        first_row, row = store.fetch_events_by_type(EventType.merge_verify)
        units = row_units(row)
        for unit_id in module_units('alpha'):
            assert units[unit_id]['outcome'] == 'cache_hit'
            assert units[unit_id]['evidence_event_id'] == first_row['id']
        assert {entry.unit_id: entry.entry_id for entry in consumed} == {
            unit_id: first_entries[unit_id].entry_id for unit_id in module_units('alpha')
        }

    @pytest.mark.asyncio
    async def test_row_9_log_mode_reads_would_hit_and_consumes_nothing(self, repo, modules, registry, store):
        config = cache_config(repo, LOG_MODES)
        pool = merge_pool(repo, config, modules, registry, store)
        await pool.dispatch(head_of(repo), _spec(config, modules))
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')
        merge_sha = head_of(repo)

        result = await pool.dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        units = row_units(store.fetch_events_by_type(EventType.merge_verify)[-1])
        assert {units[unit_id]['outcome'] for unit_id in module_units('alpha')} == {'would_hit'}
        assert await registry.consumed_for(merge_sha) == ()

    @pytest.mark.asyncio
    async def test_a_replay_the_registry_cannot_consume_is_no_verdict(
        self, repo, modules, registry, store, tmp_path,
    ):
        config = cache_config(repo)
        await merge_pool(repo, config, modules, registry, store).dispatch(head_of(repo), _spec(config, modules))
        commit_edit(repo, 'beta/src/beta/__init__.py', 'B = 2\n')

        async def _decide_then_garble(*args: Any, **kwargs: Any) -> VerifyResult:
            decided = await run_scoped_verification(*args, **kwargs)
            _garble(tmp_path / 'verify_cache.db')
            return decided

        pool = merge_pool(repo, config, modules, registry, store, run_scoped=_decide_then_garble)
        result = await pool.dispatch(head_of(repo), _spec(config, modules))

        assert outcomes_of(result, module_units('alpha')) == {UnitOutcome.CACHE_HIT}
        assert not result.passed
        assert is_no_verdict_summary(result.summary), result.summary
        assert store.fetch_events_by_type(EventType.merge_verify)[-1]['data']['passed'] is False

    @pytest.mark.asyncio
    async def test_a_red_gate_keeps_its_own_verdict_when_its_replays_cannot_be_consumed(
        self, repo, modules, registry, store, tmp_path,
    ):
        config = cache_config(repo)
        await merge_pool(repo, config, modules, registry, store).dispatch(head_of(repo), _spec(config, modules))
        commit_edit(repo, 'beta/tests/test_beta.py', 'def test_beta():\n    assert False\n')
        decided: list[VerifyResult] = []

        async def _decide_then_garble(*args: Any, **kwargs: Any) -> VerifyResult:
            decided.append(await run_scoped_verification(*args, **kwargs))
            _garble(tmp_path / 'verify_cache.db')
            return decided[-1]

        pool = merge_pool(repo, config, modules, registry, store, run_scoped=_decide_then_garble)
        result = await pool.dispatch(head_of(repo), _spec(config, modules))

        red = decided[-1]
        assert outcomes_of(red, module_units('alpha')) == {UnitOutcome.CACHE_HIT}
        assert not red.passed
        assert red.category
        assert not is_no_verdict_summary(red.summary), red.summary
        assert (result.passed, result.category, result.summary) == (False, red.category, red.summary)

    @pytest.mark.asyncio
    async def test_a_non_local_runner_records_nothing(self, repo, modules, access, registry, store):
        config = cache_config(repo)
        remote_result = await merge_gate(repo, config, modules, access)
        assert remote_result.passed, remote_result.summary
        assert remote_result.cache_record is not None
        merge_sha = head_of(repo)
        pool = VerifyRunnerPool(
            [_RemoteDouble(remote_result)], event_store=store, task_id='5653', verify_cache_registry=registry,
        )

        result = await pool.dispatch(merge_sha, _spec(config, modules))

        assert result.passed, result.summary
        assert await entries_for(registry, result) == {}
        assert await registry.consumed_for(merge_sha) == ()
