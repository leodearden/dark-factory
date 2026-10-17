"""Run orchestration: pre-run refusal, replay, metrics, post-run checks, artifacts in a run dir."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from graphiti_core.helpers import SEMAPHORE_LIMIT

from arm_harness._fakes import (
    FakeArmGraph,
    PreregRepo,
    RecordingJournal,
    embedding_spec,
    fail,
    incumbent_control_spec,
    llm_spec,
    make_prereg_repo,
)
from fused_memory.arm_harness.conformance import ConformanceLedger
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.llm_metrics import GRAPH_SAMENESS_DETAILS_FILENAME
from fused_memory.arm_harness.metrics_record import (
    LLM_METRIC_IDS,
    METRICS_DIRNAME,
    IndexConfiguration,
    LlmMetricId,
    load_metrics_record,
)
from fused_memory.arm_harness.replay import MAX_CONSECUTIVE_FAILURES
from fused_memory.arm_harness.replay_types import (
    ArmAbort,
    EpisodeOutcome,
    ReplayItem,
    ReplaySettings,
)
from fused_memory.arm_harness.run import (
    ABORT_FILENAME,
    OUTCOMES_FILENAME,
    RUN_MANIFEST_FILENAME,
    PreRunCheckError,
    load_outcomes,
    require_pre_run_checks,
    run_llm_arm,
    write_outcomes,
)
from fused_memory.arm_harness.run_manifest import load_run_manifest
from fused_memory.backends.llm_token_usage import LlmTokenUsage
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.write_journal import WriteJournal

REFERENCE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
SETTINGS = ReplaySettings(
    concurrency=1,
    episode_timeout_s=120.0,
    index_configuration=IndexConfiguration.WITH_INDICES,
)


def _items(count: int) -> list[ReplayItem]:
    return [
        ReplayItem(
            episode_id=f'ep-{i}',
            name=f'ep-{i}',
            content=f'content of episode {i}',
            source_description='lme corpus',
            reference_time=REFERENCE_TIME,
        )
        for i in range(count)
    ]


def _reference(count: int) -> list[EpisodeOutcome]:
    return [
        EpisodeOutcome(
            episode_id=f'ep-{i}',
            ok=True,
            error_class=None,
            duration_ms=12.0,
            tokens=LlmTokenUsage(input_tokens=30, output_tokens=10, llm_calls=1),
            replay_episode_uuid=f'ref-ep-{i}',
            entity_names=('alice', 'carol'),
            edge_triples=(),
        )
        for i in range(count)
    ]


def _cited_by_replay(query: str) -> list[SimpleNamespace]:
    """A search double whose one hit cites the replay of the episode the query came from."""
    name = query.removeprefix('content of episode ')
    return [SimpleNamespace(episodes=[f'replay-ep-{name}'])]


@pytest.fixture
def repo(tmp_path: Path) -> PreregRepo:
    return make_prereg_repo(tmp_path / 'repo')


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / 'run'
    directory.mkdir()
    return directory


def _audited_ledger(valid: int = 3) -> ConformanceLedger:
    ledger = ConformanceLedger()
    for _ in range(valid):
        ledger.record_valid()
    return ledger


@pytest.fixture
def run_arm(run_dir: Path, repo: PreregRepo, mock_config: FusedMemoryConfig):
    """``run_llm_arm`` wired to this test's run dir, repo and base config.

    Without an explicit journal, a real WriteJournal is opened under ``run_dir/journal``.
    """

    async def run(spec, items, *, graph, reference=None, journal=None):
        owned = journal is None
        journal = journal or WriteJournal(run_dir / 'journal')
        if owned:
            await journal.initialize()
        try:
            return await run_llm_arm(
                spec,
                items,
                graph=graph,
                conformance=_audited_ledger(),
                journal=journal,
                settings=SETTINGS,
                run_dir=run_dir,
                repo_root=repo.root,
                reference=reference,
                base_config=mock_config,
            )
        finally:
            if owned:
                await journal.close()

    return run


def _metric_files(run_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted((run_dir / METRICS_DIRNAME).glob('*.json'))}


# --- (a) happy path ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_complete_run_leaves_every_artifact_in_the_run_dir(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)
    graph = FakeArmGraph(search_results=_cited_by_replay)

    manifest = await run_arm(spec, _items(3), graph=graph, reference=_reference(3))

    assert manifest.incomplete is False
    assert manifest.abort is None
    assert manifest.episode_ids == ('ep-0', 'ep-1', 'ep-2')
    assert all(check.passed for check in manifest.check_results), manifest.check_results
    assert load_run_manifest(run_dir / RUN_MANIFEST_FILENAME) == manifest
    assert len(load_outcomes(run_dir / OUTCOMES_FILENAME)) == 3
    assert (run_dir / 'journal' / 'write_journal.db').is_file()
    assert (run_dir / GRAPH_SAMENESS_DETAILS_FILENAME).is_file()
    assert not (run_dir / ABORT_FILENAME).exists()
    assert set(_metric_files(run_dir)) == LLM_METRIC_IDS


@pytest.mark.asyncio
async def test_every_metrics_record_loads_back_with_the_spec_shas(run_arm, run_dir, repo):
    spec = llm_spec(code_sha=repo.with_prereg, preregistration_sha=repo.with_prereg)

    await run_arm(spec, _items(2), graph=FakeArmGraph())

    files = _metric_files(run_dir)
    assert files
    for path in files.values():
        record = load_metrics_record(path)
        assert record.arm_id == spec.arm_id
        assert record.code_sha == spec.code_sha
        assert record.corpus_sha == spec.corpus_sha
        assert record.preregistration_sha == spec.preregistration_sha
        assert record.incomplete is False


@pytest.mark.asyncio
async def test_without_a_reference_there_is_no_graph_sameness(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)

    manifest = await run_arm(spec, _items(2), graph=FakeArmGraph())

    assert LlmMetricId.GRAPH_SAMENESS not in _metric_files(run_dir)
    assert not (run_dir / GRAPH_SAMENESS_DETAILS_FILENAME).exists()
    assert InstrumentCheckId.REFERENCE_NONEMPTY not in {c.check_id for c in manifest.check_results}


@pytest.mark.asyncio
async def test_the_manifest_records_the_effective_run_environment(run_arm, repo, mock_config):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)

    manifest = await run_arm(spec, _items(1), graph=FakeArmGraph())

    assert manifest.spec == spec
    assert manifest.settings_summary.concurrency == SETTINGS.concurrency
    assert manifest.settings_summary.index_configuration is SETTINGS.index_configuration
    assert manifest.settings_summary.episode_timeout_s == SETTINGS.episode_timeout_s
    assert manifest.effective_embedder.model == mock_config.embedder.model
    assert manifest.effective_embedder.dimensions == mock_config.embedder.dimensions
    assert manifest.graphiti_max_coroutines == mock_config.queue.graphiti_max_coroutines
    assert manifest.graphiti_semaphore_limit == SEMAPHORE_LIMIT
    assert manifest.started_at <= manifest.finished_at


def test_outcomes_round_trip_one_canonical_line_each(tmp_path):
    outcomes = _reference(2)

    path = write_outcomes(tmp_path, outcomes)

    lines = path.read_text(encoding='utf-8').splitlines()
    assert path.name == OUTCOMES_FILENAME
    assert lines == [
        json.dumps(outcome.model_dump(mode='json'), sort_keys=True, ensure_ascii=False)
        for outcome in outcomes
    ]
    assert load_outcomes(path) == tuple(outcomes)


# --- (b) pre-run refusal -------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unresolvable_preregistration_sha_refuses_before_anything(run_arm, run_dir, repo):
    spec = llm_spec(code_sha=repo.with_prereg, preregistration_sha=repo.without_prereg)
    graph, journal = FakeArmGraph(), RecordingJournal()

    with pytest.raises(PreRunCheckError) as caught:
        await run_arm(spec, _items(2), graph=graph, journal=journal)

    failed = [check for check in caught.value.check_results if not check.passed]
    assert [check.check_id for check in failed] == [InstrumentCheckId.PREREGISTRATION_SHA]
    assert list(run_dir.iterdir()) == []
    assert graph.events == []
    assert journal.calls == []


@pytest.mark.asyncio
async def test_a_code_sha_other_than_head_refuses_before_anything(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.without_prereg)
    graph, journal = FakeArmGraph(), RecordingJournal()

    with pytest.raises(PreRunCheckError) as caught:
        await run_arm(spec, _items(2), graph=graph, journal=journal)

    failed = [check for check in caught.value.check_results if not check.passed]
    assert [check.check_id for check in failed] == [InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT]
    assert list(run_dir.iterdir()) == []
    assert graph.events == []
    assert journal.calls == []


def test_pre_run_checks_are_callable_alone_before_any_resource_is_opened(repo):
    stale = incumbent_control_spec(code_sha=repo.without_prereg)
    current = incumbent_control_spec(code_sha=repo.with_prereg)

    with pytest.raises(PreRunCheckError) as caught:
        require_pre_run_checks(stale, repo.root)
    passed = require_pre_run_checks(current, repo.root)

    assert InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT in {
        check.check_id for check in caught.value.check_results if not check.passed
    }
    assert [check.check_id for check in passed] == [
        InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT,
        InstrumentCheckId.PREREGISTRATION_SHA,
    ]
    assert all(check.passed for check in passed)


def test_pre_run_checks_accept_an_embedding_arm(repo):
    spec = embedding_spec(code_sha=repo.with_prereg, preregistration_sha=repo.with_prereg)

    passed = require_pre_run_checks(spec, repo.root)

    assert [check.check_id for check in passed] == [
        InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT,
        InstrumentCheckId.PREREGISTRATION_SHA,
    ]
    assert all(check.passed for check in passed)


# --- (c) INV-4 abort -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_inv4_abort_keeps_partial_artifacts_flagged_incomplete(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)
    failing = {f'ep-{i}': fail for i in range(2, 2 + MAX_CONSECUTIVE_FAILURES)}
    graph = FakeArmGraph(failing)

    manifest = await run_arm(spec, _items(9), graph=graph)

    expected_abort = ArmAbort(
        arm_id=spec.arm_id,
        item_ids=tuple(f'ep-{i}' for i in range(2, 2 + MAX_CONSECUTIVE_FAILURES)),
        error_classes=('RuntimeError',) * MAX_CONSECUTIVE_FAILURES,
    )
    assert ArmAbort.model_validate_json((run_dir / ABORT_FILENAME).read_text()) == expected_abort
    assert manifest.incomplete is True
    assert manifest.abort == expected_abort
    assert load_run_manifest(run_dir / RUN_MANIFEST_FILENAME) == manifest
    assert len(load_outcomes(run_dir / OUTCOMES_FILENAME)) == 2 + MAX_CONSECUTIVE_FAILURES
    files = _metric_files(run_dir)
    assert LlmMetricId.EPISODE_FAILURE_RATE in files
    assert LlmMetricId.TOKENS_PER_EPISODE in files
    for path in files.values():
        assert json.loads(path.read_text())['incomplete'] is True


# --- (d) post-run instrument failures are recorded -----------------------------------


@pytest.mark.asyncio
async def test_zero_token_usage_is_a_recorded_failed_check(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)

    manifest = await run_arm(spec, _items(2), graph=FakeArmGraph(usage=(0, 0)))

    token_checks = [
        check for check in manifest.check_results
        if check.check_id is InstrumentCheckId.TOKEN_COST_ACCOUNTING
    ]
    assert len(token_checks) == 1
    assert token_checks[0].passed is False
    assert load_run_manifest(run_dir / RUN_MANIFEST_FILENAME) == manifest


@pytest.mark.asyncio
async def test_missing_token_usage_is_a_recorded_failed_check_naming_the_episodes(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)
    unmeasurable = FakeArmGraph(usage=None, llm_client=SimpleNamespace(token_tracker=None))

    manifest = await run_arm(spec, _items(2), graph=unmeasurable)

    token_checks = [
        check for check in manifest.check_results
        if check.check_id is InstrumentCheckId.TOKEN_COST_ACCOUNTING
    ]
    assert len(token_checks) == 1
    assert token_checks[0].passed is False
    assert set(token_checks[0].offenders) == {'ep-0', 'ep-1'}
    assert 'TokenAccountingError' in token_checks[0].detail
    assert load_run_manifest(run_dir / RUN_MANIFEST_FILENAME) == manifest


@pytest.mark.asyncio
async def test_an_empty_reference_is_a_recorded_failed_check(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)

    manifest = await run_arm(
        spec, _items(2), graph=FakeArmGraph(), reference=[]
    )

    (reference_check,) = [
        check for check in manifest.check_results
        if check.check_id is InstrumentCheckId.REFERENCE_NONEMPTY
    ]
    assert reference_check.passed is False


# --- (e) retrieval utility is probed after every write -------------------------------


@pytest.mark.asyncio
async def test_retrieval_utility_searches_only_after_the_final_write(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)
    graph = FakeArmGraph(search_results=_cited_by_replay)

    await run_arm(spec, _items(4), graph=graph)

    last_write = max(i for i, event in enumerate(graph.events) if event == 'add_episode')
    first_search = graph.events.index('search')
    assert first_search > last_write
    assert len(graph.search_calls) == 4
    record = load_metrics_record(_metric_files(run_dir)[LlmMetricId.RETRIEVAL_UTILITY])
    assert record.metric.value == 1.0


@pytest.mark.asyncio
async def test_no_ok_episode_means_no_retrieval_probe(run_arm, run_dir, repo):
    spec = incumbent_control_spec(code_sha=repo.with_prereg)
    graph = FakeArmGraph(default=fail)

    await run_arm(spec, _items(2), graph=graph)

    assert graph.search_calls == []
    assert LlmMetricId.RETRIEVAL_UTILITY not in _metric_files(run_dir)
