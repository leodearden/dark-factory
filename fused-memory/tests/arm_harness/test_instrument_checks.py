"""Per-run instrument checks (PRD §Instrument validation) and the RunManifest record."""

import dataclasses
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import incumbent_control_spec, llm_spec, run_manifest_for
from fused_memory.arm_harness.instrument_checks import (
    PREREGISTRATION_DOC_PATH,
    CheckResult,
    InstrumentCheckId,
    check_code_sha_matches_checkout,
    check_preregistration_sha,
    check_reference_nonempty,
    check_token_cost_accounting,
)
from fused_memory.arm_harness.metrics_record import LlmMetricId, record_for
from fused_memory.arm_harness.replay import ArmAbort, EpisodeOutcome
from fused_memory.arm_harness.run_manifest import (
    RunManifest,
    load_run_manifest,
    serialize_run_manifest,
)

MEASURED_AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ['git', '-C', str(root), '-c', 'user.name=T', '-c', 'user.email=t@e.example',
         '-c', 'commit.gpgsign=false', *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@dataclasses.dataclass(frozen=True)
class Repo:
    root: Path
    without_prereg: str
    with_prereg: str


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    """A real two-commit repo: the second commit adds the preregistration doc."""
    root = tmp_path / 'repo'
    root.mkdir()
    _git(root, 'init', '-q', '-b', 'main')
    (root / 'seed.txt').write_text('seed\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'seed')
    without_prereg = _git(root, 'rev-parse', 'HEAD')
    doc = root / PREREGISTRATION_DOC_PATH
    doc.parent.mkdir(parents=True)
    doc.write_text('# preregistration\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'prereg')
    with_prereg = _git(root, 'rev-parse', 'HEAD')
    return Repo(root=root, without_prereg=without_prereg, with_prereg=with_prereg)


# --- CheckResult ---------------------------------------------------------------------


def test_instrument_check_ids_are_a_closed_vocabulary():
    assert {member.value for member in InstrumentCheckId} == {
        'code-sha-matches-checkout',
        'preregistration-sha',
        'token-cost-accounting',
        'reference-nonempty',
        'arm-config-symmetry',
        'single-code-sha',
    }


def test_check_result_is_frozen():
    result = CheckResult(
        check_id=InstrumentCheckId.SINGLE_CODE_SHA, passed=True, detail='ok', offenders=()
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


# --- code sha vs checkout ------------------------------------------------------------


def test_code_sha_check_passes_on_a_clean_checkout_at_the_spec_sha(repo: Repo):
    result = check_code_sha_matches_checkout(llm_spec(code_sha=repo.with_prereg), repo.root)

    assert result.check_id is InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT
    assert result.passed, result.detail
    assert result.offenders == ()


def test_code_sha_check_fails_when_head_differs(repo: Repo):
    result = check_code_sha_matches_checkout(llm_spec(code_sha=repo.without_prereg), repo.root)

    assert not result.passed
    assert result.offenders == (repo.with_prereg,)
    assert repo.without_prereg in result.detail
    assert repo.with_prereg in result.detail


def test_code_sha_check_fails_on_a_dirty_tree_listing_the_dirty_paths(repo: Repo):
    (repo.root / 'seed.txt').write_text('edited\n')
    (repo.root / 'untracked.txt').write_text('new\n')

    result = check_code_sha_matches_checkout(llm_spec(code_sha=repo.with_prereg), repo.root)

    assert not result.passed
    assert result.offenders == ('seed.txt', 'untracked.txt')
    assert 'seed.txt' in result.detail
    assert 'untracked.txt' in result.detail


def test_code_sha_check_turns_a_git_failure_into_a_failed_result(tmp_path: Path):
    result = check_code_sha_matches_checkout(llm_spec(), tmp_path)

    assert not result.passed
    assert 'git' in result.detail


# --- preregistration sha -------------------------------------------------------------


def test_a_control_arm_needs_no_preregistration(tmp_path: Path):
    result = check_preregistration_sha(incumbent_control_spec(), tmp_path)

    assert result.check_id is InstrumentCheckId.PREREGISTRATION_SHA
    assert result.passed, result.detail


def test_a_candidate_passes_when_its_sha_carries_the_preregistration_doc(repo: Repo):
    result = check_preregistration_sha(llm_spec(preregistration_sha=repo.with_prereg), repo.root)

    assert result.passed, result.detail


def test_a_candidate_fails_when_its_sha_lacks_the_preregistration_doc(repo: Repo):
    result = check_preregistration_sha(
        llm_spec(preregistration_sha=repo.without_prereg), repo.root
    )

    assert not result.passed
    assert result.offenders == (repo.without_prereg,)
    assert PREREGISTRATION_DOC_PATH in result.detail


def test_a_candidate_fails_on_a_sha_the_repo_does_not_have(repo: Repo):
    result = check_preregistration_sha(llm_spec(preregistration_sha='d' * 40), repo.root)

    assert not result.passed
    assert result.offenders == ('d' * 40,)


# --- token and cost accounting -------------------------------------------------------


def _scalar_record(spec, metric_id: LlmMetricId, value: float):
    metric = Metric(metric_id=metric_id, kind='scalar', value=value, n=3)
    return record_for(spec, metric, measured_at=MEASURED_AT, incomplete=False)


def test_a_local_arm_with_tokens_and_zero_cost_passes():
    spec = llm_spec()
    records = (
        _scalar_record(spec, LlmMetricId.TOKENS_PER_EPISODE, 60.0),
        _scalar_record(spec, LlmMetricId.USD_PER_EPISODE, 0.0),
    )

    result = check_token_cost_accounting(spec, records)

    assert result.check_id is InstrumentCheckId.TOKEN_COST_ACCOUNTING
    assert result.passed, result.detail


@pytest.mark.parametrize('tokens', [None, 0.0], ids=['missing', 'zero'])
def test_missing_or_zero_tokens_fail(tokens):
    spec = llm_spec()
    records = [_scalar_record(spec, LlmMetricId.USD_PER_EPISODE, 0.0)]
    if tokens is not None:
        records.append(_scalar_record(spec, LlmMetricId.TOKENS_PER_EPISODE, tokens))

    result = check_token_cost_accounting(spec, tuple(records))

    assert not result.passed
    assert result.offenders == ('tokens-per-episode',)


@pytest.mark.parametrize('usd', [None, 0.0], ids=['missing', 'zero'])
def test_a_metered_arm_with_missing_or_zero_cost_fails(usd):
    spec = incumbent_control_spec()
    records = [_scalar_record(spec, LlmMetricId.TOKENS_PER_EPISODE, 60.0)]
    if usd is not None:
        records.append(_scalar_record(spec, LlmMetricId.USD_PER_EPISODE, usd))

    result = check_token_cost_accounting(spec, tuple(records))

    assert not result.passed
    assert result.offenders == ('usd-per-episode',)


def test_a_metered_arm_with_tokens_and_cost_passes():
    spec = incumbent_control_spec()
    records = (
        _scalar_record(spec, LlmMetricId.TOKENS_PER_EPISODE, 60.0),
        _scalar_record(spec, LlmMetricId.USD_PER_EPISODE, 0.00004),
    )

    assert check_token_cost_accounting(spec, records).passed


# --- reference non-empty -------------------------------------------------------------


def _outcome(episode_id: str, *, ok: bool = True, entities: tuple[str, ...] = ()) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=ok,
        error_class=None if ok else 'RuntimeError',
        duration_ms=10.0,
        tokens=None,
        replay_episode_uuid=f'replay-{episode_id}' if ok else None,
        entity_names=entities,
        edge_triples=(),
    )


def test_an_empty_reference_fails():
    result = check_reference_nonempty(())

    assert result.check_id is InstrumentCheckId.REFERENCE_NONEMPTY
    assert not result.passed


def test_a_reference_without_ok_episodes_fails_naming_them():
    result = check_reference_nonempty((_outcome('e1', ok=False), _outcome('e2', ok=False)))

    assert not result.passed
    assert result.offenders == ('e1', 'e2')


def test_a_reference_without_any_entity_fails_naming_its_ok_episodes():
    result = check_reference_nonempty((_outcome('e1'), _outcome('e2'), _outcome('e3', ok=False)))

    assert not result.passed
    assert result.offenders == ('e1', 'e2')
    assert 'entit' in result.detail


def test_a_reference_with_entities_passes():
    result = check_reference_nonempty((_outcome('e1', entities=('alice',)), _outcome('e2')))

    assert result.passed, result.detail


# --- RunManifest ---------------------------------------------------------------------


def _manifest_with_everything() -> RunManifest:
    abort = ArmAbort(arm_id='incumbent-ctrl-a', item_ids=('e2',), error_classes=('RuntimeError',))
    checks = (
        CheckResult(
            check_id=InstrumentCheckId.REFERENCE_NONEMPTY,
            passed=False,
            detail='reference has no entities',
            offenders=('e1',),
        ),
    )
    return run_manifest_for(
        incumbent_control_spec(), incomplete=True, abort=abort, check_results=checks
    )


def test_run_manifest_round_trips_through_its_canonical_text(tmp_path: Path):
    manifest = _manifest_with_everything()
    path = tmp_path / 'run.json'
    path.write_text(serialize_run_manifest(manifest))

    assert load_run_manifest(path) == manifest
    assert serialize_run_manifest(load_run_manifest(path)) == path.read_text()


def test_run_manifest_text_is_canonical_json():
    text = serialize_run_manifest(run_manifest_for(llm_spec()))

    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def test_run_manifest_follows_the_null_convention():
    payload = json.loads(serialize_run_manifest(run_manifest_for(incumbent_control_spec())))

    assert payload['abort'] is None
    assert payload['spec']['preregistration_sha'] is None
    assert 'quant' not in payload['spec']['serving']
    assert 'unit_name' not in payload['spec']['serving']


def test_run_manifest_keeps_the_spec_axis():
    manifest = run_manifest_for(llm_spec())

    assert RunManifest.model_validate_json(serialize_run_manifest(manifest)).spec == llm_spec()


def test_an_aborted_run_must_be_marked_incomplete():
    abort = ArmAbort(arm_id='qwen3-8b-vllm', item_ids=('e1',), error_classes=('RuntimeError',))

    with pytest.raises(ValidationError, match='incomplete'):
        run_manifest_for(llm_spec(), abort=abort, incomplete=False)


def test_a_run_cannot_finish_before_it_starts():
    with pytest.raises(ValidationError, match='finished_at'):
        run_manifest_for(llm_spec(), finished_at=MEASURED_AT.replace(year=2025))


def test_run_manifest_rejects_repeated_episode_ids():
    with pytest.raises(ValidationError, match='e1'):
        run_manifest_for(llm_spec(), episode_ids=('e1', 'e2', 'e1'))
