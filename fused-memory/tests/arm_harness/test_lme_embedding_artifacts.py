"""Contract tests over the committed LME ι embedding artifacts (task 3722).

``plans/local-memory-models-eval-embedding/`` holds the six embedding arm specs, the
probe set they were run against, one committed run per arm and
``embedding-preregistration-inputs.json``, the committed formula's output over the
two incumbent control runs. These tests re-derive that output and the candidate specs
from their committed inputs, so neither can drift from the formula, the probe set or
the arms.yaml slate.

Lane discipline: file reads plus ``git`` subprocesses only, and NO ``integration``
marker, so the merge lane's default selection runs them.
"""

import json
import shutil
import subprocess
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from fused_memory.arm_harness.arm_spec import METERED_STACK, EmbeddingArmSpec, load_arm_spec
from fused_memory.arm_harness.embedding_preregistration import (
    EMBEDDING_PREREGISTRATION_INPUTS_FILENAME,
    check_embedding_run_symmetry,
    derive_embedding_preregistration_inputs,
    serialize_embedding_preregistration_inputs,
)
from fused_memory.arm_harness.embedding_run_manifest import (
    EmbeddingRunManifest,
    load_embedding_run_manifest,
)
from fused_memory.arm_harness.instrument_checks import (
    PREREGISTRATION_DOC_PATH,
    InstrumentCheckId,
)
from fused_memory.arm_harness.metrics_record import MetricsRecord, load_metrics_records
from fused_memory.arm_harness.probe_set import FrozenReference, load_probe_set, probe_set_sha
from fused_memory.arm_harness.run import RUN_MANIFEST_FILENAME
from fused_memory.arm_harness.slate import (
    embedding_candidate_spec,
    embedding_control_spec,
    load_embedding_slate,
)

REPO_ROOT = Path(__file__).parents[3]
EMBEDDING_RELATIVE = 'plans/local-memory-models-eval-embedding'
EMBEDDING = REPO_ROOT / EMBEDDING_RELATIVE
PROBE_SET = EMBEDDING / 'probe-set.json'
FROZEN_REFERENCE = REPO_ROOT / 'plans' / 'local-memory-models-eval-controls' / 'frozen-reference.json'
ARMS_YAML = REPO_ROOT / 'scripts' / 'local-model-serving' / 'arms.yaml'
PREREGISTRATION_SHA = 'be4a44d6243b58a3293d66eb73f0ba56fa206efd'
"""The erratum commit Leo's esc-3722-3 ruling (2026-10-08) binds θ and ι runs to."""
SEARCH_TIMEOUT_S = 30.0
"""config.yaml's queue.search_timeout_seconds, the anchor preregistration §6 names."""

CONTROLS = {'incumbent-embed-a': 'evalmem_lme_emb_ctl_a',
            'incumbent-embed-b': 'evalmem_lme_emb_ctl_b'}
CANDIDATES = (
    'qwen3-embedding-0.6b',
    'granite-embedding-english-r2',
    'qwen3-embedding-4b',
    'gte-modernbert-base',
)
ARMS = (*CONTROLS, *CANDIDATES)
INCUMBENT_MODEL, INCUMBENT_DIM = 'text-embedding-3-small', 1536


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which('git') is None:
        pytest.skip('git is not available; cannot check the committed index')
    inside = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if inside.returncode != 0:
        pytest.skip('not a git working tree; cannot check the committed index')
    return subprocess.run(
        ['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _tracked() -> list[str]:
    return _git('ls-files', '--', EMBEDDING_RELATIVE).stdout.splitlines()


def _run_dir(arm_id: str) -> Path:
    runs = EMBEDDING / 'runs' / arm_id
    stamps = sorted(path for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []
    assert len(stamps) == 1, f'{arm_id} must have exactly one committed stamp dir: {stamps}'
    return stamps[0]


@cache
def _run(arm_id: str) -> EmbeddingRunManifest:
    return load_embedding_run_manifest(_run_dir(arm_id) / RUN_MANIFEST_FILENAME)


@cache
def _records(arm_id: str) -> tuple[MetricsRecord, ...]:
    return load_metrics_records(_run_dir(arm_id))


def _spec(arm_id: str) -> EmbeddingArmSpec:
    spec = load_arm_spec(EMBEDDING / 'specs' / f'{arm_id}.json')
    assert isinstance(spec, EmbeddingArmSpec), spec
    return spec


@cache
def _shared_shas() -> tuple[str, str]:
    code_shas = {_run(arm_id).spec.code_sha for arm_id in ARMS}
    corpus_shas = {_run(arm_id).spec.corpus_sha for arm_id in ARMS}
    assert len(code_shas) == 1, code_shas
    assert len(corpus_shas) == 1, corpus_shas
    return code_shas.pop(), corpus_shas.pop()


# --- durability -----------------------------------------------------------------------


def test_exactly_the_six_arms_are_committed_each_with_one_run():
    tracked = _tracked()
    runs = {path.split('/')[3] for path in tracked if path.startswith(f'{EMBEDDING_RELATIVE}/runs/')}
    specs = {
        Path(path).stem for path in tracked if path.startswith(f'{EMBEDDING_RELATIVE}/specs/')
    }
    assert runs == set(ARMS)
    assert specs == set(ARMS)
    for name in ('README.md', 'probe-set.json', EMBEDDING_PREREGISTRATION_INPUTS_FILENAME):
        assert f'{EMBEDDING_RELATIVE}/{name}' in tracked
    for arm_id in ARMS:
        run_dir = _run_dir(arm_id)
        assert (run_dir / RUN_MANIFEST_FILENAME).is_file()
        assert (run_dir / 'metrics').is_dir()


# --- one probe, one code sha, symmetric runs --------------------------------------------


def test_every_run_shares_one_code_sha_and_the_probe_sets_corpus_sha():
    _, corpus = _shared_shas()
    assert corpus == probe_set_sha(PROBE_SET.read_bytes())


def test_every_run_is_symmetric():
    symmetry = check_embedding_run_symmetry([_run(arm_id) for arm_id in ARMS])
    assert symmetry.passed, symmetry.detail


@pytest.mark.parametrize('arm_id', ARMS)
def test_every_run_recorded_the_configured_search_timeout(arm_id):
    assert _run(arm_id).settings.search_timeout_s == SEARCH_TIMEOUT_S


def test_the_probe_sets_reference_is_the_frozen_reference():
    frozen = FrozenReference.model_validate_json(FROZEN_REFERENCE.read_bytes())
    assert load_probe_set(PROBE_SET).reference == frozen


# --- the specs ------------------------------------------------------------------------


@pytest.mark.parametrize('arm', load_embedding_slate(ARMS_YAML), ids=lambda arm: arm.arm_id)
def test_each_candidate_spec_is_the_slates_reading_of_arms_yaml(arm):
    code_sha, corpus = _shared_shas()
    expected = embedding_candidate_spec(
        arm, code_sha=code_sha, corpus_sha=corpus, preregistration_sha=PREREGISTRATION_SHA
    )
    assert _spec(arm.arm_id) == expected
    assert _run(arm.arm_id).spec == expected


def test_the_slate_is_exactly_the_four_candidates():
    assert [arm.arm_id for arm in load_embedding_slate(ARMS_YAML)] == list(CANDIDATES)


@pytest.mark.parametrize(('arm_id', 'scratch'), CONTROLS.items())
def test_each_control_is_the_metered_incumbent_embedder(arm_id, scratch):
    code_sha, corpus = _shared_shas()
    expected = embedding_control_spec(
        arm_id,
        model_id=INCUMBENT_MODEL,
        embedding_dim=INCUMBENT_DIM,
        code_sha=code_sha,
        corpus_sha=corpus,
        scratch_group_id=scratch,
    )
    assert _spec(arm_id) == expected
    assert _run(arm_id).spec == expected
    assert expected.serving.stack == METERED_STACK
    assert expected.preregistration_sha is None


def test_the_candidates_preregistration_sha_carries_the_doc_and_precedes_head():
    assert {_run(arm_id).spec.preregistration_sha for arm_id in CANDIDATES} == {
        PREREGISTRATION_SHA
    }
    assert _git('cat-file', '-e', f'{PREREGISTRATION_SHA}:{PREREGISTRATION_DOC_PATH}').returncode == 0
    assert _git('merge-base', '--is-ancestor', PREREGISTRATION_SHA, 'HEAD').returncode == 0


# --- order, checks and the pre-registration ---------------------------------------------


def test_both_controls_finished_before_any_candidate_started():
    last_control = max(_run(arm_id).finished_at for arm_id in CONTROLS)
    first_candidate = min(_run(arm_id).started_at for arm_id in CANDIDATES)
    assert last_control < first_candidate


@pytest.mark.parametrize('arm_id', ARMS)
def test_every_run_passed_every_instrument_check(arm_id):
    checks = _run(arm_id).check_results
    assert all(check.passed for check in checks), checks
    counts = Counter(check.check_id for check in checks)
    assert counts[InstrumentCheckId.FROZEN_REFERENCE_UNCHANGED] == 1
    assert counts[InstrumentCheckId.REEMBED_INTEGRITY] == 2
    assert counts[InstrumentCheckId.INDEX_CONFIGURATION] == 2


@pytest.mark.parametrize('arm_id', ARMS)
def test_every_run_records_its_raw_norms(arm_id):
    run = _run(arm_id)
    assert run.graph_reembed.raw_norms is not None
    assert run.replica_reembed.raw_norms is not None


def test_the_preregistration_inputs_re_derive_from_the_control_runs():
    arm_a, arm_b = CONTROLS
    inputs = derive_embedding_preregistration_inputs(
        _run(arm_a), _records(arm_a), _run(arm_b), _records(arm_b)
    )
    committed = (EMBEDDING / EMBEDDING_PREREGISTRATION_INPUTS_FILENAME).read_text()
    assert committed == serialize_embedding_preregistration_inputs(inputs)
    assert json.loads(committed)['control_arm_ids'] == [arm_a, arm_b]
