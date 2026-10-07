"""Contract tests over the committed LME incumbent control artifacts (task 3719, ζ).

``plans/local-memory-models-eval-controls/`` holds the four live incumbent control runs
the pre-registration is derived from, and ``preregistration-inputs.json``, the committed
formula's output over runs A and B. These tests re-derive that output from the committed
runs, so the numbers the preregistration doc quotes cannot drift from the formula.

Lane discipline: file reads plus ``git`` subprocesses only, and NO ``integration``
marker, so the merge lane's default selection runs them.
"""

import json
import math
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from fused_memory.arm_harness.arm_spec import load_arm_spec
from fused_memory.arm_harness.corpus import corpus_sha
from fused_memory.arm_harness.llm_metrics import (
    GRAPH_SAMENESS_DETAILS_FILENAME,
    GraphSamenessDetails,
)
from fused_memory.arm_harness.margins import GATED_METRICS
from fused_memory.arm_harness.metrics_record import (
    LLM_METRIC_IDS,
    MetricsRecord,
    load_metrics_records,
)
from fused_memory.arm_harness.preregistration import (
    PREREGISTRATION_INPUTS_FILENAME,
    derive_preregistration_inputs,
    serialize_preregistration_inputs,
)
from fused_memory.arm_harness.run import OUTCOMES_FILENAME, RUN_MANIFEST_FILENAME, load_outcomes
from fused_memory.arm_harness.run_manifest import RunManifest, load_run_manifest

REPO_ROOT = Path(__file__).parents[3]
CONTROLS_RELATIVE = 'plans/local-memory-models-eval-controls'
CONTROLS = REPO_ROOT / CONTROLS_RELATIVE
CORPUS_MANIFEST = (
    REPO_ROOT / 'fused-memory' / 'scripts' / 'local_memory_models_eval' / 'corpus_manifest.json'
)
FROZEN_REFERENCE = CONTROLS / 'frozen-reference.json'
ARM_A, ARM_B = 'incumbent-generic-a', 'incumbent-generic-b'
GENERIC_20, OPENAI_20 = 'incumbent-generic-20', 'incumbent-openai-20'
ARMS = (ARM_A, ARM_B, GENERIC_20, OPENAI_20)
SCREENING_EPISODES = 20
FROZEN_GRAPH = 'evalmem_lme_ref_incumbent_a'


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
    return _git('ls-files', '--', CONTROLS_RELATIVE).stdout.splitlines()


def _run_dir(arm_id: str) -> Path:
    stamps = sorted(path for path in (CONTROLS / 'runs' / arm_id).iterdir() if path.is_dir())
    assert len(stamps) == 1, f'{arm_id} must have exactly one committed stamp dir: {stamps}'
    return stamps[0]


@cache
def _run(arm_id: str) -> RunManifest:
    return load_run_manifest(_run_dir(arm_id) / RUN_MANIFEST_FILENAME)


@cache
def _records(arm_id: str) -> tuple[MetricsRecord, ...]:
    return load_metrics_records(_run_dir(arm_id))


def _manifest_episode_ids() -> tuple[str, ...]:
    return tuple(entry['uuid'] for entry in json.loads(CORPUS_MANIFEST.read_text())['episodes'])


# --- durability -----------------------------------------------------------------------


@pytest.mark.parametrize('arm_id', ARMS)
def test_each_control_run_is_committed_once(arm_id):
    prefix = f'{CONTROLS_RELATIVE}/runs/{arm_id}/'
    manifests = [
        path for path in _tracked()
        if path.startswith(prefix) and path.endswith(f'/{RUN_MANIFEST_FILENAME}')
        and path.count('/') == prefix.count('/') + 1
    ]

    assert len(manifests) == 1, manifests


def test_the_derived_artifacts_are_committed_and_no_journal_is():
    tracked = _tracked()

    for name in (PREREGISTRATION_INPUTS_FILENAME, FROZEN_REFERENCE.name):
        assert f'{CONTROLS_RELATIVE}/{name}' in tracked
    assert [path for path in tracked if '/journal/' in path] == []


# --- the runs -------------------------------------------------------------------------


@pytest.mark.parametrize('arm_id', ARMS)
def test_each_run_is_a_complete_passing_control_run(arm_id):
    run = _run(arm_id)

    assert run.spec.arm_id == arm_id
    assert not run.incomplete
    assert run.abort is None
    assert run.spec.arm_role == 'control'
    assert run.spec.preregistration_sha is None
    assert run.check_results
    assert all(check.passed for check in run.check_results), run.check_results
    assert {record.arm_id for record in _records(arm_id)} == {arm_id}


@pytest.mark.parametrize('arm_id', ARMS)
def test_each_committed_spec_is_the_spec_its_run_recorded(arm_id):
    assert load_arm_spec(CONTROLS / 'specs' / f'{arm_id}.json') == _run(arm_id).spec


def test_every_run_is_pinned_to_one_code_sha_and_the_committed_corpus():
    code_shas = {_run(arm_id).spec.code_sha for arm_id in ARMS}
    corpus_shas = {_run(arm_id).spec.corpus_sha for arm_id in ARMS}

    assert len(code_shas) == 1, code_shas
    assert corpus_shas == {corpus_sha(CORPUS_MANIFEST.read_bytes())}
    (code_sha,) = code_shas
    assert _git('merge-base', '--is-ancestor', code_sha, 'HEAD').returncode == 0


@pytest.mark.parametrize('arm_id', (ARM_A, ARM_B))
def test_the_full_runs_cover_the_whole_committed_corpus(arm_id):
    assert _run(arm_id).episode_ids == _manifest_episode_ids()


@pytest.mark.parametrize('arm_id', (GENERIC_20, OPENAI_20))
def test_the_screening_runs_cover_the_first_twenty_manifest_episodes(arm_id):
    assert _run(arm_id).episode_ids == _manifest_episode_ids()[:SCREENING_EPISODES]


# --- the pre-registration inputs ------------------------------------------------------


def _derived_inputs():
    run_a, run_b = _run_dir(ARM_A), _run_dir(ARM_B)
    return derive_preregistration_inputs(
        _run(ARM_A),
        _records(ARM_A),
        load_outcomes(run_a / OUTCOMES_FILENAME),
        _run(ARM_B),
        _records(ARM_B),
        load_outcomes(run_b / OUTCOMES_FILENAME),
        GraphSamenessDetails.model_validate_json(
            (run_b / GRAPH_SAMENESS_DETAILS_FILENAME).read_text()
        ),
    )


def test_the_committed_inputs_are_the_formula_over_the_committed_runs():
    inputs = _derived_inputs()

    committed = (CONTROLS / PREREGISTRATION_INPUTS_FILENAME).read_text()
    assert committed == serialize_preregistration_inputs(inputs)
    assert inputs.control_arm_ids == (ARM_A, ARM_B)
    assert all(math.isfinite(entry.margin) for entry in inputs.margins)
    assert {entry.metric_id for entry in inputs.margins} == set(GATED_METRICS) & LLM_METRIC_IDS


# --- client-class parity --------------------------------------------------------------


def test_the_parity_deltas_are_the_differences_of_the_two_screening_runs():
    parity_dir = _run_dir(GENERIC_20) / 'parity' / OPENAI_20
    deltas = load_metrics_records(parity_dir)
    generic = {record.metric.metric_id: record for record in _records(GENERIC_20)}
    openai = {record.metric.metric_id: record for record in _records(OPENAI_20)}

    assert {delta.metric.metric_id for delta in deltas} == generic.keys() & openai.keys()
    for delta in deltas:
        metric_id = delta.metric.metric_id
        assert delta.delta_of is not None
        assert delta.delta_of.minuend_arm_id == GENERIC_20
        assert delta.delta_of.subtrahend_arm_id == OPENAI_20
        assert delta.metric.value == generic[metric_id].metric.value - openai[metric_id].metric.value


# --- the frozen reference graph -------------------------------------------------------


def test_the_frozen_reference_names_run_as_graph():
    frozen = json.loads(FROZEN_REFERENCE.read_text())

    assert set(frozen) == {'graph', 'node_count', 'edge_count', 'topology_hash'}
    assert frozen['graph'] == _run(ARM_A).spec.scratch_group_id == FROZEN_GRAPH
    assert frozen['node_count'] > 0
