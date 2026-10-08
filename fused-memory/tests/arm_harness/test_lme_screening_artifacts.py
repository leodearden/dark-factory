"""Contract tests over the committed LME η screening artifacts (task 3720).

``plans/local-memory-models-eval-screening/`` holds one sweep's evidence: per-arm specs,
command records, α's VRAM readings, the usage tap's calls and the pinned runs. It also
holds ``screening-verdict.json``, the pre-registered survivor rule's output over that
evidence. These tests re-derive the verdict from the committed evidence and compare
what the rule decided: the pins, the envelope, the cap, the survivors, the outcome, and
each arm's gates by verdict, value, bound, margin and unit. So the committed decision
cannot drift from the rule or the evidence. Gate detail prose and the non-gating
reported block are deliberately left out of the comparison: rewording them does not
mean rewriting this dated artifact. The tests also couple the evidence to the
committed arms.yaml slate: a slate change invalidates η's verdict.

Lane discipline: file reads plus ``git`` subprocesses only, and NO ``integration``
marker, so the merge lane's default selection runs them.
"""

import json
import shutil
import subprocess
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from fused_memory.arm_harness.arm_spec import LlmArmSpec, load_arm_spec
from fused_memory.arm_harness.corpus import corpus_sha
from fused_memory.arm_harness.instrument_checks import InstrumentCheckId
from fused_memory.arm_harness.preregistration import (
    PREREGISTRATION_INPUTS_FILENAME,
    PreregistrationInputs,
    load_preregistration_inputs,
)
from fused_memory.arm_harness.run import OUTCOMES_FILENAME, load_outcomes
from fused_memory.arm_harness.screening import (
    SCREENING_VERDICT_FILENAME,
    SURVIVOR_CAP,
    ScreeningVerdict,
    derive_screening_verdict,
    load_screening_verdict,
)
from fused_memory.arm_harness.screening_evidence import (
    SCREENING_RUN_SHAPE,
    ArmEvidence,
    ArmEvidencePaths,
    load_arm_evidence,
)
from fused_memory.arm_harness.slate import LOOPBACK_HOST, SlateArm, candidate_spec, load_llm_slate

REPO_ROOT = Path(__file__).parents[3]
SCREENING_RELATIVE = 'plans/local-memory-models-eval-screening'
SCREENING = REPO_ROOT / SCREENING_RELATIVE
CONTROLS = REPO_ROOT / 'plans' / 'local-memory-models-eval-controls'
ARMS_YAML = REPO_ROOT / 'scripts' / 'local-model-serving' / 'arms.yaml'
CORPUS_MANIFEST = (
    REPO_ROOT / 'fused-memory' / 'scripts' / 'local_memory_models_eval' / 'corpus_manifest.json'
)
CONTROL_A = 'incumbent-generic-a'
PREREGISTRATION_DOC = 'plans/local-memory-models-eval-preregistration.md'
SLATE: tuple[SlateArm, ...] = load_llm_slate(ARMS_YAML)
SLATE_IDS = [arm.arm_id for arm in SLATE]


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
    return _git('ls-files', '--', SCREENING_RELATIVE).stdout.splitlines()


@cache
def _inputs() -> PreregistrationInputs:
    return load_preregistration_inputs(CONTROLS / PREREGISTRATION_INPUTS_FILENAME)


@cache
def _evidence(arm_id: str) -> ArmEvidence:
    arm = next(arm for arm in SLATE if arm.arm_id == arm_id)
    return load_arm_evidence(ArmEvidencePaths(SCREENING, arm_id), arm)


def _control_a_outcomes_path() -> Path:
    stamps = sorted(path for path in (CONTROLS / 'runs' / CONTROL_A).iterdir() if path.is_dir())
    assert len(stamps) == 1, stamps
    return stamps[0] / OUTCOMES_FILENAME


def _control_a_params():
    spec = load_arm_spec(CONTROLS / 'specs' / f'{CONTROL_A}.json')
    assert isinstance(spec, LlmArmSpec)
    return spec.params


def _first_manifest_episodes() -> tuple[str, ...]:
    episodes = json.loads(CORPUS_MANIFEST.read_text())['episodes']
    return tuple(entry['uuid'] for entry in episodes[: SCREENING_RUN_SHAPE.limit])


def _committed_verdict() -> ScreeningVerdict:
    return load_screening_verdict(SCREENING / SCREENING_VERDICT_FILENAME)


# --- durability -----------------------------------------------------------------------


def test_the_verdict_and_its_readme_are_committed_and_no_journal_is():
    tracked = _tracked()

    assert f'{SCREENING_RELATIVE}/{SCREENING_VERDICT_FILENAME}' in tracked
    assert f'{SCREENING_RELATIVE}/README.md' in tracked
    assert [path for path in tracked if '/journal/' in path] == []


def test_the_evidence_covers_exactly_the_committed_slate():
    spec_ids = sorted(path.stem for path in (SCREENING / 'specs').glob('*.json'))
    arm_dirs = sorted(path.name for path in (SCREENING / 'arms').iterdir() if path.is_dir())

    assert spec_ids == sorted(SLATE_IDS)
    assert arm_dirs == sorted(SLATE_IDS)


# --- the specs ------------------------------------------------------------------------


@pytest.mark.parametrize('arm_id', SLATE_IDS)
def test_each_spec_is_the_slate_arms_candidate_spec_through_a_loopback_tap(arm_id):
    evidence = _evidence(arm_id)
    spec = evidence.spec

    assert spec == candidate_spec(
        evidence.arm,
        base_url=spec.serving.base_url,
        code_sha=_inputs().code_sha,
        corpus_sha=corpus_sha(CORPUS_MANIFEST.read_bytes()),
        preregistration_sha=spec.preregistration_sha or '',
        params=_control_a_params(),
    )
    assert urlsplit(spec.serving.base_url).hostname == LOOPBACK_HOST


def test_one_preregistration_sha_that_carries_the_doc_and_is_on_this_history():
    shas = {_evidence(arm_id).spec.preregistration_sha for arm_id in SLATE_IDS}

    assert len(shas) == 1
    (sha,) = shas
    assert sha is not None
    assert _git('cat-file', '-e', f'{sha}:{PREREGISTRATION_DOC}').returncode == 0
    assert _git('merge-base', '--is-ancestor', sha, 'HEAD').returncode == 0


# --- the runs -------------------------------------------------------------------------


@pytest.mark.parametrize('arm_id', SLATE_IDS)
def test_each_served_run_is_a_pinned_screening_run_of_the_committed_spec(arm_id):
    evidence = _evidence(arm_id)
    if not evidence.served:
        assert evidence.run is None
        return
    run = evidence.run
    assert run is not None

    assert run.episode_ids == _first_manifest_episodes()
    assert run.spec == evidence.spec
    assert run.spec.code_sha == _inputs().code_sha
    checks = {check.check_id: check.passed for check in run.check_results}
    assert checks[InstrumentCheckId.CODE_SHA_MATCHES_CHECKOUT] is True
    assert checks[InstrumentCheckId.PREREGISTRATION_SHA] is True


# --- the verdict ----------------------------------------------------------------------


def _decision(verdict: ScreeningVerdict) -> dict[str, object]:
    return {
        'pins': (verdict.preregistration_sha, verdict.code_sha, verdict.corpus_sha),
        'envelope': verdict.envelope,
        'cap': verdict.cap,
        'survivors': verdict.survivors,
        'outcome': verdict.outcome,
        'arms': [
            (
                arm.arm_id, arm.stack, arm.reasoning, arm.served, arm.survives,
                [
                    (gate.gate, gate.verdict, gate.value, gate.bound, gate.margin, gate.unit)
                    for gate in arm.gates
                ],
            )
            for arm in verdict.arms
        ],
    }


def test_the_committed_verdict_is_the_rule_over_the_committed_evidence():
    verdict = derive_screening_verdict(
        SLATE,
        {arm_id: _evidence(arm_id) for arm_id in SLATE_IDS},
        _inputs(),
        load_outcomes(_control_a_outcomes_path()),
    )

    assert _decision(_committed_verdict()) == _decision(verdict)


def test_each_arm_was_screened_in_its_manifest_reasoning_mode():
    by_arm = {arm.arm_id: arm.reasoning for arm in _committed_verdict().arms}

    assert by_arm == {arm.arm_id: arm.reasoning for arm in SLATE}


def test_the_survivor_cap_holds_and_does_not_bind():
    verdict = _committed_verdict()

    assert len(verdict.survivors) <= SURVIVOR_CAP
    assert verdict.cap.binds is False
