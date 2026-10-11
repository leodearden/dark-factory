"""Contract tests over the committed LME θ LLM-axis artifacts (task 3721).

``plans/local-memory-models-eval-llm/`` holds the incumbent's measured cost: the selected
production telemetry window, the ``incumbent-cost.json`` derived from it and from the two
ζ control runs, and the graphiti write history the availability assessment reads. θ ran no
arm, because η's screening verdict has no survivor. These tests re-derive the committed
cost byte for byte from the committed window, and pin the no-run premise to η's verdict:
if that verdict ever changes, θ's report is stale and this file fails.

Lane discipline: file reads plus ``git`` subprocesses only, and NO ``integration``
marker, so the merge lane's default selection runs them.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from fused_memory.arm_harness.arm_spec import LlmArmSpec, load_arm_spec
from fused_memory.arm_harness.incumbent_cost import (
    INCUMBENT_COST_FILENAME,
    PRODUCTION_TELEMETRY_FILENAME,
    TelemetryWindow,
    derive_incumbent_cost,
    load_incumbent_cost,
    load_llm_attempts,
    serialize_incumbent_cost,
)
from fused_memory.arm_harness.metrics_record import load_metrics_records
from fused_memory.arm_harness.screening import (
    SCREENING_VERDICT_FILENAME,
    ScreeningOutcome,
    load_screening_verdict,
)

REPO_ROOT = Path(__file__).parents[3]
LLM_AXIS_RELATIVE = 'plans/local-memory-models-eval-llm'
LLM_AXIS = REPO_ROOT / LLM_AXIS_RELATIVE
CONTROLS = REPO_ROOT / 'plans' / 'local-memory-models-eval-controls'
SCREENING_VERDICT = (
    REPO_ROOT / 'plans' / 'local-memory-models-eval-screening' / SCREENING_VERDICT_FILENAME
)
PRICING_ARM_ID = 'incumbent-generic-a'
CONTROL_RUNS = (
    CONTROLS / 'runs' / 'incumbent-generic-a' / '20261007T011000Z',
    CONTROLS / 'runs' / 'incumbent-generic-b' / '20261007T013800Z',
)
COMMITTED_FILES = frozenset({
    'README.md',
    PRODUCTION_TELEMETRY_FILENAME,
    INCUMBENT_COST_FILENAME,
    'graphiti-write-history.jsonl',
})


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


def _pricing_spec() -> LlmArmSpec:
    spec = load_arm_spec(CONTROLS / 'specs' / f'{PRICING_ARM_ID}.json')
    assert isinstance(spec, LlmArmSpec)
    return spec


def test_the_evidence_directory_commits_exactly_the_derived_artifacts_and_no_run():
    tracked = _git('ls-files', '--', LLM_AXIS_RELATIVE).stdout.splitlines()

    assert {path.removeprefix(f'{LLM_AXIS_RELATIVE}/') for path in tracked} == COMMITTED_FILES
    assert len(tracked) == len(COMMITTED_FILES)


def test_the_committed_cost_re_derives_byte_for_byte_from_the_committed_window():
    committed_text = (LLM_AXIS / INCUMBENT_COST_FILENAME).read_text()
    committed = load_incumbent_cost(LLM_AXIS / INCUMBENT_COST_FILENAME)
    window = TelemetryWindow(
        start=committed.production.window_start,
        end=committed.production.window_end,
        attempts=load_llm_attempts(LLM_AXIS / PRODUCTION_TELEMETRY_FILENAME),
    )

    derived = derive_incumbent_cost(
        window,
        pricing_spec=_pricing_spec(),
        control_records=[load_metrics_records(run) for run in CONTROL_RUNS],
    )

    assert serialize_incumbent_cost(derived) == committed_text


def test_the_production_spend_is_priced_at_the_controls_price():
    committed = load_incumbent_cost(LLM_AXIS / INCUMBENT_COST_FILENAME)

    assert committed.pricing_arm_id == PRICING_ARM_ID
    assert committed.pricing == _pricing_spec().pricing


def test_theta_ran_no_arm_because_screening_left_no_survivor():
    verdict = load_screening_verdict(SCREENING_VERDICT)

    assert verdict.survivors == ()
    assert verdict.outcome is ScreeningOutcome.NEGATIVE_VERDICT
