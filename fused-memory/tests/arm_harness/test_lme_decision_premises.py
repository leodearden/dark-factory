"""Premise pins for LME λ's two decision records (task 3723).

Each test re-applies the pre-registered rule's committed code to the committed evidence the
record rests on. When one fails, the record it names is stale: its proposed verdict no
longer follows from the evidence, and λ's synthesis must be redone before μ (task 3725)
rules on it.

Lane discipline: file reads plus ``git`` subprocesses only, and NO ``integration``
marker, so the merge lane's default selection runs them.
"""

import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from fused_memory.arm_harness.embedding_preregistration import (
    DECIDING_CONFIGURATION,
    EMBEDDING_PREREGISTRATION_INPUTS_FILENAME,
    EmbeddingComparison,
    compare_embedding_arm,
    load_embedding_preregistration_inputs,
)
from fused_memory.arm_harness.embedding_run_manifest import load_embedding_run_manifest
from fused_memory.arm_harness.metrics_record import EmbeddingMetricId, load_metrics_records
from fused_memory.arm_harness.run import RUN_MANIFEST_FILENAME
from fused_memory.arm_harness.screening import (
    SCREENING_VERDICT_FILENAME,
    GateId,
    GateVerdict,
    ScreeningOutcome,
    load_screening_verdict,
)
from fused_memory.arm_harness.slate import load_embedding_slate

REPO_ROOT = Path(__file__).parents[3]
LLM_RECORD = 'plans/local-memory-models-eval-decision-llm.md'
EMBEDDING_RECORD = 'plans/local-memory-models-eval-decision-embedding.md'
SCREENING_VERDICT = (
    REPO_ROOT / 'plans' / 'local-memory-models-eval-screening' / SCREENING_VERDICT_FILENAME
)
EMBEDDING = REPO_ROOT / 'plans' / 'local-memory-models-eval-embedding'
ARMS_YAML = REPO_ROOT / 'scripts' / 'local-model-serving' / 'arms.yaml'
NON_INFERIOR_EMBEDDERS = frozenset({'granite-embedding-english-r2', 'qwen3-embedding-0.6b'})
INFERIOR_EMBEDDERS = frozenset({'qwen3-embedding-4b', 'gte-modernbert-base'})


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


def test_the_llm_decision_record_is_committed():
    assert LLM_RECORD in _git('ls-files', '--', LLM_RECORD).stdout.splitlines()


def test_every_llm_arm_fell_at_the_throughput_floor_alone_so_none_reached_the_section_5_comparison():
    verdict = load_screening_verdict(SCREENING_VERDICT)

    assert verdict.outcome is ScreeningOutcome.NEGATIVE_VERDICT
    assert {
        arm.arm_id: {gate.gate for gate in arm.gates if gate.verdict is GateVerdict.FAIL}
        for arm in verdict.arms
    } == {
        'qwen3.5-9b': {GateId.THROUGHPUT_FLOOR},
        'phi-4-14b': {GateId.THROUGHPUT_FLOOR},
        'moe-stretch': {GateId.THROUGHPUT_FLOOR},
    }


def _run_dir(arm_id: str) -> Path:
    runs = EMBEDDING / 'runs' / arm_id
    stamps = sorted(path for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []
    assert len(stamps) == 1, f'{arm_id} must have exactly one committed stamp dir: {stamps}'
    return stamps[0]


@cache
def _embedding_comparisons() -> dict[str, EmbeddingComparison]:
    inputs = load_embedding_preregistration_inputs(
        EMBEDDING / EMBEDDING_PREREGISTRATION_INPUTS_FILENAME
    )
    comparisons: dict[str, EmbeddingComparison] = {}
    for arm in load_embedding_slate(ARMS_YAML):
        run_dir = _run_dir(arm.arm_id)
        comparisons[arm.arm_id] = compare_embedding_arm(
            inputs,
            load_embedding_run_manifest(run_dir / RUN_MANIFEST_FILENAME),
            load_metrics_records(run_dir),
        )
    return comparisons


def test_the_embedding_decision_record_is_committed():
    assert EMBEDDING_RECORD in _git('ls-files', '--', EMBEDDING_RECORD).stdout.splitlines()


def test_the_preregistered_rule_admits_exactly_granite_and_qwen3_0_6b():
    comparisons = _embedding_comparisons()

    assert {arm for arm, c in comparisons.items() if c.non_inferior} == NON_INFERIOR_EMBEDDERS
    assert set(comparisons) == NON_INFERIOR_EMBEDDERS | INFERIOR_EMBEDDERS


def test_each_inferior_embedder_fails_only_with_indices_recall_at_10_and_every_envelope_admits():
    comparisons = _embedding_comparisons()

    for arm in INFERIOR_EMBEDDERS:
        failing = {
            (row.metric_id, row.index_configuration)
            for row in comparisons[arm].margins
            if row.index_configuration is DECIDING_CONFIGURATION and not row.admits
        }
        assert failing == {(EmbeddingMetricId.KNOWN_ITEM_RECALL_AT_10, DECIDING_CONFIGURATION)}, arm
    assert all(c.envelope.admits for c in comparisons.values())
