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
from pathlib import Path

import pytest

from fused_memory.arm_harness.screening import (
    SCREENING_VERDICT_FILENAME,
    GateId,
    GateVerdict,
    ScreeningOutcome,
    load_screening_verdict,
)

REPO_ROOT = Path(__file__).parents[3]
LLM_RECORD = 'plans/local-memory-models-eval-decision-llm.md'
EMBEDDING_RECORD = 'plans/local-memory-models-eval-decision-embedding.md'
SCREENING_VERDICT = (
    REPO_ROOT / 'plans' / 'local-memory-models-eval-screening' / SCREENING_VERDICT_FILENAME
)


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
