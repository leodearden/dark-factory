"""Hermetic dry-run integration test for orchestrator.evals.prompt_opt.engine — the G2 signal.

See plans/tier1-prompt-optimization-prd.md T6. Runs the whole
run_optimization_loop over a tiny synthetic fixture corpus with fully
injected async fakes (no real invoke_agent call anywhere) and asserts the
full set of loop invariants: bounded edits (LR 4->2), a rejected-edit
buffer, frontier-optimizer minibatch reflection, acceptance ONLY on paired
+ N-repeat strict improvement of the selection split scored on the
executor with delta > measured band, a heuristics candidate + accept/reject
record + provenance, and that the CONTRACT is NEVER mutated.

Fixture design (see the docstrings on `_FakeScorer`/`_ScriptedProposeFn`):
a deterministic "quality mark" embedded in each heuristics text stands in
for the true (noiseless) quality of that heuristics block, and a small
per-(item, heuristics-text) occurrence-keyed jitter cycle stands in for the
scorer's inherent repeat-to-repeat noise (D-5: "the reviewer scorer is a
noisy haiku matcher"). Both are fully deterministic (no real randomness),
so the whole run is reproducible byte-for-byte.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from shared.prompt_artifact import ArtifactProvenance, PromptSpec, compose_prompt

from orchestrator.evals.prompt_opt import LoopResult, Scorer, run_optimization_loop
from orchestrator.evals.prompt_opt.engine import LoopResult as _EngineLoopResult
from orchestrator.evals.prompt_opt.engine import run_optimization_loop as _engine_run_optimization_loop

EXECUTOR_MODEL = 'exec-x'
OPTIMIZER_MODEL = 'frontier-y'

_CONTRACT = 'You are the executor. Follow the HEURISTICS below exactly. [CONTRACT-SENTINEL]'

_BASELINE_HEURISTICS = 'BASELINE: be concise. QUALITY_MARK=0.50'
_ACCEPTED_CANDIDATE = 'BASELINE: be concise. Also: cite line numbers explicitly. QUALITY_MARK=0.70'
# Same measured quality as _ACCEPTED_CANDIDATE (a cosmetic-only edit) — the
# "no-op/within-band" candidate for step 1.
_NOOP_CANDIDATE = (
    'BASELINE: be concise. Also: cite line numbers explicitly, please. QUALITY_MARK=0.70'
)

_QUALITY_RE = re.compile(r'QUALITY_MARK=([0-9.]+)')
_JITTER_CYCLE = (0.03, -0.03)


def _quality_of(text: str) -> float:
    match = _QUALITY_RE.search(text)
    assert match is not None, f'fixture text missing QUALITY_MARK: {text!r}'
    return float(match.group(1))


def _make_corpus(n: int = 20) -> list[str]:
    return [f'item-{i}' for i in range(n)]


async def _fake_rollout_fn(composed_prompt: str, item: Any, model: str) -> str:
    """The hermetic RolloutFn fake: always executor_model, echoes composed_prompt.

    Standing in for a real executor call, the "rollout" is just a string that
    deterministically encodes (item, composed_prompt) — enough for
    `_FakeScorer` to recover both the heuristics-conditioned quality mark and
    a stable per-(item, heuristics) occurrence key.
    """
    assert model == EXECUTOR_MODEL, f'rollout_fn must ALWAYS be called with executor_model, got {model!r}'
    return f'rollout::{item}::{composed_prompt}'


class _FakeScorer:
    """Deterministic Scorer fake: quality mark + a small occurrence-keyed jitter cycle.

    Every distinct (item, heuristics-text) pair — i.e. every distinct
    `rollout` string, since rollout_fn's output is a pure function of exactly
    those two things plus the constant CONTRACT — gets its own independent
    occurrence counter, so repeat N of that exact pair always draws the same
    jitter value. This simulates bounded, reproducible scorer noise (D-5)
    without any real randomness.
    """

    def __init__(self) -> None:
        self._occurrence_counts: dict[str, int] = {}

    async def score(self, item: Any, rollout: Any) -> float:
        count = self._occurrence_counts.get(rollout, 0)
        self._occurrence_counts[rollout] = count + 1
        jitter = _JITTER_CYCLE[count % len(_JITTER_CYCLE)]
        return _quality_of(rollout) + jitter


class _ScriptedProposeFn:
    """Fake ProposeFn: step 0 proposes an improving edit, step 1 a no-op/within-band edit."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        current_heuristics: str,
        scored_minibatch: list[Any],
        rejected_buffer: list[str],
        *,
        max_edits: int,
        optimizer_model: str,
    ) -> str:
        self.calls.append({
            'current_heuristics': current_heuristics,
            'scored_minibatch': list(scored_minibatch),
            'rejected_buffer': list(rejected_buffer),
            'max_edits': max_edits,
            'optimizer_model': optimizer_model,
        })
        step_index = len(self.calls) - 1
        return _ACCEPTED_CANDIDATE if step_index == 0 else _NOOP_CANDIDATE


class TestRunOptimizationLoopDryRun:
    @pytest.mark.asyncio
    async def test_full_hermetic_loop(self) -> None:
        spec = PromptSpec(
            prompt_id='trial-prompt',
            contract=_CONTRACT,
            baseline_heuristics=_BASELINE_HEURISTICS,
        )
        corpus = _make_corpus(20)
        scorer = _FakeScorer()
        propose_fn = _ScriptedProposeFn()
        rollout_calls_models: list[str] = []

        async def rollout_fn(composed_prompt: str, item: Any, model: str) -> str:
            rollout_calls_models.append(model)
            return await _fake_rollout_fn(composed_prompt, item, model)

        result = await run_optimization_loop(
            spec=spec,
            corpus=corpus,
            scorer=scorer,
            executor_model=EXECUTOR_MODEL,
            optimizer_model=OPTIMIZER_MODEL,
            rollout_fn=rollout_fn,
            propose_fn=propose_fn,
            n_repeats=3,
            max_steps=2,
            split_seed=7,
            minibatch_size=2,
            git_sha='deadbeef',
            date='2026-07-14',
            harness_version='harness-test-v1',
        )

        # --- bounded edits: LR requested to propose_fn is 4 then decays toward 2 ---
        assert [c['max_edits'] for c in propose_fn.calls] == [4, 3]

        # --- frontier-optimizer minibatch reflection: current heuristics + rejected
        # buffer state threaded correctly across steps ---
        assert propose_fn.calls[0]['current_heuristics'] == _BASELINE_HEURISTICS
        assert propose_fn.calls[0]['rejected_buffer'] == []
        assert propose_fn.calls[0]['scored_minibatch']  # non-empty: real rollouts scored
        assert propose_fn.calls[1]['current_heuristics'] == _ACCEPTED_CANDIDATE
        assert propose_fn.calls[1]['rejected_buffer'] == []  # nothing rejected before step 1

        # --- propose_fn ONLY ever called with the frontier optimizer model ---
        assert {c['optimizer_model'] for c in propose_fn.calls} == {OPTIMIZER_MODEL}

        # --- rollout_fn ONLY ever called with the executor model (scored on the
        # ACTUAL executor model, never the optimizer) ---
        assert rollout_calls_models
        assert set(rollout_calls_models) == {EXECUTOR_MODEL}

        # --- rejected-edit buffer holds the rejected no-op candidate ---
        assert result.rejected_buffer == [_NOOP_CANDIDATE]

        # --- acceptance ONLY on paired + N-repeat strict improvement over the
        # measured band; the within-band step is rejected, never a bare tie ---
        assert len(result.accept_records) == 2
        accepted = [r for r in result.accept_records if r.accepted]
        rejected = [r for r in result.accept_records if not r.accepted]
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert accepted[0].paired_delta > accepted[0].band
        assert rejected[0].paired_delta > 0  # not a trivial zero/tie delta
        assert rejected[0].paired_delta <= rejected[0].band  # but within the noise floor

        # --- result shape: a LoopResult with the accepted candidate as final,
        # a schema-valid 8-field ArtifactProvenance, and a TEST score ---
        assert isinstance(result, LoopResult)
        assert result.final_heuristics == _ACCEPTED_CANDIDATE
        assert isinstance(result.test_score, float)

        ArtifactProvenance.model_validate(result.provenance.model_dump())
        assert result.provenance.optimizer_model == OPTIMIZER_MODEL
        assert result.provenance.split_seed == 7
        assert result.provenance.git_sha == 'deadbeef'
        assert result.provenance.date == '2026-07-14'
        assert result.provenance.harness_version == 'harness-test-v1'
        assert result.provenance.held_out_TEST_score == result.test_score
        assert result.provenance.accept_delta == accepted[0].paired_delta

        # --- the CONTRACT is NEVER mutated: compose_prompt(contract, final)
        # still starts with the contract byte-for-byte ---
        composed = compose_prompt(spec.contract, result.final_heuristics)
        assert composed.startswith(spec.contract)

    def test_public_api_importable_from_package_top_level(self) -> None:
        # Importing at module level above already proves this; assert
        # identity with the submodule's originals for good measure.
        assert run_optimization_loop is _engine_run_optimization_loop
        assert LoopResult is _EngineLoopResult
        assert isinstance(Scorer, type)
