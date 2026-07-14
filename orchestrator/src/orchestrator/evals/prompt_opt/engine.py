"""The T6 bespoke SkillOpt-discipline prompt-optimization loop.

See plans/tier1-prompt-optimization-prd.md T6. Orchestrates: split the
corpus -> measure the repeatability band from N repeats of the baseline
heuristics on the selection split -> per step (sample a train minibatch,
roll it out + score it on the executor, propose a bounded heuristics edit,
paired N-repeat score current vs candidate on the selection split, accept
only on strict improvement beyond the band) -> score the final heuristics
on TEST once -> assemble an ArtifactProvenance sidecar.

Every external effect — the executor rollout, the optimizer reflection, the
scoring — is dependency-injected (`rollout_fn`, `propose_fn`, `scorer`), so
this module makes no real LLM calls itself and is fully exercised
hermetically by test_prompt_opt_engine.py.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from shared.prompt_artifact import ArtifactProvenance, PromptSpec, compose_prompt

from orchestrator.evals.prompt_opt.optimizer import propose_heuristics_edit
from orchestrator.evals.prompt_opt.scorer import ProposeFn, RolloutFn, ScoredItem, Scorer
from orchestrator.evals.prompt_opt.splits import split_corpus
from orchestrator.evals.prompt_opt.variance import (
    AcceptanceRecord,
    evaluate_acceptance,
    measure_repeatability_band,
    textual_lr,
)

__all__ = ['LoopResult', 'run_optimization_loop']


@dataclass(frozen=True)
class LoopResult:
    """The full output of one :func:`run_optimization_loop` run."""

    final_heuristics: str
    accept_records: list[AcceptanceRecord]
    rejected_buffer: list[str]
    provenance: ArtifactProvenance
    test_score: float


def _stable_corpus_hash(corpus: list[Any]) -> str:
    """A deterministic hash of *corpus*'s contents (no wall-clock/subprocess dependency)."""
    payload = '|'.join(repr(item) for item in corpus)
    return 'sha256:' + hashlib.sha256(payload.encode('utf-8')).hexdigest()


async def run_optimization_loop(
    spec: PromptSpec,
    corpus: list[Any],
    scorer: Scorer,
    executor_model: str,
    optimizer_model: str,
    *,
    rollout_fn: RolloutFn,
    propose_fn: ProposeFn = propose_heuristics_edit,
    n_repeats: int,
    max_steps: int,
    split_seed: int,
    minibatch_size: int,
    git_sha: str,
    date: str,
    harness_version: str,
    corpus_hash: str | None = None,
) -> LoopResult:
    """Run the bespoke SkillOpt-discipline optimization loop.

    *rollout_fn* is ALWAYS invoked with ``model=executor_model`` — never
    *optimizer_model* — so every acceptance decision is structurally
    guaranteed to be "scored on the ACTUAL executor model". *propose_fn* is
    ALWAYS invoked with the frontier *optimizer_model* and is handed ONLY
    the current heuristics text (never *spec.contract*), so the CONTRACT is
    un-editable by construction (D-3): every rollout prompt is built via
    :func:`shared.prompt_artifact.compose_prompt(spec.contract, heuristics)`,
    which guarantees the CONTRACT is always a byte-identical prefix of the
    composed text regardless of what the optimizer proposes.

    A candidate is accepted only when its paired selection-split delta
    (scored with *n_repeats* repeats of both the current and candidate
    heuristics) strictly exceeds a repeatability band pre-measured from
    *n_repeats* repeats of the STARTING heuristics on the selection split —
    never bare tie-rejection over a noisy scorer (D-5). Rejected candidates
    accumulate in the returned ``rejected_buffer`` and are visible to
    subsequent *propose_fn* calls.

    *git_sha*/*date*/*harness_version*/*corpus_hash*/*split_seed* are
    caller-supplied (never computed here) so this function has no
    wall-clock or subprocess dependency and is fully deterministic given
    deterministic *rollout_fn*/*scorer*/*propose_fn* — see
    test_prompt_opt_engine.py's hermetic dry-run.
    """
    split = split_corpus(corpus, seed=split_seed)
    current_heuristics = spec.baseline_heuristics
    rejected_buffer: list[str] = []
    accept_records: list[AcceptanceRecord] = []
    last_accepted_delta = 0.0

    async def _score_split(heuristics: str, items: list[Any]) -> list[float]:
        """Roll out + score *heuristics* over *items* once (one 'repeat'), on the executor."""
        scores = []
        for item in items:
            composed = compose_prompt(spec.contract, heuristics)
            rollout = await rollout_fn(composed, item, executor_model)
            scores.append(await scorer.score(item, rollout))
        return scores

    # Pre-measure the repeatability band from N repeats of the STARTING
    # heuristics on the selection split — the noise floor every step's
    # acceptance decision is measured against.
    band_batches = [await _score_split(current_heuristics, split.selection) for _ in range(n_repeats)]
    band = measure_repeatability_band(band_batches)

    for step in range(max_steps):
        max_edits = textual_lr(step)

        minibatch_rng = random.Random(split_seed + step)
        minibatch_items = minibatch_rng.sample(split.train, min(minibatch_size, len(split.train)))

        scored_minibatch: list[ScoredItem] = []
        for item in minibatch_items:
            composed = compose_prompt(spec.contract, current_heuristics)
            rollout = await rollout_fn(composed, item, executor_model)
            score = await scorer.score(item, rollout)
            scored_minibatch.append(ScoredItem(item=item, rollout=rollout, score=score))

        candidate = await propose_fn(
            current_heuristics,
            scored_minibatch,
            rejected_buffer,
            max_edits=max_edits,
            optimizer_model=optimizer_model,
        )

        base_repeats = [await _score_split(current_heuristics, split.selection) for _ in range(n_repeats)]
        cand_repeats = [await _score_split(candidate, split.selection) for _ in range(n_repeats)]
        record = evaluate_acceptance(base_repeats, cand_repeats, band)
        accept_records.append(record)

        if record.accepted:
            current_heuristics = candidate
            last_accepted_delta = record.paired_delta
        else:
            rejected_buffer.append(candidate)

    test_scores = await _score_split(current_heuristics, split.test)
    test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    provenance = ArtifactProvenance(
        optimizer_model=optimizer_model,
        corpus_hash=corpus_hash or _stable_corpus_hash(corpus),
        split_seed=split_seed,
        held_out_TEST_score=test_score,
        accept_delta=last_accepted_delta,
        git_sha=git_sha,
        date=date,
        harness_version=harness_version,
    )

    return LoopResult(
        final_heuristics=current_heuristics,
        accept_records=accept_records,
        rejected_buffer=rejected_buffer,
        provenance=provenance,
        test_score=test_score,
    )
