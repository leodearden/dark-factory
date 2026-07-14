#!/usr/bin/env python3
"""scripts/legibility/census.py — periodic legibility census runner.

Task eta (η) of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.7 "Census runner", contract §7, decisions 4/5/6/9, boundary test §8.7).
The LEAF that consumes every prior legibility module: beta's (β)
inventory/sampling, alpha's (α) digest, delta's (δ) trickle coder, gamma's
(γ) codebook merger, zeta's (ζ) census trigger/state.

Flow (§5.7): usage-headroom preflight -> stratified-random saturation
mining (Sonnet miners) until >=dup_rate duplicates for consecutive_batches
-> Sonnet verification of novel clusters vs current main -> Fable
synthesis into a dated ``plans/confusion-census-<date>.md`` report with
the origin x manifestation matrix -> curator-path ``submit_task``
remediation filing -> codebook update (promote/reject candidates, retire
fixed, never delete) -> advance ``docs/legibility/census-state.json``.
``--force`` for operator-initiated runs.

Every LLM / MCP / git side effect in this module is an INJECTED seam
(``invoke``, ``verify_fn``, ``synthesize_fn``, ``submit_fn``,
``escalate_fn``, ``status_fetcher``, ``commit``, ``batch_source``) --
mirrors delta's ``coder.code_digests(invoke=)`` and zeta's
``census_trigger(status_fetcher=)``. The scripts/ test env (``uv run
--project shared``) has no MCP client / httpx / live models, so every
seam is ALWAYS faked in this module's own test suite; the deterministic
core (duplicate/dup_rate, the mining batch loop + saturation stop, the
origin x manifestation matrix, census-state advance, codebook lifecycle
transforms, report rendering) is unit-tested with no network.

Model routing (ratified static policy, PRD §5/§12 -- deliberately NOT the
adaptive ``resolve_route`` ladder): Sonnet for mining + verification,
Fable ONLY for synthesis, read from ``config.Models``
(``census_miner``/``census_verify``/``census_synthesis``).

Codebook handling (PRD decision 1 -- sole writer, never delete): new
mining sightings/candidates merge through the EXISTING
``codebook.apply_coding_record``. Census-only lifecycle transitions
(promote/reject/retire) are deterministic in-memory transforms in this
module, persisted via ``codebook.dump()`` after ``codebook.validate()``
and ``codebook.assert_no_deletion()`` confirm the write is safe.
``codebook.py`` itself is NOT modified.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/census.py` runs
# -- must run BEFORE the `legibility.*` imports below, since a direct
# script invocation puts only scripts/legibility/ (not scripts/) on
# sys.path. Skipped under pytest/normal package import: __name__ is
# 'legibility.census' or 'census', never '__main__' (mirrors
# sampling.py:37-38).
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codebook  # noqa: E402
import coder  # noqa: E402
import config  # noqa: E402
import digest  # noqa: E402
import inventory  # noqa: E402
from legibility import census_trigger  # noqa: E402


# ---------------------------------------------------------------------------
# is_duplicate / batch_dup_rate — the saturation-mining novelty signal
# ---------------------------------------------------------------------------

def is_duplicate(record: dict) -> bool:
    """A §7.3 coding record is a *duplicate* (a re-observation of an
    already-known cause) iff it carries zero candidates -- a candidate is
    the sole novelty signal in a coding record; matches-only or entirely
    empty (``{"matches": [], "candidates": []}``) both mean "nothing new
    here"."""
    return len(record.get("candidates") or []) == 0


def batch_dup_rate(records: list[dict]) -> float:
    """Fraction of *records* that are duplicates (``is_duplicate`` True).
    Returns ``0.0`` for an empty batch rather than raising
    ZeroDivisionError -- an empty/all-failed batch has no signal either
    way, and the saturation loop must be able to treat it as "not
    saturated" without crashing."""
    if not records:
        return 0.0
    return sum(1 for record in records if is_duplicate(record)) / len(records)


# ---------------------------------------------------------------------------
# mine_to_saturation — stratified-random batch loop + saturation stop
# ---------------------------------------------------------------------------

@dataclass
class BatchStats:
    """Per-batch mining tally: how one ``coder.code_digests`` batch scored
    against the saturation threshold."""

    index: int
    total: int
    succeeded: int
    failed: int
    dup_rate: float
    saturated: bool


@dataclass
class MiningResult:
    """Outcome of ``mine_to_saturation``: every successfully-coded record
    across every consumed batch, per-batch stats, and why mining stopped
    (``"saturated"`` -- ``config.consecutive_batches`` consecutive batches
    at/above ``config.dup_rate``; ``"exhausted"`` -- ``batch_source`` ran
    out first)."""

    records: list[dict] = field(default_factory=list)
    batch_stats: list[BatchStats] = field(default_factory=list)
    stop_reason: str = "exhausted"


def mine_to_saturation(
    batch_source, codebook_dict: dict, *, project: str, model: str, config, invoke,
) -> MiningResult:
    """Code batches from *batch_source* against *codebook_dict* via
    ``coder.code_digests`` until novelty saturates.

    Iterates *batch_source* lazily -- a batch is pulled only when the loop
    is ready to code it, so a source that stops yielding once mining
    saturates never has its later batches materialized. Per batch: code
    via ``coder.code_digests(batch, codebook_dict, project=project,
    model=model, invoke=invoke)``, compute this batch's ``dup_rate`` over
    its own successful records (``batch_dup_rate`` -- failed codings are
    excluded from the denominator per that function's own contract), and
    track a consecutive-saturated-batch counter: incremented when
    ``dup_rate >= config.dup_rate``, reset to 0 otherwise. Mining stops
    (``stop_reason="saturated"``) the moment the counter reaches
    ``config.consecutive_batches`` -- right after that Nth consecutive
    saturated batch, so no further batch is pulled. If *batch_source*
    exhausts before that, ``stop_reason="exhausted"``.

    *config* is a ``config.Saturation``-shaped object (``.dup_rate``,
    ``.consecutive_batches`` -- i.e. a project's
    ``LegibilityConfig.census.saturation``), not the whole
    ``LegibilityConfig``. *model* is the caller's already-resolved model
    id (Sonnet miner routing per the ratified static policy -- this
    function does not read ``config.Models`` itself).
    """
    result = MiningResult()
    consecutive_saturated = 0

    for index, batch in enumerate(batch_source):
        run_result = coder.code_digests(
            list(batch), codebook_dict, project=project, model=model, invoke=invoke,
        )
        result.records.extend(run_result.records)

        dup_rate = batch_dup_rate(run_result.records)
        saturated = dup_rate >= config.dup_rate
        result.batch_stats.append(
            BatchStats(
                index=index,
                total=run_result.total,
                succeeded=run_result.succeeded,
                failed=run_result.failed,
                dup_rate=dup_rate,
                saturated=saturated,
            )
        )

        consecutive_saturated = consecutive_saturated + 1 if saturated else 0
        if consecutive_saturated >= config.consecutive_batches:
            result.stop_reason = "saturated"
            return result

    result.stop_reason = "exhausted"
    return result
