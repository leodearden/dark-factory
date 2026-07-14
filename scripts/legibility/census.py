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
