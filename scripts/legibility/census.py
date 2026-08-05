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
--project shared``) has no MCP client and no live models, so every seam is
ALWAYS faked in this module's own test suite; the deterministic core
(duplicate/dup_rate, the mining batch loop + saturation stop, the origin x
manifestation matrix, census-state advance, codebook lifecycle transforms,
report rendering) is unit-tested with no network. Note that httpx IS
available there -- a direct dependency of ``shared``
(``shared/pyproject.toml``, ``httpx>=0.27``, task 2965) -- so the seams are
what keep the suite off the network, not an absent HTTP client: an
un-faked poster would really reach whatever is listening on
``$FUSED_MEMORY_MCP_URL`` (default localhost:8002).

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

import argparse
import contextlib
import copy
import functools
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/census.py` runs
# -- must run BEFORE the `legibility.*` imports below, since a direct
# script invocation puts only scripts/legibility/ (not scripts/) on
# sys.path. Skipped under pytest/normal package import: __name__ is
# 'legibility.census' or 'census', never '__main__' (mirrors
# sampling.py:37-38).
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would scan cap-banner text using
# the MAIN checkout's marker list rather than its own. Same reasoning and same
# form as scripts/audit_combine_gate_marker_loss.py:74-84 and
# scripts/repair_wiped_metadata_files.py:65-75 (tasks 2881/2882/3329), with
# parents[2] rather than parent.parent because census.py sits one directory
# deeper (scripts/legibility/, not scripts/). Unconditional -- NOT inside the
# `__main__` guard above -- because the `shared.cap_markers` import it enables
# is module-level, so it must resolve under pytest and package import too.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

import codebook  # noqa: E402
import coder  # noqa: E402
import digest  # noqa: E402
import inventory  # noqa: E402
import sampling  # noqa: E402
from legibility import census_trigger  # noqa: E402
from shared.cap_markers import (  # noqa: E402
    BLOCKING_BANNER_MARKERS,
    looks_like_blocking_banner,
)

import config  # noqa: E402

logger = logging.getLogger("legibility.census")


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
    against the saturation threshold.

    ``status`` mirrors ``coder.RunResult.status`` (``"ok"`` or
    ``"failure"`` -- a storm, PRD §8.6, when this batch's failed/total
    fraction strictly exceeds 0.5): surfaced so a caller (``run_census``,
    or any future one) can tell a batch's ``dup_rate`` was drawn from a
    small, skewed, storm-degraded sample rather than a healthy one, even
    though ``saturated`` already accounts for it (a storm batch is never
    ``saturated=True`` -- see ``mine_to_saturation``)."""

    index: int
    total: int
    succeeded: int
    failed: int
    dup_rate: float
    saturated: bool
    status: str = "ok"


@dataclass
class MiningResult:
    """Outcome of ``mine_to_saturation``: every successfully-coded record
    across every consumed batch, per-batch stats, and why mining stopped
    (``"saturated"`` -- ``config.consecutive_batches`` consecutive batches
    at/above ``config.dup_rate``; ``"exhausted"`` -- ``batch_source`` ran
    out first; ``"capped"`` -- the operator's ``max_batches`` cap was
    reached, so coverage is deliberately PARTIAL).

    ``max_batches`` echoes the operator cap this run was given (``None``
    when uncapped -- the default). It travels on the result so
    ``render_report`` can state the cap and its coverage consequence from
    the mining facts it is already handed, without a second parameter."""

    records: list[dict] = field(default_factory=list)
    batch_stats: list[BatchStats] = field(default_factory=list)
    stop_reason: str = "exhausted"
    max_batches: int | None = None


def mine_to_saturation(
    batch_source, codebook_dict: dict, *, project: str, model: str, config, invoke,
    max_batches: int | None = None,
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
    ``dup_rate >= config.dup_rate`` AND the batch is not a storm, reset to
    0 otherwise. Mining stops (``stop_reason="saturated"``) the moment the
    counter reaches ``config.consecutive_batches`` -- right after that Nth
    consecutive saturated batch, so no further batch is pulled. If
    *batch_source* exhausts before that, ``stop_reason="exhausted"``.

    A STORM batch (``coder.RunResult.status == "failure"`` -- PRD §8.6,
    more than half the batch's digests failed to code) never counts toward
    saturation, however high its ``dup_rate`` computes: that rate is drawn
    from whatever handful of records happened to succeed, too small and
    too skewed a sample to trust, and a run of storms would otherwise be
    able to trip the saturation stop on degraded data rather than genuine
    novelty exhaustion. The batch's ``dup_rate`` is still recorded on its
    ``BatchStats`` (with ``status="failure"``) for visibility, but
    ``saturated`` is forced False and the consecutive-saturated counter is
    reset, exactly as if the batch scored below threshold.

    *max_batches* is the OPERATOR COST CAP (``--max-batches``): mining
    stops with ``stop_reason="capped"`` once that many batches have been
    coded. The cap is enforced here, inside the loop, rather than by
    islicing *batch_source* upstream, for two reasons: only this loop
    knows WHY it stopped, so ``"capped"`` stays distinguishable from a
    source that genuinely ran dry (an islice wrapper is indistinguishable
    from ``"exhausted"`` -- exactly the silent-cap failure the cap must
    not introduce), and the ``return`` happens before the next batch is
    pulled, so a capped-away batch costs no digest render and no
    ``coder.code_digests`` call. A capped run is deliberately PARTIAL
    coverage: sessions beyond the cap were never mined, and
    ``render_report`` says so in as many words. The saturation check
    deliberately PRECEDES the cap check, so a run that saturates on
    exactly the capped batch reports the stronger, more informative
    ``"saturated"`` (novelty genuinely exhausted -- coverage was
    sufficient regardless of the cap) rather than under-claiming as
    partial. ``max_batches=None`` (the default) is exactly today's
    behavior: no cap, no extra rendering, unbounded mining. A cap below 1
    raises ``ValueError`` rather than degrading silently: because the cap
    is checked AFTER a batch is coded, ``max_batches=0`` would still spend
    one full ``coder.code_digests`` call and then render a
    self-contradictory "mined 1 batch(es); operator batch cap = 0" line --
    a silently mis-honored cap on a parameter whose whole purpose is to be
    an explicit, legible bound.

    *config* is a ``config.Saturation``-shaped object (``.dup_rate``,
    ``.consecutive_batches`` -- i.e. a project's
    ``LegibilityConfig.census.saturation``), not the whole
    ``LegibilityConfig``. *model* is the caller's already-resolved model
    id (Sonnet miner routing per the ratified static policy -- this
    function does not read ``config.Models`` itself).
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError(
            f"max_batches must be >= 1 when set, got {max_batches!r} -- a cap of "
            "0 or less cannot be honored (the cap is checked after a batch is "
            "coded, so it would still spend one full coder.code_digests call) "
            "and would render a self-contradictory coverage line; omit it "
            "entirely for unbounded mining"
        )

    result = MiningResult(max_batches=max_batches)
    consecutive_saturated = 0

    for index, batch in enumerate(batch_source):
        run_result = coder.code_digests(
            list(batch), codebook_dict, project=project, model=model, invoke=invoke,
        )
        result.records.extend(run_result.records)

        dup_rate = batch_dup_rate(run_result.records)
        # A storm (run_result.status == "failure") never satisfies
        # saturation, no matter how high dup_rate computes over its few
        # successful records -- see the storm paragraph above.
        saturated = dup_rate >= config.dup_rate and run_result.status != "failure"
        result.batch_stats.append(
            BatchStats(
                index=index,
                total=run_result.total,
                succeeded=run_result.succeeded,
                failed=run_result.failed,
                dup_rate=dup_rate,
                saturated=saturated,
                status=run_result.status,
            )
        )

        consecutive_saturated = consecutive_saturated + 1 if saturated else 0
        if consecutive_saturated >= config.consecutive_batches:
            result.stop_reason = "saturated"
            return result

        # Checked AFTER saturation, deliberately -- see the max_batches
        # paragraph above. Returning here means batch N+1 is never pulled.
        if max_batches is not None and len(result.batch_stats) >= max_batches:
            logger.warning(
                "mining stopped at the operator batch cap: %d batch(es) mined "
                "(--max-batches=%d) -- coverage is PARTIAL, not saturated; "
                "sessions beyond the cap were NOT mined",
                len(result.batch_stats), max_batches,
            )
            result.stop_reason = "capped"
            return result

    result.stop_reason = "exhausted"
    return result


# ---------------------------------------------------------------------------
# compute_matrix / render_matrix — origin x manifestation matrix (PRD
# decision 6: the `unknown` phase is explicit, never inferred)
# ---------------------------------------------------------------------------

def compute_matrix(sightings: list[dict]) -> dict[str, dict[str, int]]:
    """Tally *sightings* into an origin x manifestation count structure.

    A sighting with a missing, ``None``, or empty ``origin_phase`` /
    ``manifested_phase`` is bucketed under the explicit ``"unknown"``
    phase -- NEVER inferred to a concrete one (PRD decision 6). Returns a
    nested dict ``{origin: {manifested: count}}`` covering exactly the
    phases actually observed (rows/cols this sparse by construction --
    ``"unknown"`` only appears when at least one sighting actually landed
    there), row/col-ordered per ``codebook.PHASES`` for a stable,
    deterministic iteration order. Every observed row carries every
    observed column key (0 for a combination that was never seen), so a
    caller never needs a ``.get(..., 0)`` fallback. An empty *sightings*
    list returns ``{}``.
    """
    counts: dict[tuple[str, str], int] = {}
    origins: set[str] = set()
    manifesteds: set[str] = set()

    for sighting in sightings:
        origin = sighting.get("origin_phase") or "unknown"
        manifested = sighting.get("manifested_phase") or "unknown"
        origins.add(origin)
        manifesteds.add(manifested)
        key = (origin, manifested)
        counts[key] = counts.get(key, 0) + 1

    ordered_origins = [p for p in codebook.PHASES if p in origins]
    ordered_manifesteds = [p for p in codebook.PHASES if p in manifesteds]

    return {
        origin: {
            manifested: counts.get((origin, manifested), 0)
            for manifested in ordered_manifesteds
        }
        for origin in ordered_origins
    }


def render_matrix(matrix: dict[str, dict[str, int]]) -> str:
    """Render *matrix* (``compute_matrix``'s output shape) as a
    deterministic markdown table: one row per origin phase, one column
    per manifestation phase, PHASES-ordered, 0 for any cell with no
    count. Returns a fixed placeholder (never an empty string or a
    header-only table) when *matrix* is empty."""
    if not matrix:
        return "_No sightings recorded._\n"

    manifesteds_seen: set[str] = set()
    for row in matrix.values():
        manifesteds_seen.update(row.keys())
    manifesteds = [p for p in codebook.PHASES if p in manifesteds_seen]
    origins = [p for p in codebook.PHASES if p in matrix]

    header = "| origin \\ manifested | " + " | ".join(manifesteds) + " |"
    separator = "| --- | " + " | ".join("---" for _ in manifesteds) + " |"
    rows = [
        "| " + origin + " | " + " | ".join(str(matrix[origin].get(m, 0)) for m in manifesteds) + " |"
        for origin in origins
    ]
    return "\n".join([header, separator, *rows]) + "\n"


# ---------------------------------------------------------------------------
# preflight_headroom — cheap usage-headroom probe (PRD decision 5: no
# usage API assumed -- one tiny call, scan its reply for a banner)
# ---------------------------------------------------------------------------

_HEADROOM_BANNER_MARKERS = BLOCKING_BANNER_MARKERS
"""Case-insensitive substrings that mark a usage-limit/auth banner reply
from the headless `claude` CLI, rather than a genuine model response.

A CONTRACT owned by ``shared.cap_markers``, not a literal to be edited
here. This name is kept only as an alias so a reader following the old
spelling still lands somewhere real; the list itself is validated against
verbatim real-CLI transcripts (``REAL_CLI_CAP_MESSAGES``) by both
``shared/tests/test_cap_markers.py`` and this module's own tests.

It used to be a four-entry literal here -- 'usage limit', 'rate limit',
'please run /login', 'invalid api key' -- which missed
"You've hit your weekly limit · resets Aug 5, 11am" entirely, so a weekly
cap passed this probe and every verify call after it was fail-closed
rejected as an ordinary verdict (task 3645). A near-identical list already
existed in shared/tests/_capacity_skip.py with the right coverage, but it
was test-only and unreachable from here. Add a newly-observed cap phrasing
to the corpus in ``shared.cap_markers``, with a transcript to cite, and
both suites go red until the markers cover it."""

_HEADROOM_PROBE_PROMPT = "ping"
"""The tiny probe prompt -- a cheap single round trip, not a real mining
call. Its content doesn't matter; only whether the reply carries a banner
marker or raises."""


@dataclass
class HeadroomResult:
    """Verdict from ``preflight_headroom``: ``ok=True`` means the probe
    round-tripped cleanly and mining may proceed; ``ok=False`` means it
    should be deferred to the next trigger evaluation, with ``reason``
    explaining why (a banner marker, or an invocation error)."""

    ok: bool
    reason: str | None = None


def preflight_headroom(invoke, *, model: str) -> HeadroomResult:
    """Issue one tiny probe via ``invoke(prompt, model)`` and decide
    whether the census has headroom to proceed.

    No usage API is assumed to exist (PRD decision 5) -- this is a cheap
    preflight probe, not a quota lookup. The reply is scanned
    case-insensitively for a known usage-limit/auth banner marker via
    ``shared.cap_markers.looks_like_blocking_banner`` (capacity OR auth --
    either means no useful model output is coming); a match defers, and
    the reason quotes the marker that fired. An invocation error
    raised by *invoke* (e.g. a ``CoderInvocationError``-shaped failure)
    is also treated as a deferral -- fail-safe, never a crash, since a
    probe failure is exactly the kind of "the model isn't reachable right
    now" signal this preflight exists to catch. Makes no mining decisions
    itself -- just the ok/deferred verdict + reason.
    """
    try:
        reply = invoke(_HEADROOM_PROBE_PROMPT, model)
    except Exception as exc:  # noqa: BLE001 - any probe failure must fail safe
        return HeadroomResult(ok=False, reason=f"headroom probe invocation failed: {exc}")

    marker = looks_like_blocking_banner(reply or "")
    if marker is not None:
        return HeadroomResult(
            ok=False, reason=f"headroom probe reply carries a banner marker: {marker!r}",
        )

    return HeadroomResult(ok=True, reason=None)


# ---------------------------------------------------------------------------
# build_task_payloads — verified clusters -> curator-path submit_task
# kwargs (PRD decision 9: normal task_kind, never planning_mode; PRD
# decision 4: a harness-rooted cluster may target dark_factory)
# ---------------------------------------------------------------------------

def _cluster_description(cluster: dict, *, project_id: str) -> str:
    """Factual cluster summary + evidence -- no prose routing intent
    (lesson `prose-routing-intent`: routing is expressed structurally, via
    payload fields, never as English directives embedded in the text)."""
    lines = [
        cluster.get("summary")
        or cluster.get("title")
        or "Confusion cluster observed by the periodic legibility census."
    ]

    evidence = cluster.get("evidence") or []
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"- {quote}" for quote in evidence)

    sightings = cluster.get("sightings") or []
    if sightings:
        lines.append("")
        lines.append(f"Observed in {len(sightings)} sighting(s) (project: {project_id}).")

    return "\n".join(lines)


def _resolve_target_project(
    cluster: dict, *, project_root: str, project_id: str, title: str,
) -> tuple[str, str]:
    """Resolve one cluster's target project_root/project_id, honoring the
    ``target_project_root``/``target_project_id`` override pair (PRD
    decision 4) as ALL-OR-NOTHING: the two name the SAME project and must
    move together, so a cluster carrying only one of the two (a malformed
    override -- e.g. a verify_fn/synthesis bug) can never mix an override
    root with the census's own id, or vice versa, and file into the wrong
    registry. A partial pair is logged and IGNORED entirely, falling back
    to the census's own *project_root*/*project_id* -- the same fail-safe
    default as no override at all (reviewer_comprehensive finding #2)."""
    has_root_override = "target_project_root" in cluster
    has_id_override = "target_project_id" in cluster
    if has_root_override != has_id_override:
        logger.warning(
            "census: cluster %r supplies only one of "
            "target_project_root/target_project_id (must move together, "
            "PRD decision 4) -- ignoring the partial override and filing "
            "into this census's own project %r instead",
            title, project_id,
        )
        return project_root, project_id
    return (
        cluster.get("target_project_root", project_root),
        cluster.get("target_project_id", project_id),
    )


def build_task_payloads(clusters, *, project_root: str, project_id: str) -> list[dict]:
    """Map each verified cluster to one curator-path ``submit_task`` kwarg
    dict. ``task_kind`` is always ``"normal"``; ``planning_mode`` is
    deliberately OMITTED (defaults False at the submit_task layer) --
    curator dedup against already-filed remediation is the point (PRD
    decision 9), and ``planning_mode`` is exactly the curator-bypassing
    path that would defeat it.

    A cluster observed in a hosted project whose root cause is
    harness-rooted may carry ``target_project_root``/``target_project_id``
    overrides to file into dark_factory instead of the census's own
    project (PRD decision 4, same fused-memory, different project_root);
    absent those, the payload targets the census's own *project_root*/
    *project_id*. The two overrides move together -- see
    ``_resolve_target_project``.

    Pure function -- returns payloads only; the actual ``submit_fn`` call
    happens in ``run_census``.
    """
    payloads = []
    for cluster in clusters:
        title = cluster.get("title") or "Untitled confusion cluster"
        target_project_root, target_project_id = _resolve_target_project(
            cluster, project_root=project_root, project_id=project_id, title=title,
        )
        payloads.append(
            {
                "project_root": target_project_root,
                "title": f"[legibility census] {title}",
                "description": _cluster_description(cluster, project_id=target_project_id),
                "task_kind": "normal",
                "priority": cluster.get("priority", "medium"),
                "metadata": {
                    "source": "legibility_census",
                    "origin_project_id": project_id,
                },
            }
        )
    return payloads


# ---------------------------------------------------------------------------
# promote_candidate / reject_candidate / retire_entry — census-only
# codebook lifecycle transforms (PRD decision 1: deterministic, in-memory,
# reuse gamma's validator + never-delete assertion + sole-writer dump;
# codebook.py itself is NOT modified)
# ---------------------------------------------------------------------------

def _find_by_id(items: list[dict], item_id: str, *, kind: str) -> dict:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise ValueError(f"{kind}: no id {item_id!r} found")


def promote_candidate(cb: dict, cand_id: str, entry_fields: dict) -> dict:
    """Promote candidate *cand_id* in *cb* to a new codebook entry.

    Returns a NEW codebook (a ``copy.deepcopy`` of *cb*, never mutated in
    place). Appends an entry built from *entry_fields* (expected to carry
    ``id``/``title``/``severity``/``status``/``origin_phase``/
    ``manifested_phase`` -- everything ``codebook.py``'s v2 entry schema
    requires except ``sightings``) with ``sightings`` set to the
    candidate's OWN sightings (the candidate is the source of truth for
    what was actually observed, not *entry_fields*). The candidate itself
    is stamped ``disposition="promoted"`` and ``promoted_to`` naming the
    new entry id -- RETAINED, never removed (PRD decision 3).
    ``codebook.assert_no_deletion`` is run against the pre-transform *cb*
    before returning, as a construction-independent safety net.
    """
    result = copy.deepcopy(cb)
    candidate = _find_by_id(result.get("candidates") or [], cand_id, kind="promote_candidate")

    new_entry_id = entry_fields["id"]
    entry = dict(entry_fields)
    entry["sightings"] = copy.deepcopy(candidate.get("sightings") or [])
    result.setdefault("entries", []).append(entry)

    candidate["disposition"] = "promoted"
    candidate["promoted_to"] = new_entry_id

    codebook.assert_no_deletion(cb, result)
    return result


def reject_candidate(cb: dict, cand_id: str) -> dict:
    """Reject candidate *cand_id* in *cb*: stamps ``disposition="rejected"``
    in place on a deep copy, RETAINED (never removed). Returns a NEW
    codebook; *cb* is never mutated."""
    result = copy.deepcopy(cb)
    candidate = _find_by_id(result.get("candidates") or [], cand_id, kind="reject_candidate")
    candidate["disposition"] = "rejected"

    codebook.assert_no_deletion(cb, result)
    return result


def retire_entry(cb: dict, entry_id: str) -> dict:
    """Retire entry *entry_id* in *cb*: stamps ``status="retired"`` in
    place on a deep copy, RETAINED (never removed). Returns a NEW
    codebook; *cb* is never mutated."""
    result = copy.deepcopy(cb)
    entry = _find_by_id(result.get("entries") or [], entry_id, kind="retire_entry")
    entry["status"] = "retired"

    codebook.assert_no_deletion(cb, result)
    return result


# ---------------------------------------------------------------------------
# advance_census_state — §7.5 census-state.json sole writer (zeta/2579
# MUST-persist contract: last_census_done_count is ALWAYS present)
# ---------------------------------------------------------------------------

def advance_census_state(
    path, *, now_iso: str, report_path: str, done_count: int | None,
) -> None:
    """Write the §7.5 census-state dict to *path*, atomically.

    Always writes exactly ``{"last_census_at": now_iso,
    "last_census_report": report_path, "last_census_done_count":
    done_count}`` -- ``last_census_done_count`` is NEVER conditionally
    omitted, even when ``done_count == 0`` (falsy but a real baseline).
    zeta's ``census_trigger.compute_tasks_landed()`` returns ``None`` --
    and its condition (b) permanently fails safe -- whenever that key is
    absent (census_trigger.py:410-416), and this module is
    census-state.json's SOLE writer, so this is the one place that
    baseline can ever be supplied.

    ``done_count`` is THREE-VALUED (task 3291):

    * a positive int -- a real observed done-count;
    * ``0`` -- also a real observed done-count, for a project with no done
      tasks. Unchanged, and still never dropped as falsy;
    * ``None`` -- the done-count could not be OBSERVED at census time
      (get_statuses unreachable, or it answered with something that was not
      a ``{"statuses": mapping}`` envelope). Serialised as JSON ``null``,
      so the key is still always present and the MUST-persist contract
      above holds literally. ``compute_tasks_landed`` treats ``null``
      exactly like an absent key, so condition (b) fails SAFE until the
      next successful census, with condition (a) (``max_interval_days``)
      remaining the unconditional backstop.

    Writing a fabricated ``0`` for an unobservable count, or carrying
    forward the previous file's value, are both FORBIDDEN here. A fabricated
    0 was written on 2026-07-24 and on 2026-07-31 (task 3291), and it is
    unsound rather than merely untidy: it turns the delta into
    ``current_done - 0`` -- every done task ever, ~2872 against a 120
    threshold. That stayed latent only while the get_statuses fetch was ALSO
    broken (the same defect zeroed ``current_done``, so the measured pre-fix
    delta was ``0 - 0``); repairing the fetch alone would have detonated it.
    See ``census_trigger``'s module docstring for the replayed measurements.
    A carried-forward stale count is merely a quieter guess, silently
    under-reporting the next window's delta. ``null`` is the only honest
    value for an unknown, and it is the caller's job to pass it rather than
    invent a number.

    Uses the same ``tempfile.mkstemp`` + ``os.replace`` atomic-write
    pattern as ``codebook.dump`` (temp file in the same directory as
    *path*, then an atomic rename): a crash or kill mid-write can never
    leave a partial state file for the next ``load_census_state`` to trip
    over, and a pre-existing state file is fully replaced, never merged.
    """
    state = {
        "last_census_at": now_iso,
        "last_census_report": report_path,
        "last_census_done_count": done_count,
    }
    directory = os.path.dirname(os.fspath(path)) or "."
    # WRITER-SIDE parent creation, mirroring trickle_state.py's atomic
    # writer -- deliberately belt-and-braces with run_census's own
    # _ensure_output_parents, which serves a different purpose. That call
    # buys FAIL-FAST (an un-creatable output path must cost nothing rather
    # than a whole ~$100 mining run) and only covers run_census's four
    # paths; this line makes the guarantee hold for EVERY caller of this
    # function, present and future, so nobody can reintroduce the
    # FileNotFoundError that mkstemp(dir=<missing>) raises just by writing
    # census state from somewhere new. Idempotent, so the two do not fight.
    os.makedirs(directory, exist_ok=True)
    fd, tmp_file = tempfile.mkstemp(prefix=".census-state-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_file, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_file)
        raise


# ---------------------------------------------------------------------------
# render_report — dated plans/confusion-census-<date>.md markdown assembly
# ---------------------------------------------------------------------------

@dataclass
class VerifyCoverage:
    """Coverage record for the operator verify cap (``--max-verify-clusters``).

    ``novel`` is how many novel clusters this run's mining actually
    produced; ``verified`` is how many of them were handed to ``verify_fn``
    (one Sonnet call each -- the cost being bounded); ``cap`` is the
    operator cap that produced the split. The ``novel - verified``
    remainder is DEFERRED, not dropped: those clusters still merge into
    the codebook as ``pending`` candidates via the untouched
    ``codebook.apply_coding_record`` path (which consumes raw mining
    records, not verified clusters), so ``_find_pending_candidate_id`` can
    still find them.

    That pickup is CONDITIONAL, not automatic, and the report says so:
    pending candidates are absent from ``coder.build_codebook_index``
    (entries only), so a recurrence does re-emerge as a novel cluster and
    gets promoted against the pending id -- but the sightings that produced
    the deferred cluster in THIS window are never re-mined, because
    ``advance_census_state`` re-anchors the next window at this run's
    ``last_census_at``. A deferred cluster is therefore re-adjudicated only
    if the same confusion RECURS; a one-off deferred by the cap sits pending
    until a human adjudicates it.

    ``None`` in place of this record means no verify cap was used and no
    ``## Verification`` section is rendered."""

    novel: int
    verified: int
    cap: int | None = None


@dataclass
class DryRunFiling:
    """Outcome record for ``--dry-run-filing``: every would-be
    ``submit_task`` payload this run built was written to ``path`` as JSON
    for human review, and NOTHING was filed into a live task tree.

    ``payload_count`` is how many payloads the file holds. Reused by both
    ``render_report`` (which must not let an empty ``filed_task_ids`` read
    as a normal run that filed nothing) and ``CensusOutcome`` (so
    ``main``'s summary line can name the review file instead of printing a
    misleading ``filed_tasks=0``). ``None`` in place of this record means
    the run filed normally."""

    path: str
    payload_count: int


def render_report(
    *,
    date: str,
    project_id: str,
    force: bool,
    matrix_md: str,
    mining_result: MiningResult,
    synthesis_md: str,
    filed_task_ids: list[str],
    cost_note: str,
    verify_coverage: VerifyCoverage | None = None,
    dry_run: DryRunFiling | None = None,
) -> str:
    """Assemble the dated census report as markdown, purely from the
    pieces passed in -- no clock, no model call, no I/O. *date* and every
    piece of LLM-produced prose (*synthesis_md*, *matrix_md*) are inputs,
    so the same inputs always render byte-identical output.

    NO SILENT CAPS: when the operator bounded this run, the report says
    so in as many words. The batch-cap coverage lines in ``## Saturation``
    are rendered ONLY when ``mining_result.max_batches`` is not None, so a
    FLAGLESS run's output is byte-identical to what it was before the
    operator cost-control flags existed (locked by
    ``test_render_report_flagless_output_is_byte_identical_golden``). The
    same gating applies to every other cost-control rendering here.
    """
    lines = [f"# confusion census {date}", "", f"Project: {project_id}"]

    if force:
        lines.append("")
        lines.append("_--force: operator-initiated run._")

    lines.append("")
    lines.append("## Saturation")
    lines.append("")
    lines.append(f"- batches: {len(mining_result.batch_stats)}")
    lines.append(f"- stop reason: {mining_result.stop_reason}")
    if mining_result.max_batches is not None:
        # Deliberately states only counts this function was actually handed:
        # the total number of ENUMERATED sessions is not knowable here
        # (batch_source is a generic injected iterable), and claiming
        # "X of Y" would assert a number the code never measured.
        sessions = sum(stats.total for stats in mining_result.batch_stats)
        if mining_result.stop_reason == "capped":
            lines.append(
                f"- coverage: mined {sessions} session digest(s) across "
                f"{len(mining_result.batch_stats)} batch(es); operator batch cap = "
                f"{mining_result.max_batches} batch(es) -- mining was BOUNDED BY THE CAP, "
                "not run to saturation: sessions beyond the cap were NOT mined, so this "
                "census is PARTIAL coverage, not a full sweep."
            )
            # PARTIAL is not the same as "the rest comes later" -- say which
            # one this is. run_census always calls advance_census_state, and
            # _census_window_dates anchors the NEXT window at last_census_at,
            # so the capped-away sessions fall outside every future window.
            lines.append(
                "- NOT PICKED UP LATER: this run still advances last_census_at, so the "
                "next census window starts here -- the capped-away sessions fall outside "
                "it and are never re-enumerated. Sweeping them means rolling "
                "last_census_at back in docs/legibility/census-state.json before the "
                "next run; a plain re-run will not reach them."
            )
        else:
            lines.append(
                f"- operator batch cap: {mining_result.max_batches} batch(es) "
                f"(not reached -- mining stopped by: {mining_result.stop_reason})"
            )
    for stats in mining_result.batch_stats:
        lines.append(
            f"  - batch {stats.index}: dup_rate={stats.dup_rate:.2f} "
            f"(total={stats.total}, succeeded={stats.succeeded}, failed={stats.failed}, "
            f"saturated={stats.saturated})"
        )

    if verify_coverage is not None:
        deferred = verify_coverage.novel - verify_coverage.verified
        lines.append("")
        lines.append("## Verification")
        lines.append("")
        if deferred > 0:
            lines.append(
                f"- verified {verify_coverage.verified} of {verify_coverage.novel} novel "
                f"clusters (operator verify cap: {verify_coverage.cap}); {deferred} deferred "
                "as pending candidates -- merged into the codebook by this run but NOT "
                "verified; adjudication deferred to a later census."
            )
            # Mirrors the batch-cap disclosure above: "a later census" is
            # conditional, not automatic. This window's sightings are not
            # re-mined (last_census_at re-anchors), so a deferred cluster is
            # re-adjudicated only when the same confusion shows up again.
            lines.append(
                "- a deferred candidate is re-adjudicated only if the same confusion "
                "RECURS in a later window: this run advances last_census_at, so these "
                "sightings are never re-mined. A one-off deferred by the cap stays "
                "pending until it is adjudicated by hand."
            )
        else:
            # A cap that was SET BUT NOT REACHED must not emit the deferral
            # clause -- nothing was deferred and nothing went unverified.
            lines.append(
                f"- verified all {verify_coverage.novel} novel cluster(s); operator "
                f"verify cap: {verify_coverage.cap} (not reached)."
            )

    lines.append("")
    lines.append("## Origin x Manifestation Matrix")
    lines.append("")
    lines.append(matrix_md)

    lines.append("## Synthesis")
    lines.append("")
    lines.append(synthesis_md)

    lines.append("")
    lines.append("## Filed Tasks")
    lines.append("")
    if dry_run is not None:
        # Checked FIRST: under --dry-run-filing, filed_task_ids is empty by
        # construction, and the plain "_none filed._" placeholder would read
        # as a normal run that simply had nothing to file.
        lines.append(
            f"_dry-run: {dry_run.payload_count} payload(s) written to {dry_run.path} "
            "-- NOTHING filed; review before filing._"
        )
    elif filed_task_ids:
        lines.extend(f"- {task_id}" for task_id in filed_task_ids)
    else:
        lines.append("_none filed._")

    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(cost_note)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# run_census — full orchestration: preflight -> mine -> verify -> synthesize
# -> matrix -> codebook update -> file tasks -> report -> advance state
# ---------------------------------------------------------------------------

def _novel_clusters(records: list[dict]) -> list[dict]:
    """Build one verification cluster per DISTINCT candidate title carried
    by a non-duplicate mining record (``is_duplicate`` False -- a candidate
    is itself the novelty signal, so every candidate on a novel record
    becomes its own cluster). Each cluster is shaped to satisfy both
    ``compute_matrix`` (a ``sightings`` list, each carrying
    ``origin_phase``/``manifested_phase``) and ``build_task_payloads``
    (``title``/``summary``/``evidence``/``sightings``) without needing to
    re-consult the source record. Duplicate records (zero candidates)
    contribute nothing here -- they still flow into the codebook merge via
    their ``matches``, just not into verification.

    Two defensive skips keep verification aligned with what
    ``codebook.apply_coding_record`` will actually do with these same
    records at merge time (mirrors its own title-keyed grouping,
    codebook.py:494):

    - A candidate with no title (or an empty one) is SKIPPED -- a title is
      the merge's sole grouping key, so a titleless candidate has nothing
      to resolve a promoted/rejected verdict back to.
    - A title already seen earlier in *records* is also SKIPPED
      (first-occurrence wins) -- ``apply_coding_record`` collapses every
      new candidate sharing a title into the SAME pending codebook
      candidate, so a second cluster for that title would spend a
      redundant verify_fn call on a candidate id that can never be found
      "pending" a second time (``_find_pending_candidate_id`` would return
      None for it, silently dropping its promote/reject)."""
    seen_titles: set[str] = set()
    clusters = []
    for record in records:
        if is_duplicate(record):
            continue
        session = record.get("session")
        for candidate in record.get("candidates") or []:
            title = candidate.get("title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            origin_phase = candidate.get("origin_phase") or "unknown"
            manifested_phase = candidate.get("manifested_phase") or "unknown"
            evidence_quote = candidate.get("evidence_quote")
            clusters.append(
                {
                    "title": title,
                    "summary": candidate.get("cause") or title,
                    "cause": candidate.get("cause"),
                    "area": candidate.get("area"),
                    "origin_phase": origin_phase,
                    "manifested_phase": manifested_phase,
                    "evidence": [evidence_quote] if evidence_quote else [],
                    "sightings": [
                        {
                            "session": session,
                            "origin_phase": origin_phase,
                            "manifested_phase": manifested_phase,
                        }
                    ],
                }
            )
    return clusters


def _find_pending_candidate_id(cb: dict, title: str | None) -> str | None:
    """Locate a still-``pending`` candidate in *cb* by *title* -- the same
    key ``codebook.apply_coding_record`` groups new candidates by, so a
    verified/rejected cluster (built pre-merge from a raw mining record) can
    be resolved to the REAL candidate id the merge just assigned it. Returns
    ``None`` if no such pending candidate exists (defensive -- should not
    happen for a title that came from this run's own mining records)."""
    for candidate in cb.get("candidates") or []:
        if candidate.get("title") == title and candidate.get("disposition") == "pending":
            return candidate.get("id")
    return None


def _free_payloads_path(path: Path, *, limit: int = 1000) -> Path:
    """Return *path* if it is free, else the first unused numbered sibling
    (``{stem}-2{suffix}``, ``{stem}-3{suffix}``, ...).

    A ``--dry-run-filing`` payload file is a human-review deliverable AND
    the only remaining handle on that run's remediation work -- the run
    already advanced the codebook and census-state, so nothing can
    regenerate it (see the dry-run WARNING in ``run_census``). A second
    dry run on the same date must therefore never overwrite the first.

    The probe is bounded by *limit*; exhausting it raises ``RuntimeError``
    naming the directory. Raising here is safe precisely because this
    write precedes ``codebook.dump``/``advance_census_state`` -- the
    module's ordering invariant means an abort at this point leaves
    nothing persisted, so a re-run starts clean.
    """
    if not path.exists():
        return path
    for n in range(2, limit + 1):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"census: could not find a free dry-run payload path near {path} -- "
        f"{limit} numbered siblings already exist in {path.parent}; clear out "
        "the reviewed ones before running another dry-run census"
    )


def _ensure_output_parents(*paths) -> None:
    """Create the parent directory of every non-``None`` output path."""
    for path in paths:
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)


_VALID_ENTRY_SEVERITIES = ("high", "medium", "low")
"""codebook.py's own ``_ENTRY_SCHEMA["severity"]`` enum, duplicated here
since codebook.py is NOT modified by this module (see the module
docstring's "Codebook handling" section). ``run_census`` clamps an
untrusted verify_fn-returned cluster's severity to this closed set before
it ever reaches ``codebook.validate()`` -- an out-of-enum value (e.g. an
escalation-style ``"critical"``/``"urgent"``, valid elsewhere in this
codebase but NOT in the codebook's own severity enum) would otherwise fail
validation deep into the pipeline, after mining/verify/synthesis work is
already spent (reviewer_comprehensive finding #3)."""


@dataclass
class CensusOutcome:
    """Outcome of ``run_census``. ``status`` is ``"deferred"`` (headroom
    preflight failed -- no further work was done) or ``"done"`` (the full
    pipeline ran to completion)."""

    status: str
    reason: str | None = None
    report_path: str | None = None
    filed_task_ids: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    dry_run: DryRunFiling | None = None


def run_census(
    *,
    batch_source,
    invoke,
    verify_fn,
    synthesize_fn,
    submit_fn,
    escalate_fn,
    status_fetcher,
    commit,
    codebook_dict: dict,
    config,
    project_root: str,
    project_id: str,
    codebook_path,
    census_state_path,
    report_path,
    date: str,
    force: bool = False,
    max_batches: int | None = None,
    max_verify_clusters: int | None = None,
    dry_run_payloads_path: str | Path | None = None,
) -> CensusOutcome:
    """Run one periodic legibility census end to end.

    Every LLM / MCP / git side effect is one of the seam parameters above
    -- this function performs no I/O and calls no model except through
    them. *config* is a ``config.LegibilityConfig`` (``config.census.
    saturation`` drives the mining stop condition; ``config.models.*``
    drives the ratified static model routing).

    Order: ``preflight_headroom`` first (PRD decision 5 -- one tiny probe,
    Haiku-tier per ``config.models.trickle``). A failed probe DEFERS the
    whole census: files one INFO escalation via *escalate_fn* naming the
    deferral, logs loudly, and returns immediately -- no batch is pulled
    from *batch_source*, no ``submit_fn``/``codebook.dump``/
    ``advance_census_state`` call happens. Otherwise falls through to the
    happy path: ``mine_to_saturation`` (Sonnet miners, ``census_miner``) ->
    split records into duplicates vs novel clusters -> ``verify_fn``
    (Sonnet, ``census_verify``, asserts observations vs current main, never
    diagnoses -- lesson guards-assert-unverified-diagnoses) ->
    ``synthesize_fn`` (Fable, ``census_synthesis`` -- clustering/prose ONLY
    here) -> ``compute_matrix``/``render_matrix`` over verified sightings ->
    merge every mining record via ``codebook.apply_coding_record`` -> apply
    ``promote_candidate``/``reject_candidate`` for verified/rejected
    clusters (resolved to the merge's real candidate ids by title, severity
    clamped to the codebook's own enum -- an untrusted out-of-enum severity
    from *verify_fn* would otherwise fail ``codebook.validate`` deep into
    the pipeline) and ``retire_entry`` for any entry ids *verify_fn*
    reports fixed -> ``codebook.validate`` (raises and aborts BEFORE
    anything is written, on an invalid merge) -> ``build_task_payloads`` +
    *submit_fn* per payload, best-effort (a raised exception, or a result
    with no usable id, is logged and excluded from ``filed_task_ids``
    rather than aborting the run or inflating the filed-task count) ->
    ``render_report`` -> write the report to *report_path* ->
    ``codebook.dump`` -> ``advance_census_state`` (done-count from
    *status_fetcher*) -> best-effort *commit* of report + codebook + state.

    The report write, ``codebook.dump``, and ``advance_census_state`` run
    in that fixed order, with nothing else in between the latter two, to
    keep as small as possible the window in which a mid-pipeline failure
    could leave on-disk state inconsistent: a failure before
    ``codebook.dump`` leaves at most the report written (harmless -- a
    plain re-run regenerates it); a failure inside ``advance_census_state``
    itself (the one residual risk -- e.g. a disk-full atomic-write failure)
    leaves the codebook advanced ahead of census-state.json on disk.
    Recovery for that residual case is a plain re-run:
    ``codebook.apply_coding_record`` dedups merged sightings/candidates by
    session, and a promoted/rejected candidate is no longer ``"pending"``,
    so re-mining the same window is idempotent rather than double-counted.
    A storm batch encountered during mining (see ``mine_to_saturation``) is
    logged loudly and called out in the report's cost note rather than
    silently folded into a clean-looking result.

    OPERATOR COST CONTROL. *max_batches* bounds mining to that many
    batches (``--max-batches``). It is the reusable spend brake for a run
    that cannot rely on saturation to bound itself -- most sharply a FIRST
    census against an empty codebook, where every batch's ``dup_rate``
    only measures "the miner found nothing to match" and mining therefore
    runs to source exhaustion. It defaults to ``None`` = today's
    behavior: unbounded, no extra rendering. A capped run is deliberately
    PARTIAL coverage and never pretends otherwise: the report states the
    cap and says so in as many words, and ``CensusOutcome.stop_reason``
    is ``"capped"`` -- distinct from the ``"exhausted"`` a source that
    genuinely ran dry produces. PARTIAL here does NOT mean "the rest is
    picked up next time": this run still calls ``advance_census_state``,
    and ``_census_window_dates`` anchors the next window at
    ``last_census_at``, so the capped-away sessions fall outside every
    future window and are never re-enumerated. Sweeping them means rolling
    ``last_census_at`` back in ``docs/legibility/census-state.json`` first;
    the report bullet says exactly that, so a bounded run cannot be read as
    a deferred-but-recoverable one. A cap below 1 raises ``ValueError``
    (see ``mine_to_saturation``) rather than degrading into a half-applied
    cap.

    *max_verify_clusters* bounds the per-cluster verification spend
    (``--max-verify-clusters``) to the first N novel clusters in mining
    order. Selection is a plain ``[:N]`` slice of ``_novel_clusters``'s
    own first-occurrence-wins ordering -- deterministic given the run,
    with no invented "most important first" heuristic (any such ranking
    would need a signal the census does not have pre-verification). The
    deferred remainder is NOT dropped: the codebook merge below consumes
    the raw ``mining_result.records``, not the verified clusters, so a
    deferred cluster still lands as a ``pending`` candidate that
    ``_find_pending_candidate_id`` can resolve later. That later pickup is
    CONDITIONAL, though, and the report says so: this run advances
    ``last_census_at``, so this window's sightings are never re-mined, and
    a deferred cluster is re-adjudicated only if the same confusion RECURS
    in a later window (pending candidates are absent from
    ``coder.build_codebook_index``, so a recurrence does re-emerge as novel
    and promote against the pending id). A one-off deferred by the cap sits
    pending until a human adjudicates it. What a deferred cluster
    unconditionally forgoes is this run's adjudication: the matrix and the
    synthesis necessarily cover only the VERIFIED subset, and the report's
    ``## Verification`` section states that split rather than letting a
    bounded run read as a complete one. A cap below 1 raises ``ValueError``
    up front -- a negative cap would slice from the END of the cluster list
    rather than bounding it, which is precisely the silently mis-honored cap
    this flag exists to make impossible.

    *dry_run_payloads_path* switches filing to review mode
    (``--dry-run-filing``): every would-be ``submit_task`` payload is
    written there as JSON for a human to read, *submit_fn* is never
    called, and ``filed_task_ids`` stays empty. ONLY the external filing
    is stubbed -- mining, verification, synthesis, the matrix, the
    codebook merge and promotions, the report write, ``codebook.dump``
    and ``advance_census_state`` all proceed exactly as on a normal run,
    so the payload file is a faithful preview of what a real run would
    file rather than the output of a half-executed census. The write sits
    at the same point in the sequence the filing loop occupies (after
    ``build_task_payloads``, before ``codebook.dump``), preserving the
    ordering invariant above, and the file is appended to the best-effort
    *commit* paths -- a dry run's deliverable IS the payload file, so it
    is versioned alongside the report and codebook it came from.

    A dry run is consequently NOT resumable by re-running: the mining, the
    codebook merge, the promotions, ``codebook.dump`` and
    ``advance_census_state`` all really happened, so a later census codes
    these same confusions as ``matches`` (no novel clusters, empty
    payloads) over a window that has re-anchored at this run's
    ``last_census_at``. The payload file is the sole remaining handle on
    the remediation work, which is why it is never overwritten: a
    colliding path is left untouched and the payloads go to a numbered
    sibling instead (see ``_free_payloads_path``), so ``dry_run.path`` --
    not the requested path -- is what the report, the commit paths and
    ``CensusOutcome`` name.
    """
    # Validated BEFORE the headroom probe spends anything: a nonsense cap on a
    # flag whose entire purpose is to be an explicit, legible bound must be
    # rejected outright, never half-applied. Left unchecked,
    # max_verify_clusters=-1 would slice novel_clusters[:-1] -- silently
    # verifying all but the LAST cluster and reporting cap=-1 as if honored.
    if max_batches is not None and max_batches < 1:
        raise ValueError(
            f"max_batches must be >= 1 when set, got {max_batches!r} -- omit it "
            "entirely for unbounded mining"
        )
    if max_verify_clusters is not None and max_verify_clusters < 1:
        raise ValueError(
            f"max_verify_clusters must be >= 1 when set, got {max_verify_clusters!r} "
            "-- a negative cap would slice novel_clusters[:N] from the END rather "
            "than bounding it; omit it entirely to verify every novel cluster"
        )

    headroom = preflight_headroom(invoke, model=config.models.trickle)
    if not headroom.ok:
        reason = headroom.reason or "headroom preflight failed"
        logger.warning("census deferred: %s", reason)
        escalate_fn(
            category="infra_issue",
            severity="info",
            summary=f"legibility census deferred: {reason}",
            detail=reason,
        )
        return CensusOutcome(status="deferred", reason=reason)

    # Every output parent, created ONCE, here -- not at each of the four
    # write sites below. Three reasons, in order of how much they cost when
    # ignored:
    #
    # (1) FAIL-FAST. On 2026-08-03 a census of /home/leo/src/reify mined to
    #     saturation (~12.5h, ~$100) and only THEN died on its first output
    #     write: `[Errno 2] No such file or directory:
    #     '/home/leo/src/reify/plans/confusion-census-2026-08-02-payloads.json'`
    #     -- reify simply has no plans/ directory. Creating (or failing to
    #     create) the parents before mining means an un-creatable output path
    #     costs nothing instead of a whole mining run.
    # (2) ONE PLACE. report_path, the dry-run payloads, codebook_path and
    #     census_state_path are four writers of the SAME run's outputs; a
    #     mkdir at each is lockstep duplication (INV-5) that drifts the moment
    #     a fifth output is added. Note this covers codebook.dump and
    #     advance_census_state too, not just the two write_text calls: both
    #     write atomically via `tempfile.mkstemp(dir=os.path.dirname(path))`,
    #     which raises the same FileNotFoundError on a missing directory as
    #     write_text does. reify tripped only the plans/ pair because main()
    #     loads its config from <root>/docs/legibility/legibility.yaml, so
    #     that directory necessarily existed; a target reached via --config
    #     pointing elsewhere trips the codebook/state pair identically.
    #     dark-factory's own plans/ + docs/legibility/ are the ONLY reason
    #     none of this was ever hit before.
    #     This "one place" argument is honest only for run_census's OWN four
    #     paths, so it is not the whole guard: advance_census_state also
    #     creates its parent writer-side (mirroring trickle_state), which
    #     covers every OTHER caller too. codebook.dump has no such
    #     writer-side guard yet -- codebook.py sits outside this task's lock
    #     scope -- so nightly.py's codebook.dump call remains exposed to the
    #     same FileNotFoundError. Follow-up, not fixed here.
    # (3) AFTER THE GATE. Sitting below the headroom-defer return rather than
    #     at the top of run_census keeps the DEFER branch side-effect-free --
    #     a deferred census creates no empty directories in the target
    #     project.
    #
    # Creating an empty directory is not persisted census state, so the
    # documented report -> codebook.dump -> advance_census_state ordering
    # invariant (see this function's docstring and the comments at those
    # three sites) is untouched by this call.
    _ensure_output_parents(
        report_path, codebook_path, census_state_path, dry_run_payloads_path,
    )

    mining_result = mine_to_saturation(
        batch_source,
        codebook_dict,
        project=project_id,
        model=config.models.census_miner,
        config=config.census.saturation,
        invoke=invoke,
        max_batches=max_batches,
    )

    novel_clusters = _novel_clusters(mining_result.records)
    if max_verify_clusters is None:
        clusters_to_verify = novel_clusters
        verify_coverage = None
    else:
        clusters_to_verify = novel_clusters[:max_verify_clusters]
        verify_coverage = VerifyCoverage(
            novel=len(novel_clusters),
            verified=len(clusters_to_verify),
            cap=max_verify_clusters,
        )
        deferred_count = len(novel_clusters) - len(clusters_to_verify)
        if deferred_count:
            logger.warning(
                "census: operator verify cap (--max-verify-clusters=%d) reached -- "
                "%d of %d novel cluster(s) DEFERRED, not verified this run; they still "
                "merge into the codebook as pending candidates (deferred, never "
                "dropped), but a later census re-adjudicates them only if the same "
                "confusion RECURS -- this run advances last_census_at, so these "
                "sightings are not re-mined",
                max_verify_clusters, deferred_count, len(novel_clusters),
            )

    verify_result = verify_fn(clusters_to_verify, model=config.models.census_verify) or {}
    verified = verify_result.get("verified") or []
    rejected = verify_result.get("rejected") or []
    fixed_entry_ids = verify_result.get("fixed") or []

    # DETECTOR for a systemic verifier failure. Scoping the subprocess cwd
    # removed one CAUSE of the 2026-08-03 silent mass rejection; it is not a
    # detector for the class. _build_default_verify_fn fails CLOSED per
    # cluster -- correctly, an unverifiable claim must reject rather than
    # crash -- so ANY systemic failure (model unreachable mid-run, a
    # different permission denial, a persistent parse failure) still
    # presents as an ordinary all-rejected run, and the report below states
    # zero verified clusters in the same voice it would use for a genuinely
    # unremarkable census. "clusters were offered and NONE survived" is the
    # observable signature of that incident, so say it out loud.
    #
    # Not an error and not a defer: an all-rejected run is legitimately
    # possible, so this must not fail a census whose mining is already
    # paid for. The escalate_fn call is wrapped because, unlike the
    # defer-path escalation above (which returns immediately afterwards),
    # this one sits between the mining spend and the output writes -- a
    # raising escalate_fn must not be what discards the run's results.
    if clusters_to_verify and not verified:
        suspect = (
            f"census: ALL {len(clusters_to_verify)} verified-candidate cluster(s) were "
            "REJECTED and none survived -- suspect a systemic verifier failure "
            "(model unreachable, tool access denied, or unparseable verdicts) rather "
            "than genuinely unfounded claims. This is the observable signature of the "
            "2026-08-03 sandbox incident, where the verify subprocess was rooted "
            "outside the censused tree and every read was permission-denied. Check "
            "the per-cluster 'verify failed' warnings above; a run with real findings "
            "is being reported as an empty census if this is systemic."
        )
        logger.warning("%s", suspect)
        try:
            escalate_fn(
                category="infra_issue",
                severity="info",
                summary=(
                    f"legibility census: all {len(clusters_to_verify)} cluster(s) rejected "
                    "-- possible systemic verifier failure"
                ),
                detail=suspect,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; the warning above is the real signal
            logger.warning("census: mass-rejection escalation failed (best-effort): %s", exc)

    synthesis_md = synthesize_fn(verified, model=config.models.census_synthesis)

    verified_sightings = [s for cluster in verified for s in (cluster.get("sightings") or [])]
    matrix_md = render_matrix(compute_matrix(verified_sightings))

    updated_codebook = codebook_dict
    for record in mining_result.records:
        updated_codebook, _stats = codebook.apply_coding_record(updated_codebook, record)

    for cluster in verified:
        cand_id = _find_pending_candidate_id(updated_codebook, cluster.get("title"))
        if cand_id is None:
            continue
        severity = cluster.get("severity")
        if severity not in _VALID_ENTRY_SEVERITIES:
            if severity is not None:
                logger.warning(
                    "census: cluster %r carries out-of-enum severity %r; "
                    "clamping to 'medium'", cluster.get("title"), severity,
                )
            severity = "medium"
        entry_fields = {
            "id": f"entry-{cand_id}",
            "title": cluster.get("title"),
            "severity": severity,
            "status": "open",
            "origin_phase": cluster.get("origin_phase") or "unknown",
            "manifested_phase": cluster.get("manifested_phase") or "unknown",
        }
        updated_codebook = promote_candidate(updated_codebook, cand_id, entry_fields)

    for cluster in rejected:
        cand_id = _find_pending_candidate_id(updated_codebook, cluster.get("title"))
        if cand_id is not None:
            updated_codebook = reject_candidate(updated_codebook, cand_id)

    for entry_id in fixed_entry_ids:
        updated_codebook = retire_entry(updated_codebook, entry_id)

    validation_errors = codebook.validate(updated_codebook)
    if validation_errors:
        raise RuntimeError(
            f"census: codebook merge produced an invalid codebook: {validation_errors}"
        )

    # Best-effort per payload -- a raised exception, or a result with no
    # usable id, is logged and EXCLUDED from filed_task_ids rather than
    # aborting the run or silently inflating the filed-task count
    # (reviewer_comprehensive finding #1: an id-less result must never
    # render as a "- None" report bullet, nor count as a genuinely-filed
    # task). Mirrors the best-effort handling used for commit() below.
    # Positioned BEFORE codebook.dump()/advance_census_state() below
    # (reviewer_comprehensive finding #4): a bug in payload construction can
    # then only abort the run before anything is persisted, never strand an
    # already-advanced codebook.
    task_payloads = build_task_payloads(verified, project_root=project_root, project_id=project_id)
    filed_task_ids = []
    dry_run_filing = None
    if dry_run_payloads_path is not None:
        # --dry-run-filing: write the payloads for human review and file
        # NOTHING. submit_fn is deliberately left untouched (not swapped for
        # a collector) so a test can assert it was never reached, and so the
        # id-less-result WARNING below can never fire for an intentional
        # operator mode.
        requested_path = Path(dry_run_payloads_path)
        resolved_path = _free_payloads_path(requested_path)
        if resolved_path != requested_path:
            logger.warning(
                "census: dry-run payload path %s already exists and was left "
                "UNTOUCHED (an earlier review artifact no re-run can "
                "regenerate) -- this run's payloads were written to %s instead",
                requested_path, resolved_path,
            )
        resolved_path.write_text(
            json.dumps(task_payloads, indent=2) + "\n", encoding="utf-8",
        )
        # The WARNING deliberately does NOT offer a repeat census as the
        # recovery path, because that path is dead: this run merges the
        # candidates into the codebook and dumps it, so the same confusions
        # code as `matches` next time, _novel_clusters() comes back empty and
        # build_task_payloads() returns [] -- and advance_census_state() moves
        # last_census_at, which _census_window_dates() anchors on, so the
        # window these payloads came from is never enumerated again. Hand
        # filing is the only remaining handle on the work; do not reintroduce
        # the "just run it again without the flag" guidance.
        logger.warning(
            "census: --dry-run-filing -- %d task payload(s) written to %s; "
            "NOTHING was filed into a live task tree. This run HAS ALREADY "
            "advanced the codebook and census-state, so filing those payloads "
            "by hand is the only remaining way to land the work: a later "
            "census will NOT re-file them (these confusions now code as "
            "matches, not candidates, and the census window has re-anchored).",
            len(task_payloads), resolved_path,
        )
        # Every downstream consumer -- the report's Filed Tasks section, the
        # commit paths and CensusOutcome -- reads the path off this one
        # object, so they all name the file actually written.
        dry_run_filing = DryRunFiling(
            path=str(resolved_path), payload_count=len(task_payloads),
        )
    else:
        for payload in task_payloads:
            try:
                submit_result = submit_fn(**payload)
            except Exception as exc:  # noqa: BLE001 - best-effort, see comment above
                logger.warning(
                    "census: submit_fn failed for payload %r: %s", payload.get("title"), exc,
                )
                continue
            task_id = submit_result.get("id") if isinstance(submit_result, dict) else None
            if task_id is None:
                logger.warning(
                    "census: submit_fn returned no usable id for payload %r (result=%r) "
                    "-- not counted as filed", payload.get("title"), submit_result,
                )
                continue
            filed_task_ids.append(task_id)

    storm_batch_indices = [s.index for s in mining_result.batch_stats if s.status == "failure"]
    if storm_batch_indices:
        logger.warning(
            "census: %d storm batch(es) (>50%% coding failures, PRD §8.6) at "
            "indices %s -- dup_rate for those batches was excluded from the "
            "saturation decision but is still recorded in the report",
            len(storm_batch_indices), storm_batch_indices,
        )

    # verify is ONE call PER CLUSTER (_build_default_verify_fn), not one call
    # total -- and the per-cluster count is precisely what an operator using
    # --max-verify-clusters reads this line to check.
    cost_note = (
        f"invoke calls: {config.models.census_miner} miner="
        f"{sum(s.total for s in mining_result.batch_stats)}, "
        f"{config.models.census_verify} verify={len(clusters_to_verify)}, "
        f"{config.models.census_synthesis} synthesis=1, "
        f"{config.models.trickle} headroom-probe=1"
    )
    if storm_batch_indices:
        cost_note += (
            f"; WARNING: {len(storm_batch_indices)} storm batch(es) at indices "
            f"{storm_batch_indices} (>50% coding failures -- degraded dup-rate "
            "signal, excluded from the saturation decision)"
        )
    report_md = render_report(
        date=date,
        project_id=project_id,
        force=force,
        matrix_md=matrix_md,
        mining_result=mining_result,
        synthesis_md=synthesis_md,
        filed_task_ids=filed_task_ids,
        cost_note=cost_note,
        verify_coverage=verify_coverage,
        dry_run=dry_run_filing,
    )
    # Written BEFORE codebook.dump()/advance_census_state() below -- a
    # failure here (e.g. a disk-full write_text) leaves nothing but this one
    # file touched (no codebook write, no state advance), so a re-run starts
    # clean rather than resuming from a partially-persisted pipeline
    # (reviewer_comprehensive finding #4).
    Path(report_path).write_text(report_md, encoding="utf-8")

    # An unobservable done-count degrades ONLY the next window's condition-(b)
    # baseline -- it must never abandon a run whose mining has already been
    # paid for and whose dated report is already on disk above. Aborting would
    # also leave last_census_at unadvanced, so condition (a) would re-fire the
    # census every single night -- a strictly more over-eager loop than any
    # this task set out to fix.
    #
    # Both the fetch and the extraction are guarded. extract_done_count is what
    # stops fused-memory's {"error", "error_type"} envelope -- which rides an
    # isError:false JSON-RPC response and so arrives here looking like a
    # perfectly good dict -- from being counted as a done-count of 0 and
    # persisted as a REAL baseline. Such a fabricated 0 arms condition (b)
    # with a delta of every-done-task-ever the moment the fetch works again
    # (task 3291; see census_trigger's module docstring for the replayed
    # measurements). See advance_census_state's docstring for why null, not 0
    # and not a carried-forward value, is the honest degradation.
    try:
        done_count = census_trigger.extract_done_count(status_fetcher())
    except Exception as exc:  # noqa: BLE001 - a bad baseline must not fail the census
        done_count = None
        logger.warning(
            "census: done-count unobservable (%s) -- persisting a null "
            "last_census_done_count; the tasks-landed trigger condition will "
            "fail safe until the next successful census",
            exc,
        )

    # codebook.dump() and advance_census_state() are adjacent on purpose
    # (reviewer_comprehensive finding #4): nothing else here can raise
    # between them, so the ONLY way census-state.json can end up lagging
    # behind an already-persisted codebook is a failure inside
    # advance_census_state itself. Should that happen, recovery is a plain
    # re-run -- see this function's docstring for why that is safe.
    codebook.dump(updated_codebook, codebook_path)
    advance_census_state(
        census_state_path, now_iso=date, report_path=str(report_path), done_count=done_count,
    )

    commit_paths = [str(report_path), str(codebook_path), str(census_state_path)]
    if dry_run_filing is not None:
        commit_paths.append(dry_run_filing.path)
    try:
        commit(paths=commit_paths, message=f"legibility census {date}")
    except Exception as exc:  # noqa: BLE001 - best-effort, never fails the census
        logger.warning("census: best-effort commit failed: %s", exc)

    return CensusOutcome(
        status="done",
        report_path=str(report_path),
        filed_task_ids=filed_task_ids,
        stop_reason=mining_result.stop_reason,
        dry_run=dry_run_filing,
    )


# ---------------------------------------------------------------------------
# Real default seams for the standalone CLI -- every one of these is
# ALWAYS replaced by a fake in this module's own test suite; nothing here
# is unit-tested directly (integration-only, exercised by a live --force
# run per this task's acceptance criterion). Mirrors nightly.py's /
# census_trigger.py's identical "module-level default_* builders behind
# the real subprocess/httpx boundary" convention.
# ---------------------------------------------------------------------------

DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
"""Root of the encoded ~/.claude/projects session-transcript tree that
inventory.enumerate_sessions scans -- distinct from a censused project's
OWN project_root (which only ever holds that project's docs/legibility/
state, not any transcripts)."""

_DEFAULT_CENSUS_LOOKBACK_DAYS = 30
"""Fallback mining-window length (days) when a project has never been
censused before (no last_census_at anchor to start the window from) -- a
first-ever census must not attempt to mine the dawn of time."""

_DEFAULT_CENSUS_BATCH_SIZE = 20
"""Sessions per mine_to_saturation batch for the real batch_source."""


def _census_window_dates(project_root, *, now: datetime) -> list[date]:
    """The list of calendar dates to enumerate sessions for: from the last
    census's `last_census_at` (read via census_trigger.load_census_state)
    through *now*, inclusive -- falling back to a fixed
    `_DEFAULT_CENSUS_LOOKBACK_DAYS`-day lookback when never censused or the
    state is malformed/unparseable (fail-safe: a bounded window, never an
    unbounded one)."""
    state_path = Path(project_root) / "docs" / "legibility" / "census-state.json"
    status, state = census_trigger.load_census_state(state_path)

    start_date = None
    if status == "ok" and state and state.get("last_census_at"):
        try:
            start_date = datetime.fromisoformat(state["last_census_at"]).date()
        except (TypeError, ValueError):
            start_date = None
    if start_date is None:
        start_date = (now - timedelta(days=_DEFAULT_CENSUS_LOOKBACK_DAYS)).date()

    end_date = now.date()
    span_days = (end_date - start_date).days
    if span_days < 0:
        return [end_date]
    return [start_date + timedelta(days=offset) for offset in range(span_days + 1)]


def _stratified_random_order(by_stratum: dict, *, rng: random.Random) -> list:
    """Interleave *by_stratum* (``{stratum: [ScoredRecord, ...]}``)
    round-robin, each stratum's own list independently shuffled first, so
    every batch mine_to_saturation draws is a representative random
    cross-section of every active stratum rather than front-loading one
    stratum's sessions ahead of another's (the "stratified-RANDOM"
    sampling this task's design decisions call for)."""
    queues = [list(records) for records in by_stratum.values()]
    for queue in queues:
        rng.shuffle(queue)

    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop())
    return order


def default_batch_source(cfg, *, projects_root, now: datetime,
                          batch_size: int = _DEFAULT_CENSUS_BATCH_SIZE, rng=None):
    """Real stratified-RANDOM batch_source: enumerate every session across
    the census window's inclusive ``[start, end]`` date range
    (:func:`_census_window_dates` first/last) in ONE walk via
    :func:`inventory.enumerate_sessions_in_range` — O(total_files), not
    O(window_days × files) — under *projects_root* matching *cfg*'s
    ``cwd_prefixes``, score + classify each (mirrors
    ``nightly.select_scored_records``'s one-pass loop), interleave into a
    random cross-stratum order, and lazily render each session to a digest
    via ``digest.build_digest`` one ``batch_size``-sized batch at a time.

    A generator: NOTHING here executes until the mining loop actually
    iterates it, so a caller than never reaches the happy path (e.g. a
    headroom-preflight defer) never enumerates a single session. A single
    session whose digest fails to render is logged and skipped -- it never
    aborts the whole batch (mirrors ``nightly.build_digests``'s per-record
    isolation).
    """
    rng = rng if rng is not None else random.Random()

    # _census_window_dates always returns a non-empty list (even its span<0
    # branch returns [end_date]), so window[0]/window[-1] are always safe.
    # The inclusive [start, end] range predicate yields the same record set as
    # the per-date union — each file matches at most one date — but walks the
    # tree ONCE (O(total_files), not O(window_days × files)).
    window = _census_window_dates(cfg.project_root, now=now)
    start_date, end_date = window[0], window[-1]
    scored = []
    for session in inventory.enumerate_sessions_in_range(
        projects_root, cfg.cwd_prefixes, start_date, end_date,
        agent_transcript_roots=inventory.resolve_agent_transcript_roots(
            cfg.project_root, cfg.agent_transcript_roots
        ),
    ):
        counts, first_turn = sampling._score_and_find_first_turn(session.path)
        stratum = sampling.classify_agent_class(first_turn, session.path)
        scored.append(sampling.ScoredRecord(session=session, stratum=stratum, counts=counts))

    by_stratum: dict[str, list] = {}
    for record in scored:
        by_stratum.setdefault(record.stratum, []).append(record)
    ordered = _stratified_random_order(by_stratum, rng=rng)

    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        digests = []
        for record in chunk:
            try:
                digests.append(
                    digest.build_digest(record.path, agent_class_override=record.stratum)
                )
            except Exception as exc:  # noqa: BLE001 - isolate one bad transcript, keep mining
                logger.warning("census: failed to build digest for %s: %s", record.path, exc)
        if digests:
            yield digests


def _verify_prompt(cluster: dict, *, project_root: str) -> str:
    """Prompt for the real Sonnet verify_fn: confirm-or-refute one novel
    cluster against *project_root*'s current main via targeted file reads.

    The CLI subprocess this prompt is delivered to runs with its cwd set to
    *project_root* (``_build_stage_invokes`` binds it). That is load-bearing
    for a reason that is NOT relative-path resolution: ``claude -p``
    SANDBOXES tool access to the cwd tree, so a verifier launched from
    anywhere else has every Read/Bash against this tree permission-denied,
    with no interactive prompt to approve (proven 2026-08-03). The
    absolute-paths instruction below is retained as belt-and-braces on top
    of that scoping, not as a substitute for it.

    Asks for an OBSERVATION, never a diagnosis (codebook lesson
    guards-assert-unverified-diagnoses)."""
    return (
        "You are the periodic-census verifier for the dark-factory "
        "agent-confusion codebook (plans/confusion-reduction-prd.md "
        "section 5.7). A trickle miner flagged the confusion cluster below "
        "as novel. Using targeted reads of " + str(project_root) + " "
        "(ABSOLUTE paths only), confirm whether this is still an "
        "OBSERVABLE fact about the CURRENT state of that tree -- never a "
        "diagnosis or a guess about root cause you cannot directly "
        "verify.\n\n"
        "Respond with STRICT JSON ONLY (no prose, no markdown fences), "
        'exactly this shape: {"verified": true|false, "reason": "..."}.\n\n'
        "=== CLUSTER ===\n" + json.dumps(cluster)
    )


def _synthesis_prompt(verified: list) -> str:
    """Prompt for the real Fable synthesize_fn: cluster + write prose for
    the dated census report from the VERIFIED findings only."""
    return (
        "You are the periodic-census synthesis writer for the dark-factory "
        "agent-confusion codebook (plans/confusion-reduction-prd.md "
        "section 5.7). Cluster the VERIFIED confusion findings below and "
        "write clear, factual prose for a dated census report -- "
        "observations only, never a diagnosis that was not itself "
        "verified.\n\n"
        "=== VERIFIED CLUSTERS ===\n" + json.dumps(verified)
    )


def _build_default_verify_fn(project_root: str, invoke):
    """Build the real ``verify_fn(clusters, *, model)`` seam: one Sonnet
    call per cluster via *invoke* (default ``coder._invoke_cli`` --
    headless ``claude -p --model``), parsed via ``coder.parse_coder_output``.
    Any per-cluster failure (invocation error, unparseable output) rejects
    that cluster rather than crashing the whole census -- a conservative
    fail-closed default for an unverifiable claim. This default never
    reports a "fixed" entry (a stronger claim than a per-cluster verify
    prompt is designed to elicit).

    That fail-closed default is correct, and it is also what made the
    2026-08-03 sandbox gap SILENT: with the CLI subprocess rooted outside
    the censused tree, every permission-denied verifier read became an
    ordinary per-cluster rejection, so a whole census mass-rejected without
    a single error surfacing. What keeps the default honest rather than
    indiscriminate is ``_build_stage_invokes`` scoping the subprocess cwd to
    *project_root* -- a rejection then means the claim really could not be
    verified, not that the verifier could not see the tree.

    Scoping the cwd removed one CAUSE, not the class: a model that goes
    unreachable mid-run, a different permission denial, or persistently
    unparseable verdicts all still land here as ordinary rejections. So
    ``run_census`` additionally DETECTS the signature -- clusters offered,
    none verified -- and says so loudly rather than quietly reporting an
    empty census."""
    def _verify_fn(clusters, *, model):
        verified, rejected = [], []
        for cluster in clusters:
            prompt = _verify_prompt(cluster, project_root=project_root)
            try:
                raw = invoke(prompt, model)
                verdict = coder.parse_coder_output(raw)
            except Exception as exc:  # noqa: BLE001 - an unverifiable claim rejects, never crashes
                logger.warning(
                    "census: verify failed for cluster %r: %s", cluster.get("title"), exc,
                )
                rejected.append(cluster)
                continue
            (verified if verdict.get("verified") else rejected).append(cluster)
        return {"verified": verified, "rejected": rejected, "fixed": []}

    return _verify_fn


def _build_default_synthesize_fn(invoke):
    """Build the real ``synthesize_fn(verified, *, model)`` seam: one
    Fable call via *invoke* clustering + writing prose for the verified
    findings. An empty *verified* list is handled without a model call --
    there is nothing to synthesize."""
    def _synthesize_fn(verified, *, model):
        if not verified:
            return "No novel, verified confusion clusters this census."
        return invoke(_synthesis_prompt(verified), model)

    return _synthesize_fn


_FUSED_MEMORY_URL_ENV_VAR = "FUSED_MEMORY_MCP_URL"
_DEFAULT_FUSED_MEMORY_URL = "http://localhost:8002"
"""Mirrors census_trigger.default_status_fetcher's identical env
var/default -- submit_task lives on the same fused-memory MCP server this
project's get_statuses call already targets."""


def _post_mcp_tool_call(url: str, tool_name: str, arguments: dict) -> dict:
    """POST one JSON-RPC ``tools/call`` envelope to *url* and unwrap the
    result via ``census_trigger._extract_tool_result`` (reused rather than
    reimplemented -- the same MCP envelope shape applies everywhere in this
    codebase). ``httpx`` is imported lazily so importing this module for its
    unit-tested pure core never needs it, and so the tests can substitute a
    stub for the real POST -- not for availability, since httpx is a direct
    dependency of ``shared`` (``httpx>=0.27``, task 2965) and this module
    runs under ``uv run --project shared``. Mirrors
    ``census_trigger.default_status_fetcher`` / ``nightly._default_poster``."""
    import httpx

    response = httpx.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        # Required by the streamable-HTTP MCP transport -- single-sourced
        # in census_trigger (already imported here) so a transport change is
        # a one-line edit, not four lockstep edits with a silent-406 risk.
        headers=census_trigger.MCP_STREAMABLE_HTTP_HEADERS,
        timeout=30.0,
    )
    response.raise_for_status()
    return census_trigger._extract_tool_result(response.json())


def default_submit_fn(**kwargs) -> dict:
    """Real curator-path ``submit_task`` poster (fused-memory MCP server) --
    build_task_payloads' payloads are forwarded verbatim as this call's
    arguments."""
    url = os.environ.get(_FUSED_MEMORY_URL_ENV_VAR, _DEFAULT_FUSED_MEMORY_URL) + "/mcp"
    return _post_mcp_tool_call(url, "submit_task", kwargs)


def _build_default_escalate_fn(cfg):
    """Build the real info-escalation poster (this project's escalation
    server, PRD decision 8's poster pattern -- mirrors
    ``nightly.post_escalation``'s envelope shape): a synthetic ``task_id``
    labels the census as its source, since it is a CLI-driven run, not a
    Taskmaster task. Best-effort -- any failure is logged and swallowed,
    since ``run_census``'s own defer path already logs loudly and the
    CLI's own exit code is the authoritative signal regardless of whether
    this POST succeeds."""
    def _escalate_fn(**kwargs) -> dict:
        url = f"http://localhost:{cfg.escalation_port}/mcp"
        arguments = {
            "task_id": f"legibility-census-{cfg.project_id}",
            "agent_role": "legibility-census",
            **kwargs,
        }
        try:
            return _post_mcp_tool_call(url, "escalate_info", arguments)
        except Exception as exc:  # noqa: BLE001 - best-effort; run_census already logged loudly
            logger.warning("census: escalation post failed (best-effort): %s", exc)
            return {}

    return _escalate_fn


def _build_default_commit(project_root):
    """Build the real best-effort git-commit seam: ``git commit --only
    <paths> -m message`` in *project_root* (CLAUDE.md's scoped-commit
    convention -- never a bare ``git commit``, never ``git stash``). A path
    git has never tracked before (this run's first-ever census-state.json
    / dated report) makes ``--only`` fail with "did not match any file";
    that one case is retried once after a scoped ``git add -- <paths>``
    (mirrors ``nightly._git_commit_docs_only``'s identical fallback).
    Raises on any other failure -- ``run_census`` already wraps this call
    in a best-effort try/except."""
    def _commit(*, paths, message) -> None:
        str_paths = [str(p) for p in paths]

        def _run():
            return subprocess.run(
                ["git", "-C", str(project_root), "commit", "--only", *str_paths, "-m", message],
                capture_output=True, text=True,
            )

        result = _run()
        if result.returncode != 0 and "did not match any file" in (result.stderr or "").lower():
            subprocess.run(
                ["git", "-C", str(project_root), "add", "--", *str_paths],
                capture_output=True, text=True,
            )
            result = _run()
        if result.returncode != 0:
            raise RuntimeError(f"git commit failed: {(result.stderr or '').strip()}")

    return _commit


# ---------------------------------------------------------------------------
# main(argv) -- CLI: `python scripts/legibility/census.py [--force]`
# ---------------------------------------------------------------------------

def _parse_cli_date(value: str) -> date:
    return date.fromisoformat(value)


def _positive_int(value: str) -> int:
    """``argparse`` ``type=`` for the operator cost caps: an int >= 1.

    A cap of 0 or a negative value is rejected at the CLI boundary (exit
    2, with a message) rather than half-applied downstream: ``--max-batches
    0`` would still code one full batch (the cap is checked after a batch
    is coded) and then render a self-contradictory coverage line, and
    ``--max-verify-clusters -1`` would slice ``novel_clusters[:-1]``,
    verifying all but the LAST cluster while reporting ``cap=-1`` as if
    honored. Both are exactly the silent, mis-honored cap these flags exist
    to make impossible -- so a nonsense value fails loud. Omitting the flag,
    not passing 0, is how you ask for no cap."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be 1 or greater, got {parsed} -- omit the flag entirely for no cap"
        )
    return parsed


def _build_stage_invokes(cfg, *, project_root):
    """Build the three per-stage ``invoke(prompt, model)`` seams, each
    carrying its OWN claude-CLI subprocess timeout from ``cfg.timeouts``
    (see ``config.Timeouts`` for the rationale — why each stage needs its
    own budget) and each scoped to *project_root* as its subprocess cwd.

    Returns ``(mining_invoke, verify_invoke, synthesis_invoke)``. Every
    census stage calls its invoke as ``invoke(prompt, model)`` with two
    positional args and no kwargs, so a ``functools.partial`` that
    pre-binds the keyword-only ``timeout`` is a drop-in ``invoke``.
    ``mining_invoke`` also backs the headroom probe (``run_census`` routes
    both through its single ``invoke`` param).

    *project_root* is bound as ``cwd`` on ALL THREE partials, not on verify
    alone. VERIFY is where the gap was proven fatal (fleet session
    census-reify-3386101, 2026-08-03): it is the only stage whose prompt
    directs the model to read the target tree, and ``claude -p`` sandboxes
    tool access to its cwd tree, so censusing a project from some other
    directory permission-denied every verifier read. Scoping only verify
    would nonetheless leave mining and synthesis silently rooted in
    whatever directory the operator launched from — an asymmetry with no
    defensible reason that a later reader would file as a bug. Binding all
    three makes "the census subprocess runs inside the censused project" a
    uniform invariant instead of a verify-only patch.

    That uniformity is NOT free, and the cost belongs on the record.
    Mining and synthesis take no tool action, but their cwd is still
    observable: ``claude -p`` assembles context from the directory it runs
    in — CLAUDE.md, ``.claude/settings.json`` (hooks included) and
    ``.mcp.json`` are all cwd-relative. Scoping the subprocess to the
    censused project therefore prepends THAT project's CLAUDE.md to every
    mining call — and mining is both the highest-volume stage (hundreds of
    calls) and the dominant cost of a ~$100 census — and can fire that
    project's session hooks inside the census subprocess. It is still the
    right trade: reading the censused project's own conventions is if
    anything more correct for mining, and the alternative is two stages
    silently rooted in an arbitrary directory. But if the added per-call
    context ever measures material, the lever is an explicit
    ``--settings``/``--strict-mcp-config``-style flag on the mining and
    synthesis partials — NOT un-scoping their cwd, which would restore the
    asymmetry this paragraph exists to rule out.

    ``coder._invoke_cli`` is looked up here at call time (inside this
    function, invoked from ``main``), never bound at import, so
    monkeypatching ``coder._invoke_cli`` in tests takes effect.
    """
    cwd = str(project_root)
    return (
        functools.partial(
            coder._invoke_cli, timeout=cfg.timeouts.census_mining_secs, cwd=cwd,
        ),
        functools.partial(
            coder._invoke_cli, timeout=cfg.timeouts.census_verify_secs, cwd=cwd,
        ),
        functools.partial(
            coder._invoke_cli, timeout=cfg.timeouts.census_synthesis_secs, cwd=cwd,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Unless ``--force``, gates on ``census_trigger.decide_for_project``
    (belt-and-braces for a manual/operator run -- the production caller,
    task epsilon's nightly trickle, only launches this entrypoint when
    zeta already fired) and no-ops -- prints the NO-FIRE reasons, exit 0 --
    when it would not fire. ``--force`` bypasses the gate entirely and is
    recorded in the report header as operator-initiated (``run_census``'s
    own ``force`` parameter).

    Otherwise builds the real default seams (headless-CLI ``invoke`` for
    mining/verify/synthesis, MCP posters for ``submit_fn``/``escalate_fn``,
    ``census_trigger.default_status_fetcher`` for the done-count baseline,
    a stratified-random ``batch_source`` over the mining window, and a
    scoped git-commit helper) and runs the full pipeline via
    ``run_census``.

    Three OPERATOR COST-CONTROL flags bound what a single run may spend,
    each defaulting to today's unbounded behavior so a flagless
    invocation -- notably the nightly trickle's, which passes no extra
    argv -- is unchanged: ``--max-batches N`` bounds mining (the
    capped-away sessions are NOT re-mined by a later census -- this run
    still advances ``last_census_at``, so the next window starts here),
    ``--max-verify-clusters N`` bounds per-cluster verification (a deferred
    cluster is re-adjudicated only if that confusion RECURS in a later
    window), and ``--dry-run-filing`` writes every would-be task payload to
    ``plans/confusion-census-<date>-payloads.json`` (alongside the dated
    report) for human review instead of filing it -- those payloads must
    then be filed by hand, since this run still advances the codebook and
    census-state and a later census will not re-file them. They are composable
    and reusable, not first-census-only, though an attended FIRST census
    against an empty codebook is where all three matter most: saturation
    cannot bound that run, since every batch's dup_rate then only
    measures "the miner found nothing to match". Both numeric caps take
    ``type=_positive_int``, so a nonsense value (0 or negative) exits 2 at
    the CLI boundary instead of being half-applied. None of the three lets
    a bounded run masquerade as a complete one, or as one whose remainder
    is automatically picked up later -- see ``run_census`` and
    ``render_report``.

    Returns non-zero only on a genuine fail-loud error (a config-load
    failure, or an uncaught exception from ``run_census``) -- a deferred
    (headroom-preflight) outcome still exits 0, mirroring
    ``census_trigger``'s own CLI contract of reserving a non-zero exit for
    an operator-facing failure, not an expected defer/no-fire outcome.

    Configures logging FIRST, before arg parsing and so before the trigger
    gate, which is what gets this module's INFO lines (the empty-codebook
    line, the per-stage progress lines) into the journal on EVERY
    invocation -- including one that no-ops at the gate. Without it root
    sits at its WARNING default and they are all silently dropped. Shares
    ``config.configure_logging`` with ``nightly.main()`` rather than
    inlining a second ``basicConfig``, so the env var
    (``LEGIBILITY_LOG_LEVEL``), the default level and the format cannot
    drift between the two entrypoints (INV-5 no-lockstep-duplication).
    """
    config.configure_logging()

    parser = argparse.ArgumentParser(
        prog="census",
        description="Legibility periodic-census runner "
        "(plans/confusion-reduction-prd.md section 5.7, task eta).",
    )
    parser.add_argument(
        "--project-root", default=".",
        help="Root of the project being censused (default: %(default)s).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to the project's legibility.yaml "
        "(default: <project-root>/docs/legibility/legibility.yaml).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the census_trigger gate (operator-initiated run).",
    )
    parser.add_argument(
        "--date", default=None, type=_parse_cli_date,
        help="Census date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--max-batches", type=_positive_int, default=None,
        help="Operator cost control: stop mining after N batches (N >= 1; omit "
        "for no cap). Omit for today's behavior -- mine until novelty saturates "
        "or the source runs out. A capped run is deliberately PARTIAL coverage "
        "and says so in the report; its stop_reason is 'capped', not "
        "'exhausted'. The capped-away sessions are NOT re-mined later: this run "
        "still advances last_census_at, so the next census window starts here.",
    )
    parser.add_argument(
        "--max-verify-clusters", type=_positive_int, default=None,
        help="Operator cost control: verify at most N novel clusters (one "
        "Sonnet call each), taken in mining order (N >= 1; omit for no cap). "
        "Omit to verify every novel cluster. The deferred remainder still "
        "merges into the codebook as pending candidates -- deferred, never "
        "dropped -- but is re-adjudicated only if the same confusion RECURS in "
        "a later window; this window's sightings are not re-mined.",
    )
    parser.add_argument(
        "--dry-run-filing", action="store_true",
        help="Operator cost control: write every would-be task payload to "
        "plans/confusion-census-<date>-payloads.json for human review and "
        "file NOTHING. Everything else (codebook update, promotions, report, "
        "census-state advance) proceeds normally -- and because the codebook "
        "and census-state DO advance, the payloads must be filed by hand; a "
        "later census will not re-file them. An existing payload file is "
        "never overwritten (the payloads go to a numbered sibling instead).",
    )
    args = parser.parse_args(argv)

    # Resolved, not used raw: --project-root defaults to "." and is routinely
    # passed relative. An unresolved relative root would make the stage cwd
    # binding below vacuous (cwd="." IS the launcher's cwd — exactly the bug
    # that binding exists to close) and would silently falsify
    # _verify_prompt's own "ABSOLUTE paths only" contract, since it
    # interpolates str(project_root) straight into the prompt. One resolve()
    # at the CLI boundary makes the prompt text, the subprocess cwd and all
    # four output paths name the same absolute tree.
    project_root = Path(args.project_root).resolve()

    # Rejected LOUDLY here, at the CLI boundary, rather than left to fail
    # somewhere downstream. Now that project_root is also the stage
    # subprocess cwd, a typo'd root surfaces on the FIRST invoke -- the
    # headroom probe -- as a CoderInvocationError, and preflight_headroom
    # deliberately folds ANY probe exception into HeadroomResult(ok=False).
    # Without this check `census --project-root /typo --config <real one>`
    # would exit 0 with "census deferred: headroom probe invocation
    # failed: ..." plus an INFO escalation: an operator typo wearing the
    # exact costume of a usage-limit defer, on every subsequent run. A
    # non-existent root is never a deferral -- it is a bad argument.
    if not project_root.is_dir():
        print(
            f"census: --project-root {args.project_root!r} resolves to "
            f"{project_root}, which is not an existing directory",
            file=sys.stderr,
        )
        return 1

    config_path = (
        Path(args.config) if args.config
        else project_root / "docs" / "legibility" / "legibility.yaml"
    )

    try:
        cfg = config.load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - a broken/missing config fails loud at CLI startup
        print(f"census: failed to load config at {config_path}: {exc}", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    date_str = args.date.isoformat() if args.date is not None else now.date().isoformat()
    status_fetcher = census_trigger.default_status_fetcher(project_root)

    if not args.force:
        decision = census_trigger.decide_for_project(
            project_root, now=now, status_fetcher=status_fetcher,
        )
        if not decision.fire:
            print(f"census: NO-FIRE for {cfg.project_id} -- pass --force to run anyway")
            for reason in decision.reasons:
                print(f"  {reason}")
            return 0

    codebook_path = project_root / "docs" / "legibility" / "confusion-codebook.yaml"
    census_state_path = project_root / "docs" / "legibility" / "census-state.json"
    report_path = project_root / "plans" / f"confusion-census-{date_str}.md"
    # Derived from report_path.parent so the human-review payload file stays
    # co-located with the dated report even if the report location moves.
    dry_run_payloads_path = (
        report_path.parent / f"confusion-census-{date_str}-payloads.json"
        if args.dry_run_filing else None
    )

    try:
        codebook_dict = codebook.load(codebook_path)
    except FileNotFoundError:
        logger.info(
            "census: no codebook at %s yet, starting from an empty v2 document", codebook_path,
        )
        codebook_dict = {"version": 2, "entries": [], "candidates": []}

    # Each census stage gets its OWN claude-CLI subprocess timeout; see
    # config.Timeouts for the rationale.
    mining_invoke, verify_invoke, synthesis_invoke = _build_stage_invokes(
        cfg, project_root=project_root,
    )

    # One escalate_fn closure shared by BOTH consumers: run_census's
    # headroom-defer path (lines ~800) and main()'s hard-failure catch-all
    # below -- so defer and hard-failure escalations share one census
    # escalation source and one never-mask-the-exit contract.
    escalate_fn = _build_default_escalate_fn(cfg)

    try:
        outcome = run_census(
            batch_source=default_batch_source(
                cfg, projects_root=DEFAULT_PROJECTS_ROOT, now=now,
            ),
            invoke=mining_invoke,
            verify_fn=_build_default_verify_fn(str(project_root), verify_invoke),
            synthesize_fn=_build_default_synthesize_fn(synthesis_invoke),
            submit_fn=default_submit_fn,
            escalate_fn=escalate_fn,
            status_fetcher=status_fetcher,
            commit=_build_default_commit(project_root),
            codebook_dict=codebook_dict,
            config=cfg,
            project_root=str(project_root),
            project_id=cfg.project_id,
            codebook_path=codebook_path,
            census_state_path=census_state_path,
            report_path=report_path,
            date=date_str,
            force=args.force,
            max_batches=args.max_batches,
            max_verify_clusters=args.max_verify_clusters,
            dry_run_payloads_path=dry_run_payloads_path,
        )
    except Exception as exc:  # noqa: BLE001 - fail loud: escalate (PRD decision 8) AND exit non-zero, never a silent crash
        print(f"census: FAILED -- {exc}", file=sys.stderr)
        # PRD decision 8 -- degradation never silent: file a best-effort
        # escalation via the shared closure so a hard failure leaves an
        # operator signal, not just a stderr line. The closure swallows all
        # POST errors internally (logging a best-effort warning), so this
        # never masks the exit and `return 1` always runs.
        escalate_fn(
            category="infra_issue",
            severity="info",
            summary=f"legibility census run failed ({cfg.project_id}): {exc}",
            detail=traceback.format_exc(),
        )
        return 1

    if outcome.status == "deferred":
        print(f"census: deferred -- {outcome.reason}")
        return 0

    if outcome.dry_run is not None:
        # A bare filed_tasks=0 here would read as "a normal run that had
        # nothing to file" -- name the review file and the count instead.
        print(
            f"census: done -- report={outcome.report_path} "
            f"dry-run-filing: {outcome.dry_run.payload_count} payload(s) -> "
            f"{outcome.dry_run.path} (nothing filed) "
            f"stop_reason={outcome.stop_reason}"
        )
        return 0

    print(
        f"census: done -- report={outcome.report_path} "
        f"filed_tasks={len(outcome.filed_task_ids)} stop_reason={outcome.stop_reason}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
