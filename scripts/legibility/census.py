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

import argparse
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
from datetime import date, datetime, timedelta, timezone
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
import sampling  # noqa: E402
from legibility import census_trigger  # noqa: E402

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
    behavior: no cap, no extra rendering, unbounded mining.

    *config* is a ``config.Saturation``-shaped object (``.dup_rate``,
    ``.consecutive_batches`` -- i.e. a project's
    ``LegibilityConfig.census.saturation``), not the whole
    ``LegibilityConfig``. *model* is the caller's already-resolved model
    id (Sonnet miner routing per the ratified static policy -- this
    function does not read ``config.Models`` itself).
    """
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

_HEADROOM_BANNER_MARKERS = (
    "usage limit",
    "rate limit",
    "please run /login",
    "invalid api key",
)
"""Case-insensitive substrings that mark a usage-limit/auth banner reply
from the headless `claude` CLI, rather than a genuine model response."""

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
    case-insensitively for a known usage-limit/auth banner marker
    (``_HEADROOM_BANNER_MARKERS``); a match defers. An invocation error
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

    lowered = (reply or "").lower()
    for marker in _HEADROOM_BANNER_MARKERS:
        if marker in lowered:
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
    path, *, now_iso: str, report_path: str, done_count: int,
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
    fd, tmp_file = tempfile.mkstemp(prefix=".census-state-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp_file, path)
    except BaseException:
        try:
            os.remove(tmp_file)
        except OSError:
            pass
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
    records, not verified clusters), so a later census picks them up
    exactly as designed. ``None`` in place of this record means no verify
    cap was used and no ``## Verification`` section is rendered."""

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
        lines.append(
            f"- verified {verify_coverage.verified} of {verify_coverage.novel} novel "
            f"clusters (operator verify cap: {verify_coverage.cap}); {deferred} deferred "
            "as pending candidates -- merged into the codebook by this run but NOT "
            "verified; adjudication deferred to a later census."
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
    genuinely ran dry produces.

    *max_verify_clusters* bounds the per-cluster verification spend
    (``--max-verify-clusters``) to the first N novel clusters in mining
    order. Selection is a plain ``[:N]`` slice of ``_novel_clusters``'s
    own first-occurrence-wins ordering -- deterministic given the run,
    with no invented "most important first" heuristic (any such ranking
    would need a signal the census does not have pre-verification). The
    deferred remainder is NOT dropped: the codebook merge below consumes
    the raw ``mining_result.records``, not the verified clusters, so a
    deferred cluster still lands as a ``pending`` candidate and
    ``_find_pending_candidate_id`` will find it on a later census exactly
    as designed. What a deferred cluster does forgo is this run's
    adjudication: the matrix and the synthesis necessarily cover only the
    VERIFIED subset, and the report's ``## Verification`` section states
    that split rather than letting a bounded run read as a complete one.
    """
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
                "merge into the codebook as pending candidates for a later census "
                "(deferred, never dropped)",
                max_verify_clusters, deferred_count, len(novel_clusters),
            )

    verify_result = verify_fn(clusters_to_verify, model=config.models.census_verify) or {}
    verified = verify_result.get("verified") or []
    rejected = verify_result.get("rejected") or []
    fixed_entry_ids = verify_result.get("fixed") or []

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

    cost_note = (
        f"invoke calls: {config.models.census_miner} miner="
        f"{sum(s.total for s in mining_result.batch_stats)}, "
        f"{config.models.census_verify} verify=1, "
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
    )
    # Written BEFORE codebook.dump()/advance_census_state() below -- a
    # failure here (e.g. a disk-full write_text) leaves nothing but this one
    # file touched (no codebook write, no state advance), so a re-run starts
    # clean rather than resuming from a partially-persisted pipeline
    # (reviewer_comprehensive finding #4).
    Path(report_path).write_text(report_md, encoding="utf-8")

    status = status_fetcher()
    done_count = sum(1 for v in (status.get("statuses") or {}).values() if v == "done")

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

    try:
        commit(
            paths=[str(report_path), str(codebook_path), str(census_state_path)],
            message=f"legibility census {date}",
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, never fails the census
        logger.warning("census: best-effort commit failed: %s", exc)

    return CensusOutcome(
        status="done",
        report_path=str(report_path),
        filed_task_ids=filed_task_ids,
        stop_reason=mining_result.stop_reason,
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
    Absolute paths only -- this function never sets a subprocess cwd, so a
    relative path in the model's own tool calls would resolve against
    whatever directory the CLI process happens to inherit. Asks for an
    OBSERVATION, never a diagnosis (codebook lesson
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
    prompt is designed to elicit)."""
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
    codebase). ``httpx`` is imported lazily since it is not a ``scripts/``
    dependency (mirrors ``census_trigger.default_status_fetcher`` /
    ``nightly._default_poster``)."""
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


def _build_stage_invokes(cfg):
    """Build the three per-stage ``invoke(prompt, model)`` seams, each
    carrying its OWN claude-CLI subprocess timeout from ``cfg.timeouts``
    (see ``config.Timeouts`` for the rationale — why each stage needs its
    own budget).

    Returns ``(mining_invoke, verify_invoke, synthesis_invoke)``. Every
    census stage calls its invoke as ``invoke(prompt, model)`` with two
    positional args and no kwargs, so a ``functools.partial`` that
    pre-binds the keyword-only ``timeout`` is a drop-in ``invoke``.
    ``mining_invoke`` also backs the headroom probe (``run_census`` routes
    both through its single ``invoke`` param).

    ``coder._invoke_cli`` is looked up here at call time (inside this
    function, invoked from ``main``), never bound at import, so
    monkeypatching ``coder._invoke_cli`` in tests takes effect.
    """
    return (
        functools.partial(coder._invoke_cli, timeout=cfg.timeouts.census_mining_secs),
        functools.partial(coder._invoke_cli, timeout=cfg.timeouts.census_verify_secs),
        functools.partial(coder._invoke_cli, timeout=cfg.timeouts.census_synthesis_secs),
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

    Returns non-zero only on a genuine fail-loud error (a config-load
    failure, or an uncaught exception from ``run_census``) -- a deferred
    (headroom-preflight) outcome still exits 0, mirroring
    ``census_trigger``'s own CLI contract of reserving a non-zero exit for
    an operator-facing failure, not an expected defer/no-fire outcome.
    """
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
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    config_path = (
        Path(args.config) if args.config
        else project_root / "docs" / "legibility" / "legibility.yaml"
    )

    try:
        cfg = config.load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - a broken/missing config fails loud at CLI startup
        print(f"census: failed to load config at {config_path}: {exc}", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
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

    try:
        codebook_dict = codebook.load(codebook_path)
    except FileNotFoundError:
        logger.info(
            "census: no codebook at %s yet, starting from an empty v2 document", codebook_path,
        )
        codebook_dict = {"version": 2, "entries": [], "candidates": []}

    # Each census stage gets its OWN claude-CLI subprocess timeout; see
    # config.Timeouts for the rationale.
    mining_invoke, verify_invoke, synthesis_invoke = _build_stage_invokes(cfg)

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

    print(
        f"census: done -- report={outcome.report_path} "
        f"filed_tasks={len(outcome.filed_task_ids)} stop_reason={outcome.stop_reason}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
